import torch
import numpy as np
import pandas as pd
from io import StringIO
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerFast

from constants.prompts import SYSTEM_PROMPT, ASSISTANT_PREFIX

class DataCollatorForCellTokenizer():
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int,
        is_train: bool = True,
        use_relative_relation_ids: bool = False,
        # column_number: int = 64,
        system_prompt: str = SYSTEM_PROMPT,
        assistant_prefix: str = ASSISTANT_PREFIX,
        table_cell_separator: str = "",
        row_offset: int = 0,
        column_offset: int = 0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_train = is_train
        self.system_prompt = system_prompt
        self.assistant_prefix = assistant_prefix
        
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.start_header_id = 128006
        self.end_header_id = 128007
        self.table_cell_separator = table_cell_separator

        self.use_relative_relation_ids = use_relative_relation_ids
        self.row_offset = row_offset
        self.column_offset = column_offset
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list = []
        row_ids_list = []
        column_ids_list = []
        segment_ids_list = []
        for example in examples:
            input_ids, row_ids, column_ids, segment_ids = self._get_relation_ids_for_example(example)
            if input_ids is None:
                continue
            input_ids_list.append(input_ids)
            row_ids_list.append(row_ids)
            column_ids_list.append(column_ids)
            segment_ids_list.append(segment_ids)

        # Pad input_ids, row_ids, column_ids, segment_ids and turn into tensors
        padding_side = "right" if self.is_train else "left"
        max_length = max([len(ids) for ids in input_ids_list])
        input_ids_list = torch.tensor(self._pad_sequence(input_ids_list, padding_value=self.tokenizer.pad_token_id, padding_side=padding_side, max_length=max_length), dtype=torch.long)
        row_ids_list = torch.tensor(self._pad_sequence(row_ids_list, padding_value=-100, padding_side=padding_side, max_length=max_length), dtype=torch.long)
        column_ids_list = torch.tensor(self._pad_sequence(column_ids_list, padding_value=-100, padding_side=padding_side, max_length=max_length), dtype=torch.long)
        segment_ids_list = torch.tensor(self._pad_sequence(segment_ids_list, padding_value=0, padding_side=padding_side, max_length=max_length), dtype=torch.long)
        
        # Get attention_mask
        attention_mask = torch.ones_like(input_ids_list)
        attention_mask[input_ids_list == self.tokenizer.pad_token_id] = 0
        if not self.is_train:
            last_eot_indices = self._get_occurrence_indices(input_ids_list, self.tokenizer.eos_token_id, is_last=True)
            assert (last_eot_indices == last_eot_indices[0]).all()
            # Remove every token (including the <|end_of_text|> token) after the last <|end_of_text|> token
            input_ids_list = input_ids_list[:, :last_eot_indices[0]]
            row_ids_list = row_ids_list[:, :last_eot_indices[0]]
            column_ids_list = column_ids_list[:, :last_eot_indices[0]]
            segment_ids_list = segment_ids_list[:, :last_eot_indices[0]]
            attention_mask = attention_mask[:, :last_eot_indices[0]]

        # Get labels
        labels = input_ids_list.clone()
        if self.is_train:
            # Set all labels before the last occurrence of the end_header_id to -100
            last_header_indices = self._get_occurrence_indices(input_ids_list, self.end_header_id, is_last=True)
            for i in range(input_ids_list.size(0)):
                labels[i, :last_header_indices[i] + 2] = -100
            labels[attention_mask == 0] = -100
            
        if self.is_train:
            return {"input_ids": input_ids_list, "attention_mask": attention_mask, "row_ids": row_ids_list, "column_ids": column_ids_list, "segment_ids": segment_ids_list, "labels": labels}
        else:
            return {"input_ids": input_ids_list, "attention_mask": attention_mask, "row_ids": row_ids_list, "column_ids": column_ids_list, "segment_ids": segment_ids_list}


    def _pad_sequence(self, sequence_list: np.ndarray, padding_value: int, padding_side: str = "right", max_length: int = None) -> np.ndarray:
        if max_length is None:
            max_length = max(len(ids) for ids in sequence_list)
        
        padded_sequences = []
        for ids in sequence_list:
            if padding_side == "right":
                padded = np.pad(ids, (0, max_length - len(ids)), constant_values=padding_value)
            else:
                padded = np.pad(ids, (max_length - len(ids), 0), constant_values=padding_value)
            padded_sequences.append(padded)
        
        return np.array(padded_sequences)


    def _get_relation_ids_for_example(self, example: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """Returns a relative (attention) relation id matrix.

            The method expects that for a given query-table structure, we have the
            following structural information in the `features`:
            * "row_ids": 0 for query, 0 for header, 1-n for other rows (where n is the
            number of rows excluding the header).
            * "column_ids": 0 for query, 1-m for the table columns (where m is the
            number of columns).
            * "segment_ids": 0 for query, 1 for table tokens.
            * "input_mask": 1 for query+table tokens.
            NB: All features are 0 for [PAD] tokens

            For example, given a query+table structure:
            Q1 Q2 A0 B0 C0
                A1 B1 C1
                A2 B2 C2 Q3 PAD1 PAD2 PAD3 PAD4
            The following features must exist:
            features['row_ids'] =
                <int32>[[0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0]]
            features['column_ids'] =
                <int32>[[0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 0, 0, 0, 0]]
            features['segment_ids'] =
                <int32>[[0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]]

        Args:
            batch: Mapping of all token type ids.

        Returns:
            a dictionary with the following keys: input_ids, row_ids, column_ids, segment_ids
        """
        # Tokenize the system prompt
        # System prompt
        before_table_input_ids, before_table_row_ids, before_table_column_ids = self._tokenize_string(
            self.system_prompt, 
            include_header=True,
            header_content="system",
            include_eot=True,
            is_begin_of_text=True,
        )

        # User prompt
        user_prompt_input_ids, user_prompt_row_ids, user_prompt_column_ids = self._tokenize_string(
            "",
            include_header=True,
            header_content="user",
            include_eot=False,
        )

        # Tokenize the table
        table = self._convert_csv_string_to_table(example["table"])
        if table is None:
            return None, None, None, None
        table_input_ids, table_row_ids, table_column_ids = self._cell_tokenize_table(table)

        # Tokenize the after table text
        after_table_input_ids, after_table_row_ids, after_table_column_ids = self._tokenize_string(
            example["question"],
            include_header=False,
            include_eot=True,
        )

        # Assistant prompt
        if self.is_train:
            assistant_content = self.assistant_prefix + str(example["answer"])
        else:
            assistant_content = self.assistant_prefix
            
        assistant_prompt_input_ids, assistant_prompt_row_ids, assistant_prompt_column_ids = self._tokenize_string(
            assistant_content,
            include_header=True,
            header_content="assistant",
            include_eot=True
        )

        # Concatenate all the tokenized ids
        input_ids = np.concatenate([before_table_input_ids, user_prompt_input_ids, table_input_ids, after_table_input_ids, assistant_prompt_input_ids])
        row_ids = np.concatenate([before_table_row_ids, user_prompt_row_ids, table_row_ids, after_table_row_ids, assistant_prompt_row_ids])
        column_ids = np.concatenate([before_table_column_ids, user_prompt_column_ids, table_column_ids, after_table_column_ids, assistant_prompt_column_ids])
        segment_ids = np.zeros((len(input_ids)), dtype=int)
        segment_ids[len(before_table_input_ids) + len(user_prompt_input_ids):len(before_table_input_ids) + len(user_prompt_input_ids) + len(table_input_ids)] = 1

        return input_ids, row_ids, column_ids, segment_ids


    def _convert_csv_string_to_table(self, csv_string: str) -> List[List[str]]:
        """
        Convert a csv string to a table, including the header
        """
        try:
            # Attempt to read the CSV
            df = pd.read_csv(StringIO(csv_string), delimiter=",", on_bad_lines="skip")
            columns = df.columns.astype(str).tolist()
            columns = [col.replace("Unnamed: 0", "") for col in columns]
            return [columns] + df.values.tolist()
        except pd.errors.ParserError as e:
            print("ParserError:", e)
            # discard this example
            return None
    

    def _cell_tokenize_table(self, table: List[List[str]]) -> List[int]:
        """
        cell tokenize a table. We just need to get the position ids of each cell.
        """
        # Tokenize the table and record the token length for each cell
        cell_token_lengths = np.zeros((len(table), len(table[0])), dtype=int)
        input_ids = []
        for i in range(len(table)):
            for j in range(len(table[0])):
                # Add a row separator"\n" if it's the last cell in the row
                row_separator = "\n" if j == len(table[0]) - 1 else ""
                tokenized_cell = self.tokenizer.encode(str(table[i][j]) + self.table_cell_separator + row_separator, add_special_tokens=False)
                input_ids.extend(tokenized_cell)
                cell_token_lengths[i, j] = len(tokenized_cell)
                
        # Build input_ids, row ids, column ids
        table_length = np.sum(cell_token_lengths)

        row_ids = np.zeros((table_length), dtype=int)
        column_ids = np.zeros((table_length), dtype=int)
        accumulated_token_counter = 0
        for i in range(len(table)):
            for j in range(len(table[0])):
                row_ids[accumulated_token_counter:accumulated_token_counter + cell_token_lengths[i, j]] = i + self.row_offset
                column_ids[accumulated_token_counter:accumulated_token_counter + cell_token_lengths[i, j]] = j + self.column_offset
                accumulated_token_counter += cell_token_lengths[i, j]

        return np.array(input_ids), row_ids, column_ids


    def _tokenize_string(
            self, 
            string: str, 
            is_begin_of_text: bool = False, 
            include_header: bool = False, 
            header_content: str = "", 
            include_eot: bool = False) -> List[int]:
        input_ids = self.tokenizer.encode(string, add_special_tokens=False)

        prefix_tokens = []
        
        if is_begin_of_text:
            prefix_tokens.append(self.tokenizer.bos_token_id)
        
        if include_header:
            if not header_content:
                raise ValueError("header_content is required when include_header is True")
            prefix_tokens.extend([self.start_header_id] + self.tokenizer.encode(header_content, add_special_tokens=False) + [self.end_header_id, 271])
        
        input_ids = prefix_tokens + input_ids

        if include_eot:
            input_ids.append(self.tokenizer.eos_token_id)

        # Build row_ids and column_ids
        input_ids = np.array(input_ids)
        row_ids = np.zeros((len(input_ids)), dtype=int)
        column_ids = np.zeros((len(input_ids)), dtype=int)
        return input_ids, row_ids, column_ids


    def _get_occurrence_indices(self, sequence_ids, X, is_last: bool = True):
        """
        sequence_ids: 2D tensor of shape B x L
        X: the integer value to find in the tensor
        is_last: if True, find the last occurrence; if False, find the first occurrence
        """

        # Create a boolean mask where elements equal to X are True
        mask = (sequence_ids == X)  # Shape: B x L

        # Reverse the mask along the sequence dimension (dimension 1)
        if is_last:
            reversed_mask = torch.flip(mask, dims=[1])  # Shape: B x L
        else:
            reversed_mask = mask

        # Find the index of the first occurrence of True in the reversed mask
        # Convert boolean mask to float to use argmax (True becomes 1.0, False becomes 0.0)
        idx_in_reversed = reversed_mask.float().argmax(dim=1)  # Shape: B

        # Calculate the last occurrence index in the original tensor
        last_indices = sequence_ids.size(1) - idx_in_reversed - 1  # Shape: B

        # Handle rows where X does not occur
        # If X does not occur in a row, the entire mask row is False, and argmax returns 0
        # We need to set last_indices for these rows to -1 or any invalid index as per your requirements
        has_X = mask.any(dim=1)  # Shape: B (True if X is in the row)
        last_indices[~has_X] = -1  # Set to -1 where X does not occur

        return last_indices.unsqueeze(1)  # Shape: B x 1
