import torch
import numpy as np
import pandas as pd
from io import StringIO
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerFast

from constants.prompts import SYSTEM_PROMPT, ASSISTANT_PREFIX

class DataCollatorForGridTokenization():
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int,
        is_train: bool = True,
        is_grid_tokenization: bool = False,
        line_length: int = 32,
        system_prompt: str = SYSTEM_PROMPT,
        assistant_prefix: str = ASSISTANT_PREFIX,
        end_header_id: int = 128007,
        
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_train = is_train
        self.end_header_id = end_header_id
        self.system_prompt = system_prompt
        self.assistant_prefix = assistant_prefix
        self.is_grid_tokenization = is_grid_tokenization
        self.line_length = line_length
        
        self.start_header_id = 128006
        self.end_header_id = 128007
        self.padding_token_id = 128002 # <|reserved_special_token_0|>
        self.table_separator_id = 128003 # <|reserved_special_token_1|>
        
        
    def _get_label(self, batch: Dict[str, torch.Tensor]):
        """
        Get the label for the batch
        Set all labels before the last occurrence of the end_header_id to -100
        """
        if not self.is_train:
            return batch
        
        batch["labels"] = batch["input_ids"].clone()
        # Set all labels before the last occurrence of the end_header_id to -100
        last_occurrence_indices = self._get_last_occurrence_indices(batch["input_ids"], self.end_header_id)
        for i in range(batch["input_ids"].size(0)):
            batch["labels"][i, :last_occurrence_indices[i] + 2] = -100
            
        return batch
    
    def _call_normal(self, examples: List[Dict[str, Any]]):
        text_list = []
        for example in examples:
            user_prompt = str(example["table"]) + "\n" + str(example["question"])
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                
            ]
            if self.is_train:
                message.append({"role": "assistant", "content": self.assistant_prefix + str(example["answer"])})
            if self.is_train:
                message_string = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            else:
                message_string = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            text_list.append(message_string)
            
        batch = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_seq_length, 
            add_special_tokens=False,
        )
        
        batch = self._get_label(batch)
            
        return batch
   
    def _get_last_occurrence_indices(self, input_ids, X):
        """
        input_ids: 2D tensor of shape B x L
        X: the integer value to find in the tensor
        """

        # Create a boolean mask where elements equal to X are True
        mask = (input_ids == X)  # Shape: B x L

        # Reverse the mask along the sequence dimension (dimension 1)
        reversed_mask = torch.flip(mask, dims=[1])  # Shape: B x L

        # Find the index of the first occurrence of True in the reversed mask
        # Convert boolean mask to float to use argmax (True becomes 1.0, False becomes 0.0)
        idx_in_reversed = reversed_mask.float().argmax(dim=1)  # Shape: B

        # Calculate the last occurrence index in the original tensor
        last_indices = input_ids.size(1) - idx_in_reversed - 1  # Shape: B

        # Handle rows where X does not occur
        # If X does not occur in a row, the entire mask row is False, and argmax returns 0
        # We need to set last_indices for these rows to -1 or any invalid index as per your requirements
        has_X = mask.any(dim=1)  # Shape: B (True if X is in the row)
        last_indices[~has_X] = -1  # Set to -1 where X does not occur

        return last_indices.unsqueeze(1)  # Shape: B x 1
        
    def __call__(self, examples: List[Dict[str, Any]]):
        """
        Columns: question, answer, context, id, task, direction, size, table
        Only use question, table, and answer
        """
        if self.is_grid_tokenization:
            return self._call_grid_tokenization(examples)
        else:
            return self._call_normal(examples)
        

    def _call_grid_tokenization(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        examples: a list of dictionaries with the following keys: question, answer, context, id, task, direction, size, table
        return a dictionary with the following keys: input_ids, attention_mask, labels
        """
        pass
    def _grid_tokenize_example(self, example: Dict[str, Any]) -> List[int]:
        """
        Tokenize an example
        """
        # System prompt
        system_prompt_tokenized = self._grid_tokenize_string(
            self.system_prompt, 
            include_header=True,
            header_content="system",
            include_eot=True,
        )
        # User prompt
        before_table_tokenized = self._grid_tokenize_string(
            "",
            include_header=True,
            header_content="user",
            include_eot=False,
        )
        # Table
        table_list = self._convert_csv_string_to_table(example["table"])
        table_tokenized = self._grid_tokenize_table(table_list)
        # After table
        after_table_tokenized = self._grid_tokenize_string(
            example["question"],
            include_header=False,
            include_eot=True,
        )
        # Assistant prompt
        if self.is_train:
            assistant_content = self.assistant_prefix + str(example["answer"])
        else:
            assistant_content = self.assistant_prefix
            
        assistant_prompt_tokenized = self._grid_tokenize_string(
            assistant_content,
            include_header=True,
            header_content="assistant",
            include_eot=True,
            put_eot_at_the_end=False,
        )
        
        return system_prompt_tokenized + before_table_tokenized + table_tokenized + after_table_tokenized + assistant_prompt_tokenized
        
        
    
    def _grid_tokenize_string(
        self, 
        string: str,
        is_begin_of_text: bool = False,
        include_header: bool = False,
        header_content: str = "",
        include_eot: bool = False,
        put_eot_at_the_end: bool = True,
        ) -> List[int]:
        """
        Grid tokenize a string.
        The output is a list of token ids with a length of multiple of line_length.
        The eot token must be the last token in the output.
        """
        # Tokenize the string, without any special tokens
        tokenized_string = self.tokenizer.encode(string, add_special_tokens=False)
        
        if is_begin_of_text:
            tokenized_string = [self.tokenizer.bos_token_id] + tokenized_string
        
        if include_header:
            if not header_content:
                raise ValueError("header_content is required when include_header is True")
            tokenized_string = [self.start_header_id] + self.tokenizer.encode(header_content, add_special_tokens=False) + [self.end_header_id, 271] + tokenized_string
            
        if include_eot:
            # Pad the tokenized string to the nearest multiple of line_length (reserve the last token for eot)
            padding_length = (self.line_length - (len(tokenized_string) + 1) % self.line_length ) % self.line_length
            if put_eot_at_the_end:
                tokenized_string = tokenized_string + [self.padding_token_id] * padding_length + [self.tokenizer.eos_token_id]
            else:
                tokenized_string = tokenized_string + [self.tokenizer.eos_token_id] + [self.padding_token_id] * padding_length
        else:
            # Pad the tokenized string to the nearest multiple of line_length
            padding_length = (self.line_length - len(tokenized_string) % self.line_length) % self.line_length
            tokenized_string = tokenized_string + [self.padding_token_id] * padding_length

        return tokenized_string
    
    def _grid_tokenize_table(self, table: List[List[str]]) -> List[int]:
        """
        Grid tokenize a table. For each row, we need to pad the tokens to the nearest multiple of line_length.
        And we also need to align each column.
        """
        # Find the max length for each column
        token_lengths = np.zeros((len(table), len(table[0])), dtype=int)
        tokenized_table = []
        for i in range(len(table)):
            tokenized_row = []
            for j in range(len(table[0])):
                tokenized_cell = self.tokenizer.encode(table[i][j], add_special_tokens=False)
                tokenized_row.append(tokenized_cell)
                token_lengths[i, j] = len(tokenized_cell)
            tokenized_table.append(tokenized_row)
        
        # Find the max length for each column
        max_lengths = np.max(token_lengths, axis=0)
        
        # Pad each cell to the max length of the column
        for i in range(len(tokenized_table)):
            for j in range(len(tokenized_table[0])):
                tokenized_table[i][j] = [self.padding_token_id] * (max_lengths[j] - len(tokenized_table[i][j])) + tokenized_table[i][j]
                if j != len(tokenized_table[0]) - 1:
                    tokenized_table[i][j].append(self.table_separator_id)

        # Concatenate the lines
        final_tokenized_table = []
        for row in tokenized_table:
            # Crop the table to line_length
            concatenated_row = []
            for cell in row:
                concatenated_row.extend(cell)
            # Pad the row line length and add a "\n" at the end
            padding_length = self.line_length - len(concatenated_row) - 1
            if padding_length > 0:
                concatenated_row = concatenated_row + [self.padding_token_id] * padding_length + [self.tokenizer.encode("\n", add_special_tokens=False)[0]]
            
            # Crop the row to line_length
            concatenated_row = concatenated_row[:self.line_length]
            final_tokenized_table.extend(concatenated_row)
            
        return final_tokenized_table
        
    def _convert_csv_string_to_table(self, csv_string: str) -> List[List[str]]:
        """
        Convert a csv string to a table, including the header
        """
        df = pd.read_csv(StringIO(csv_string))
        columns = df.columns.astype(str).tolist()
        columns = [col.replace("Unnamed: 0", "") for col in columns]
        rows = df.values.astype(str).tolist()
        return [columns] + rows
        
        
    
    