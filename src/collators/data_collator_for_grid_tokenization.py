import torch
import re
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerFast
import numpy as np

from constants.prompts import SYSTEM_PROMPT, ASSISTANT_PREFIX

class DataCollatorForGridTokenization():
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int,
        is_train: bool = True,
        is_grid_tokenization: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
        assistant_prefix: str = ASSISTANT_PREFIX,
        end_header_id: int = 128007,
        line_length: int = 1000,
        
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_train = is_train
        self.end_header_id = end_header_id
        self.system_prompt = system_prompt
        self.assistant_prefix = assistant_prefix
        self.is_grid_tokenization = is_grid_tokenization
        self.line_length = line_length
        self.table_pad_token = tokenizer.eos_token
        self.start_of_line_token = "[START_OF_LINE]"
        self.end_of_line_token = "[END_OF_LINE]"
        self.table_cell_separator_token = "[TABLE_CELL_SEPARATOR]"
        self.extension_separator_map = {
            "csv": ",",
            "html": " ",
            "tsv": "\t"
        }

        if self.is_grid_tokenization:
            new_tokens = [self.start_of_line_token, self.end_of_line_token, self.table_cell_separator_token]
            self.tokenizer.add_tokens(new_tokens)
        # Set tokenizer padding and padding side
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.tokenizer.return_tensors = "pt"
        self.tokenizer.max_length = 1024
    
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
    
    
    def _get_table_with_grid(self, context: str):
        # TODO: check how to read the table correctly
        # Remove .csv extension from the context if present
        context = re.sub(r"\.csv$", "", context)
        separator = self.extension_separator_map[self.table_extension]
        modified_lines = []

        for line in f:
            cells = line.strip().split(separator)
            modified_line = self.start_of_line_token + self.table_cell_separator_token.join(cells) + self.end_of_line_token
            modified_lines.append(modified_line.strip())
        # Join the modified lines into a single string
        return "\n".join(modified_lines)


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
    
        
    def _call_grid_tokenization(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        examples: a list of dictionaries with the following keys: question, answer, context, id, task, direction, size, table
        return a dictionary with the following keys: input_ids, attention_mask, labels
        """
        # Retrieve the table content based on the context provided in the example
        max_length = 0

        for example in examples:
            table = self._get_table_with_grid(example["table"])
            
            # Construct the user prompt using the specified order of fields
            user_prompt = str(table) + "\n" + str(example["question"])
            # Create the system and user messages for the chat template
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            # Apply the chat template to create the input string
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = self._grid_it(text)
            example["input_ids"] = input_ids
            if len(input_ids) > max_length:
                max_length = len(input_ids)

        # Batch the examples

        for i in range(len(examples["input_ids"])):
            if len(examples["input_ids"][i]) < max_length:
                examples["input_ids"][i] = [self.tokenizer.pad_token_id] * (max_length - len(examples["input_ids"][i])) + examples["input_ids"][i]

        examples["attention_mask"] = [[1] * max_length] * len(examples["input_ids"])
        return examples
    
    def _grid_it(self, text):
        # Seperate the text into before_table, table, and after_table
        table_pattern = r"(\[START_OF_LINE\].*?\[END_OF_LINE\](?:\n\[START_OF_LINE\].*?\[END_OF_LINE\])*)"
        parts = re.split(table_pattern, text, maxsplit=1)
        before_table = parts[0].strip()
        table = parts[1].strip() if len(parts) > 1 else ""
        after_table = parts[2].strip() if len(parts) > 2 else ""

        # Special token ids
        start_of_line_token = self.tokenizer.encode(self.start_of_line_token, add_special_tokens=False)[0]
        end_of_line_token = self.tokenizer.encode(self.end_of_line_token, add_special_tokens=False)[0]
        table_cell_separator_token = self.tokenizer.encode(self.table_cell_separator_token, add_special_tokens=False)[0]
        pad_token_id = self.tokenizer.encode(self.table_pad_token, add_special_tokens=False)[0]

        # before_table
        before_table_tokens = self.tokenizer.encode(before_table, add_special_tokens=False)
        before_table_pad_count = self.line_length - len(before_table_tokens) % self.line_length
        before_table_tokens.extend([pad_token_id] * before_table_pad_count)

        # table
        table_tokens = self.tokenizer.encode(table, add_special_tokens=False)
        rows = table.strip().split("\n")
        col_count = len(rows[0].split(self.table_cell_separator_token))
        row_count = len(rows)
        table_grid = np.full((row_count, self.line_length), pad_token_id, dtype=object)

        # Get the max token number per column
        token_num_per_cell = []
        token_row = []
        token_counter_in_row = 0
        token_counter_in_cell = 0
        for id in table_tokens:
            token_counter_in_row += 1
            
            if id == start_of_line_token:
                token_row = []
                token_counter_in_cell = 0
                token_counter_in_row = 1
            elif id == end_of_line_token:
                token_row.append(token_counter_in_cell)
                token_num_per_cell.append(token_row)
                token_row = []
                token_counter_in_cell = 0
            elif id == table_cell_separator_token:
                token_row.append(token_counter_in_cell)
                token_counter_in_cell = 0
            else:
                token_counter_in_cell += 1
        token_num_per_cell = np.array(token_num_per_cell)
        max_token_num_per_cell = np.max(token_num_per_cell, axis = 0)
        pad_token_count = self.line_length - sum(max_token_num_per_cell)
        if pad_token_count < 0:
            print("The token number in the row exceeds the line length")
            # TODO: discard this data
        

        # Fill the table grid
        token_col_cursor = self.line_length - 1
        token_row_cursor = row_count - 1
        cell_col_cursor = col_count - 1
        cell_inner_token_counter = 0

        for id in reversed(table_tokens):
            current_cell_token_num = max_token_num_per_cell[cell_col_cursor]
            if id == start_of_line_token:
                token_row_cursor -= 1
            elif id == end_of_line_token:
                cell_col_cursor = col_count - 1
                token_col_cursor = self.line_length - 1
                table_grid[token_row_cursor, token_col_cursor - pad_token_count + 1:] = pad_token_id
                token_col_cursor -= pad_token_count
                cell_inner_token_counter = 0
            elif id == table_cell_separator_token:
                need_to_pad_token_count = current_cell_token_num - cell_inner_token_counter
                table_grid[token_row_cursor, token_col_cursor - need_to_pad_token_count + 1 : token_col_cursor + 1] = pad_token_id
                token_col_cursor -= need_to_pad_token_count
                cell_inner_token_counter = 0
            else:
                table_grid[token_row_cursor, token_col_cursor] = id
                token_col_cursor -= 1
                cell_inner_token_counter += 1
        
        # after_table
        after_table_tokens = self.tokenizer.encode(after_table, add_special_tokens=False)
        after_table_pad_count = self.line_length - len(after_table_tokens) % self.line_length
        after_table_tokens.extend([pad_token_id] * after_table_pad_count)

        # Concatenate the three grids and flatten the result
        result = np.concatenate((before_table_tokens, table_grid.flatten(), after_table_tokens), axis=0)
        # for i in result:
        #     print(i, self.tokenizer.decode(i))
        # print(result)
        return result

        
    def __call__(self, examples: List[Dict[str, Any]]):
        """
        Columns: question, answer, context, id, task, direction, size, table
        Only use question, table, and answer
        """
        if self.is_grid_tokenization:
            return self._call_grid_tokenization(examples)
        else:
            return self._call_normal(examples)
        

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