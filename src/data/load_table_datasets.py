import os
import re
import datasets
import numpy as np
from transformers import PreTrainedTokenizerFast
from typing import List

SYSTEM_PROMPT = "You are a helpful assistant that answers questions about the table. You only answer the question right after 'Answer: '"
ASSISTANT_PROMPT = "Answer: "
SHUFFLE_SEED = 42

class TableDatasetLoader:
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 4,
        table_extension: str = "html",
        test_max_samples: int = None,
        val_max_samples: int = None,
        train_max_samples: int = None,
        system_prompt: str = SYSTEM_PROMPT,
        assistant_prompt: str = ASSISTANT_PROMPT,
        user_prompt_order: List[str] = ["question", "table"],
        grid_it: bool = False,
        line_length: int = 10,
        skip_validation: bool = False
    ):
        # Initialize instance variables with given parameters
        self.dataset_root = dataset_root
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.table_extension = table_extension
        self.test_max_samples = test_max_samples
        self.val_max_samples = val_max_samples
        self.train_max_samples = train_max_samples
        self.system_prompt = system_prompt
        self.assistant_prompt = assistant_prompt
        self.user_prompt_order = user_prompt_order
        self.grid_it = grid_it
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

        # Validate the input parameters
        if not skip_validation:
            self._validate_inputs()
        if self.grid_it:
            new_tokens = [self.start_of_line_token, self.end_of_line_token, self.table_cell_separator_token]
            self.tokenizer.add_tokens(new_tokens)
            # TODO: model.resize_token_embeddings(len(tokenizer))
        # Set the path to the dataset directory 
        self.dataset_path = os.path.join(self.dataset_root, self.dataset_name)

    def _validate_inputs(self):
        # Validate dataset name
        if self.dataset_name not in ["self_generated", "wtq"]:
            raise ValueError(f"Invalid dataset name: {self.dataset_name}")
        # Validate table file extension
        if self.table_extension not in ["csv", "html", "tsv"]:
            raise ValueError(f"Invalid table extension: {self.table_extension}")
        # For self_generated datasets, ensure sample sizes are multiples of 80
        if self.dataset_name == "self_generated":
            if self.test_max_samples is not None and self.test_max_samples % 80 != 0:
                raise ValueError("The number of samples for self-generated dataset must be a multiple of 80")
            if self.val_max_samples is not None and self.val_max_samples % 80 != 0:
                raise ValueError("The number of samples for self-generated dataset must be a multiple of 80")
            if self.train_max_samples is not None and self.train_max_samples % 80 != 0:
                raise ValueError("The number of samples for self-generated dataset must be a multiple of 80")

    def _get_table(self, context: str):
        context = re.sub(f".csv$", "", context)
        with open(os.path.join(self.dataset_path, context + "." + self.table_extension), "r", encoding="utf-8") as f:
            return f.read()
    
    def _get_table_with_grid(self, context: str):
        # Remove .csv extension from the context if present
        context = re.sub(r"\.csv$", "", context)
        separator = self.extension_separator_map[self.table_extension]
        modified_lines = []

        with open(os.path.join(self.dataset_path, context + "." + self.table_extension), "r", encoding="utf-8") as f:
            for line in f:
                cells = line.strip().split(separator)
                modified_line = self.start_of_line_token + self.table_cell_separator_token.join(cells) + self.end_of_line_token
                modified_lines.append(modified_line.strip())
        # Join the modified lines into a single string
        return "\n".join(modified_lines)

    def _preprocess_single_example_to_string(self, example):
        # Retrieve the table content based on the context provided in the example
        table = self._get_table(example["context"])
        example["table"] = table

        # Construct the user prompt using the specified order of fields
        user_prompt = "\n".join([example[col_name] for col_name in self.user_prompt_order if col_name in example])
        # Create the system and user messages for the chat template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply the chat template to create the input string
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + self.assistant_prompt
        example["input_string"] = text
        return example

    def _preprocess_single_example_to_string_with_grid(self, example):
        # Retrieve the table content based on the context provided in the example
        table = self._get_table_with_grid(example["context"])
        example["table"] = table

        # Construct the user prompt using the specified order of fields
        user_prompt = "\n".join([example[col_name] for col_name in self.user_prompt_order if col_name in example])
        # Create the system and user messages for the chat template
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Apply the chat template to create the input string
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        example["input_ids"] = self._grid_it(text)
        return example

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

    def _tokenize_function(self, examples):
        # Tokenize the input strings with padding and truncation
        print(examples)
        return self.tokenizer(examples["input_string"], padding=True, truncation=True)
    

    def _batch_grid_data(self, examples):
        # Tokenize the input strings with padding and truncation
        max_length = 0
        for ids in examples["input_ids"]:
            if len(ids) > max_length:
                max_length = len(ids)

        for i in range(len(examples["input_ids"])):
            if len(examples["input_ids"][i]) < max_length:
                examples["input_ids"][i] = [self.tokenizer.pad_token_id] * (max_length - len(examples["input_ids"][i])) + examples["input_ids"][i]

        examples["attention_mask"] = [[1] * max_length] * len(examples["input_ids"])
        return examples
    

    def load(self):
        # Load the dataset from CSV files
        dataset = datasets.load_dataset("csv", data_files={
            "train": os.path.join(self.dataset_path, "data", "train.csv"),
            "test": os.path.join(self.dataset_path, "data", "test.csv"),
            "validation": os.path.join(self.dataset_path, "data", "val.csv")
        })

        # Shuffle the dataset based on the dataset name
        if self.dataset_name == "wtq":
            dataset = dataset.shuffle(seed=SHUFFLE_SEED)
        elif self.dataset_name == "self_generated":
            dataset["train"] = dataset["train"].shuffle(seed=SHUFFLE_SEED)

        # Select a subset of samples if max sample limits are specified
        if self.test_max_samples is not None:
            dataset["test"] = dataset["test"].select(range(self.test_max_samples))
        if self.val_max_samples is not None:
            dataset["validation"] = dataset["validation"].select(range(self.val_max_samples))
        if self.train_max_samples is not None:
            dataset["train"] = dataset["train"].select(range(self.train_max_samples))

        # Set tokenizer padding and padding side
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Preprocess each example to generate input strings
        if self.grid_it:
            dataset = dataset.map(self._preprocess_single_example_to_string_with_grid, batched=False)
            dataset = dataset.map(self._batch_grid_data, batched=True, batch_size=self.batch_size)
        else:
            dataset = dataset.map(self._preprocess_single_example_to_string, batched=False)
            dataset = dataset.map(self._tokenize_function, batched=True, batch_size=self.batch_size)

        return dataset


if __name__ == "__main__":
    tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", token='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')
    loader = TableDatasetLoader(dataset_root="../../datasets", dataset_name="self_generated", tokenizer=tokenizer, grid_it=True)
    dataset = loader.load()
    print(dataset)