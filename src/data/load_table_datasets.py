import os
import re
import datasets
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


        # Validate the input parameters
        self._validate_inputs()
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
        # Remove .csv extension from the context if present
        context = re.sub(r"\.csv$", "", context)
        # Read the table file with the specified extension
        with open(os.path.join(self.dataset_path, context + "." + self.table_extension), "r", encoding="utf-8") as f:
            return f.read()

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

    def _tokenize_function(self, examples):
        # Tokenize the input strings with padding and truncation
        return self.tokenizer(examples["input_string"], padding=True, truncation=True)

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

        # Preprocess each example to generate input strings
        dataset = dataset.map(self._preprocess_single_example_to_string, batched=False)

        # Set tokenizer padding and padding side
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Tokenize the dataset
        dataset = dataset.map(self._tokenize_function, batched=True, batch_size=self.batch_size)
        return dataset
