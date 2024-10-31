import os
import re
import datasets
from transformers import PreTrainedTokenizerFast
from typing import List
SYSTEM_PROMPT = "You are a helpful assistant that answers questions about the table. You only answer the question right after 'Answer: '"
ASSISTANT_PROMPT = "Answer: "
SHUFFLE_SEED = 42

def load_table_datasets(
    dataset_root: str, 
    dataset_name: str, # self_generated, wtq
    tokenizer: PreTrainedTokenizerFast, 
    batch_size: int = 4,
    table_extension: str = "html", # csv, html, tsv
    test_max_samples: int = None, 
    val_max_samples: int = None, 
    train_max_samples: int = None,
    system_prompt: str = SYSTEM_PROMPT,
    assistant_prompt: str = ASSISTANT_PROMPT,
    user_prompt_order: List[str] = ["question", "table"]
):
    """
    Load the table datasets from the given path.

    Args:
        dataset_path: The path to the dataset.
        tokenizer: The tokenizer to use for the dataset.
        table_extension: The extension of the table file.
        test_max_samples: The maximum number of samples in the test set.
        val_max_samples: The maximum number of samples in the validation set.
        train_max_samples: The maximum number of samples in the train set.
        system_prompt: The system prompt to use for the dataset.
        assistant_prompt: The assistant prompt to use for the dataset.
    
    Returns:
        A dataset with the following keys: 
        "question", "answer", "context", "id", "task", "direction", "size", "table", "text"
    """
    # Assertions
    if dataset_name not in ["self_generated", "wtq"]:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    if table_extension not in ["csv", "html", "tsv"]:
        raise ValueError(f"Invalid table extension: {table_extension}")
    if dataset_name == "self_generated":
        if test_max_samples is not None and test_max_samples % 80 != 0:
            raise ValueError("The number of samples for self-generated dataset must be a multiple of 80")
        if val_max_samples is not None and val_max_samples % 80 != 0:
            raise ValueError("The number of samples for self-generated dataset must be a multiple of 80")
        if train_max_samples is not None and train_max_samples % 80 != 0:
            raise ValueError("The number of samples for self-generated dataset must be a multiple of 80")
    
    
    dataset_path = os.path.join(dataset_root, dataset_name)
    def get_table(context: str):
        context = re.sub(f".csv$", "", context)
        with open(os.path.join(dataset_path, context + "." + table_extension), "r", encoding="utf-8") as f:
            return f.read()
    
    dataset = datasets.load_dataset("csv", data_files={
        "train": os.path.join(dataset_path, "data", "train.csv"),
        "test": os.path.join(dataset_path, "data", "test.csv"),
        "validation": os.path.join(dataset_path, "data", "val.csv")
    })
    
    # If the dataset is wtq, shuffle the dataset
    if dataset_name == "wtq":
        dataset = dataset.shuffle(seed=SHUFFLE_SEED)
    # If the dataset is self-generated, only shuffle the train set
    if dataset_name == "self_generated":
        dataset["train"] = dataset["train"].shuffle(seed=SHUFFLE_SEED)
        
    if test_max_samples is not None:
        dataset["test"] = dataset["test"].select(range(test_max_samples))
    if val_max_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(val_max_samples))
    if train_max_samples is not None:
        dataset["train"] = dataset["train"].select(range(train_max_samples))
    
    def preprocess_single_example_to_string(example):
        table = get_table(example["context"])
        example["table"] = table
        
        user_prompt = "\n".join([example[col_name] for col_name in user_prompt_order])
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + assistant_prompt
        example["input_string"] = text
        return example
    
    dataset = dataset.map(preprocess_single_example_to_string, batched=False)
    
    # Set tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def tokenize_function(examples):
        return tokenizer(examples["input_string"], return_tensors="pt", padding=True, truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True, batch_size=batch_size)
    return dataset
    