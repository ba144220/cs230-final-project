import os
import re
import datasets
from transformers import PreTrainedTokenizerFast

SYSTEM_PROMPT = "You are a helpful assistant that can answer questions about the table."
ASSISTANT_PROMPT = "Answer: "

def load_table_datasets(
    dataset_path: str, 
    tokenizer: PreTrainedTokenizerFast, 
    batch_size: int = 4,
    table_extension: str = "html", 
    shuffle: bool = False, 
    test_max_samples: int = None, 
    val_max_samples: int = None, 
    train_max_samples: int = None,
    system_prompt: str = SYSTEM_PROMPT,
    assistant_prompt: str = ASSISTANT_PROMPT
):
    """
    Load the table datasets from the given path.

    Args:
        dataset_path: The path to the dataset.
        tokenizer: The tokenizer to use for the dataset.
        table_extension: The extension of the table file.
        shuffle: Whether to shuffle the dataset. Shuffling is done before selecting the maximum number of samples.
        test_max_samples: The maximum number of samples in the test set.
        val_max_samples: The maximum number of samples in the validation set.
        train_max_samples: The maximum number of samples in the train set.
        system_prompt: The system prompt to use for the dataset.
        assistant_prompt: The assistant prompt to use for the dataset.
    
    Returns:
        A dataset with the following keys: 
        "question", "answer", "context", "id", "task", "direction", "size", "table", "text"
    """
    def get_table(context: str):
        context = re.sub(f".csv$", "", context)
        with open(os.path.join(dataset_path, context + "." + table_extension), "r", encoding="utf-8") as f:
            return f.read()
    
    dataset = datasets.load_dataset("csv", data_files={
        "train": os.path.join(dataset_path, "data", "train.csv"),
        "test": os.path.join(dataset_path, "data", "test.csv"),
        "validation": os.path.join(dataset_path, "data", "val.csv")
    })
    
    # Only shuffle the train set
    if shuffle:
        dataset["train"] = dataset["train"].shuffle(seed=42)
        
    if test_max_samples is not None:
        dataset["test"] = dataset["test"].select(range(test_max_samples))
    if val_max_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(val_max_samples))
    if train_max_samples is not None:
        dataset["train"] = dataset["train"].select(range(train_max_samples))
    
    def preprocess_single_example_to_string(example):
        table = get_table(example["context"])
        example["table"] = table
        
        user_prompt = example["question"] + "\n" + table
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + assistant_prompt
        example["text"] = text
        return example
    
    dataset = dataset.map(preprocess_single_example_to_string, batched=False)
    
    # Set tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="pt", padding=True, truncation=True)
    
    dataset = dataset.map(tokenize_function, batched=True, batch_size=batch_size)
    return dataset
    