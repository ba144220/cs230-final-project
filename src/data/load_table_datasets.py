import os
import re
import datasets
from typing import Dict

def load_table_datasets(
    dataset_path: str,
    table_extension: str = "html",
    shuffle: bool = False,
    test_max_samples: int = None,
    val_max_samples: int = None,
    train_max_samples: int = None
):
  
    dataset = datasets.load_dataset("csv", data_files={
        "train": os.path.join(dataset_path, "data", "train.csv"),
        "test": os.path.join(dataset_path, "data", "test.csv"),
        "validation": os.path.join(dataset_path, "data", "val.csv")
    })
    
    if shuffle:
        dataset = dataset.shuffle(seed=42)
    
    if test_max_samples is not None:
        dataset["test"] = dataset["test"].select(range(test_max_samples))
    if val_max_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(val_max_samples))
    if train_max_samples is not None:
        dataset["train"] = dataset["train"].select(range(train_max_samples))
    
    def preprocess_function(examples):
        # Get the `context` column
        context = examples["context"]
        # Load the table file as string
        # Remove the extension from the context
        context = re.sub(f".csv$", "", context)
        with open(os.path.join(dataset_path, context + "." + table_extension), "r", encoding="utf-8") as f:
            table = f.read()
        examples["table"] = table
        return examples

    dataset = dataset.map(preprocess_function, batched=False)
    
    return dataset
