
import os
import re
from typing import Dict, Any

from parsers.argument_classes import DatasetArguments
from datasets import load_dataset, concatenate_datasets

def load_datasets(
    dataset_args: DatasetArguments, 
):
    temp_datasets = []
    for dataset_name in dataset_args.dataset_names:
        temp_datasets.append(load_single_dataset(dataset_name, dataset_args))
    concatenated_datasets = temp_datasets[0]
    for dataset in temp_datasets[1:]:
        concatenated_datasets["train"] = concatenate_datasets([concatenated_datasets["train"], dataset["train"]])
        concatenated_datasets["validation"] = concatenate_datasets([concatenated_datasets["validation"], dataset["validation"]])
        concatenated_datasets["test"] = concatenate_datasets([concatenated_datasets["test"], dataset["test"]])
    
    # Shuffle train dataset
    concatenated_datasets["train"] = concatenated_datasets["train"].shuffle(seed=dataset_args.shuffle_seed)
    
    return concatenated_datasets
    
def load_single_dataset(
    dataset_name: str, 
    dataset_args: DatasetArguments, 
):
    """
    dataset columns: question, answer, context, id, task, direction, size
    output columns: question, answer, context, id, task, direction, size, table
    """
    dataset_path = os.path.join(dataset_args.dataset_root_dir, dataset_name)
    dataset = load_dataset("csv", data_files={
        "train": os.path.join(dataset_path, "data", "train.csv"),
        "validation": os.path.join(dataset_path, "data", "val.csv"),
        "test": os.path.join(dataset_path, "data", "test.csv"),
    })
    
    # Shuffle train dataset
    if dataset_name == "wtq":
        for split in ["train", "validation", "test"]:
            dataset[split] = dataset[split].shuffle(seed=dataset_args.shuffle_seed)
    if dataset_name == "self_generated":
        dataset["train"] = dataset["train"].shuffle(seed=dataset_args.shuffle_seed)
        
    # Limit the number of samples
    if dataset_args.train_max_samples_for_each_dataset != -1:
        dataset["train"] = dataset["train"].select(range(dataset_args.train_max_samples_for_each_dataset))
    if dataset_args.val_max_samples_for_each_dataset != -1:
        dataset["validation"] = dataset["validation"].select(range(dataset_args.val_max_samples_for_each_dataset))
    if dataset_args.test_max_samples_for_each_dataset != -1:
        dataset["test"] = dataset["test"].select(range(dataset_args.test_max_samples_for_each_dataset))

    # Get the table 
    def get_table(example: Dict[str, Any]):
        context = re.sub(f".csv$", "", example["context"])
        
        with open(os.path.join(dataset_path, context + "." + dataset_args.table_extension), "r", encoding="utf-8") as f:
            table = f.read()
        example["table"] = table
        
        return example
    dataset = dataset.map(get_table)
    
    return dataset
