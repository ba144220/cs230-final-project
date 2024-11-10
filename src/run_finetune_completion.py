import os
import re
import sys
from typing import List, Dict, Any
from dataclasses import dataclass, field

from datasets import load_dataset, concatenate_datasets

from transformers import (
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig,
    HfArgumentParser,
)

from transformers.utils import logging

logger = logging.get_logger(__name__)

@dataclass
class TrainingArguments:
    model_name: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    gradient_accumulation_steps: int = field(default=1)
    batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    save_total_limit: int = field(default=3)
    logging_steps: int = field(default=10)
    max_seq_length: int = field(default=1024)
    dry_run: bool = field(default=False)
    
@dataclass
class DatasetArguments:
    dataset_root_dir: str = field(default="./datasets")
    dataset_names: List[str] = field(default_factory=lambda: ["self_generated", "wtq"])
    table_extension: str = field(default="html")
    user_prompt_order: List[str] = field(default_factory=lambda: ["table", "question"])
    train_max_samples_for_each_dataset: int = field(default=-1)
    val_max_samples_for_each_dataset: int = field(default=-1)
    test_max_samples_for_each_dataset: int = field(default=-1)
    shuffle_seed: int = field(default=42)

@dataclass
class PeftArguments:
    r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: int = field(default=0.05)
    bias:str = field(default="none")
    task_type:str = field(default="CAUSAL_LM")


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
    
    # Sanity check
    if dataset_name == "self_generated":
        if dataset_args.train_max_samples_for_each_dataset % 80 != 0:
            logging.warning(f"train_max_samples_for_each_dataset for {dataset_name} is not a multiple of 80")
        if dataset_args.val_max_samples_for_each_dataset % 80 != 0:
            logging.warning(f"val_max_samples_for_each_dataset for {dataset_name} is not a multiple of 80")
        if dataset_args.test_max_samples_for_each_dataset % 80 != 0:
            logging.warning(f"test_max_samples_for_each_dataset for {dataset_name} is not a multiple of 80")
    
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
    

def main():
    parser = HfArgumentParser((DatasetArguments, TrainingArguments, PeftArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        dataset_args, training_args, peft_args = parser.parse_yaml_file(sys.argv[1])
    else:
        dataset_args, training_args, peft_args = parser.parse_args_into_dataclasses()
    
    # Load datasets
    datasets = load_datasets(dataset_args)
    
    print(datasets)
    
    # # Tokenizer
    # tokenizer = PreTrainedTokenizerFast.from_pretrained(training_args.model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "right"
    
    
    
    

if __name__ == "__main__":
    main()