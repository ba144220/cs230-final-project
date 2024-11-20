from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelArguments:
    model_name: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    adapter_path: str = field(default="")
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    
    # TableLlama arguments
    line_length: int = field(default=None)
    x_channels_start: int = field(default=None)
    x_channels_end: int = field(default=None)
    x_channels_step: int = field(default=None)
    y_channels_start: int = field(default=None)
    y_channels_end: int = field(default=None)
    y_channels_step: int = field(default=None)

@dataclass
class TrainingArguments:
    output_dir: str = field(default="./outputs")
    gradient_accumulation_steps: int = field(default=1)
    batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    save_total_limit: int = field(default=3)
    save_steps: int = field(default=100)
    logging_steps: int = field(default=10)
    eval_steps: int = field(default=200)
    max_seq_length: int = field(default=1024)
    dry_run: bool = field(default=False)
    
    # Run ID
    run_id_prefix: str = field(default="run")
    
    # Wandb arguments
    wandb_entity: str = field(default=None)
    wandb_project: str = field(default=None)
    
    # Huggingface arguments
    hf_organization: str = field(default="cs230-table-llama")
    push_to_hub: bool = field(default=False)

@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=100)
    do_sample: bool = field(default=False)
    top_k: int = field(default=None)
    top_p: float = field(default=None)
    temperature: float = field(default=None)

@dataclass
class DatasetArguments:
    dataset_root_dir: str = field(default="./datasets")
    dataset_names: List[str] = field(default_factory=lambda: ["self_generated", "wtq"])
    table_extension: str = field(default="csv")
    train_max_samples_for_each_dataset: int = field(default=-1)
    val_max_samples_for_each_dataset: int = field(default=-1)
    test_max_samples_for_each_dataset: int = field(default=-1)
    shuffle_seed: int = field(default=42)
    shuffle: bool = field(default=True)
    
    max_table_row_num: int = field(default=None)
    max_table_width: int = field(default=None)

@dataclass
class PeftArguments:
    r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: int = field(default=0.05)
    bias:str = field(default="none")
    task_type:str = field(default="CAUSAL_LM")
