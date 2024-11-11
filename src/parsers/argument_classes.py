from dataclasses import dataclass, field
from typing import List

@dataclass
class ModelArguments:
    model_name: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    adapter_path: str = field(default="")
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    line_length: int = field(default=32)
    channel_period: int = field(4)
    x_channel_offset: int = field(2)
    y_channel_offset: int = field(3)

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

@dataclass
class GenerationArguments:
    max_new_tokens: int = field(default=100)
    do_sample: bool = field(default=False)
    top_k: int = field(default=50)
    top_p: float = field(default=0.95)

@dataclass
class DatasetArguments:
    dataset_root_dir: str = field(default="./datasets")
    dataset_names: List[str] = field(default_factory=lambda: ["self_generated", "wtq"])
    table_extension: str = field(default="html")
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
