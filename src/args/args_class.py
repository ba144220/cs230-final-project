from dataclasses import dataclass, field
from typing import List

@dataclass
class TrainArguments:
    # Model args
    model_name: str = field(default="meta-llama/Llama-3.2-1B-Instruct")
    # Dataset args
    dataset_name: str = field(default="self_generated")
    table_extension: str = field(default="html")
    user_prompt_order: List[str] = field(default_factory=lambda: ["table", "question"])
    batch_size: int = field(default=8)
    output_dir: str = field(default="./outputs")
    train_max_samples: int = field(default=160)
    val_max_samples: int = field(default=160)
    test_max_samples: int = field(default=960)
    max_new_tokens: int = field(default=32)
    # Training args
    per_device_train_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=30)
    learning_rate: float = field(default=2e-4)
    save_total_limit: int = field(default=3)
    logging_steps: int = field(default=10)
    max_seq_length: int = field(default=2048)