import os
import sys
import torch
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import wandb

from transformers import (
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig,
    HfArgumentParser,
)
from transformers.integrations.integration_utils import WandbCallback

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

from collators.data_collator_for_grid_tokenization import DataCollatorForGridTokenization
from parsers.argument_classes import ModelArguments, DatasetArguments, TrainingArguments, PeftArguments
from utils.datasets_loader import load_datasets
from models.modeling_table_llama import (
    TableLlamaConfig,
    TableLlamaForCausalLM
)

from callbacks.fixed_wandb_callback import FixedWandbCallback

def generate_run_id(training_args: TrainingArguments):
    """
    Generate a run number in the format of run_YYMMDD_HHMMSS
    """
    # Set timezone to Pacific Time
    now = datetime.now(ZoneInfo("US/Pacific"))
    current_time = now.strftime("%y%m%d_%H%M%S")
    return f'{training_args.run_id_prefix}_{current_time}' if not training_args.dry_run else "dry_run"


def main():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments, PeftArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, dataset_args, training_args, peft_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, dataset_args, training_args, peft_args = parser.parse_args_into_dataclasses()
        
    # Wandb setup
    run_id = generate_run_id(training_args)
    if training_args.wandb_entity and training_args.wandb_project:
        wandb.init(
            entity=training_args.wandb_entity,
            project=training_args.wandb_project,
            name=run_id,
        )
    
    # Load datasets
    def filter_function(example):
        if dataset_args.max_table_row_num is not None and example["table_row_num"] > dataset_args.max_table_row_num:
            return False
        if dataset_args.max_table_width is not None and example["table_width"] > dataset_args.max_table_width:
            return False
        return True
    datasets = load_datasets(dataset_args, filter_function=filter_function)
    
    # Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=model_args.load_in_4bit,
        load_in_8bit=model_args.load_in_8bit,
        bnb_4bit_compute_dtype=torch.bfloat16 if model_args.load_in_4bit else None,
        bnb_4bit_use_double_quant=model_args.load_in_4bit,
    )
    # TableLlama
    table_llama_config = TableLlamaConfig.from_pretrained(model_args.model_name)
    table_llama_config.rope_table_llama = {
        "line_length": model_args.line_length,
        "x_channels_start": model_args.x_channels_start,
        "x_channels_end": model_args.x_channels_end,
        "x_channels_step": model_args.x_channels_step,
        "y_channels_start": model_args.y_channels_start,
        "y_channels_end": model_args.y_channels_end,
        "y_channels_step": model_args.y_channels_step,
    }

    model = TableLlamaForCausalLM.from_pretrained(
        model_args.model_name, 
        quantization_config=bnb_config if model_args.load_in_4bit or model_args.load_in_8bit else None,
        device_map="auto",
        config=table_llama_config
    )
        
    # Peft config
    peft_config = LoraConfig(
        task_type=peft_args.task_type,
        r=peft_args.r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias=peft_args.bias,
    )
    
    # SFT config
    if training_args.wandb_project:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
        os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        os.environ["WANDB_DISABLED"] = "false"
    
    sft_config = SFTConfig(
        output_dir=os.path.join(training_args.output_dir, run_id),
        
        do_train=True,
        do_eval=True,
        do_predict=False,
        
        per_device_train_batch_size=training_args.batch_size,
        per_device_eval_batch_size=training_args.batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        max_seq_length=training_args.max_seq_length,
        
        num_train_epochs=training_args.num_train_epochs,
        save_total_limit=training_args.save_total_limit,
        save_strategy="steps",
        save_steps=training_args.save_steps,
        logging_steps=training_args.logging_steps,
        
        # Learning rate arguments
        learning_rate=training_args.learning_rate,
        lr_scheduler_type=training_args.lr_scheduler_type,
        warmup_ratio=training_args.warmup_ratio,
        
        remove_unused_columns=False,
        
        eval_strategy="steps",
        eval_steps=training_args.eval_steps,
        
        report_to=["wandb"] if training_args.wandb_project else [],
        run_name=run_id,
        
        dataset_kwargs={
            "skip_prepare_dataset": True,
        }
    )
    
    collator = DataCollatorForGridTokenization(
        tokenizer=tokenizer,
        max_seq_length=training_args.max_seq_length,
        is_train=True,
        is_grid_tokenization=model_args.line_length is not None,
        line_length=model_args.line_length if model_args.line_length is not None else 64,
    )
    # SFT trainer
    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        peft_config=peft_config,
        args=sft_config,
        data_collator=collator,
    )
    # Fix the wandb callback
    sft_trainer.remove_callback(WandbCallback)
    sft_trainer.add_callback(FixedWandbCallback)
        
    sft_trainer.train()
    
    sft_trainer.save_model(sft_config.output_dir)
    
    # Copy the sys.argv[1] to the output directory
    shutil.copy(sys.argv[1], os.path.join(sft_config.output_dir, os.path.basename(sys.argv[1])))
    
    if training_args.push_to_hub:
        sft_trainer.push_to_hub(f"{training_args.hf_organization}/{run_id}")
    
if __name__ == "__main__":
    main()