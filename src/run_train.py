import os
import sys
import torch
import shutil
from datetime import datetime

from transformers import (
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig,
    HfArgumentParser,
)

from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

import transformers

from collators.data_collator_for_grid_tokenization import DataCollatorForGridTokenization
from parsers.argument_classes import ModelArguments, DatasetArguments, TrainingArguments, PeftArguments
from utils.datasets_loader import load_datasets
from models.modeling_table_llama import (
    TableLlamaConfig,
    TableLlamaForCausalLM
)

transformers.logging.set_verbosity_info()


def generate_run_id(is_dry_run: bool):
    """
    Generate a run number in the format of run_YYYYMMDD_HHMMSS
    """
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    return 'run_{0}'.format(current_time) if not is_dry_run else "dry_run"


def main():
    parser = HfArgumentParser((ModelArguments, DatasetArguments, TrainingArguments, PeftArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, dataset_args, training_args, peft_args = parser.parse_yaml_file(sys.argv[1])
    else:
        model_args, dataset_args, training_args, peft_args = parser.parse_args_into_dataclasses()
    
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
    table_llama_config.rope_table_llama["channel_period"] = model_args.channel_period
    table_llama_config.rope_table_llama["x_channel_offset"] = model_args.x_channel_offset
    table_llama_config.rope_table_llama["y_channel_offset"] = model_args.y_channel_offset
    table_llama_config.rope_table_llama["line_length"] = model_args.line_length

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
    run_id = generate_run_id(training_args.dry_run)
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
        
    sft_trainer.train()
    
    sft_trainer.save_model(sft_config.output_dir)
    
    # Copy the sys.argv[1] to the output directory
    shutil.copy(sys.argv[1], os.path.join(sft_config.output_dir, os.path.basename(sys.argv[1])))
    
if __name__ == "__main__":
    main()