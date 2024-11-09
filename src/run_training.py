import sys
import json
from datetime import datetime
import wandb
import os

from transformers import (
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig,
    HfArgumentParser,
)

from trl import (
    SFTTrainer,
    SFTConfig
)

from peft import LoraConfig
from data.load_table_datasets import load_table_datasets
from args.args_class import (
    TrainArguments,
    PeftArguments,
    DataArguments
)
from dotenv import load_dotenv

def wandb_init(run_id):
    os.environ["WANDB_PROJECT"] = "table-llama"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_NAME"] = run_id
    wandb.init(project="table-llama", name=run_id)

def generate_run_number():
    """
    Generate a run number in the format of run_YYYYMMDD_HHMMSS
    """
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    return 'run_{0}'.format(current_time)

def save_training_args_to_json(training_args, output_dir, run_file_name):  
    with open('{0}/{1}_training.json'.format(output_dir, run_file_name), 'w') as fout:
        fout.write(training_args.to_json_string())

def main():
    load_dotenv()
    ################
    # ArgumentParser
    ################
    parser = HfArgumentParser([PeftArguments, TrainArguments, DataArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        peft_args, train_args, data_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        peft_args, train_args, data_args = parser.parse_args_into_dataclasses()
    run_id = generate_run_number()
    output_dir = os.path.join(data_args.output_dir, run_id)

    if not train_args.dry_run:
        wandb_init(run_id)

    ################
    # Model init & Tokenizer
    ################    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = LlamaForCausalLM.from_pretrained(train_args.model_name, quantization_config=quantization_config, device_map="auto")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(train_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ################
    # Dataset
    ################
    dataset = load_table_datasets(
        dataset_root="datasets",
        dataset_name=data_args.dataset_name,
        tokenizer=tokenizer,
        table_extension=data_args.table_extension,
        batch_size=data_args.batch_size,
        train_max_samples=data_args.train_max_samples,
        val_max_samples=data_args.val_max_samples,
        test_max_samples=data_args.test_max_samples,
        user_prompt_order=data_args.user_prompt_order
    )

    ################
    # Training
    ################
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    peft_config = LoraConfig(
        r=peft_args.r,
        lora_alpha=peft_args.lora_alpha,
        lora_dropout=peft_args.lora_dropout,
        bias=peft_args.bias,
        task_type=peft_args.task_type,
    )
    

    training_args = SFTConfig(
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        num_train_epochs=train_args.num_train_epochs,
        bf16=True,
        save_total_limit=train_args.save_total_limit,
        logging_steps=train_args.logging_steps,
        output_dir=output_dir,
        report_to="wandb" if not train_args.dry_run else None,
    )
    trainer = SFTTrainer(
        model,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        max_seq_length=train_args.max_seq_length,
        args=training_args,
        peft_config=peft_config
    )
    trainer.train()
    save_training_args_to_json(training_args, output_dir, run_id)
    print("Model trained successfully.")

if __name__ == "__main__":
    main()