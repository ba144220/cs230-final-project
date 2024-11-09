import sys
import os
import shutil
import wandb
from datetime import datetime

from datasets import concatenate_datasets

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
from data.load_table_datasets import load_single_dataset
from args.args_class import (
    TrainArguments,
    PeftArguments,
    DataArguments
)

def wandb_init(run_id):
    os.environ["WANDB_PROJECT"] = "table-llama"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_NAME"] = run_id
    wandb.init(project="table-llama", name=run_id)

def generate_run_number(dry_run: bool):
    """
    Generate a run number in the format of run_YYYYMMDD_HHMMSS
    """
    now = datetime.now()
    current_time = now.strftime("%Y%m%d_%H%M%S")
    return 'run_{0}'.format(current_time) if not dry_run else "dry_run"

def save_args_to_json(training_args, output_dir, run_file_name):  
    with open('{0}/{1}_training.json'.format(output_dir, run_file_name), 'w') as fout:
        fout.write(training_args.to_json_string())

def main():
    ################
    # ArgumentParser
    ################
    parser = HfArgumentParser([PeftArguments, TrainArguments, DataArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        peft_args, train_args, data_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        peft_args, train_args, data_args = parser.parse_args_into_dataclasses()
    run_id = generate_run_number(train_args.dry_run)
    output_dir = os.path.join(data_args.output_dir, run_id)
    
    if not train_args.dry_run:
        wandb_init(run_id)
    else:
        wandb.finish()

    ################
    # Model init & Tokenizer
    ################    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = LlamaForCausalLM.from_pretrained(train_args.model_name, quantization_config=quantization_config, device_map="auto")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(train_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    ################
    # Dataset
    ################
    datasets = []
    for dataset_name in data_args.dataset_names:
        datasets.append( 
            load_single_dataset(
                dataset_root="datasets",
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                table_extension=data_args.table_extension,
                batch_size=train_args.batch_size,
                train_max_samples=data_args.train_max_samples,
                val_max_samples=data_args.val_max_samples,
                test_max_samples=data_args.test_max_samples,
                user_prompt_order=data_args.user_prompt_order
            )
        )
    concatenated_datasets = datasets[0]
    for dataset in datasets[1:]:
        concatenated_datasets["train"] = concatenate_datasets([concatenated_datasets["train"], dataset["train"]])
        concatenated_datasets["validation"] = concatenate_datasets([concatenated_datasets["validation"], dataset["validation"]])
        concatenated_datasets["test"] = concatenate_datasets([concatenated_datasets["test"], dataset["test"]])

    dataset = concatenated_datasets
    
    # Shuffle the train set
    dataset["train"] = dataset["train"].shuffle(seed=42)
    
    # Set tokenizer padding
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    def tokenize_function(examples):
        return tokenizer(examples["input_string"], return_tensors="pt", padding=True, truncation=True, max_length=train_args.max_seq_length)
    
    dataset = dataset.map(tokenize_function, batched=True, batch_size=train_args.batch_size)
    
    def generate_labels(examples):
        # Set every label to be the same as the input
        examples["labels"] = examples["input_ids"]
        # Set all labels to -100
        # examples["labels"] = [-100] * len(examples["labels"])
        return examples
    
    dataset = dataset.map(generate_labels, batched=True, batch_size=train_args.batch_size)
    
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
        # Training
        do_train=True,
        per_device_train_batch_size=train_args.batch_size,
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        num_train_epochs=train_args.num_train_epochs,
        bf16=True,
        # Evaluation
        # do_eval=True,
        per_device_eval_batch_size=train_args.batch_size,
        # eval_strategy="steps",
        # eval_steps=100,
        # Prediction
        do_predict=False,
        # Save
        save_total_limit=train_args.save_total_limit,
        output_dir=output_dir,
        # Logging and reporting
        logging_steps=train_args.logging_steps,
        report_to="wandb" if not train_args.dry_run else None,
        
    )

    trainer = SFTTrainer(
        model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        max_seq_length=train_args.max_seq_length,
        args=training_args,
        peft_config=peft_config
    )
    
    # Do evaluation before training
    trainer.evaluate()
    trainer.train()
    if not train_args.dry_run:
        wandb.finish()
        save_args_to_json(training_args, output_dir, run_id)
        # Copy the ./scripts/training_config.json to the output directory
        shutil.copy("./scripts/training_config.json", os.path.join(output_dir, "training_config.json"))
        
    print("Model trained successfully.")

if __name__ == "__main__":
    main()