import sys

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

def main():
    ################
    # ArgumentParser
    ################
    # Add json option
    
    parser = HfArgumentParser([PeftArguments, TrainArguments, DataArguments])
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        peft_args, train_args, data_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        peft_args, train_args, data_args = parser.parse_args_into_dataclasses()

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
        learning_rate=train_args.learning_rate,
        bf16=True,
        save_total_limit=train_args.save_total_limit,
        logging_steps=train_args.logging_steps,
        output_dir=data_args.output_dir,
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
    trainer.save_model(data_args.output_dir)
    print("Model trained successfully.")
   
if __name__ == "__main__":
    main()