from transformers import (
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments
)

from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM
)

from peft import LoraConfig, get_peft_model
from data.load_table_datasets import load_table_datasets
from args.args_class import (
    TrainArguments,
    PeftArguments,
    DataArguments
)

def formatting_prompts_func(dataset):
    output_texts = []
    for i in range(len(dataset['question'])):
        text = f"### Question: {dataset['question'][i]}\n ### Answer: {dataset['answer'][i]}"
        output_texts.append(text)
    return output_texts

def main():
    ################
    # ArgumentParser
    ################
    peft_parser = HfArgumentParser(PeftArguments)
    peft_args = peft_parser.parse_args_into_dataclasses()[0]

    train_parser = HfArgumentParser(TrainArguments)
    train_args = train_parser.parse_args_into_dataclasses()[0]

    data_parser = HfArgumentParser(DataArguments)
    data_args = data_parser.parse_args_into_dataclasses()[0]
    
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
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    ################
    # Training
    ################
    peft_config = LoraConfig(
    r=peft_args.r,
    lora_alpha=peft_args.lora_alpha,
    lora_dropout=peft_args.lora_dropout,
    bias=peft_args.bias,
    task_type=peft_args.task_type,
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.generation_config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        num_train_epochs=train_args.num_train_epochs,
        learning_rate=train_args.learning_rate,
        bf16=True,
        save_total_limit=train_args.save_total_limit,
        logging_steps=train_args.logging_steps,
        output_dir=train_args.output_dir
    )
    trainer = SFTTrainer(
        peft_model,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        max_seq_length=train_args.max_seq_length,
        formatting_func=formatting_prompts_func,
        args=training_args
    )
    trainer.train() 
    trainer.save_model(train_args.output_dir)
    print("Model trained successfully.")
   
if __name__ == "__main__":
    main()