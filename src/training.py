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
from args.args_class import TrainArguments

def formatting_prompts_func(dataset):
    output_texts = []
    for i in range(len(dataset['question'])):
        text = f"### Question: {dataset['question'][i]}\n ### Answer: {dataset['answer'][i]}"
        output_texts.append(text)
    return output_texts

def main():
    ################
    # Model init & Tokenizer
    ################
    parser = HfArgumentParser(TrainArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = LlamaForCausalLM.from_pretrained(args.model_name, quantization_config=quantization_config, device_map="auto")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ################
    # Dataset
    ################
    dataset = load_table_datasets(
        dataset_root="datasets",
        dataset_name=args.dataset_name,
        tokenizer=tokenizer,
        table_extension=args.table_extension,
        batch_size=args.batch_size,
        train_max_samples=args.train_max_samples,
        val_max_samples=args.val_max_samples,
        test_max_samples=args.test_max_samples,
        user_prompt_order=args.user_prompt_order
    )
    response_template = " ### Answer:"
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    ################
    # Training
    ################
    peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.generation_config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        per_device_train_batch_size=args.per_device_train_batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=True,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        output_dir=args.output_dir
    )
    trainer = SFTTrainer(
        peft_model,
        train_dataset=dataset['train'],
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        formatting_func=formatting_prompts_func,
        args=training_args
    )
    trainer.train() 
    trainer.save_model(args.output_dir)
    print("Model trained successfully.")
   
if __name__ == "__main__":
    main()