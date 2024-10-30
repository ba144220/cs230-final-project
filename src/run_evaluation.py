
import torch
import numpy as np
from transformers import (
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

from peft import LoraConfig, get_peft_model
from data.load_table_datasets import load_table_datasets

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "self_generated"
TABLE_EXTENSION = "csv"
BATCH_SIZE = 4


def main():
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load dataset
    dataset = load_table_datasets(
        dataset_path="datasets/self_generated",
        tokenizer=tokenizer,
        table_extension=TABLE_EXTENSION,
        batch_size=BATCH_SIZE,
        shuffle=True,
        train_max_samples=100,
        val_max_samples=100,
        test_max_samples=100
    )
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])    

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = LlamaForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quantization_config, device_map="auto")
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    
    peft_model = get_peft_model(model, peft_config)
    peft_model.generation_config.pad_token_id = tokenizer.pad_token_id

    training_args = Seq2SeqTrainingArguments(
        output_dir="./outputs",
        do_train=False,
        do_predict=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        remove_unused_columns=True,
        predict_with_generate=True,
    )
    peft_model.generation_config.max_new_tokens = 32

    trainer = Seq2SeqTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
    
    )

    results = trainer.predict(
        dataset["test"],
    )
    
    predictions = np.where(results.predictions == -100, tokenizer.pad_token_id, results.predictions)
    print(tokenizer.decode(predictions[5], skip_special_tokens=True))



if __name__ == "__main__":
    main()
