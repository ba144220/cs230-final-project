
import numpy as np
import pandas as pd
import re
import copy
from transformers import (
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig, 
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)

from peft import LoraConfig, get_peft_model
from data.load_table_datasets import load_table_datasets

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "self_generated"
TABLE_EXTENSION = "csv"
BATCH_SIZE = 4
OUTPUT_DIR = "outputs"


def main():
    # Load tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Load dataset
    dataset = load_table_datasets(
        dataset_root="datasets",
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer,
        table_extension=TABLE_EXTENSION,
        batch_size=BATCH_SIZE,
        train_max_samples=160,
        val_max_samples=160,
        test_max_samples=160
    )
    # Copy the dataset to pt_dataset
    pt_dataset = copy.deepcopy(dataset)
    pt_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])    

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

    trainer = Seq2SeqTrainer(
        model=peft_model,
        tokenizer=tokenizer,
        train_dataset=pt_dataset["train"],
        eval_dataset=pt_dataset["validation"],
        args=training_args,
    )
    
    
    peft_model.generation_config.max_new_tokens = 32
    results = trainer.predict(
        pt_dataset["test"],
    )
    
    predictions = np.where(results.predictions == -100, tokenizer.pad_token_id, results.predictions)
    # Remove the first token of every prediction
    # * Notes: Llama3.2 always start with an extra <|begin_of_text|> token
    predictions = predictions[:, 1:]
    
    pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=False)
    for i in range(len(pred_str)):
        pred_str[i] = pred_str[i][len(dataset["test"][i]["text"]):]
        # Remove the <|eot_id|> token
        pred_str[i] = re.sub(r"<\|eot_id\|>", "", pred_str[i])
    
    # Save predictions to a csv
    # Extend the dataset with the predictions
    df = dataset["test"].to_pandas()
    # Remove the "input_ids", "attention_mask" columns
    df = df.drop(columns=["input_ids", "attention_mask"])
    df["predictions"] = pred_str
    df.to_csv(f"{OUTPUT_DIR}/{DATASET_NAME}_eval.csv", index=False)
    
if __name__ == "__main__":
    main()
