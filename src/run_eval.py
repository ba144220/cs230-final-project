import sys
import os
from datetime import datetime
from tqdm import tqdm
from transformers import (
    HfArgumentParser, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig
)
import torch
from torch.utils.data.dataloader import DataLoader

from parsers.argument_classes import DatasetArguments, ModelArguments, TrainingArguments, GenerationArguments
from utils.datasets_loader import load_datasets
from collators.data_collator_for_grid_tokenization import DataCollatorForGridTokenization

from models.modeling_table_llama import (
    TableLlamaConfig,
    TableLlamaForCausalLM
)

def main():
    parser = HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments, GenerationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        dataset_args, model_args, training_args, generation_args = parser.parse_yaml_file(sys.argv[1])
    else:
        dataset_args, model_args, training_args, generation_args = parser.parse_args_into_dataclasses()
        
    # Tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
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
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    # Load adapter
    if model_args.adapter_path:
        model.load_adapter(model_args.adapter_path)
    
    # Load datasets
    def filter_function(example):
        if dataset_args.max_table_row_num is not None and example["table_row_num"] > dataset_args.max_table_row_num:
            return False
        if dataset_args.max_table_width is not None and example["table_width"] > dataset_args.max_table_width:
            return False
        return True
    datasets = load_datasets(dataset_args, filter_function=filter_function)
    
    # Data collator
    data_collator = DataCollatorForGridTokenization(
        tokenizer=tokenizer,
        max_seq_length=training_args.max_seq_length,
        is_train=False,
    )
    
    # Inference loop
    pred_dataloader = DataLoader(
        datasets["test"],
        collate_fn=data_collator,
        batch_size=training_args.batch_size,
    )
    
    # Predict
    
    predictions = []
    for idx, batch in enumerate(tqdm(pred_dataloader)):
        with torch.no_grad():
            input_length = batch["input_ids"].size(1)
            # Move the batch to the device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            outputs = model.generate(
                **batch, 
                max_new_tokens=generation_args.max_new_tokens,
                do_sample=generation_args.do_sample,
                top_k=generation_args.top_k,
                top_p=generation_args.top_p,
                pad_token_id=tokenizer.eos_token_id
            )
            output_strings = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=False)
            predictions.extend(output_strings)
    
    # Create a new column for predictions
    df = datasets["test"].to_pandas()
    df["raw_predictions"] = predictions
    
    def clean_predictions(predictions):
        return [pred.replace("<|eot_id|>", "").replace("Answer:", "").strip() for pred in predictions]
    
    df["predictions"] = clean_predictions(df["raw_predictions"])
    df["correct"] = df["answer"] == df["predictions"]
    
    print(f"Base model: {model_args.model_name}")
    print(f"Adapter: {model_args.adapter_path}")
    print(f"Total samples: {df.shape[0]}")
    
    # Count accuracy for each task and direction
    list_item_row_total = df[(df["task"] == "list_items") & (df["direction"] == "row")].shape[0]
    list_item_col_total = df[(df["task"] == "list_items") & (df["direction"] == "column")].shape[0]
    arithmetic_row_total = df[(df["task"] == "arithmetic") & (df["direction"] == "row")].shape[0]
    arithmetic_col_total = df[(df["task"] == "arithmetic") & (df["direction"] == "column")].shape[0]
    
    list_item_row_correct = df[(df["task"] == "list_items") & (df["direction"] == "row") & (df["correct"])].shape[0] 
    list_item_col_correct = df[(df["task"] == "list_items") & (df["direction"] == "column") & (df["correct"])].shape[0] 
    arithmetic_row_correct = df[(df["task"] == "arithmetic") & (df["direction"] == "row") & (df["correct"])].shape[0] 
    arithmetic_col_correct = df[(df["task"] == "arithmetic") & (df["direction"] == "column") & (df["correct"])].shape[0] 
    
    self_generated_total = list_item_row_total + list_item_col_total + arithmetic_row_total + arithmetic_col_total
    self_generated_correct = list_item_row_correct + list_item_col_correct + arithmetic_row_correct + arithmetic_col_correct
    
    print(f"List item row correct: {list_item_row_correct} / {list_item_row_total} = {list_item_row_correct / list_item_row_total * 100:.2f}%")
    print(f"List item column correct: {list_item_col_correct} / {list_item_col_total} = {list_item_col_correct / list_item_col_total * 100:.2f}%")
    print(f"Arithmetic row correct: {arithmetic_row_correct} / {arithmetic_row_total} = {arithmetic_row_correct / arithmetic_row_total * 100:.2f}%")
    print(f"Arithmetic column correct: {arithmetic_col_correct} / {arithmetic_col_total} = {arithmetic_col_correct / arithmetic_col_total * 100:.2f}%")
    print(f"Self-generated correct: {self_generated_correct} / {self_generated_total} = {self_generated_correct / self_generated_total * 100:.2f}%")
    
    wtq_total = df[df["task"] == "wtq"].shape[0]
    wtq_correct = df[(df["task"] == "wtq") & (df["correct"])].shape[0]
    
    print(f"WTQ correct: {wtq_correct} / {wtq_total} = {wtq_correct / wtq_total * 100:.2f}%")
    
    total_correct = self_generated_correct + wtq_correct
    total_total = self_generated_total + wtq_total
    
    print(f"Total correct: {total_correct} / {total_total} = {total_correct / total_total * 100:.2f}%")
    
    
    
    # Save it to the adapter path
    if model_args.adapter_path:
        output_path = os.path.join(model_args.adapter_path, f"predictions.csv")
    else:
        output_path = os.path.join(training_args.output_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_predictions.csv")
        
    df.to_csv(output_path, index=False)
        
    

if __name__ == "__main__":
    main()

