from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    HfArgumentParser
)
import pandas as pd
from parsers.argument_classes import (
    DatasetArguments,
    GenerationArguments,
    ModelArguments,
    TrainingArguments
)
from utils.datasets_loader import load_datasets
from io import StringIO
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import torch
import os
from datetime import datetime

def _convert_csv_string_to_table(csv_string: str) -> pd.DataFrame:
    """
    Convert a csv string to a table, including the header
    """
    df = pd.read_csv(StringIO(csv_string), delimiter=",", on_bad_lines="skip")
    columns = df.columns.astype(str).tolist()
    columns = [col.replace("Unnamed: 0", "") for col in columns]
    rows = df.values.astype(str).tolist()
    return pd.DataFrame(rows, columns = columns)

def main():
    parser = HfArgumentParser((DatasetArguments, ModelArguments, TrainingArguments, GenerationArguments))
    dataset_args, model_args, training_args, generation_args = parser.parse_args_into_dataclasses()

    # Load datasets
    def filter_function(example):
        if dataset_args.max_table_row_num is not None and example["table_row_num"] > dataset_args.max_table_row_num:
            return False
        if dataset_args.max_table_width is not None and example["table_width"] > dataset_args.max_table_width:
            return False
        return True
    datasets = load_datasets(dataset_args, filter_function=filter_function)
    # Tapex Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    
    # Model
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/tapex-large-finetuned-wtq")
            
    # Inference loop
    pred_dataloader = DataLoader(
        datasets["test"],
        batch_size=training_args.batch_size,
    )
    
    # Predict
    predictions = []
    for idx, batch in enumerate(tqdm(pred_dataloader)):
        with torch.no_grad():
            # Move the batch to the device
            # batch = {k: v.to(model.device) for k, v in batch.items()}
            question = batch["question"]
            table = batch["table"][0]
            df = _convert_csv_string_to_table(table)
            encoding = tokenizer(df, question, padding='max_length', truncation=True, return_tensors="pt")
            outputs = model.generate(
                **encoding, 
            )
            output_strings = tokenizer.batch_decode(outputs, skip_special_tokens=False)
            predictions.extend(output_strings)
    print("Predictions", predictions)

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

    if "wtq" in dataset_args.dataset_names:
        wtq_total = df[df["task"] == "wtq"].shape[0]
        wtq_correct = df[(df["task"] == "wtq") & (df["correct"])].shape[0]
            
        print(f"WTQ correct: {wtq_correct} / {wtq_total} = {wtq_correct / wtq_total * 100:.2f}%")
    
    total_correct = df["correct"].sum()
    total_total = df.shape[0]
    
    print(f"Total correct: {total_correct} / {total_total} = {total_correct / total_total * 100:.2f}%")
    
        
    # Save it to the adapter path
    if model_args.adapter_path:
        output_path = os.path.join(model_args.adapter_path, f"predictions.csv")
    else:
        # Create the output directory if not exists
        os.makedirs(training_args.output_dir, exist_ok=True)
        output_path = os.path.join(training_args.output_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_predictions.csv")
        
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()