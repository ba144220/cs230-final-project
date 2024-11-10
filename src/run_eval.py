import sys
from tqdm import tqdm
from transformers import (
    HfArgumentParser, 
    PreTrainedTokenizerFast, 
    LlamaForCausalLM, 
    BitsAndBytesConfig
)
import torch
from torch.utils.data.dataloader import DataLoader

from parsers.argument_classes import DatasetArguments, ModelArguments, TrainingArguments
from utils.datasets_loader import load_datasets
from collators.data_collator_for_assistant_completion import DataCollatorForAssistantCompletion

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
    model = LlamaForCausalLM.from_pretrained(
        model_args.model_name,
        quantization_config=bnb_config if model_args.load_in_4bit or model_args.load_in_8bit else None,
        device_map="auto",
    )
    # Load adapter
    if model_args.adapter_path:
        model.load_adapter(model_args.adapter_path)
    
    # Load datasets
    datasets = load_datasets(dataset_args)
    
    # Data collator
    data_collator = DataCollatorForAssistantCompletion(
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
    for idx, batch in tqdm(enumerate(pred_dataloader)):
        with torch.no_grad():
            input_length = batch["input_ids"].size(1)
            outputs = model.generate(
                **batch, 
                max_new_tokens=generation_args.max_new_tokens,
                do_sample=generation_args.do_sample,
                top_k=generation_args.top_k,
                top_p=generation_args.top_p,
            )
            output_strings = tokenizer.batch_decode(outputs[:, input_length:], skip_special_tokens=False)
            print(output_strings, datasets["test"][idx]["answer"])
            predictions.extend(output_strings)
    
    print(predictions)
    

if __name__ == "__main__":
    main()

