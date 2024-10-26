import datasets
import re
from trl import SFTTrainer, SFTConfig
from transformers import (
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig, 
    TrainerCallback
)
from peft import LoraConfig
from data.load_table_datasets import TableDataset
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "self_generated"
TABLE_EXTENSION = "csv"


def main():
    tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    dataset = TableDataset(
        dataset_path="datasets/self_generated",
        tokenizer=tokenizer,
        table_extension=TABLE_EXTENSION,
        train_max_samples=100,
        val_max_samples=100,
        test_max_samples=100
    ).get_dataset()
    print(dataset)
    print(dataset["train"])
    print(dataset["train"][0]["text"])
    
    # bnb_config = BitsAndBytesConfig(
    #     load_in_8bit=True,
    # )

    # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", quantization_config=bnb_config, low_cpu_mem_usage=True)
    # # Set the padding side to left
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.eos_token
    
    # dataset = load_table_datasets(
    #     "datasets/self_generated", 
    #     tokenizer,
    #     TABLE_EXTENSION, 
    #     shuffle=True, 
    #     test_max_samples=100, 
    #     val_max_samples=100, 
    #     train_max_samples=100
    # )
    # print(dataset)
    # print(dataset["train"][0])
    # print(dataset["train"][0]["input_ids"])
    

    # peft_config = LoraConfig(
    #     r=16,
    #     lora_alpha=32,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )
    
   
    
    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=dataset["train"],
    #     eval_dataset=dataset["validation"],
    #     peft_config=peft_config,
    #     args=SFTConfig(
    #         max_seq_length=2048,
    #         dataset_text_field="question",
    #         output_dir="../outputs",
    #         remove_unused_columns=False,
    #     ),
       
    # )
    
    # print(model.config)

    # results = trainer.predict(dataset["test"])
    # print(results)


if __name__ == "__main__":
    main()
