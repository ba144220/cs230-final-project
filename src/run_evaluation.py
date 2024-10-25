import datasets
import re
from trl import SFTTrainer, SFTConfig
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from data.load_table_datasets import load_table_datasets
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "self_generated"
TABLE_EXTENSION = "html"


def main():
    dataset = load_table_datasets("datasets/self_generated", TABLE_EXTENSION, shuffle=True, test_max_samples=100, val_max_samples=100, train_max_samples=100)
    print(dataset)
    # print(dataset["train"][0])
    # print(dataset["train"][0]["table"])

    # model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    # tokenizer = LlamaTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


if __name__ == "__main__":
    main()
