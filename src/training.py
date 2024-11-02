from transformers import (
    LlamaForCausalLM, 
    PreTrainedTokenizerFast, 
    BitsAndBytesConfig, 
)

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser
)

from trl import SFTConfig, SFTTrainer
from data.load_table_datasets import load_table_datasets

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
DATASET_NAME = "self_generated"
TABLE_EXTENSION = "html"
USER_PROMPT_ORDER = ["table", "question"]
BATCH_SIZE = 8
OUTPUT_DIR = "./outputs"
TRAIN_MAX_SAMPLES = 160
VAL_MAX_SAMPLES = 160
TEST_MAX_SAMPLES = 960
MAX_NEW_TOKENS = 32

def main():
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_config = parser.parse_args_and_config()
    
    ################
    # Model init kwargs & Tokenizer
    ################
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ################
    # Dataset
    ################
    dataset = load_table_datasets(
        dataset_root="datasets",
        dataset_name=DATASET_NAME,
        tokenizer=tokenizer,
        table_extension=TABLE_EXTENSION,
        batch_size=BATCH_SIZE,
        train_max_samples=TRAIN_MAX_SAMPLES,
        val_max_samples=VAL_MAX_SAMPLES,
        test_max_samples=TEST_MAX_SAMPLES,
        user_prompt_order=USER_PROMPT_ORDER
    )

    ################
    # Training
    ################
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

    trainer = SFTTrainer(
        model,
        train_dataset=dataset,
        args=SFTConfig(),
        peft_config=peft_config,
    )
    trainer.train()
    
if __name__ == "__main__":
    main()