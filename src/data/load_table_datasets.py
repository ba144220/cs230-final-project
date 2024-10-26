import os
import re
import datasets
from transformers import PreTrainedTokenizerFast

SYSTEM_PROMPT = "You are a helpful assistant that can answer questions about the table."
ASSISTANT_PROMPT = "Answer: "

class TableDataset():
    def __init__(
        self, 
        dataset_path: str,
        tokenizer: PreTrainedTokenizerFast,
        table_extension: str = "html",
        shuffle: bool = False,
        test_max_samples: int = None,
        val_max_samples: int = None,
        train_max_samples: int = None,
        system_prompt: str = SYSTEM_PROMPT,
        assistant_prompt: str = ASSISTANT_PROMPT
    ):
        
        self.dataset_path = dataset_path
        self.dataset = datasets.load_dataset("csv", data_files={
            "train": os.path.join(dataset_path, "data", "train.csv"),
            "test": os.path.join(dataset_path, "data", "test.csv"),
            "validation": os.path.join(dataset_path, "data", "val.csv")
        })
        if shuffle:
            self.dataset = self.dataset.shuffle(seed=42)
        if test_max_samples is not None:
            self.dataset["test"] = self.dataset["test"].select(range(test_max_samples))
        if val_max_samples is not None:
            self.dataset["validation"] = self.dataset["validation"].select(range(val_max_samples))
        if train_max_samples is not None:
            self.dataset["train"] = self.dataset["train"].select(range(train_max_samples))
            
        self.tokenizer = tokenizer
        self.table_extension = table_extension
        self.system_prompt = system_prompt
        self.assistant_prompt = assistant_prompt
        
        self.dataset = self.dataset.map(self._preprocess_single_example_to_string, batched=False)
    
    def get_dataset(self):
        return self.dataset
    
    def _get_table(self, context: str):
        context = re.sub(f".csv$", "", context)
        with open(os.path.join(self.dataset_path, context + "." + self.table_extension), "r", encoding="utf-8") as f:
            return f.read()
    def _preprocess_single_example_to_string(self, example):
        
        table = self._get_table(example["context"])
        example["table"] = table
        
        user_prompt = example["question"] + "\n" + table
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        text = text + self.assistant_prompt
        example["text"] = text
        return example
                
       
    

# def load_table_datasets(
#     dataset_path: str,
#     tokenizer: PreTrainedTokenizerFast,
#     table_extension: str = "html",
#     shuffle: bool = False,
#     test_max_samples: int = None,
#     val_max_samples: int = None,
#     train_max_samples: int = None
# ):
  
#     dataset = datasets.load_dataset("csv", data_files={
#         "train": os.path.join(dataset_path, "data", "train.csv"),
#         "test": os.path.join(dataset_path, "data", "test.csv"),
#         "validation": os.path.join(dataset_path, "data", "val.csv")
#     })
    
#     if shuffle:
#         dataset = dataset.shuffle(seed=42)
    
#     if test_max_samples is not None:
#         dataset["test"] = dataset["test"].select(range(test_max_samples))
#     if val_max_samples is not None:
#         dataset["validation"] = dataset["validation"].select(range(val_max_samples))
#     if train_max_samples is not None:
#         dataset["train"] = dataset["train"].select(range(train_max_samples))
    
#     def preprocess_function(examples, tokenizer: PreTrainedTokenizerFast):
#         # Get the `context` column
#         context = examples["context"]
#         # Load the table file as string
#         # Remove the extension from the context
#         context = re.sub(f".csv$", "", context)
#         with open(os.path.join(dataset_path, context + "." + table_extension), "r", encoding="utf-8") as f:
#             table = f.read()
#         examples["table"] = table
#         # Combine 'question' and 'context' if necessary
#         inputs = [q + ' ' + c for q, c in zip(examples['question'], examples['table'])]
        
#         # Tokenize the inputs
#         model_inputs = tokenizer(
#             inputs, 
#             max_length=512, 
#             truncation=True, 
#             padding='max_length'  # or 'longest' or False
#         )
        
#         # Tokenize labels if you have them (e.g., 'answer')
#         if 'answer' in examples:
#             with tokenizer.as_target_tokenizer():
#                 labels = tokenizer(
#                     examples['answer'], 
#                     max_length=512, 
#                     truncation=True, 
#                     padding='max_length'
#                 )
#             model_inputs['labels'] = labels['input_ids']
        
#         return model_inputs

    
#     dataset = dataset.map(preprocess_function, batched=False, fn_kwargs={"tokenizer": tokenizer})

#     return dataset

