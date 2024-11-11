import torch
from typing import List, Dict, Any
from transformers import PreTrainedTokenizerFast

from constants.prompts import SYSTEM_PROMPT, ASSISTANT_PREFIX

class DataCollatorForGridTokenization():
    def __init__(
        self, 
        tokenizer: PreTrainedTokenizerFast,
        max_seq_length: int,
        is_train: bool = True,
        is_grid_tokenization: bool = False,
        system_prompt: str = SYSTEM_PROMPT,
        assistant_prefix: str = ASSISTANT_PREFIX,
        end_header_id: int = 128007,
        
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_train = is_train
        self.end_header_id = end_header_id
        self.system_prompt = system_prompt
        self.assistant_prefix = assistant_prefix
        self.is_grid_tokenization = is_grid_tokenization
    
    def _get_label(self, batch: Dict[str, torch.Tensor]):
        """
        Get the label for the batch
        Set all labels before the last occurrence of the end_header_id to -100
        """
        if not self.is_train:
            return batch
        
        batch["labels"] = batch["input_ids"].clone()
        # Set all labels before the last occurrence of the end_header_id to -100
        last_occurrence_indices = self._get_last_occurrence_indices(batch["input_ids"], self.end_header_id)
        for i in range(batch["input_ids"].size(0)):
            batch["labels"][i, :last_occurrence_indices[i] + 2] = -100
            
        return batch
    
    def _call_normal(self, examples: List[Dict[str, Any]]):
        text_list = []
        for example in examples:
            user_prompt = str(example["table"]) + "\n" + str(example["question"])
            message = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
                
            ]
            if self.is_train:
                message.append({"role": "assistant", "content": self.assistant_prefix + str(example["answer"])})
            if self.is_train:
                message_string = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False)
            else:
                message_string = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            text_list.append(message_string)
            
        batch = self.tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.max_seq_length, 
            add_special_tokens=False,
        )
        
        batch = self._get_label(batch)
            
        return batch
        
    def _call_grid_tokenization(self, examples: List[Dict[str, Any]]):
        pass
        
    def __call__(self, examples: List[Dict[str, Any]]):
        """
        Columns: question, answer, context, id, task, direction, size, table
        Only use question, table, and answer
        """
        if self.is_grid_tokenization:
            return self._call_grid_tokenization(examples)
        else:
            return self._call_normal(examples)
        

    def _get_last_occurrence_indices(self, input_ids, X):
        """
        input_ids: 2D tensor of shape B x L
        X: the integer value to find in the tensor
        """

        # Create a boolean mask where elements equal to X are True
        mask = (input_ids == X)  # Shape: B x L

        # Reverse the mask along the sequence dimension (dimension 1)
        reversed_mask = torch.flip(mask, dims=[1])  # Shape: B x L

        # Find the index of the first occurrence of True in the reversed mask
        # Convert boolean mask to float to use argmax (True becomes 1.0, False becomes 0.0)
        idx_in_reversed = reversed_mask.float().argmax(dim=1)  # Shape: B

        # Calculate the last occurrence index in the original tensor
        last_indices = input_ids.size(1) - idx_in_reversed - 1  # Shape: B

        # Handle rows where X does not occur
        # If X does not occur in a row, the entire mask row is False, and argmax returns 0
        # We need to set last_indices for these rows to -1 or any invalid index as per your requirements
        has_X = mask.any(dim=1)  # Shape: B (True if X is in the row)
        last_indices[~has_X] = -1  # Set to -1 where X does not occur

        return last_indices.unsqueeze(1)  # Shape: B x 1