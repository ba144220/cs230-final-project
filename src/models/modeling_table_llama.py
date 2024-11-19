"""
This file contains the implementation of the TableLlama model.
"""

import torch
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


import transformers.utils.logging as logging
logger = logging.get_logger(__name__)



"""
TableLlamaConfig
"""

class TableLlamaConfig(LlamaConfig):
    # Add a new parameter `rope_table_llama` to the LlamaConfig class
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        rope_table_llama = kwargs.pop("rope_table_llama", None)
        if rope_table_llama is None:
            self.rope_table_llama = {
                "line_length": None,
                "x_channels_start": None,
                "x_channels_end": None,
                "x_channels_step": None,
                "y_channels_start": None,
                "y_channels_end": None,
                "y_channels_step": None,
            }
        else:
            self.rope_table_llama = rope_table_llama
       

"""
TableLlamaRotaryEmbedding
"""


class TableLlamaRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        config: TableLlamaConfig,
        device=None,
    ):
        super().__init__()

        self.config = config
        self.rope_type = getattr(config.rope_scaling, "rope_type", "llama3")
    
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        
        # Table Llama specific initialization
        self.rope_table_llama = getattr(self.config, "rope_table_llama")

        x_channels_start = self.rope_table_llama["x_channels_start"]
        x_channels_end = self.rope_table_llama["x_channels_end"]
        x_channels_step = self.rope_table_llama["x_channels_step"]
        y_channels_start = self.rope_table_llama["y_channels_start"]
        y_channels_end = self.rope_table_llama["y_channels_end"]
        y_channels_step = self.rope_table_llama["y_channels_step"]
        line_length = self.rope_table_llama["line_length"]
        
        if line_length is None:
            # Set a large number to avoid the RoPE effect
            line_length = 10**8

            
        if x_channels_end is None:
            x_channels_start = 10**8
            x_channels_end = 10**8
            x_channels_step = 10**8
        else:
            if x_channels_step is None or x_channels_start is None:
                raise ValueError("You have set x_channels_end but not x_channels_step or x_channels_start")
          
        if y_channels_end is None:
            y_channels_start = 10**8
            y_channels_end = 10**8
            y_channels_step = 10**8
        else:
            if y_channels_step is None or y_channels_start is None:
                raise ValueError("You have set y_channels_end but not y_channels_step or y_channels_start")

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.num_channels = inv_freq.shape[0]
        self.line_length = line_length
        self.x_channels_start = x_channels_start
        self.x_channels_end = x_channels_end
        self.x_channels_step = x_channels_step
        self.y_channels_start = y_channels_start
        self.y_channels_end = y_channels_end
        self.y_channels_step = y_channels_step
        
        
        
  
    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # (1, num_channels, 1)
        position_ids_expanded = position_ids[:, None, :].float() # (batch_size, 1, seq_len)
        position_ids_expanded = position_ids_expanded.repeat(1, self.num_channels, 1) # (batch_size, num_channels, seq_len)
        
        
        x_position_ids = position_ids_expanded % self.line_length
        y_position_ids = position_ids_expanded // self.line_length
        
        # Replace the position_ids_expanded with x_position_ids and y_position_ids
        x_start = self.x_channels_start
        x_end = self.x_channels_end
        x_step = self.x_channels_step
        y_start = self.y_channels_start
        y_end = self.y_channels_end
        y_step = self.y_channels_step
        
        position_ids_expanded[:, x_start:x_end:x_step, :] = x_position_ids[:, x_start:x_end:x_step, :]
        position_ids_expanded[:, y_start:y_end:y_step, :] = y_position_ids[:, y_start:y_end:y_step, :]

        
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
                        
            freqs = position_ids_expanded * inv_freq_expanded # (batch_size, num_channels, seq_len)
            freqs = freqs.transpose(1, 2) # (batch_size, seq_len, num_channels)
            
            emb = torch.cat((freqs, freqs), dim=-1)

            cos = emb.cos()
            sin = emb.sin()
        

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling # (batch_size, seq_len, dim)
        sin = sin * self.attention_scaling # (batch_size, seq_len, dim)

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

"""
TableLlamaModel
"""


class TableLlamaModel(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.rotary_emb = TableLlamaRotaryEmbedding(config)


class TableLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: TableLlamaConfig):
        # Change the config class to TableLlamaConfig
        super().__init__(config)
        # if getattr(config, "rope_table_llama", None) is None:
        #     logger.warning("[TableLlamaForCausalLM] `rope_table_llama` is None. Using default values.")
        #     config.rope_table_llama = DEFAULT_ROPE_TABLE_LLAMA
        
        self.model = TableLlamaModel(config)
