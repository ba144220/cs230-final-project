"""
This file contains the implementation of the TableLlama model.
"""

import torch
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


import transformers.utils.logging as logging
logger = logging.get_logger(__name__)


DEFAULT_ROPE_TABLE_LLAMA = {
    "line_length": 32,
    "channel_period": 4,
    "x_channel_offset": 2,
    "y_channel_offset": 3,
}

"""
TableLlamaConfig
"""

def rope_table_llama_config_validation(config):
    
    rope_table_llama = getattr(config, "rope_table_llama", None)
    # Sanity check for `rope_table_llama`
    if rope_table_llama is None:
        raise ValueError("[TableLlamaConfig] `rope_table_llama` must be specified")
    
    # `x_channel_offset` must be less than `channel_period`
    line_length = rope_table_llama.get("line_length", None)
    channel_period = rope_table_llama.get("channel_period", None)
    x_channel_offset = rope_table_llama.get("x_channel_offset", None)
    y_channel_offset = rope_table_llama.get("y_channel_offset", None)
    if line_length is None or channel_period is None or x_channel_offset is None or y_channel_offset is None:
        raise ValueError("`x_channel_offset`, `y_channel_offset`, and `channel_period` must be specified")
    if x_channel_offset == y_channel_offset:
        raise ValueError("`x_channel_offset` and `y_channel_offset` must be different")
    if x_channel_offset < 0 or y_channel_offset < 0:
        raise ValueError("`x_channel_offset` and `y_channel_offset` must be non-negative")



class TableLlamaConfig(LlamaConfig):
    # Add a new parameter `rope_table_llama` to the LlamaConfig class
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        rope_table_llama = kwargs.pop("rope_table_llama", None)
        if rope_table_llama is None:
            logger.warning("[TableLlamaConfig] `rope_table_llama` is None. Using default values.")
            self.rope_table_llama = DEFAULT_ROPE_TABLE_LLAMA
        else:
            self.rope_table_llama = rope_table_llama
        
        rope_table_llama_config_validation(self)
       

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
        period = self.rope_table_llama["channel_period"]
        x_offset = self.rope_table_llama["x_channel_offset"]
        y_offset = self.rope_table_llama["y_channel_offset"]
        line_length = self.rope_table_llama["line_length"]
        
        # Get the default rope parameters
        default_rope_init_fn = ROPE_INIT_FUNCTIONS["default"]
        default_inv_freq, _ = default_rope_init_fn(self.config, device) # (dim // 2)
        
        # Repeat the default_inv_freq for `channel_period` times
        inv_freq_2d = torch.repeat_interleave(default_inv_freq, period, dim=0) # (dim // 2 * channel_period)
        # Get the first d//2 elements
        inv_freq_2d = inv_freq_2d[:inv_freq.shape[0]] # (dim // 2)
        
        # Replace inv_freq for every N*channel_period + x_channel_offset elements
        inv_freq[x_offset::period] = inv_freq_2d[x_offset::period]
        inv_freq[y_offset::period] = inv_freq_2d[y_offset::period]

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        self.num_channels = inv_freq_2d.shape[0]
        self.period = period
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.line_length = line_length
        
        
  
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
        position_ids_expanded[:, self.x_offset::self.period, :] = x_position_ids[:, self.x_offset::self.period, :]
        position_ids_expanded[:, self.y_offset::self.period, :] = y_position_ids[:, self.y_offset::self.period, :]

        
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
