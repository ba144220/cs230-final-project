import torch
from typing import Optional
from transformers import LlamaModel, LlamaForCausalLM
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from models.modeling_table_llama import TableLlamaConfig


class LlamaNoGridRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        device=None,
        config: Optional[TableLlamaConfig] = None,
    ):        
        super().__init__()

        self.config = config
        self.rope_type = getattr(config.rope_scaling, "rope_type", "llama3")
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq 
    
    @torch.no_grad()
    def forward(
        self, 
        x, 
        position_ids: torch.LongTensor
    ):
        """
        Args:
            x: The input tensor.
            position_ids: The position ids with shape (batch_size, head_dim//2, seq_len).
        """
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1) # (1, head_dim//2, 1)
        position_ids_expanded = position_ids.float() # (batch_size, head_dim//2, seq_len)
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
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

class LlamaNoGridModel(LlamaModel):
    def __init__(self, config: TableLlamaConfig):
        super().__init__(config)
        self.rotary_emb = LlamaNoGridRotaryEmbedding(config=config)
        

class LlamaNoGridForCausalLM(LlamaForCausalLM):
    def __init__(self, config: TableLlamaConfig):
        super().__init__(config)
        
        # Get replace rules
        rope_table_llama = getattr(config, "rope_table_llama", {
            "x_channels_start": None,
            "x_channels_end": None,
            "x_channels_step": None,
            "y_channels_start": None,
            "y_channels_end": None,
            "y_channels_step": None,
        })
        x_channels_start = rope_table_llama["x_channels_start"]
        x_channels_end = rope_table_llama["x_channels_end"]
        x_channels_step = rope_table_llama["x_channels_step"]
        y_channels_start = rope_table_llama["y_channels_start"]
        y_channels_end = rope_table_llama["y_channels_end"]
        y_channels_step = rope_table_llama["y_channels_step"]

            
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

        self.x_channels_start = x_channels_start
        self.x_channels_end = x_channels_end
        self.x_channels_step = x_channels_step
        self.y_channels_start = y_channels_start
        self.y_channels_end = y_channels_end
        self.y_channels_step = y_channels_step
        
        self.config.rope_table_llama = {
            "x_channels_start": x_channels_start,
            "x_channels_end": x_channels_end,
            "x_channels_step": x_channels_step,
            "y_channels_start": y_channels_start,
            "y_channels_end": y_channels_end,
            "y_channels_step": y_channels_step,
        }
        
        self.model = LlamaNoGridModel(self.config)

    def forward(
        self, 
        position_ids: Optional[torch.LongTensor] = None, # (batch_size, seq_len)
        column_ids: Optional[torch.LongTensor] = None, # (batch_size, input_len)
        row_ids: Optional[torch.LongTensor] = None, # (batch_size, input_len)
        segment_ids: Optional[torch.LongTensor] = None, # (batch_size, input_len), not used yet
        *args, 
        **kwargs
    ):
        # Calculate the head_dim
        head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.config.num_attention_heads)
        # Expand the position ids
        position_ids = position_ids[:, None, :] # (batch_size, 1, seq_len)
        position_ids = position_ids.repeat(1, head_dim//2, 1) # (batch_size, head_dim//2, seq_len)
        
        # Sanity check
        # The shape of column_ids, row_ids, and segment_ids should be the same
        
        if column_ids is not None:
            input_len = column_ids.shape[1]
        elif row_ids is not None:
            input_len = row_ids.shape[1]
        else:
            input_len = -1
        
        if position_ids.shape[2] == input_len:
            # Replace the position ids with the x and y channels
            if column_ids is not None:
                column_ids_expanded = column_ids[:, None, :] # (batch_size, 1, input_len)
                column_ids_expanded = column_ids_expanded.repeat(1, head_dim//2, 1) # (batch_size, head_dim//2, input_len)
                x_start = self.x_channels_start
                x_end = self.x_channels_end
                x_step = self.x_channels_step
                position_ids[:, x_start:x_end:x_step, :] = column_ids_expanded[:, x_start:x_end:x_step, :]
            if row_ids is not None:
                row_ids_expanded = row_ids[:, None, :] # (batch_size, 1, input_len)
                row_ids_expanded = row_ids_expanded.repeat(1, head_dim//2, 1) # (batch_size, head_dim//2, input_len)
                y_start = self.y_channels_start
                y_end = self.y_channels_end
                y_step = self.y_channels_step
                position_ids[:, y_start:y_end:y_step, :] = row_ids_expanded[:, y_start:y_end:y_step, :]

        return super().forward(position_ids=position_ids, *args, **kwargs)