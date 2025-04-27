# integration/atlas_wrappers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import math # Needed for manual attention calculation fallback
import time # For debug timing
import os # For PID

# Use relative imports assuming standard structure
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from atlasinfer_core.quant.adaptive_quantizer import QuantizedTensor
    from atlasinfer_core.memory.unified_memory_manager import UnifiedMemoryManager

# Import concrete classes needed at runtime
from atlasinfer_core.quant.adaptive_quantizer import dequantize_tensor, clear_dequantized_cache


# --- Generic Quantized Linear ---
class QuantizedLinear(nn.Module):
    """ A Linear layer that uses a QuantizedTensor for weights. """
    def __init__(self, quantized_weights: 'QuantizedTensor', bias: Optional[torch.Tensor]):
        super().__init__()
        self.quantized_weights = quantized_weights
        self.bias = bias
        self.original_shape = getattr(quantized_weights, 'original_shape', None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_id = id(self); input_dev = x.device; pid_str=f"[PID:{os.getpid()}]"
        print(f"{pid_str}    >> QLinear Enter ID: {module_id} (Shape: {self.original_shape}), Input dev: {input_dev}") # DEBUG

        compute_device = input_dev

        # --- Ensure Weights/Bias are on Compute Device ---
        q_tensor = self.quantized_weights
        if not hasattr(q_tensor, 'fp8_data'):
             raise TypeError(f"Weight in QuantizedLinear ({module_id}) is not a valid QuantizedTensor: {type(q_tensor)}")

        if q_tensor.fp8_data.device != compute_device:
            print(f"{pid_str}    !! QLinear WARN ID: {module_id}: Weights on {q_tensor.fp8_data.device}, moving to {compute_device}")
            self.quantized_weights = q_tensor.to(compute_device)
            q_tensor = self.quantized_weights

        current_bias = self.bias
        if current_bias is not None and current_bias.device != compute_device:
            print(f"{pid_str}    !! QLinear WARN ID: {module_id}: Bias on {current_bias.device}, moving to {compute_device}")
            self.bias = current_bias.to(compute_device)
            current_bias = self.bias

        # --- Dequantize Weights ---
        print(f"{pid_str}    >> QLinear Dequant START ID: {module_id} on {compute_device}...") # Clearer START
        dequant_start = time.time()
        try:
            dequantized_weight = dequantize_tensor(q_tensor, compute_device)
        except Exception as e:
             print(f"{pid_str}    !! QLinear DEQUANT ERROR {module_id}: {e}")
             raise
        dequant_end = time.time()
        print(f"{pid_str}    << QLinear Dequant END ID: {module_id}. Time: {dequant_end - dequant_start:.4f}s") # Clearer END

        # --- Perform Linear Operation ---
        print(f"{pid_str}    >> QLinear F.linear START ID: {module_id}...") # Clearer START
        linear_start = time.time()
        try:
             output = F.linear(x, dequantized_weight, current_bias)
        except Exception as e:
             print(f"{pid_str}    !! QLinear F.linear ERROR {module_id}: {e}")
             raise
        linear_end = time.time()
        print(f"{pid_str}    << QLinear F.linear END ID: {module_id}. Time: {linear_end - linear_start:.4f}s") # Clearer END
        print(f"{pid_str}    << QLinear Exit ID: {module_id}")
        return output

    def extra_repr(self) -> str:
        """ String representation for print(model). """
        q_shape, q_device = None, 'N/A'
        if hasattr(self, 'quantized_weights') and self.quantized_weights:
             q_shape = self.original_shape
             try: q_device = self.quantized_weights.fp8_data.device
             except AttributeError: pass # Keep 'N/A' if error
        return f'quantized_shape={q_shape}, bias={self.bias is not None}, device={q_device}'


# --- Generic Atlas Attention Wrapper ---
class AtlasAttentionWrapper(nn.Module):
    """
    Wraps the attention LOGIC and uses explicitly provided projection layers.
    Manages the KV cache using UnifiedMemoryManager.
    Includes more robust parameter finding logic. Includes Debug Prints.
    """
    def __init__(self,
                 q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module, o_proj: nn.Module,
                 parent_layer: nn.Module, memory_manager: 'UnifiedMemoryManager',
                 layer_idx: int, model_type: str):
        super().__init__()
        self.memory_manager = memory_manager
        self.layer_idx = layer_idx
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = q_proj, k_proj, v_proj, o_proj

        # Extract parameters robustly
        original_attn_module = getattr(parent_layer, 'self_attn', None)
        config = getattr(parent_layer, 'config', getattr(original_attn_module, 'config', None))
        # ... (Robust finding logic for num_heads, num_kv_heads, head_dim from previous version) ...
        self.num_heads = getattr(parent_layer, 'num_heads', None) or \
                         getattr(parent_layer, 'num_attention_heads', None) or \
                         getattr(original_attn_module, 'num_heads', None) or \
                         getattr(original_attn_module, 'num_attention_heads', None) or \
                         (getattr(config, 'num_attention_heads', None) if config else None)
        self.num_kv_heads = getattr(parent_layer, 'num_key_value_heads', None) or \
                            getattr(original_attn_module, 'num_key_value_heads', None) or \
                            (getattr(config, 'num_key_value_heads', None) if config else None)
        if self.num_kv_heads is None: self.num_kv_heads = self.num_heads
        self.head_dim = getattr(parent_layer, 'head_dim', None) or \
                        getattr(original_attn_module, 'head_dim', None) or \
                        (getattr(config, 'head_dim', None) if config else None)
        hidden_size = getattr(parent_layer, 'hidden_size', None) or \
                      getattr(original_attn_module, 'hidden_size', None) or \
                      (getattr(config, 'hidden_size', None) if config else None) or \
                      (getattr(config, 'd_model', None) if config else None)
        if self.head_dim is None and self.num_heads is not None and hidden_size is not None and hidden_size % self.num_heads == 0:
            self.head_dim = hidden_size // self.num_heads
        elif self.num_heads is None and self.head_dim is not None and hidden_size is not None and hidden_size % self.head_dim == 0:
             self.num_heads = hidden_size // self.head_dim
             if self.num_kv_heads is None or self.num_kv_heads == 1: self.num_kv_heads = self.num_heads
        if self.num_heads is None or self.head_dim is None or self.num_kv_heads is None:
             raise ValueError(f"CRITICAL: Cannot determine head/dim/kv info layer {layer_idx}")

        # Other attributes
        self.scaling = getattr(parent_layer, 'scaling', getattr(original_attn_module, 'scaling', 1.0))
        self.rotary_emb, self._apply_rope_func = None, None
        rope_models = ['llama', 'mistral', 'gemma', 'falcon']
        if model_type in rope_models:
            self.rotary_emb = getattr(parent_layer, 'rotary_emb', getattr(original_attn_module, 'rotary_emb', None))
            apply_func = getattr(parent_layer, '_apply_rotary_pos_emb', getattr(original_attn_module, '_apply_rotary_pos_emb', None))
            if apply_func: self._apply_rope_func = apply_func
            if self.rotary_emb is None: print(f"Warning: RoPE expected but not found layer {layer_idx}")
        dropout_val = getattr(parent_layer, 'dropout', None) or \
                      getattr(original_attn_module, 'dropout', None) or \
                      getattr(original_attn_module, 'attn_dropout', None)
        if dropout_val is None and config:
             dropout_val = getattr(config, 'attention_dropout', getattr(config, 'attn_pdrop', getattr(config, 'dropout', 0.0)))
        self.dropout = dropout_val if dropout_val is not None else 0.0
        self.training = getattr(parent_layer, 'training', False)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # ... (No changes needed from previous version) ...
        try:
            if self.head_dim is None or self.num_heads is None or self.num_kv_heads is None: raise ValueError("Missing head/dim info")
            expected_q_dim = self.num_heads * self.head_dim; expected_kv_dim = self.num_kv_heads * self.head_dim
        except TypeError: raise ValueError(f"Invalid head/dim info L{self.layer_idx}")
        current_dim = tensor.shape[-1]; num_heads_to_use = self.num_heads
        if current_dim == expected_kv_dim: num_heads_to_use = self.num_kv_heads
        elif current_dim != expected_q_dim: raise ValueError(f"Unexpected tensor dim {current_dim} L{self.layer_idx}. Expected Q={expected_q_dim} or KV={expected_kv_dim}.")
        try: return tensor.view(bsz, seq_len, num_heads_to_use, self.head_dim).transpose(1, 2).contiguous()
        except Exception as e: print(f"ERROR reshape L{self.layer_idx}: {e}"); raise


    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, past_key_value: Any = None,
        output_attentions: bool = False, use_cache: bool = False, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        layer_str = f"L{self.layer_idx}"; pid_str=f"[PID:{os.getpid()}]"
        print(f"{pid_str} ----> Attn Enter {layer_str}, HS dev: {hidden_states.device}, shape: {hidden_states.shape}") # DEBUG START
        forward_start_time = time.time()
        bsz, q_len, _ = hidden_states.size(); compute_device = hidden_states.device

        # --- 1. Compute Q, K, V ---
        proj_start_time = time.time()
        print(f"{pid_str}      {layer_str} Proj Q START..."); query_states = self.q_proj(hidden_states); print(f"{pid_str}      {layer_str} Proj Q END.")
        print(f"{pid_str}      {layer_str} Proj K START..."); key_states = self.k_proj(hidden_states); print(f"{pid_str}      {layer_str} Proj K END.")
        print(f"{pid_str}      {layer_str} Proj V START..."); value_states = self.v_proj(hidden_states); print(f"{pid_str}      {layer_str} Proj V END.")
        proj_time = time.time() - proj_start_time
        print(f"{pid_str}      {layer_str} Proj All Done (Q:{query_states.device}). Time: {proj_time:.4f}s") # DEBUG

        if hasattr(self, 'scaling') and self.scaling != 1.0: query_states = query_states * self.scaling
        query_states = self._shape(query_states, q_len, bsz); key_states = self._shape(key_states, q_len, bsz); value_states = self._shape(value_states, q_len, bsz)

        # --- 2. KV Cache Handling ---
        kv_get_start = time.time(); print(f"{pid_str}      {layer_str} KV Get START...")
        try: past_k, past_v = self.memory_manager.get_kv(self.layer_idx)
        except Exception as e: print(f"{pid_str}      !! {layer_str} KV Get ERROR: {e}"); raise
        kv_get_time = time.time() - kv_get_start; print(f"{pid_str}      {layer_str} KV Get END. Found: {past_k is not None}. Time: {kv_get_time:.4f}s")

        kv_seq_len = q_len
        if past_k is not None:
            if past_k.device != compute_device: past_k = past_k.to(compute_device)
            if past_v.device != compute_device: past_v = past_v.to(compute_device)
            try:
                cat_start = time.time()
                key_states = torch.cat([past_k, key_states], dim=2); value_states = torch.cat([past_v, value_states], dim=2)
                kv_seq_len = key_states.shape[2]; cat_time = time.time() - cat_start
                # print(f"{pid_str}      {layer_str} KV Concatenated. New len: {kv_seq_len}. Time: {cat_time:.4f}s") # DEBUG
            except Exception as e: print(f"ERROR concat KV {layer_str}: {e}"); raise

        # --- 3. RoPE Embeddings ---
        if self.rotary_emb is not None:
             rope_start = time.time(); print(f"{pid_str}      {layer_str} RoPE START...")
             if position_ids is None:
                  past_kv_len = past_k.shape[2] if past_k is not None else 0
                  current_pos = torch.arange(past_kv_len, kv_seq_len, dtype=torch.long, device=compute_device)
                  position_ids_to_pass = current_pos[-q_len:].unsqueeze(0).expand(bsz, q_len) if q_len==current_pos.shape[0] else current_pos.unsqueeze(0).expand(bsz, q_len)
             else: position_ids_to_pass = position_ids
             try:
                 try: cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                 except TypeError: cos, sin = self.rotary_emb(value_states)
                 if self._apply_rope_func: query_states, key_states = self._apply_rope_func(query_states, key_states, cos, sin, position_ids_to_pass)
                 else: print(f"Warn: RoPE apply func missing {layer_str}")
             except Exception as e: print(f"Error applying RoPE {layer_str}: {e}")
             rope_time = time.time() - rope_start; print(f"{pid_str}      {layer_str} RoPE END. Time: {rope_time:.4f}s") # DEBUG

        # --- 4. Store updated K, V ---
        kv_put_start = time.time(); print(f"{pid_str}      {layer_str} KV Put START...")
        try: self.memory_manager.put_kv(self.layer_idx, key_states, value_states)
        except Exception as e: print(f"ERROR putting KV cache {layer_str}: {e}")
        kv_put_time = time.time() - kv_put_start; print(f"{pid_str}      {layer_str} KV Put END. Time: {kv_put_time:.4f}s")

        # --- 5. Multi-Head Attention ---
        attn_calc_start = time.time(); print(f"{pid_str}      {layer_str} Attn Calc START...")
        if self.num_kv_heads != self.num_heads:
            if self.num_heads % self.num_kv_heads != 0: raise ValueError("num_heads % num_kv_heads != 0")
            num_groups = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(num_groups, dim=1); value_states = value_states.repeat_interleave(num_groups, dim=1)

        attn_output, attn_weights = None, None; use_sdpa = hasattr(F, 'scaled_dot_product_attention') and not output_attentions; attn_method = "N/A"
        if use_sdpa:
            attn_mask_sdpa, is_causal_sdpa = None, False # Simplified mask handling
            if attention_mask is not None:
                 if attention_mask.dim() == 4: attn_mask_sdpa = attention_mask < -1.0
                 elif attention_mask.dim() == 2: attn_mask_sdpa = (attention_mask == 0)[:, None, None, :kv_seq_len].expand(bsz, self.num_heads, q_len, kv_seq_len)
            # is_causal_sdpa = q_len > 1 and attn_mask_sdpa is None # More precise check needed maybe
            try:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask_sdpa, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal_sdpa)
                attn_method = "SDPA"
            except Exception as e: print(f"SDPA fail {layer_str}, fallback. E: {e}"); attn_output = None; attn_method = "SDPA->Fail"

        if attn_output is None: # Fallback BMM
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len): raise ValueError("Attn weights shape error")
            if attention_mask is not None: # Additive mask
                try: attn_weights = attn_weights + attention_mask
                except RuntimeError as e: print(f"Mask add fail {layer_str}: {e}"); raise
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if self.dropout > 0.0 and self.training: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_method = "BMM"
        attn_calc_time = time.time() - attn_calc_start; print(f"{pid_str}      {layer_str} Attn Calc END ({attn_method}). Time: {attn_calc_time:.4f}s")

        # --- 6. Reshape and Output Projection ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        h_size = self.num_heads * self.head_dim; attn_output = attn_output.reshape(bsz, q_len, h_size)
        out_proj_start = time.time(); print(f"{pid_str}      {layer_str} Out Proj START...")
        attn_output = self.o_proj(attn_output)
        out_proj_time = time.time() - out_proj_start; print(f"{pid_str}      {layer_str} Out Proj END. Time: {out_proj_time:.4f}s")

        forward_time = time.time() - forward_start_time
        print(f"{pid_str} <---- Attn Exit {layer_str}. Total Time: {forward_time:.4f}s") # DEBUG END
        return attn_output, attn_weights if output_attentions else None, None