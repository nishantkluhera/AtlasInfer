# integration/atlas_wrappers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import math # Needed for manual attention calculation fallback
import time # Added for debug timing

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
    # Required type hint even if only used at runtime
    # Use string literal if QuantizedTensor is defined later or causes import issues
    # from atlasinfer_core.quant.adaptive_quantizer import QuantizedTensor

    def __init__(self, quantized_weights: 'QuantizedTensor', bias: Optional[torch.Tensor]):
        super().__init__()
        # Store weights and bias. Device placement (CPU/GPU) is handled
        # either by initial model placement or the offload hook.
        self.quantized_weights = quantized_weights
        self.bias = bias
        # Store original shape from quantized tensor for reference/debugging
        self.original_shape = getattr(quantized_weights, 'original_shape', None)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        module_id = id(self)
        input_dev = x.device
        print(f"    >> QLinear Enter ID: {module_id} (Shape: {self.original_shape}), Input dev: {input_dev}") # DEBUG

        compute_device = input_dev # Assume computation happens on input device

        # --- Ensure Weights/Bias are on Compute Device ---
        q_tensor = self.quantized_weights
        if not hasattr(q_tensor, 'fp8_data'): # Check if it's a valid QuantizedTensor
             raise TypeError(f"Weight in QuantizedLinear ({module_id}) is not a valid QuantizedTensor: {type(q_tensor)}")

        if q_tensor.fp8_data.device != compute_device:
            print(f"    !! QLinear WARN ID: {module_id}: Weights on {q_tensor.fp8_data.device}, moving to {compute_device}")
            self.quantized_weights = q_tensor.to(compute_device)
            q_tensor = self.quantized_weights # Update local reference

        current_bias = self.bias
        if current_bias is not None and current_bias.device != compute_device:
            print(f"    !! QLinear WARN ID: {module_id}: Bias on {current_bias.device}, moving to {compute_device}")
            self.bias = current_bias.to(compute_device)
            current_bias = self.bias

        # --- Dequantize Weights ---
        print(f"    >> QLinear Dequantizing ID: {module_id} on device {compute_device}...") # DEBUG
        dequant_start_time = time.time()
        dequantized_weight = dequantize_tensor(q_tensor, compute_device)
        dequant_time = time.time() - dequant_start_time
        print(f"    >> QLinear Dequantized ID: {module_id}. Weight dev: {dequantized_weight.device}. Time: {dequant_time:.4f}s") # DEBUG

        # --- Perform Linear Operation ---
        print(f"    >> QLinear Performing F.linear ID: {module_id}...") # DEBUG
        linear_start_time = time.time()
        output = F.linear(x, dequantized_weight, current_bias)
        linear_time = time.time() - linear_start_time
        print(f"    << QLinear Exit ID: {module_id}. Output dev: {output.device}. Linear Time: {linear_time:.4f}s") # DEBUG
        return output

    def extra_repr(self) -> str:
        """ String representation for print(model). """
        if hasattr(self, 'quantized_weights') and self.quantized_weights:
             q_shape = self.original_shape # Use stored shape
             try: q_device = self.quantized_weights.fp8_data.device
             except AttributeError: q_device = 'N/A (init?)'
        else: q_shape, q_device = None, 'N/A'
        return f'quantized_shape={q_shape}, bias={self.bias is not None}, device={q_device}'


# --- Generic Atlas Attention Wrapper ---
class AtlasAttentionWrapper(nn.Module):
    """
    Wraps the attention LOGIC and uses explicitly provided projection layers.
    Manages the KV cache using UnifiedMemoryManager.
    Keeps parent_layer reference for accessing RoPE methods etc.
    """
    def __init__(self,
                 q_proj: nn.Module,             # Pass the Q projection layer
                 k_proj: nn.Module,             # Pass the K projection layer
                 v_proj: nn.Module,             # Pass the V projection layer
                 o_proj: nn.Module,             # Pass the O projection layer
                 parent_layer: nn.Module,       # Keep reference to parent needed for RoPE etc.
                 memory_manager: 'UnifiedMemoryManager',
                 layer_idx: int,
                 model_type: str):              # Model type to check for RoPE etc.
        super().__init__()
        self.memory_manager = memory_manager
        self.layer_idx = layer_idx
        self.parent_layer = parent_layer # Store reference

        # Store projection layers directly
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj

        # --- Extract necessary parameters using parent_layer ---
        original_attn_module = getattr(parent_layer, 'self_attn', None) # Might be self or the wrapper

        # Extract parameters (num_heads, head_dim, scaling, dropout)
        self.num_heads = getattr(parent_layer, 'num_heads', None) or \
                         getattr(parent_layer, 'num_attention_heads', None)
        self.num_kv_heads = getattr(parent_layer, 'num_key_value_heads', self.num_heads) # Default to num_heads
        self.head_dim = getattr(parent_layer, 'head_dim', None)

        if self.head_dim is None and self.num_heads is not None:
            hidden_size = getattr(parent_layer, 'hidden_size', None)
            embed_dim = getattr(parent_layer, 'embed_dim', None)
            model_dim = getattr(parent_layer, 'd_model', None)
            h_size = hidden_size or embed_dim or model_dim
            if h_size: self.head_dim = h_size // self.num_heads
        if self.num_heads is None or self.head_dim is None:
            if hasattr(parent_layer, 'config'):
                 self.num_heads = getattr(parent_layer.config, 'num_attention_heads', None)
                 h_size = getattr(parent_layer.config, 'hidden_size', None)
                 if self.num_heads and h_size: self.head_dim = h_size // self.num_heads
            if self.num_heads is None or self.head_dim is None:
                  raise ValueError(f"CRITICAL: Missing head/dim info layer {layer_idx}")

        # --- Other attributes ---
        self.scaling = getattr(parent_layer, 'scaling', 1.0) # Get scaling from parent (OPT)
        self.rotary_emb = getattr(parent_layer, 'rotary_emb', None) # Get RoPE from parent
        self._apply_rope_func = getattr(parent_layer, '_apply_rotary_pos_emb', None) # Get helper from parent

        dropout_val = getattr(parent_layer, 'dropout', None) or \
                      getattr(original_attn_module, 'attn_dropout', None) # Check original module too
        if dropout_val is None and hasattr(parent_layer, 'config'):
             dropout_val = getattr(parent_layer.config, 'attention_dropout', None) or \
                           getattr(parent_layer.config, 'attn_pdrop', None) or \
                           getattr(parent_layer.config, 'dropout', 0.0)
        self.dropout = dropout_val if dropout_val is not None else 0.0
        self.training = getattr(parent_layer, 'training', False)


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """ Reshapes tensor for multi-head attention. Handles GQA/MQA. """
        try:
            expected_q_dim = self.num_heads * self.head_dim
            expected_kv_dim = self.num_kv_heads * self.head_dim
        except TypeError:
             raise ValueError(f"Cannot reshape layer {self.layer_idx}: num_heads={self.num_heads}, head_dim={self.head_dim}")

        current_dim = tensor.shape[-1]
        num_heads_to_use = self.num_heads
        if current_dim == expected_kv_dim: num_heads_to_use = self.num_kv_heads
        elif current_dim != expected_q_dim:
            raise ValueError(f"Unexpected tensor dim {current_dim} layer {self.layer_idx}. Expected Q={expected_q_dim} or KV={expected_kv_dim}.")

        try: reshaped = tensor.view(bsz, seq_len, num_heads_to_use, self.head_dim).transpose(1, 2).contiguous()
        except Exception as e:
             print(f"ERROR reshape layer {self.layer_idx}: {e}"); print(f"  Input={tensor.shape}, bsz={bsz}, seq_len={seq_len}, num_heads={num_heads_to_use}, head_dim={self.head_dim}"); raise
        return reshaped


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # For RoPE
        past_key_value: Any = None, # Ignored
        output_attentions: bool = False,
        use_cache: bool = False, # We force behavior equivalent to use_cache=True
        **kwargs # Catch other model-specific args like layer_head_mask etc.
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        layer_str = f"L{self.layer_idx}" # Short string for logging
        print(f" ----> Attn Enter {layer_str}, HS dev: {hidden_states.device}, shape: {hidden_states.shape}") # DEBUG START
        forward_start_time = time.time()

        bsz, q_len, _ = hidden_states.size()
        compute_device = hidden_states.device

        # --- 1. Compute Q, K, V ---
        proj_start_time = time.time()
        print(f"      {layer_str} Proj Q...") # DEBUG
        query_states = self.q_proj(hidden_states)
        print(f"      {layer_str} Proj K...") # DEBUG
        key_states = self.k_proj(hidden_states)
        print(f"      {layer_str} Proj V...") # DEBUG
        value_states = self.v_proj(hidden_states)
        proj_time = time.time() - proj_start_time
        print(f"      {layer_str} Proj Done (Q:{query_states.device}). Time: {proj_time:.4f}s") # DEBUG

        # Apply OPT scaling if applicable
        if self.scaling != 1.0: query_states = query_states * self.scaling

        # Reshape for multi-head attention
        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        # --- 2. KV Cache Handling ---
        kv_get_start = time.time()
        print(f"      {layer_str} KV Get...") # DEBUG
        past_k, past_v = self.memory_manager.get_kv(self.layer_idx)
        kv_get_time = time.time() - kv_get_start
        print(f"      {layer_str} KV Get Done. Found: {past_k is not None}. Time: {kv_get_time:.4f}s") # DEBUG

        kv_seq_len = q_len
        if past_k is not None:
            if past_k.device != compute_device: past_k = past_k.to(compute_device)
            if past_v.device != compute_device: past_v = past_v.to(compute_device)
            try:
                cat_start = time.time()
                key_states = torch.cat([past_k, key_states], dim=2)
                value_states = torch.cat([past_v, value_states], dim=2)
                kv_seq_len = key_states.shape[2]
                cat_time = time.time() - cat_start
                # print(f"      {layer_str} KV Concatenated. New len: {kv_seq_len}. Time: {cat_time:.4f}s") # Verbose Debug
            except Exception as e: print(f"ERROR concat KV {layer_str}: {e}"); raise

        # --- 3. RoPE Embeddings (if applicable) ---
        if self.rotary_emb is not None:
            rope_start = time.time()
            if position_ids is None:
                 past_kv_len = past_k.shape[2] if past_k is not None else 0
                 position_ids = torch.arange(
                     past_kv_len, kv_seq_len, dtype=torch.long, device=compute_device
                 ).unsqueeze(0).expand(bsz, q_len) # Use q_len here

            try:
                # Attempt calling rotary_emb with necessary args
                try: cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                except TypeError: cos, sin = self.rotary_emb(value_states) # Try without seq_len

                if self._apply_rope_func: # Use stored function reference
                     query_states, key_states = self._apply_rope_func(query_states, key_states, cos, sin, position_ids)
                else: print(f"Warning: RoPE active but no apply method found/stored {layer_str}")
                rope_time = time.time() - rope_start
                # print(f"      {layer_str} RoPE Applied. Time: {rope_time:.4f}s") # Verbose Debug
            except Exception as e: print(f"Error applying RoPE {layer_str}: {e}")


        # --- 4. Store updated K, V back into memory manager ---
        kv_put_start = time.time()
        # print(f"      {layer_str} KV Put...") # Verbose Debug
        try: self.memory_manager.put_kv(self.layer_idx, key_states, value_states)
        except Exception as e: print(f"ERROR putting KV cache {layer_str}: {e}")
        kv_put_time = time.time() - kv_put_start
        # print(f"      {layer_str} KV Put Done. Time: {kv_put_time:.4f}s") # Verbose Debug


        # --- 5. Multi-Head Attention Calculation ---
        attn_calc_start = time.time()
        # print(f"      {layer_str} Attn Calc...") # Verbose Debug
        if self.num_kv_heads != self.num_heads:
            if self.num_heads % self.num_kv_heads != 0: raise ValueError(f"num_heads % num_kv_heads != 0 {layer_str}")
            num_groups = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(num_groups, dim=1)
            value_states = value_states.repeat_interleave(num_groups, dim=1)

        attn_output = None; attn_weights = None
        use_sdpa = hasattr(F, 'scaled_dot_product_attention') and not output_attentions

        if use_sdpa:
            attn_mask_sdpa = None; is_causal_sdpa = False
            if attention_mask is not None:
                 if attention_mask.dim() == 4: attn_mask_sdpa = attention_mask < -1.0
                 elif attention_mask.dim() == 2: attn_mask_sdpa = (attention_mask == 0)[:, None, None, :kv_seq_len].expand(bsz, self.num_heads, q_len, kv_seq_len)

            try:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask=attn_mask_sdpa, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal_sdpa)
                attn_method = "SDPA"
            except Exception as e: print(f"SDPA fail {layer_str}, fallback. E: {e}"); attn_output = None; attn_method = "SDPA->Fail"

        if attn_output is None: # Fallback BMM
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2)) / math.sqrt(self.head_dim)
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len): raise ValueError("Attn weights shape err")
            if attention_mask is not None:
                try: attn_weights = attn_weights + attention_mask
                except Exception as e: print(f"Mask add fail {layer_str}: {e}"); raise
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if self.dropout > 0.0 and self.training: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_method = "BMM"
        attn_calc_time = time.time() - attn_calc_start
        # print(f"      {layer_str} Attn Calc Done ({attn_method}). Time: {attn_calc_time:.4f}s") # Verbose Debug


        # --- 6. Reshape and Output Projection ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        h_size = self.num_heads * self.head_dim
        attn_output = attn_output.reshape(bsz, q_len, h_size)

        out_proj_start = time.time()
        # print(f"      {layer_str} Out Proj...") # Verbose Debug
        attn_output = self.o_proj(attn_output)
        out_proj_time = time.time() - out_proj_start
        # print(f"      {layer_str} Out Proj Done. Time: {out_proj_time:.4f}s") # Verbose Debug

        forward_time = time.time() - forward_start_time
        print(f" <---- Attn Exit {layer_str}. Total Time: {forward_time:.4f}s") # DEBUG END
        return attn_output, attn_weights if output_attentions else None, None