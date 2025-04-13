# integration/atlas_wrappers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import math # Needed for manual attention calculation fallback

# Use relative imports assuming standard structure
# Need TYPE_CHECKING to avoid circular import if QuantizedTensor type hint is needed later
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
        self.original_shape = quantized_weights.original_shape # Store for reference/debugging

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_device = x.device # Assume computation happens on input device

        # --- Ensure Weights/Bias are on Compute Device ---
        # This is primarily a safety check; the offload hook OR initial placement
        # should handle device management. If offloading is NOT used,
        # weights/bias should already be on the compute_device. If offloading
        # IS used, the pre_forward hook should have moved them here.
        q_tensor = self.quantized_weights
        if q_tensor.fp8_data.device != compute_device:
            # This might indicate a problem with hook attachment or model placement
            # print(f"Warning: QuantizedLinear weights were on {q_tensor.fp8_data.device}, moving to {compute_device}")
            self.quantized_weights = q_tensor.to(compute_device)
            q_tensor = self.quantized_weights # Update local reference

        current_bias = self.bias
        if current_bias is not None and current_bias.device != compute_device:
            # print(f"Warning: QuantizedLinear bias was on {current_bias.device}, moving to {compute_device}")
            self.bias = current_bias.to(compute_device) # Update the bias attribute
            current_bias = self.bias # Update local reference


        # --- Dequantize Weights ---
        # Uses the cache within dequantize_tensor for efficiency if called rapidly
        # before potential offload clears it.
        dequantized_weight = dequantize_tensor(q_tensor, compute_device)

        # --- Perform Linear Operation ---
        output = F.linear(x, dequantized_weight, current_bias)

        # --- Cleanup ---
        # DO NOT clear cache here. The offload hook's post_forward
        # is responsible for clearing the cache *after* the parent layer finishes.
        # If we clear it here, subsequent uses within the same layer (rare) fail.

        return output

    def extra_repr(self) -> str:
        """ String representation for print(model). """
        if hasattr(self, 'quantized_weights') and self.quantized_weights:
             q_shape = self.quantized_weights.original_shape
             # Safely access device - might not be fully initialized in some edge cases?
             try:
                q_device = self.quantized_weights.fp8_data.device
             except AttributeError:
                q_device = 'N/A (init?)'
        else:
             q_shape = None
             q_device = 'N/A'
        return f'quantized_shape={q_shape}, bias={self.bias is not None}, device={q_device}'


# --- Generic Atlas Attention Wrapper ---
class AtlasAttentionWrapper(nn.Module):
    """
    Wraps the attention LOGIC and uses explicitly provided projection layers.
    Manages the KV cache using UnifiedMemoryManager.
    """
    # === Modified __init__ signature ===
    def __init__(self,
                 # original_attn_module: nn.Module, # No longer strictly needed if forward is self-contained
                 q_proj: nn.Module,             # Pass the Q projection layer
                 k_proj: nn.Module,             # Pass the K projection layer
                 v_proj: nn.Module,             # Pass the V projection layer
                 o_proj: nn.Module,             # Pass the O projection layer
                 parent_layer: nn.Module,       # Still needed for potential RoPE/config access
                 memory_manager: 'UnifiedMemoryManager',
                 layer_idx: int):
        super().__init__()
        self.memory_manager = memory_manager
        self.layer_idx = layer_idx
        self.parent_layer = parent_layer # Keep for context if needed

        # === Store projection layers directly ===
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        # ========================================

        # --- Extract necessary parameters (must be found on parent/original) ---
        # Try getting from parent first, then original module (less reliable now)
        original_attn_module = getattr(parent_layer, 'self_attn', None) # Get original ref if possible

        self.num_heads = getattr(parent_layer, 'num_heads', None) or \
                         getattr(parent_layer, 'num_attention_heads', None) or \
                         getattr(original_attn_module, 'num_heads', None) or \
                         getattr(original_attn_module, 'num_attention_heads', None)

        self.num_kv_heads = getattr(parent_layer, 'num_key_value_heads', None)
        if self.num_kv_heads is None and original_attn_module:
             self.num_kv_heads = getattr(original_attn_module, 'num_key_value_heads', self.num_heads) # Default to num_heads
        elif self.num_kv_heads is None:
             self.num_kv_heads = self.num_heads # Default if not found anywhere

        self.head_dim = getattr(parent_layer, 'head_dim', None) or \
                        getattr(original_attn_module, 'head_dim', None)

        # Infer head_dim if possible
        if self.head_dim is None and self.num_heads is not None:
            hidden_size = getattr(parent_layer, 'hidden_size', None) or \
                          getattr(original_attn_module, 'hidden_size', None)
            embed_dim = getattr(parent_layer, 'embed_dim', None) or \
                        getattr(original_attn_module, 'embed_dim', None)
            model_dim = getattr(parent_layer, 'd_model', None) or \
                       getattr(original_attn_module, 'd_model', None)
            h_size = hidden_size or embed_dim or model_dim
            if h_size:
                self.head_dim = h_size // self.num_heads
                # print(f"Inferred head_dim={self.head_dim} for layer {layer_idx}") # Debug
        # Final check
        if self.num_heads is None or self.head_dim is None:
             print(f"CRITICAL WARNING: Could not determine num_heads/head_dim for layer {layer_idx}.")
             raise ValueError(f"Missing head/dim info in layer {layer_idx}")


        # --- Other attributes ---
        self.scaling = getattr(parent_layer, 'scaling', getattr(original_attn_module, 'scaling', 1.0))
        self.rotary_emb = getattr(parent_layer, 'rotary_emb', getattr(original_attn_module, 'rotary_emb', None))
        # Try getting dropout from config if not found directly
        dropout_val = getattr(parent_layer, 'dropout', None) or \
                      getattr(original_attn_module, 'dropout', None) or \
                      getattr(original_attn_module, 'attn_dropout', None)
        if dropout_val is None and hasattr(parent_layer, 'config'): # Try config on parent
             dropout_val = getattr(parent_layer.config, 'attention_dropout', 0.0)

        self.dropout = dropout_val if dropout_val is not None else 0.0

        self.training = getattr(parent_layer, 'training', False) # Get training status


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # ... (This method should now be reliable as head_dim/num_heads are validated in init) ...
        expected_q_dim = self.num_heads * self.head_dim
        expected_kv_dim = self.num_kv_heads * self.head_dim
        current_dim = tensor.shape[-1]
        num_heads_to_use = self.num_heads

        if current_dim == expected_kv_dim:
            num_heads_to_use = self.num_kv_heads
        elif current_dim != expected_q_dim:
            raise ValueError(f"Unexpected tensor dimension {current_dim} in _shape layer {self.layer_idx}. "
                             f"Expected Q={expected_q_dim} or KV={expected_kv_dim}.")

        return tensor.view(bsz, seq_len, num_heads_to_use, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        # The forward logic remains exactly the same as the previous full version,
        # as it already uses self.q_proj, self.k_proj etc. which are now guaranteed
        # to be correctly assigned during __init__.

        # --- 1. Compute Q, K, V ---
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        bsz, q_len, _ = hidden_states.size()
        compute_device = hidden_states.device

        # Apply OPT scaling if applicable
        if self.scaling != 1.0:
            query_states = query_states * self.scaling

        # Reshape for multi-head attention
        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)

        # --- 2. KV Cache Handling ---
        past_k, past_v = self.memory_manager.get_kv(self.layer_idx)

        kv_seq_len = q_len
        if past_k is not None:
            if past_k.device != compute_device: past_k = past_k.to(compute_device)
            if past_v.device != compute_device: past_v = past_v.to(compute_device)
            try:
                key_states = torch.cat([past_k, key_states], dim=2)
                value_states = torch.cat([past_v, value_states], dim=2)
                kv_seq_len = key_states.shape[2]
            except Exception as e:
                print(f"ERROR concatenating KV cache layer {self.layer_idx}: {e}")
                raise

        # --- 3. RoPE Embeddings (if applicable) ---
        if self.rotary_emb is not None:
            if position_ids is None:
                 past_kv_len = past_k.shape[2] if past_k is not None else 0
                 position_ids = torch.arange(
                     past_kv_len, kv_seq_len, dtype=torch.long, device=compute_device
                 ).unsqueeze(0).expand(bsz, -1)

            try:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
                apply_rope_func = getattr(self.parent_layer, '_apply_rotary_pos_emb', None)
                if apply_rope_func:
                     query_states, key_states = apply_rope_func(query_states, key_states, cos, sin, position_ids)
                else:
                     print(f"Warning: RoPE found but no apply method layer {self.layer_idx}")
            except Exception as e:
                 print(f"Error applying RoPE layer {self.layer_idx}: {e}")


        # --- 4. Store updated K, V back into memory manager ---
        try:
            self.memory_manager.put_kv(self.layer_idx, key_states, value_states)
        except Exception as e:
            print(f"ERROR putting KV cache layer {self.layer_idx}: {e}")

        # --- 5. Multi-Head Attention Calculation ---
        if self.num_kv_heads != self.num_heads:
            if self.num_heads % self.num_kv_heads != 0:
                 raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}) layer {self.layer_idx}")
            num_key_value_groups = self.num_heads // self.num_kv_heads
            key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)

        attn_output = None
        attn_weights = None
        use_sdpa = hasattr(F, 'scaled_dot_product_attention') and not output_attentions

        if use_sdpa:
            attn_mask_sdpa = None
            # Simplified SDPA mask handling (more robust conversion needed for general case)
            if attention_mask is not None and attention_mask.dim() == 4:
                 attn_mask_sdpa = attention_mask < -1.0 # Convert additive to boolean

            try:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=attn_mask_sdpa,
                    dropout_p=self.dropout if self.training else 0.0
                )
            except Exception as e:
                print(f"SDPA failed layer {self.layer_idx}, fallback. E: {e}")
                attn_output = None

        if attn_output is None: # Fallback BMM
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            attn_weights = attn_weights / math.sqrt(self.head_dim)
            if attention_mask is not None:
                try: attn_weights = attn_weights + attention_mask
                except Exception as e: print(f"Mask add failed layer {self.layer_idx}: {e}"); raise
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            if self.dropout > 0.0 and self.training: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout)
            attn_output = torch.matmul(attn_weights, value_states)

        # --- 6. Reshape and Output Projection ---
        attn_output = attn_output.transpose(1, 2).contiguous()
        h_size = self.num_heads * self.head_dim
        attn_output = attn_output.reshape(bsz, q_len, h_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights if output_attentions else None, None