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
    Wraps an existing attention module OR the logic block containing it.
    Manages the KV cache using UnifiedMemoryManager.
    Looks for QKV projections on the parent layer if not found directly.
    """
    # Required type hints
    # from atlasinfer_core.memory.unified_memory_manager import UnifiedMemoryManager # Already imported

    # === Added parent_layer parameter ===
    def __init__(self,
                 original_attn_module: nn.Module, # The module being wrapped/replaced (e.g., OPTSdpaAttention)
                 parent_layer: nn.Module,      # The containing layer (e.g., OPTDecoderLayer)
                 memory_manager: 'UnifiedMemoryManager', # Use string hint for manager too
                 layer_idx: int):
        super().__init__()
        # --- Store original module and core components ---
        self.original_attn_module = original_attn_module
        self.parent_layer = parent_layer # Store reference to parent
        self.memory_manager = memory_manager
        self.layer_idx = layer_idx

        # --- Extract/Assume necessary components and attributes ---
        # PRIORITIZE looking on the PARENT layer first, as SDPA variants might not hold proj layers directly
        self.q_proj = getattr(parent_layer, 'q_proj', getattr(original_attn_module, 'q_proj', None))
        self.k_proj = getattr(parent_layer, 'k_proj', getattr(original_attn_module, 'k_proj', None))
        self.v_proj = getattr(parent_layer, 'v_proj', getattr(original_attn_module, 'v_proj', None))
        self.o_proj = getattr(parent_layer, 'o_proj', getattr(original_attn_module, 'o_proj', None)) # Output projection

        # Try alternate names if standard ones fail (still looking on parent first)
        if not all([self.q_proj, self.k_proj, self.v_proj, self.o_proj]):
             if not self.q_proj: self.q_proj = getattr(parent_layer, 'Wq', getattr(original_attn_module, 'Wq', None))
             if not self.k_proj: self.k_proj = getattr(parent_layer, 'Wk', getattr(original_attn_module, 'Wk', None))
             if not self.v_proj: self.v_proj = getattr(parent_layer, 'Wv', getattr(original_attn_module, 'Wv', None))
             if not self.o_proj: self.o_proj = getattr(parent_layer, 'Wo', getattr(original_attn_module, 'Wo', None))

        # If still missing after checking parent and original module, raise error
        if not all([self.q_proj, self.k_proj, self.v_proj, self.o_proj]):
              raise AttributeError(f"CRITICAL: Could not find Q/K/V/O projection layers on parent ({type(parent_layer).__name__}) "
                                   f"or original module ({type(original_attn_module).__name__}) for layer {layer_idx}")

        # --- Attention parameters (Get from original module preferentially, fallback to parent) ---
        self.num_heads = getattr(original_attn_module, 'num_heads', None) or \
                         getattr(original_attn_module, 'num_attention_heads', None) or \
                         getattr(parent_layer, 'num_heads', None) or \
                         getattr(parent_layer, 'num_attention_heads', None)

        self.num_kv_heads = getattr(original_attn_module, 'num_key_value_heads', None)
        if self.num_kv_heads is None: # Fallback if not on original
             self.num_kv_heads = getattr(parent_layer, 'num_key_value_heads', self.num_heads) # Default to num_heads if missing

        self.head_dim = getattr(original_attn_module, 'head_dim', None) or \
                        getattr(parent_layer, 'head_dim', None)

        # Infer head_dim if possible
        if self.head_dim is None and self.num_heads is not None:
            # Look for hidden_size/embed_dim on parent or original module
            hidden_size = getattr(parent_layer, 'hidden_size', None) or \
                          getattr(original_attn_module, 'hidden_size', None)
            embed_dim = getattr(parent_layer, 'embed_dim', None) or \
                        getattr(original_attn_module, 'embed_dim', None)
            model_dim = getattr(parent_layer, 'd_model', None) or \
                       getattr(original_attn_module, 'd_model', None)

            h_size = hidden_size or embed_dim or model_dim
            if h_size:
                self.head_dim = h_size // self.num_heads
                print(f"Inferred head_dim={self.head_dim} for layer {layer_idx}")
            else:
                print(f"Warning: Cannot determine head_dim for layer {layer_idx}.")

        # Final check for num_heads/head_dim, crucial for calculations
        if self.num_heads is None or self.head_dim is None:
             print(f"CRITICAL WARNING: Could not determine num_heads/head_dim for layer {layer_idx}. Attention may fail.")
             # Attempt reasonable defaults, but this is risky
             self.num_heads = 1
             self.num_kv_heads = 1
             self.head_dim = 1


        # --- Other potential attributes (get from original preferentially) ---
        self.scaling = getattr(original_attn_module, 'scaling', getattr(parent_layer,'scaling', 1.0))
        self.rotary_emb = getattr(original_attn_module, 'rotary_emb', getattr(parent_layer,'rotary_emb', None))
        self.dropout = getattr(original_attn_module, 'dropout', None) or \
                       getattr(original_attn_module, 'attn_dropout', None) or \
                       getattr(parent_layer, 'dropout', 0.0) # Default dropout 0 if not found

        # Store training flag if original module had it
        self.training = getattr(original_attn_module, 'training', False) # Default to eval


    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        """ Reshapes tensor for multi-head attention. Handles GQA/MQA. """
        # Output shape should be (bsz, num_heads, seq_len, head_dim)
        # Input tensor shape: (bsz, seq_len, num_heads * head_dim) for Q
        #                     (bsz, seq_len, num_kv_heads * head_dim) for K, V
        try:
            expected_q_dim = self.num_heads * self.head_dim
            expected_kv_dim = self.num_kv_heads * self.head_dim
        except TypeError: # Handle case where num_heads or head_dim is None
             print(f"ERROR in _shape layer {self.layer_idx}: num_heads or head_dim is None. Cannot reshape.")
             # Return tensor unchanged? Or raise error? Raise might be better.
             raise ValueError(f"Cannot reshape tensor in layer {self.layer_idx} due to missing head/dim info.")


        current_dim = tensor.shape[-1]
        num_heads_to_use = self.num_heads # Default for Q

        if current_dim == expected_kv_dim: # K or V
             num_heads_to_use = self.num_kv_heads
        elif current_dim != expected_q_dim: # Neither Q nor K/V expected dim
             print(f"Warning: Unexpected tensor dimension {current_dim} in _shape layer {self.layer_idx}. "
                   f"Expected Q={expected_q_dim} or KV={expected_kv_dim}. Using num_heads={self.num_heads}.")
             # Proceed with caution, might error later

        try:
            # Need valid num_heads_to_use and head_dim
            reshaped = tensor.view(bsz, seq_len, num_heads_to_use, self.head_dim).transpose(1, 2).contiguous()
        except Exception as e:
             print(f"ERROR during tensor.view/transpose in _shape layer {self.layer_idx}: {e}")
             print(f"  Input shape={tensor.shape}, bsz={bsz}, seq_len={seq_len}, num_heads={num_heads_to_use}, head_dim={self.head_dim}")
             raise # Re-raise error

        return reshaped


    def forward(
        self,
        hidden_states: torch.Tensor,
        # Common arguments across models (add more if needed)
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None, # For RoPE
        past_key_value: Any = None, # Ignored, handled by memory manager
        output_attentions: bool = False,
        use_cache: bool = False, # We force behavior equivalent to use_cache=True
        **kwargs # Catch other model-specific args like layer_head_mask etc.
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        compute_device = hidden_states.device

        # --- 1. Compute Q, K, V ---
        # Projections (q_proj, etc.) are attributes of the wrapper, pointing to the actual layers
        # These might be QuantizedLinear or standard nn.Linear
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Apply OPT scaling if applicable
        if self.scaling != 1.0:
            query_states = query_states * self.scaling

        # Reshape for multi-head attention
        query_states = self._shape(query_states, q_len, bsz) # (bsz, num_heads, q_len, head_dim)
        key_states = self._shape(key_states, q_len, bsz)   # (bsz, num_kv_heads, q_len, head_dim)
        value_states = self._shape(value_states, q_len, bsz) # (bsz, num_kv_heads, q_len, head_dim)

        # --- 2. KV Cache Handling ---
        # Retrieve past K, V from memory manager (will be on target device or None)
        past_k, past_v = self.memory_manager.get_kv(self.layer_idx)

        kv_seq_len = q_len # Assume full sequence length if no past state
        if past_k is not None:
            # Device checks (should be handled by manager, but good to verify)
            if past_k.device != compute_device:
                 # print(f"Warning: Moving past_k to {compute_device} in layer {self.layer_idx}")
                 past_k = past_k.to(compute_device)
            if past_v.device != compute_device:
                 # print(f"Warning: Moving past_v to {compute_device} in layer {self.layer_idx}")
                 past_v = past_v.to(compute_device)

            # Concatenate along sequence length dimension (usually dim=2 for B,H,S,D)
            try:
                key_states = torch.cat([past_k, key_states], dim=2)
                value_states = torch.cat([past_v, value_states], dim=2)
                kv_seq_len = key_states.shape[2] # Update sequence length
            except Exception as e:
                print(f"ERROR concatenating KV cache for layer {self.layer_idx}: {e}")
                print(f"  Past K shape: {past_k.shape}, New K shape: {key_states.shape if 'key_states' in locals() else 'N/A'}")
                print(f"  Past V shape: {past_v.shape}, New V shape: {value_states.shape if 'value_states' in locals() else 'N/A'}")
                raise # Re-raise error, as this is critical

        # --- 3. RoPE Embeddings (if applicable) ---
        if self.rotary_emb is not None:
            # RoPE needs position_ids. If not provided, assume standard range.
            if position_ids is None:
                 # Need the *cumulative* sequence length for RoPE calculation
                 past_kv_len = past_k.shape[2] if past_k is not None else 0
                 # seq_len_indices = kv_seq_len - past_kv_len # Should match q_len
                 position_ids = torch.arange(
                     past_kv_len, kv_seq_len, dtype=torch.long, device=compute_device
                 ).unsqueeze(0).expand(bsz, -1) # Shape (bsz, q_len) for the new tokens

            # Apply RoPE - varies slightly across models
            try:
                cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len) # Some need seq_len
                # Try to find the specific application method on the PARENT layer first now
                apply_rope_func = getattr(self.parent_layer, '_apply_rotary_pos_emb',
                                          getattr(self.original_attn_module, '_apply_rotary_pos_emb', None))
                if apply_rope_func:
                     query_states, key_states = apply_rope_func(query_states, key_states, cos, sin, position_ids)
                else: # Fallback maybe needed for other RoPE implementations
                     # If no helper, RoPE might be applied manually in original forward
                     # This wrapper would need to replicate that logic, which is complex.
                     print(f"Warning: RoPE found but no '_apply_rotary_pos_emb' method on parent/original module layer {self.layer_idx}. RoPE may not be applied correctly.")

            except TypeError as e:
                 # Handle cases where rotary_emb might have different signature
                 try:
                      cos, sin = self.rotary_emb(value_states) # Try without seq_len
                      apply_rope_func = getattr(self.parent_layer, '_apply_rotary_pos_emb',
                                                getattr(self.original_attn_module, '_apply_rotary_pos_emb', None))
                      if apply_rope_func:
                           query_states, key_states = apply_rope_func(query_states, key_states, cos, sin, position_ids)
                      else: print(f"Warning: RoPE fallback failed in layer {self.layer_idx}. RoPE may not be applied.")
                 except Exception as e2:
                      print(f"Error applying RoPE layer {self.layer_idx} (TypeError fallback): {e} / {e2}")


        # --- 4. Store updated K, V back into memory manager ---
        # Detach tensors before storing? Typically okay for inference.
        try:
            # Ensure K/V are contiguous? Usually okay.
            self.memory_manager.put_kv(self.layer_idx, key_states, value_states)
        except Exception as e:
            print(f"ERROR putting KV cache for layer {self.layer_idx}: {e}")
            # How to recover? Maybe proceed without updating cache? Drop context?

        # --- 5. Multi-Head Attention Calculation ---
        # Handle Grouped Query Attention (GQA/MQA) where num_kv_heads < num_heads
        if self.num_kv_heads != self.num_heads:
            # Check if num_heads is divisible by num_kv_heads
            if self.num_heads % self.num_kv_heads != 0:
                 raise ValueError(f"num_heads ({self.num_heads}) must be divisible by num_kv_heads ({self.num_kv_heads}) in layer {self.layer_idx}")
            num_key_value_groups = self.num_heads // self.num_kv_heads
            # Expand K and V - ensure dimensions are correct before expanding
            # K/V shape: (bsz, num_kv_heads, kv_seq_len, head_dim)
            key_states = key_states.repeat_interleave(num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(num_key_value_groups, dim=1)


        # Use F.scaled_dot_product_attention if available (PyTorch 2.0+) for efficiency
        attn_output = None
        attn_weights = None # Only calculated if not using SDPA or requested

        # Check Pytorch version for SDPA availability
        use_sdpa = hasattr(F, 'scaled_dot_product_attention') and not output_attentions

        if use_sdpa:
            # Prepare attention mask for SDPA: Needs shape (bsz, num_heads, q_len, kv_seq_len)
            # or broadcastable, bool type where True means MASKED (ignore)
            attn_mask_sdpa = None
            if attention_mask is not None:
                # Input mask format can vary. Common: (bsz, kv_seq_len) padding mask (1=keep, 0=ignore)
                # SDPA needs (bsz, H, q, k) bool mask (True=ignore)
                if attention_mask.dim() == 2: # Likely (bsz, kv_seq_len) padding mask
                    # Expand and invert: (bsz, 1, 1, kv_seq_len) -> (bsz, num_heads, q_len, kv_seq_len)
                    attn_mask_sdpa = (attention_mask == 0)[:, None, None, :].expand(bsz, self.num_heads, q_len, kv_seq_len)
                elif attention_mask.dim() == 4: # Assumes (bsz, 1, q_len, kv_seq_len) additive mask
                     # Convert additive mask (-inf where masked) to boolean mask (True where masked)
                     attn_mask_sdpa = attention_mask < -1.0 # Thresholding near zero might be needed
                     # Expand H dimension if needed
                     if attn_mask_sdpa.shape[1] == 1 and self.num_heads > 1:
                          attn_mask_sdpa = attn_mask_sdpa.expand(bsz, self.num_heads, q_len, kv_seq_len)
                else:
                     print(f"Warning: Unsupported attention mask shape {attention_mask.shape} for SDPA in layer {self.layer_idx}. Ignoring mask.")

            try:
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states,
                    attn_mask=attn_mask_sdpa,
                    # is_causal should be handled by the mask preparation, generally False here if mask is provided
                    dropout_p=self.dropout if self.training and self.dropout is not None else 0.0
                )
            except Exception as e:
                print(f"Warning: F.scaled_dot_product_attention failed in layer {self.layer_idx}, falling back to manual bmm. Error: {e}")
                attn_output = None # Ensure fallback runs


        # Fallback to manual attention calculation (BMM) if SDPA fails or not available/suitable
        if attn_output is None:
            # Q K^T / sqrt(d_k)
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
            # Apply scaling factor (sqrt(head_dim))
            if self.head_dim is None or self.head_dim == 0: raise ValueError("head_dim not valid for attention scaling")
            attn_weights = attn_weights / math.sqrt(self.head_dim)

            # Check shape
            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(f"Attention weights have wrong size: {attn_weights.size()} "
                                 f"Expected: {(bsz, self.num_heads, q_len, kv_seq_len)}")

            # Apply attention mask (additive mask: 0 for keep, -inf for ignore)
            if attention_mask is not None:
                # Ensure mask is broadcastable to (bsz, num_heads, q_len, kv_seq_len)
                # Additive mask expects large negative values where masked
                try:
                    attn_weights = attn_weights + attention_mask # Assumes mask is correctly shaped and valued
                except RuntimeError as e:
                     print(f"Error adding attention mask layer {self.layer_idx}: {e}")
                     print(f"  attn_weights shape: {attn_weights.shape}, mask shape: {attention_mask.shape}")
                     # Fallback: ignore mask? Or raise? Raising is safer.
                     raise

            # Softmax
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Apply dropout (optional)
            if self.dropout is not None and self.training:
                 attn_weights = nn.functional.dropout(attn_weights, p=self.dropout)

            # Weighted sum V = A * V
            attn_output = torch.matmul(attn_weights, value_states)


        # --- 6. Reshape and Output Projection ---
        # Reshape attn_output: (bsz, num_heads, q_len, head_dim) -> (bsz, q_len, num_heads * head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Ensure hidden_size attribute exists or infer it
        if self.head_dim is None or self.num_heads is None: raise ValueError("Cannot reshape attention output, missing head/dim info")
        h_size = self.num_heads * self.head_dim # Calculate expected hidden size
        attn_output = attn_output.reshape(bsz, q_len, h_size)

        # Apply output projection (might be QuantizedLinear)
        attn_output = self.o_proj(attn_output)

        # Return signature expected by most models: attn_output, attn_weights (optional), past_kv (None)
        # Returning None for past_kv signals HF generate loop correctly
        return attn_output, attn_weights if output_attentions else None, None # KV state is external