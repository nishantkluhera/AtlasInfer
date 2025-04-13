# examples/run_atlas_inference.py
import torch
import argparse
import time
import os
import sys
import gc # Explicit GC just in case
import traceback # For detailed error printing
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate import init_empty_weights # Import the context manager
import numpy as np

# --- Make project modules importable ---
# Add the parent directory (atlasinfer) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from atlasinfer_core.memory.unified_memory_manager import UnifiedMemoryManager
from atlasinfer_core.utils.helpers import estimate_quantized_model_vram, cleanup_memory
# Import the modification functions
from integration.model_modifier import (
    apply_atlas_quantization_to_model,
    apply_atlas_attention_wrapper,
    setup_offloading_hooks,
    _get_model_patterns # Helper to check if model is supported early
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run AtlasInfer Inference")
    # --- Model Args ---
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Hugging Face model identifier (e.g., 'facebook/opt-1.3b', 'mistralai/Mistral-7B-v0.1')")
    parser.add_argument("--model_type", type=str, default=None, help="Model type ('opt', 'llama', 'mistral', 'gemma'). If None, infer from config.")
    parser.add_argument("--revision", type=str, default="main", help="Specific model revision (branch, tag, commit hash)")
    parser.add_argument("--trust_remote_code", action="store_true", help="Allow loading models with custom code")

    # --- Generation Args ---
    parser.add_argument("--prompt", type=str, default="The universe is", help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (higher -> more random)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling K value")
    parser.add_argument("--do_sample", action="store_true", help="Enable sampling (uses temp/top_k), otherwise greedy")

    # --- AtlasInfer Config ---
    parser.add_argument("--quant_block_size", type=int, default=128, help="Adaptive quantization block size")
    parser.add_argument("--quant_z_threshold", type=float, default=3.0, help="Adaptive quantization Z-score threshold")
    parser.add_argument("--vram_limit_gb", type=float, default=None, help="Target GPU VRAM limit for KV cache (GB). If None, estimate available VRAM.")
    parser.add_argument("--ram_cache_limit_gb", type=float, default=4.0, help="CPU RAM limit for KV cache (GB)")
    parser.add_argument("--disk_cache_path", type=str, default="./atlas_disk_cache", help="Path for disk cache (set to 'None' or '' to disable)")
    parser.add_argument("--disk_cache_limit_gb", type=int, default=50, help="Max size for disk cache (GB)")
    parser.add_argument("--force_cpu_offload", action='store_true', help="Force use of CPU weight offloading even if model might fit VRAM.")
    parser.add_argument("--vram_only_kv_cache", action='store_true', help="Try to keep KV cache only in VRAM (minimal RAM/Disk use for cache).")

    # --- Baselines / Comparison ---
    parser.add_argument("--use_hf_baseline", action="store_true", help="Run standard Hugging Face FP16 baseline instead of AtlasInfer")
    parser.add_argument("--use_bnb_4bit", action="store_true", help="Run baseline using BitsAndBytes 4-bit quantization")

    # --- Misc ---
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    parser.add_argument("--profile_memory", action='store_true', help="Print VRAM usage during generation steps (if GPU available)")
    return parser.parse_args()

def get_effective_vram_limit(requested_gb, gpu_id=0):
    """Gets available VRAM or uses requested limit, returns bytes."""
    if not torch.cuda.is_available():
        return 0
    try:
        total_vram = torch.cuda.get_device_properties(gpu_id).total_memory
        # Use mem_get_info for potentially more accurate free memory
        free_vram, _gpu_total_vram_check = torch.cuda.mem_get_info(gpu_id)
        available_gb = free_vram / (1024**3)
        total_gb = total_vram / (1024**3)
        print(f"VRAM - Total: {total_gb:.2f} GB, Available: {available_gb:.2f} GB")

        if requested_gb is None:
            # Default: Use 90% of currently free VRAM as the limit
            limit_gb = available_gb * 0.90
            print(f"Using estimated VRAM limit for KV Cache: {limit_gb:.2f} GB")
        else:
            limit_gb = requested_gb
            # Check if requested exceeds available, use slightly less than total available if so
            if limit_gb > available_gb:
                print(f"Warning: Requested KV Cache VRAM limit ({limit_gb:.2f} GB) > available ({available_gb:.2f} GB). Capping at 95% of available.")
                limit_gb = available_gb * 0.95
            else:
                 print(f"Using requested VRAM limit for KV Cache: {limit_gb:.2f} GB")

        # Ensure limit is not negative or extremely small
        limit_gb = max(0.01, limit_gb) # Minimum 10MB VRAM for cache seems reasonable
        return int(limit_gb * (1024**3)) # Return bytes
    except Exception as e:
         print(f"Error getting VRAM info: {e}. Defaulting VRAM limit for KV Cache to 1GB.")
         return int(1 * (1024**3)) # 1GB in bytes

def profile_memory_usage(step, log_prefix=""):
    """ Simple VRAM profiler. """
    if not torch.cuda.is_available(): return
    try:
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        reserved = torch.cuda.memory_reserved(0) / (1024**2)
        # Use carriage return '\r' to update the line in place during generation
        print(f"\r{log_prefix} Step {step}: VRAM Allocated={allocated:.1f} MB, Reserved={reserved:.1f} MB", end='')
    except Exception as e:
         print(f"\rError profiling memory: {e}", end='')


# ==============================================
# Main Inference Logic
# ==============================================
def main():
    args = parse_args()
    # --- Seeding ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cleanup_memory() # Start clean

    print("--- AtlasInfer Inference Script ---")
    print(f"Model: {args.model_name_or_path} (Revision: {args.revision})")
    if args.use_hf_baseline: print("Mode: Hugging Face FP16 Baseline")
    elif args.use_bnb_4bit: print("Mode: Hugging Face + BitsAndBytes 4-bit Baseline")
    else: print("Mode: AtlasInfer")
    print("-" * 30) # Separator

    # --- Setup Devices ---
    gpu_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cpu_device = torch.device("cpu")
    print(f"Using GPU: {gpu_device}, CPU: {cpu_device}")
    if gpu_device == cpu_device and args.force_cpu_offload:
        print("Warning: CPU offload forced but no GPU available. Disabling offload.")
        args.force_cpu_offload = False

    # --- Load Tokenizer & Config ---
    print("\nLoading tokenizer and config...")
    try:
        # Consider adding use_fast=False if tokenizer issues arise
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, revision=args.revision)
        config = AutoConfig.from_pretrained(args.model_name_or_path, revision=args.revision, trust_remote_code=args.trust_remote_code)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer or config for {args.model_name_or_path}: {e}")
        return 1 # Indicate error exit

    # Set pad token if missing (required by generate)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
             tokenizer.pad_token_id = tokenizer.eos_token_id
             print(f"Set pad_token_id to eos_token_id: {tokenizer.eos_token_id}")
        else:
             # Add a pad token if EOS is also missing (highly unlikely for generative models)
             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             config.pad_token_id = tokenizer.pad_token_id # Ensure config also knows
             print(f"Added [PAD] token as pad_token (ID: {tokenizer.pad_token_id}).")

    # Determine model type and check if supported by AtlasInfer early
    model_type = args.model_type or getattr(config, "model_type", None)
    if not model_type:
         print("ERROR: Could not determine model_type from config. Please specify with --model_type.")
         return 1
    print(f"Determined Model Type: {model_type}")

    if not args.use_hf_baseline and not args.use_bnb_4bit:
        if _get_model_patterns(model_type) is None:
             print(f"ERROR: AtlasInfer does not have modification patterns defined for model type '{model_type}'.")
             print("Please add patterns to integration/model_modifier.py or use a baseline mode.")
             return 1
        print(f"AtlasInfer patterns found for '{model_type}'.")

    # --- Prepare Model Kwargs (for HF loading) ---
    model_kwargs = {
        "revision": args.revision,
        "trust_remote_code": args.trust_remote_code,
    }
    # Determine loading strategy based on mode
    load_on_cpu = True # Default to loading on CPU first
    if args.use_hf_baseline and not args.use_bnb_4bit:
        model_kwargs["torch_dtype"] = torch.float16
        # Keep load_on_cpu=True for baseline too, move later if possible
    elif args.use_bnb_4bit:
        # Check if bitsandbytes is available
        try:
            import bitsandbytes
            print("BitsAndBytes library found.")
        except ImportError:
            print("ERROR: --use_bnb_4bit requires the 'bitsandbytes' library.")
            print("Installation might be tricky on Windows. Try 'pip install bitsandbytes' or check documentation.")
            return 1

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, # Compute in FP16 for accuracy/speed
            bnb_4bit_use_double_quant=True,       # Nested quantization for more savings
            bnb_4bit_quant_type="nf4"             # Recommended quant type (Normal Float 4)
        )
        # Let accelerate place BnB layers, often requires device_map
        model_kwargs["device_map"] = "auto"
        load_on_cpu = False # BnB loading handles device placement directly

    # --- Load Model ---
    print("\nLoading model...")
    model = None
    memory_manager = None
    use_cpu_offload = False

    try:
        if not args.use_hf_baseline and not args.use_bnb_4bit:
            # --- AtlasInfer Path ---
            # === ORDER CHANGE ===
            # 1. Load FP16 model fully onto CPU FIRST
            print("  Loading FULL FP16 model onto CPU (RAM intensive)...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    revision=args.revision,
                    trust_remote_code=args.trust_remote_code
                ).to(cpu_device) # Ensure on CPU
                print("  FP16 model loaded to CPU.")
                cleanup_memory()
            except Exception as e:
                 print(f"\nERROR: Failed loading FP16 weights to CPU: {e}")
                 print("Ensure you have enough free CPU RAM (> FP16 model size).")
                 raise

            # 2. Apply Quantization (Modifies model in place)
            quantization_config = {"block_size": args.quant_block_size, "z_threshold": args.quant_z_threshold}
            apply_atlas_quantization_to_model(model, model_type, quantization_config)
            cleanup_memory()
            print("  Model quantized.")

            # 3. Initialize Memory Manager
            print("  Initializing Unified Memory Manager...")
            disk_path = None if not args.disk_cache_path or args.disk_cache_path.lower() == 'none' else args.disk_cache_path
            kv_cache_vram_limit_bytes = get_effective_vram_limit(args.vram_limit_gb)
            memory_manager = UnifiedMemoryManager(
                vram_limit_gb=kv_cache_vram_limit_bytes / (1024**3),
                ram_limit_gb=args.ram_cache_limit_gb,
                disk_path=disk_path,
                disk_limit_gb=args.disk_cache_limit_gb if disk_path else 0,
                gpu_device=gpu_device,
                cpu_device=cpu_device,
                vram_only_kv_cache=args.vram_only_kv_cache
            )

            # 4. Apply Attention Wrappers (Modifies model in place AFTER quantization)
            # Now the wrapper finds QuantizedLinear layers for q_proj etc.
            apply_atlas_attention_wrapper(model, model_type, memory_manager)
            cleanup_memory()
            print("  Attention layers wrapped.")

            # === NO load_state_dict needed here, weights are already loaded ===

            # 5. Estimate VRAM needed & Determine Offloading Strategy
            # ... (Keep the VRAM estimation logic as before) ...
            total_available_vram_bytes = get_effective_vram_limit(None)
            safe_kv_cache_limit_bytes = min(kv_cache_vram_limit_bytes, total_available_vram_bytes)
            vram_for_model_bytes = max(0, total_available_vram_bytes - safe_kv_cache_limit_bytes)
            activation_buffer_bytes = min(int(0.5 * 1024**3), int(vram_for_model_bytes * 0.1))
            effective_vram_for_params = max(0, vram_for_model_bytes - activation_buffer_bytes)
            vram_for_model_gb = vram_for_model_bytes / (1024**3)
            effective_vram_for_params_gb = effective_vram_for_params / (1024**3)
            print(f"\nEstimated VRAM available for model parameters: {vram_for_model_gb:.2f} GB")
            print(f"  (Reserving ~{(activation_buffer_bytes / (1024**2)):.0f} MB for activations)")
            print(f"  Effective VRAM for parameters: {effective_vram_for_params_gb:.2f} GB")

            estimated_model_params_bytes = estimate_quantized_model_vram(model) # Estimate AFTER quantization
            estimated_model_params_gb = estimated_model_params_bytes / (1024**3)
            print(f"Estimated VRAM needed for quantized parameters: {estimated_model_params_gb:.2f} GB")

            # Determine offload necessity
            if args.force_cpu_offload: use_cpu_offload = True; print("CPU weight offload ENABLED (forced).")
            elif gpu_device != cpu_device and estimated_model_params_bytes > effective_vram_for_params:
                use_cpu_offload = True; print(f"CPU weight offload ENABLED (Est Size > Limit).")
            elif gpu_device == cpu_device: use_cpu_offload = False; print("CPU weight offload DISABLED (no GPU).")
            else: use_cpu_offload = False; print("CPU weight offload DISABLED (fits VRAM).")

            # 6. Place Model and Setup Hooks if Offloading
            model.to(cpu_device) # Ensure starting on CPU
            if use_cpu_offload:
                 setup_offloading_hooks(model, model_type, gpu_device)
                 print("Model remains on CPU; hooks manage GPU execution.")
            elif gpu_device != cpu_device:
                 print(f"Moving model parameters to GPU ({gpu_device})...")
                 try:
                      model.to(gpu_device)
                      cleanup_memory()
                      print("Model moved to GPU.")
                 except Exception as e: print(f"ERROR moving model to GPU: {e}"); raise
            else:
                 print("Model remains on CPU (no GPU available).")

        # === Baseline Path remains the same ===
        else:
            # ... (Baseline loading logic as before) ...
             mode_str = 'FP16' if not args.use_bnb_4bit else 'BitsAndBytes 4bit'
             print(f"\nLoading baseline model ({mode_str})...")
             try:
                 if load_on_cpu: # Load FP16 baseline to CPU first
                      print("  Loading weights to CPU...")
                      model = AutoModelForCausalLM.from_pretrained(
                          args.model_name_or_path,
                          **model_kwargs
                      ).to(cpu_device)
                      if gpu_device != cpu_device:
                           print(f"  Moving FP16 baseline model to GPU ({gpu_device})...")
                           try: model.to(gpu_device); print("  Model moved to GPU.")
                           except Exception as e: print(f"ERROR moving FP16 baseline to GPU: {e}"); raise
                 else: # BnB loading
                      model = AutoModelForCausalLM.from_pretrained(
                          args.model_name_or_path,
                          **model_kwargs
                      )
                      print(f"BitsAndBytes model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")
             except Exception as e: print(f"ERROR loading baseline model: {e}"); traceback.print_exc(); return 1

        # --- Set Model to Evaluation Mode ---
        model.eval()

        # --- Prepare Inputs ---
        # ... (Input preparation logic remains the same, determines input_device) ...
        inputs = tokenizer(args.prompt, return_tensors="pt")
        input_device = cpu_device # Default to CPU
        if args.use_bnb_4bit:
             try: input_device = next(model.parameters()).device
             except Exception: print("Warning: BnB device unknown, input->CPU."); input_device = cpu_device
        elif not args.use_hf_baseline and use_cpu_offload: input_device = cpu_device # Atlas offload: input->CPU
        elif gpu_device != cpu_device: input_device = gpu_device # Atlas/FP16 full GPU: input->GPU
        print(f"\nPlacing initial inputs on device: {input_device}")
        input_ids = inputs.input_ids.to(input_device)
        attention_mask = inputs.attention_mask.to(input_device)
        start_len = input_ids.shape[1]


        # --- Generation ---
        # ... (Generation logic using model.generate remains the same) ...
        print(f"\n--- Generating {args.max_new_tokens} tokens ---")
        print(f"Prompt: '{args.prompt}'")
        print(f"Using {'Sampling (T='+str(args.temperature)+', K='+str(args.top_k)+')' if args.do_sample else 'Greedy'} decoding.")
        output_sequences = None; total_time = 0.0; start_time = time.time()
        generate_kwargs = { # ... (setup generate_kwargs) ...
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            **({"do_sample": True, "temperature": args.temperature, "top_k": args.top_k} if args.do_sample else {"do_sample": False}),
        }
        if not args.use_hf_baseline and not args.use_bnb_4bit: generate_kwargs["use_cache"] = True
        if args.profile_memory: profile_memory_usage(0, log_prefix="Pre-Gen")
        print("Starting generation...")
        try:
            with torch.no_grad():
                output_sequences = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)
            total_time = time.time() - start_time
            print("\nGeneration finished.")
        except Exception as e: # ... (Error handling for generate) ...
             print(f"\n!!! ERROR during model.generate(): {e} !!!"); traceback.print_exc()
             output_sequences = input_ids # Fallback
        if args.profile_memory: profile_memory_usage(args.max_new_tokens, log_prefix="Post-Gen")
        print()


        # --- Decode and Print Results ---
        # ... (Result decoding and printing remains the same) ...
        if output_sequences is not None:
             # ... (decode logic) ...
             generated_ids = output_sequences[0]
             num_generated = len(generated_ids) - start_len
             total_time = max(total_time, 1e-6)
             tokens_per_sec = num_generated / total_time if num_generated > 0 else 0
             print("\n--- Generation Results ---")
             final_output = tokenizer.decode(generated_ids.cpu(), skip_special_tokens=True)
             print(f"\nOutput:\n{final_output}")
             print("-" * 20)
             print(f"Time taken: {total_time:.2f} seconds")
             print(f"Tokens generated: {num_generated}")
             print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        else: print("\n--- Generation Failed ---")


        # --- Print Cache Stats (AtlasInfer only) ---
        # ... (Stats printing remains the same) ...
        if not args.use_hf_baseline and not args.use_bnb_4bit and memory_manager:
             print("\n--- Memory Manager Stats ---")
             stats = memory_manager.get_stats()
             for key, value in stats.items(): print(f"  {key}: {value}")

    finally: # Use finally block for cleanup
        # --- Cleanup ---
        # ... (Cleanup logic remains the same) ...
        print("\nCleaning up resources...")
        if 'model' in locals() and model is not None: del model
        if 'memory_manager' in locals() and memory_manager is not None: memory_manager.close(); del memory_manager
        cleanup_memory()
        print("Cleanup Done.")

if __name__ == "__main__":
    # Wrap main call in try-except for final error catching
    try:
        main()
    except Exception as e:
         print(f"\nUnhandled Exception in main: {e}")
         traceback.print_exc()
         sys.exit(1) # Exit with error code
    sys.exit(0) # Exit normally