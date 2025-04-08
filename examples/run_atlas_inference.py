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
    memory_manager = None # Define here for finally block
    use_cpu_offload = False # Define here for finally block

    try:
        if not args.use_hf_baseline and not args.use_bnb_4bit:
            # --- AtlasInfer Path ---
            # 1. Load structure with empty weights on CPU
            print("  Loading empty model structure on CPU...")
            with init_empty_weights():
                model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
            model.eval() # Set eval mode early

            # 2. Initialize Memory Manager (before applying wrappers)
            print("  Initializing Unified Memory Manager...")
            disk_path = None if not args.disk_cache_path or args.disk_cache_path.lower() == 'none' else args.disk_cache_path
            # Calculate VRAM limit *for KV cache only*
            kv_cache_vram_limit_bytes = get_effective_vram_limit(args.vram_limit_gb)

            memory_manager = UnifiedMemoryManager(
                vram_limit_gb=kv_cache_vram_limit_bytes / (1024**3), # Pass GB value
                ram_limit_gb=args.ram_cache_limit_gb,
                disk_path=disk_path,
                disk_limit_gb=args.disk_cache_limit_gb if disk_path else 0,
                gpu_device=gpu_device,
                cpu_device=cpu_device,
                vram_only_kv_cache=args.vram_only_kv_cache
            )

            # 3. Apply Attention Wrappers (needs manager instance)
            apply_atlas_attention_wrapper(model, model_type, memory_manager)

            # 4. Load FP16 weights onto CPU structure
            print("  Loading FP16 weights onto CPU structure (RAM intensive)...")
            # This requires enough CPU RAM (> FP16 model size)
            try:
                base_model_fp16 = AutoModelForCausalLM.from_pretrained(
                    args.model_name_or_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True, # Important for large models
                    revision=args.revision,
                    trust_remote_code=args.trust_remote_code
                )
                # Ensure base model is loaded to CPU before state dict transfer
                base_model_fp16.to(cpu_device)
                model.load_state_dict(base_model_fp16.state_dict(), strict=True, assign=True)
                del base_model_fp16 # Free RAM asap
                cleanup_memory()
                print("  Weights loaded onto CPU structure.")
            except Exception as e:
                 print(f"\nERROR: Failed loading FP16 weights to CPU: {e}")
                 print("Ensure you have enough free CPU RAM (more than the model's FP16 size).")
                 raise # Re-raise to stop execution

            # 5. Apply Quantization (in place on CPU model)
            quantization_config = {"block_size": args.quant_block_size, "z_threshold": args.quant_z_threshold}
            apply_atlas_quantization_to_model(model, model_type, quantization_config)
            cleanup_memory()

            # 6. Estimate VRAM needed & Determine Offloading Strategy
            # VRAM available *after* reserving space for KV cache limit
            total_available_vram_bytes = get_effective_vram_limit(None) # Get current available
            # Ensure kv_cache_vram_limit_bytes isn't larger than total available
            safe_kv_cache_limit_bytes = min(kv_cache_vram_limit_bytes, total_available_vram_bytes)
            # Calculate remaining VRAM for model parameters & activations
            vram_for_model_bytes = max(0, total_available_vram_bytes - safe_kv_cache_limit_bytes)
            # Add a buffer for activations/intermediate tensors (e.g., 500MB or 10%)
            activation_buffer_bytes = min(int(0.5 * 1024**3), int(vram_for_model_bytes * 0.1))
            effective_vram_for_params = max(0, vram_for_model_bytes - activation_buffer_bytes)

            vram_for_model_gb = vram_for_model_bytes / (1024**3)
            effective_vram_for_params_gb = effective_vram_for_params / (1024**3)
            print(f"\nEstimated VRAM available for model parameters & activations: {vram_for_model_gb:.2f} GB")
            print(f"  (Reserving ~{(activation_buffer_bytes / (1024**2)):.0f} MB for activations)")
            print(f"  Effective VRAM for parameters: {effective_vram_for_params_gb:.2f} GB")


            estimated_model_params_bytes = estimate_quantized_model_vram(model)
            estimated_model_params_gb = estimated_model_params_bytes / (1024**3)
            print(f"Estimated VRAM needed for quantized parameters: {estimated_model_params_gb:.2f} GB")

            # Determine offload necessity based on effective VRAM for parameters
            if args.force_cpu_offload:
                use_cpu_offload = True
                print("CPU weight offload ENABLED (forced by user).")
            elif gpu_device != cpu_device and estimated_model_params_bytes > effective_vram_for_params:
                use_cpu_offload = True
                print(f"CPU weight offload ENABLED (estimated size {estimated_model_params_gb:.2f} GB > effective limit {effective_vram_for_params_gb:.2f} GB).")
            elif gpu_device == cpu_device:
                 use_cpu_offload = False # Ensure false if no GPU
                 print("CPU weight offload DISABLED (no GPU).")
            else:
                use_cpu_offload = False
                print("CPU weight offload DISABLED (model parameters estimated to fit VRAM).")

            # 7. Place Model and Setup Hooks if Offloading
            model.to(cpu_device) # Ensure starting on CPU
            if use_cpu_offload:
                 setup_offloading_hooks(model, model_type, gpu_device)
                 # Model stays primarily on CPU, hooks manage layer movement
                 print("Model remains on CPU; hooks will manage GPU layer execution.")
            elif gpu_device != cpu_device:
                 print(f"Moving model parameters to GPU ({gpu_device})...")
                 try:
                      model.to(gpu_device)
                      cleanup_memory()
                      print("Model moved to GPU.")
                 except Exception as e:
                      print(f"\nERROR: Failed to move model to GPU despite estimate: {e}")
                      print("Try reducing VRAM usage (e.g., smaller KV cache limit) or use --force_cpu_offload.")
                      raise
            else:
                 print("Model remains on CPU (no GPU available).")

        else:
            # --- Baseline Path (HF or BnB) ---
            mode_str = 'FP16' if not args.use_bnb_4bit else 'BitsAndBytes 4bit'
            print(f"\nLoading baseline model ({mode_str})...")
            try:
                if load_on_cpu: # Load FP16 baseline to CPU first
                     print("  Loading weights to CPU...")
                     model = AutoModelForCausalLM.from_pretrained(
                         args.model_name_or_path,
                         **model_kwargs
                     ).to(cpu_device)
                     # Now try moving to GPU if applicable
                     if gpu_device != cpu_device:
                          print(f"  Moving FP16 baseline model to GPU ({gpu_device})...")
                          try:
                               model.to(gpu_device)
                               print("  Model moved to GPU.")
                          except Exception as e:
                              print(f"\nERROR: Failed to move FP16 baseline model to GPU: {e}")
                              print("Model might be too large for VRAM. Try BnB 4-bit or AtlasInfer.")
                              raise # Stop execution if baseline doesn't fit
                else: # BnB loading with device_map='auto'
                     model = AutoModelForCausalLM.from_pretrained(
                         args.model_name_or_path,
                         **model_kwargs
                     )
                     print(f"BitsAndBytes 4-bit model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

            except Exception as e:
                print(f"ERROR loading baseline model: {e}")
                traceback.print_exc()
                return 1 # Exit

        # --- Set Model to Evaluation Mode ---
        model.eval()

        # --- Prepare Inputs ---
        inputs = tokenizer(args.prompt, return_tensors="pt")
        # Determine device for initial inputs based on final model placement/strategy
        input_device = cpu_device # Default to CPU
        if args.use_bnb_4bit:
             try:
                  # For device_map="auto", place inputs on the device of the first parameter (usually embeddings)
                  input_device = next(model.parameters()).device
             except Exception:
                  print("Warning: Could not determine BnB model device, defaulting input to CPU.")
                  input_device = cpu_device
        elif not args.use_hf_baseline and use_cpu_offload:
             # If AtlasInfer offloading, inputs start on CPU for embedding layer
             input_device = cpu_device
        elif gpu_device != cpu_device: # Model fully on GPU (Atlas or FP16 baseline)
             input_device = gpu_device
        # else: input_device remains cpu_device

        print(f"\nPlacing initial inputs on device: {input_device}")
        try:
            input_ids = inputs.input_ids.to(input_device)
            # Create attention mask here as well, place on same device
            attention_mask = inputs.attention_mask.to(input_device)
        except Exception as e:
            print(f"ERROR moving inputs to {input_device}: {e}. Check resources.")
            return 1

        start_len = input_ids.shape[1]

        # --- Generation ---
        print(f"\n--- Generating {args.max_new_tokens} tokens ---")
        print(f"Prompt: '{args.prompt}'")
        print(f"Using {'Sampling (T='+str(args.temperature)+', K='+str(args.top_k)+')' if args.do_sample else 'Greedy'} decoding.")

        output_sequences = None
        total_time = 0.0
        start_time = time.time()

        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            # Add sampling parameters
            **({"do_sample": True, "temperature": args.temperature, "top_k": args.top_k} if args.do_sample else {"do_sample": False}),
        }

        # Ensure use_cache behavior is correct for generate() + AtlasInfer
        if not args.use_hf_baseline and not args.use_bnb_4bit:
             generate_kwargs["use_cache"] = True # Our wrapper relies on this flag being True

        # Pre-generation memory profile
        if args.profile_memory: profile_memory_usage(0, log_prefix="Pre-Gen")
        print("Starting generation...")

        try:
            with torch.no_grad():
                # Use model's generate method
                output_sequences = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask, # Pass mask
                    **generate_kwargs
                )

            total_time = time.time() - start_time
            print("\nGeneration finished.") # Newline after potential profiling prints

        except Exception as e:
            print(f"\n!!! ERROR during model.generate(): {e} !!!")
            traceback.print_exc()
            if "CUDA out of memory" in str(e):
                 print(">>> CUDA Out of Memory Error! Suggestions:")
                 print("    - Reduce --vram_limit_gb (if AtlasInfer KV Cache)")
                 print("    - Use --force_cpu_offload (if AtlasInfer)")
                 print("    - Reduce RAM/Disk cache limits if system RAM is bottleneck")
                 print("    - Use a smaller model")
                 print("    - Check if other processes are using VRAM/RAM")
            # Try partial output if possible
            output_sequences = input_ids # Fallback to just input

        # Post-generation memory profile
        if args.profile_memory: profile_memory_usage(args.max_new_tokens, log_prefix="Post-Gen")
        print() # Ensure newline after final profile print

        # --- Decode and Print Results ---
        if output_sequences is not None:
            # Handle potential list output from generate
            if isinstance(output_sequences, list):
                 generated_ids = output_sequences[0]
            elif isinstance(output_sequences, torch.Tensor):
                 generated_ids = output_sequences[0] # Get first sequence in batch
            else:
                 print("Warning: Unexpected output type from model.generate()")
                 generated_ids = input_ids # Fallback

            num_generated = len(generated_ids) - start_len
            # Ensure time is not zero
            total_time = max(total_time, 1e-6) # Avoid division by zero
            tokens_per_sec = num_generated / total_time if num_generated > 0 else 0

            print("\n--- Generation Results ---")
            # Move generated_ids to CPU for decoding
            final_output = tokenizer.decode(generated_ids.cpu(), skip_special_tokens=True)
            print(f"\nOutput:\n{final_output}")
            print("-" * 20)
            print(f"Time taken: {total_time:.2f} seconds")
            print(f"Tokens generated: {num_generated} (approx. {args.max_new_tokens} requested)")
            print(f"Throughput: {tokens_per_sec:.2f} tokens/sec")
        else:
             print("\n--- Generation Failed ---")

        # --- Print Cache Stats (AtlasInfer only) ---
        if not args.use_hf_baseline and not args.use_bnb_4bit and memory_manager:
            print("\n--- Memory Manager Stats ---")
            stats = memory_manager.get_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")

    finally: # Use finally block for cleanup
        # --- Cleanup ---
        print("\nCleaning up resources...")
        # Explicitly delete model and manager to release references
        if 'model' in locals() and model is not None:
             del model
        if 'memory_manager' in locals() and memory_manager is not None:
            memory_manager.close()
            del memory_manager
        # Final cleanup
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