"""
AtlasInfer Inference Engine - Main entry point for LLM inference
"""
import torch
import argparse
import time
import sys
import gc
from typing import Optional
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from .patcher import quantize_model, quantize_model_ladq, get_model_info
from .offload import setup_cpu_offload, estimate_model_memory
from .sensitivity import compute_sensitivity_scores
from .allocator import PrecisionAllocator, PrecisionLevel, get_layer_sizes


class AtlasInference:
    """
    High-level API for efficient LLM inference with quantization and offloading.
    
    Example:
        >>> engine = AtlasInference("facebook/opt-1.3b")
        >>> output = engine.generate("The meaning of life is", max_tokens=50)
        >>> print(output)
    """
    
    def __init__(
        self,
        model_name: str,
        quantize: bool = True,
        cpu_offload: bool = False,
        memory_budget_gb: Optional[float] = None,
        block_size: int = 128,
        outlier_threshold: float = 3.0,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_name: HuggingFace model name or path
            quantize: Whether to apply quantization
            cpu_offload: Whether to use CPU offloading (for models larger than VRAM)
            memory_budget_gb: Memory budget for LADQ (enables mixed precision if set)
            block_size: Block size for quantization
            outlier_threshold: Z-score threshold for outlier detection
            device: Target device ('cuda', 'cpu', or None for auto)
            verbose: Print loading progress
        """
        self.model_name = model_name
        self.verbose = verbose
        
        # Determine device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self._log(f"Loading model: {model_name}")
        self._log(f"Device: {self.device}")
        
        # Load tokenizer
        self._log("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Handle missing pad token
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        # Load model to CPU first
        self._log("Loading model weights to CPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self._cleanup_memory()
        
        # Apply quantization
        if quantize:
            if memory_budget_gb is not None:
                # LADQ: Layer-Adaptive Dynamic Quantization (NOVEL)
                self._log(f"Applying LADQ with {memory_budget_gb:.1f} GB budget...")
                self._apply_ladq(memory_budget_gb, verbose)
            else:
                # Uniform FP8 quantization
                self._log("Applying uniform FP8 quantization...")
                quantize_model(
                    self.model,
                    block_size=block_size,
                    outlier_threshold=outlier_threshold,
                    verbose=verbose
                )
            self._cleanup_memory()
        
        # Setup CPU offload or move to GPU
        if cpu_offload and self.device.type == 'cuda':
            self._log("Setting up CPU offloading...")
            setup_cpu_offload(self.model, self.device)
            self._log("Model stays on CPU, layers move to GPU during forward pass")
        elif self.device.type == 'cuda':
            self._log(f"Moving model to {self.device}...")
            self.model.to(self.device)
            self._cleanup_memory()
        
        self.model.eval()
        self._log("Model ready!")
        
        if verbose:
            mem_info = estimate_model_memory(self.model)
            self._log(f"Model memory: {mem_info['total_gb']:.2f} GB")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        do_sample: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-K sampling parameter
            top_p: Top-P (nucleus) sampling parameter
            do_sample: Whether to use sampling (vs greedy decoding)
            **kwargs: Additional generate() arguments
            
        Returns:
            Generated text (including prompt)
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to appropriate device
        if hasattr(self.model, '_hf_hook'):
            # CPU offload mode - inputs stay on CPU
            input_device = torch.device('cpu')
        else:
            input_device = self.device
        
        input_ids = inputs.input_ids.to(input_device)
        attention_mask = inputs.attention_mask.to(input_device)
        
        # Generate
        generate_kwargs = {
            'max_new_tokens': max_tokens,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'do_sample': do_sample,
        }
        
        if do_sample:
            generate_kwargs.update({
                'temperature': temperature,
                'top_k': top_k,
                'top_p': top_p,
            })
        
        generate_kwargs.update(kwargs)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs
            )
        
        # Decode
        output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return output_text
    
    def _apply_ladq(self, memory_budget_gb: float, verbose: bool) -> None:
        """
        Apply Layer-Adaptive Dynamic Quantization (LADQ).
        
        This is the NOVEL contribution of AtlasInfer:
        1. Profile layer sensitivities
        2. Allocate precision within budget
        3. Apply mixed-precision quantization
        """
        self._log("Step 1/3: Profiling layer sensitivities...")
        sensitivities = compute_sensitivity_scores(
            self.model,
            cache_path=f".atlasinfer_cache/{self.model_name.replace('/', '_')}_sensitivity.json"
        )
        
        self._log("Step 2/3: Allocating precision within budget...")
        layer_sizes = get_layer_sizes(self.model)
        allocator = PrecisionAllocator()
        
        memory_budget_bytes = int(memory_budget_gb * 1024**3)
        allocation = allocator.allocate(
            sensitivities=sensitivities,
            layer_sizes=layer_sizes,
            memory_budget_bytes=memory_budget_bytes
        )
        
        if verbose:
            print(f"  {allocation.summary()}")
        
        self._log("Step 3/3: Applying mixed-precision quantization...")
        # Convert PrecisionLevel enum to strings
        precision_dict = {
            name: level.value for name, level in allocation.allocations.items()
        }
        quantize_model_ladq(
            self.model,
            precision_allocation=precision_dict,
            verbose=verbose
        )
    
    def _log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(f"[AtlasInfer] {message}")
    
    def _cleanup_memory(self) -> None:
        """Force garbage collection and CUDA cache flush."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AtlasInfer - Efficient LLM Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference with quantization
  python -m atlasinfer.inference --model facebook/opt-1.3b --prompt "Hello, world!"
  
  # Force CPU offloading for larger models
  python -m atlasinfer.inference --model mistralai/Mistral-7B-v0.1 --prompt "The future of AI" --cpu-offload
  
  # Run without quantization (baseline)
  python -m atlasinfer.inference --model facebook/opt-125m --prompt "Test" --no-quantize
        """
    )
    
    # Model arguments
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='HuggingFace model name or path'
    )
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default="The meaning of life is",
        help='Input prompt for generation'
    )
    
    # Generation arguments
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=50,
        help='Maximum tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Top-K sampling parameter'
    )
    parser.add_argument(
        '--do-sample',
        action='store_true',
        help='Use sampling instead of greedy decoding'
    )
    
    # AtlasInfer arguments
    parser.add_argument(
        '--no-quantize',
        action='store_true',
        help='Disable quantization (run FP16 baseline)'
    )
    parser.add_argument(
        '--memory-budget',
        type=float,
        default=None,
        help='Memory budget in GB for LADQ (enables mixed-precision)'
    )
    parser.add_argument(
        '--cpu-offload',
        action='store_true',
        help='Use CPU offloading for large models'
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=128,
        help='Block size for quantization'
    )
    parser.add_argument(
        '--outlier-threshold',
        type=float,
        default=3.0,
        help='Z-score threshold for outlier detection'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda, cpu, or auto)'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )
    
    return parser.parse_args()


def main():
    """CLI entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("AtlasInfer - Efficient LLM Inference")
    print("=" * 60)
    
    try:
        # Initialize engine
        engine = AtlasInference(
            model_name=args.model,
            quantize=not args.no_quantize,
            cpu_offload=args.cpu_offload,
            memory_budget_gb=args.memory_budget,
            block_size=args.block_size,
            outlier_threshold=args.outlier_threshold,
            device=args.device,
            verbose=not args.quiet
        )
        
        # Generate
        print("\n" + "-" * 60)
        print(f"Prompt: {args.prompt}")
        print("-" * 60)
        
        start_time = time.time()
        output = engine.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            do_sample=args.do_sample
        )
        elapsed = time.time() - start_time
        
        print(f"\nGenerated Output:\n{output}")
        print("-" * 60)
        
        # Stats
        prompt_tokens = len(engine.tokenizer.encode(args.prompt))
        output_tokens = len(engine.tokenizer.encode(output))
        generated_tokens = output_tokens - prompt_tokens
        tokens_per_sec = generated_tokens / elapsed if elapsed > 0 else 0
        
        print(f"Time: {elapsed:.2f}s | Generated: {generated_tokens} tokens | Speed: {tokens_per_sec:.1f} tok/s")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
