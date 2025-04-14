# AtlasInfer

**Efficient Large Language Model Inference on Consumer GPUs**

<!-- Badges will go here: Build Status, License, etc. -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

AtlasInfer aims to bridge the gap between powerful Large Language Models (LLMs) and accessible consumer hardware. It enables running demanding LLMs, potentially with longer context lengths than usually possible, on systems with limited VRAM (like gaming laptops or older desktop GPUs) by employing a combination of memory-saving techniques.

The project focuses on:

1.  **Reducing Memory Footprint:** Using an adaptive hybrid quantization scheme (FP8/FP16) to compress model weights while preserving important outlier values, aiming for significant memory reduction with minimal accuracy loss.
2.  **Extending Effective Memory:** Implementing a Unified Memory Manager that treats GPU VRAM, CPU RAM, and optionally NVMe SSD storage as a tiered cache (primarily for the KV cache during generation), allowing models to handle context lengths that would otherwise exceed VRAM capacity.
3.  **Enabling Larger Models:** Incorporating CPU weight offloading (leveraging `accelerate` hooks) to run models whose parameters alone might not fit entirely into VRAM.

**Target Hardware Profile (Initial Development):** Systems like an NVIDIA RTX 3060 Laptop GPU (6GB VRAM), 16GB CPU RAM, and a fast NVMe SSD. The framework aims to be adaptable to higher-end GPUs by automatically disabling offloading if sufficient VRAM is available.

## Key Features

*   **Adaptive Hybrid Quantization:** Quantizes model weights to an emulated FP8 format while keeping significant outlier values in FP16, balancing memory savings and accuracy.
*   **Tiered KV Cache:** Manages attention Key/Value cache across VRAM (fastest), CPU RAM, and NVMe SSD (slowest but largest), using an LRU policy for eviction.
*   **CPU Weight Offloading:** Dynamically keeps only necessary model layers on the GPU during inference, swapping others to/from CPU RAM, enabling models larger than available VRAM.
*   **Generic Integration:** Aims to modify standard Hugging Face `transformers` models (OPT, Llama, Mistral, Gemma supported initially) with minimal model-specific code.
*   **PyTorch Ecosystem:** Built primarily using Python and PyTorch, integrating with `transformers` and `accelerate`.

## Current Status (Alpha / Under Development)

*   **Core Components Implemented:**
    *   Adaptive Quantizer (`atlasinfer_core/quant`)
    *   Unified Memory Manager (VRAM/RAM/Disk KV Cache) (`atlasinfer_core/memory`)
    *   CPU Offloading Hook (`atlasinfer_core/utils`)
    *   Generic Model Wrappers/Modifier (`integration`)
*   **Integration:** Basic integration with Hugging Face `AutoModelForCausalLM` is implemented. Tested initially with `facebook/opt-1.3b`. Support for Llama, Mistral, Gemma relies on defined patterns and needs further testing.
*   **Functionality:** Can load models, apply quantization, wrap attention layers, set up offloading, and run generation using `model.generate()`.
*   **Performance:** **Not yet optimized.** Current implementation focuses on correctness. Python-based quantization, synchronous disk I/O, and lack of custom kernels (like Triton) mean performance will be significantly lower than optimized frameworks (like llama.cpp, vLLM, TGI) or even standard HF + bitsandbytes on capable hardware. The primary goal *at this stage* is enabling models/contexts that would otherwise cause Out-of-Memory errors.
*   **Known Limitations:**
    *   Speed requires significant optimization (Triton kernels, async I/O).
    *   Error handling is basic.
    *   Generality across all model architectures needs more testing.
    *   Requires sufficient CPU RAM to load the initial FP16 model weights before quantization.
    *   `bitsandbytes` baseline might be difficult to install/run on Windows.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/atlasinfer.git # Replace YOUR_USERNAME
    cd atlasinfer
    ```

2.  **Create a Virtual Environment:** (Recommended)
    ```bash
    python -m venv .venv
    ```
    Activate it:
    *   Windows (Command Prompt): `.venv\Scripts\activate.bat`
    *   Windows (PowerShell): `.venv\Scripts\Activate.ps1` (May require `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)
    *   Linux/macOS (bash/zsh): `source .venv/bin/activate`

3.  **Install PyTorch:** Install PyTorch matching your CUDA version. Go to [pytorch.org](https://pytorch.org/) and use the configuration tool. Example (check website for current commands):
    ```bash
    # Example for CUDA 11.8
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    # Example for CUDA 12.1
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

4.  **Create `requirements.txt`:** Create a file named `requirements.txt` in the root directory with the following content:
    ```txt
    # Core ML and numerics
    # torch torchvision torchaudio (Installed separately above)
    numpy
    scipy

    # Disk Caching
    lmdb
    # Note: May need 'python-lmdb' on some systems if 'lmdb' install fails

    # LLM Framework & Utilities
    transformers>=4.30.0 # Ensure a reasonably recent version
    accelerate>=0.20.0 # Needed for hooks and device management
    datasets # For evaluation later
    safetensors

    # Optional: Add later for performance/comparison
    # triton # Requires Linux/WSL and specific GPU/CUDA setup
    # bitsandbytes # Requires Linux/WSL or specific Windows build (tricky)
    ```

5.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The main entry point for running inference is `examples/run_atlas_inference.py`.

**Basic Command:**

```bash
python examples/run_atlas_inference.py --model_name_or_path <model_id> --prompt "<your_prompt>" [OPTIONS]
```

**Key Arguments:**

*   `--model_name_or_path`: Required. Hugging Face model ID (e.g., `facebook/opt-1.3b`, `mistralai/Mistral-7B-v0.1`).
*   `--prompt`: The starting text for generation.
*   `--max_new_tokens`: How many tokens to generate.

**AtlasInfer Specific Arguments:**

*   `--quant_block_size`: Block size for adaptive quantization (default: 128).
*   `--quant_z_threshold`: Z-score threshold for outliers (default: 3.0).
*   `--vram_limit_gb`: Max **KV Cache** VRAM in GB (e.g., `1.0`). If omitted, uses ~90% of *currently free* VRAM.
*   `--ram_cache_limit_gb`: Max **KV Cache** CPU RAM in GB (default: 4.0).
*   `--disk_cache_path`: Path for **KV Cache** disk storage (e.g., `./my_disk_cache`). Set to `None` or `""` to disable disk caching.
*   `--force_cpu_offload`: Always use CPU weight offloading, even if the model might fit in estimated available VRAM. Useful if VRAM estimation is inaccurate or you need maximum VRAM for KV cache.
*   `--vram_only_kv_cache`: Tries to keep KV cache only in VRAM, minimizing RAM/Disk usage for cache (useful on high-VRAM GPUs).

**Baseline Modes:**

*   `--use_hf_baseline`: Run standard FP16 Hugging Face inference (likely requires high VRAM).
*   `--use_bnb_4bit`: Run using `bitsandbytes` 4-bit quantization (requires `bitsandbytes` installation).

**Examples:**

*   **Run OPT 1.3B with AtlasInfer (Low VRAM target):**
    ```bash
    python examples/run_atlas_inference.py ^
        --model_name_or_path "facebook/opt-1.3b" ^
        --prompt "The advantages of functional programming are" ^
        --max_new_tokens 50 ^
        --vram_limit_gb 0.5 ^
        --ram_cache_limit_gb 4.0 ^
        --disk_cache_path "./opt1b_cache"
    ```
    *(Note: `^` is the line continuation character for Windows CMD. Use `\` for PowerShell/Linux/macOS)*

*   **Run Gemma 2B, forcing CPU offload:**
    ```bash
    python examples/run_atlas_inference.py ^
        --model_name_or_path "google/gemma-2b" ^
        --prompt "A poem about the moon:" ^
        --max_new_tokens 60 ^
        --vram_limit_gb 1.0 ^
        --ram_cache_limit_gb 6.0 ^
        --force_cpu_offload
    ```

*   **Run Mistral 7B (Requires ample CPU RAM for initial load, expect slowness):**
    ```bash
    python examples/run_atlas_inference.py ^
        --model_name_or_path "mistralai/Mistral-7B-v0.1" ^
        --prompt "Ingredients for a good life:" ^
        --max_new_tokens 40 ^
        --vram_limit_gb 0.5 ^
        --ram_cache_limit_gb 8.0 ^
        --force_cpu_offload
    ```
*   **Run OPT 1.3B on High-VRAM GPU (No Offload, No Disk Cache):**
    ```bash
    python examples/run_atlas_inference.py ^
        --model_name_or_path "facebook/opt-1.3b" ^
        --prompt "Machine learning is" ^
        --max_new_tokens 50 ^
        --vram_limit_gb 10.0 ^
        --ram_cache_limit_gb 1.0 ^
        --disk_cache_path "None"
    ```


## Architecture Overview

AtlasInfer modifies the standard transformer inference pipeline:

1.  **Loading:** Model structure loaded, weights loaded to CPU (FP16).
2.  **Wrapping:** Attention layers are wrapped by `AtlasAttentionWrapper`.
3.  **Quantization:** Linear layers (excluding embeddings/lm_head by default) are quantized *in-place* on the CPU model using `apply_atlas_quantization_to_model`.
4.  **Placement/Offload:**
    *   If sufficient VRAM exists (estimated after accounting for KV cache VRAM limit), the quantized model is moved to the GPU.
    *   If VRAM is insufficient or `--force_cpu_offload` is used, offloading hooks (`QuantizedTensorOffloadHook`) are attached to decoder layers (and potentially lm_head). The model largely stays on CPU.
5.  **Inference (`model.generate`)**:
    *   Inputs are placed on the initial device (CPU if offloading, GPU otherwise).
    *   **Embedding:** Happens on CPU (if offloading) or GPU.
    *   **Decoder Layers:**
        *   If offloading, the `QuantizedTensorOffloadHook` moves the current layer's quantized weights/bias to GPU (`pre_forward`).
        *   `QuantizedLinear` layers dequantize weights on-the-fly on the GPU.
        *   `AtlasAttentionWrapper` intercepts KV state:
            *   Retrieves past K/V from `UnifiedMemoryManager` (VRAM -> RAM -> Disk).
            *   Computes new K/V.
            *   Stores updated K/V back to `UnifiedMemoryManager` (tries VRAM -> RAM -> Disk).
        *   If offloading, the hook moves weights/bias back to CPU (`post_forward`) and clears the dequantization cache.
    *   **LM Head:** Final prediction computation (hooked if offloading and quantized).
    *   Logits are processed (sampling/greedy) to get the next token.


## Roadmap / Future Work

*   **Performance Optimization:**
    *   Implement custom Triton kernels for fused dequantization + linear layers.
    *   Implement asynchronous disk I/O for the KV cache manager.
    *   Optimize Z-score calculation and quantization process.
    *   Integrate Flash Attention / SDPA more deeply.
*   **Accuracy Evaluation:** Rigorous benchmarking of perplexity and downstream task performance vs FP16 and other quantization methods (BnB, GPTQ, AWQ).
*   **Broader Model Support:** Add and test patterns for more model architectures (Falcon, MPT, Phi, etc.).
*   **Activation Quantization:** Explore applying similar adaptive quantization techniques to activations (more complex).
*   **Advanced Caching:** Investigate more sophisticated cache eviction policies beyond LRU.
*   **Error Handling & Stability:** Improve robustness and error reporting.
*   **Documentation:** Add more detailed API documentation.

## Contributing

Contributions are welcome! Please feel free to open an issue to report bugs, suggest features, or discuss potential improvements. If you'd like to contribute code, please open an issue first to discuss the change and then submit a pull request.

## License

This project is licensed under the **MIT License**. See the `LICENSE` file (you'll need to create one) for details.
