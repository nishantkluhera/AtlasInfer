# AtlasInfer v2.0

## Overview

AtlasInfer enables efficient LLM inference on consumer hardware with **LADQ** (Layer-Adaptive Dynamic Quantization) - a novel approach that allocates precision per-layer based on sensitivity analysis.

## What Makes LADQ Novel?

| Existing Methods | LADQ |
| --- | --- |
| AWQ: Uses activation statistics to optimize quantization *scales* | Uses sensitivity to allocate *precision levels* |
| GPTQ: One-time calibration, static precision | Dynamic per-layer precision (FP16/FP8/FP4) |
| bitsandbytes: Uniform bit-width for all layers | Budget-aware mixed precision |

**Key insight**: Given a fixed memory budget, spend precision where it matters most.

## Installation

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/AtlasInfer.git
cd AtlasInfer

# Install PyTorch (from pytorch.org for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Inference (Uniform FP8)

```bash
python -m atlasinfer.inference --model facebook/opt-1.3b --prompt "Hello world"
```

### LADQ with Memory Budget

```bash
# Allocate 2GB memory budget - LADQ decides precision per layer
python -m atlasinfer.inference --model facebook/opt-1.3b --prompt "Hello" --memory-budget 2.0
```

### Python API

```python
from atlasinfer import AtlasInference

# Basic
engine = AtlasInference("facebook/opt-1.3b")
print(engine.generate("The future of AI is"))

# With LADQ
engine = AtlasInference("facebook/opt-1.3b", memory_budget_gb=2.0)
```

### Advanced: Manual LADQ Pipeline

```python
from transformers import AutoModelForCausalLM
from atlasinfer import (
    compute_sensitivity_scores,
    PrecisionAllocator,
    get_layer_sizes,
    quantize_model_ladq
)

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("facebook/opt-1.3b", torch_dtype="auto")

# 2. Profile sensitivities
sensitivities = compute_sensitivity_scores(model)

# 3. Allocate precision within budget
allocator = PrecisionAllocator()
layer_sizes = get_layer_sizes(model)
allocation = allocator.allocate(
    sensitivities,
    layer_sizes,
    memory_budget_bytes=2 * 1024**3  # 2 GB
)

# 4. Apply mixed-precision quantization
quantize_model_ladq(model, allocation.allocations)
```

## Architecture

```text
atlasinfer/
├── quantizer.py      # FP8 + FP4 quantization
├── linear.py         # QuantizedLinear, QuantizedLinear4bit
├── sensitivity.py    # Layer sensitivity profiler (NOVEL)
├── allocator.py      # Budget-aware precision allocator (NOVEL)
├── patcher.py        # Model patching (uniform + LADQ)
├── offload.py        # CPU↔GPU offload hooks
└── inference.py      # Main API + CLI
```

## CLI Arguments

| Argument | Description |
| --- | --- |
| `--model, -m` | HuggingFace model name |
| `--prompt, -p` | Input prompt |
| `--memory-budget` | VRAM budget in GB (enables LADQ) |
| `--cpu-offload` | Enable CPU offloading |
| `--no-quantize` | Disable quantization (FP16 baseline) |

## License

MIT License
