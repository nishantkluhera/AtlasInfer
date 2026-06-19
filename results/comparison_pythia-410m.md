### AtlasInfer vs bitsandbytes — EleutherAI/pythia-410m

WikiText-2 perplexity (20k eval tokens) and resident weight memory (params +
buffers), measured on an RTX 3060. Reproduce with
`python compare_baselines.py --model EleutherAI/pythia-410m`.

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 773.1 | 15.711 | +0.000 |
| AtlasInfer int8 | 8 | 499.0 | 15.735 | +0.024 |
| bnb int8 (LLM.int8) | 8 | 485.1 | 15.832 | +0.121 |
| AtlasInfer mixed-5bit | 5.0 | 408.7 | 17.195 | +1.484 |
| AtlasInfer gptq-nf4 | 4 | 378.7 | 19.730 | +4.019 |
| AtlasInfer nf4 | 4 | 378.7 | 21.392 | +5.681 |
| bnb nf4 | 4 | 341.1 | 21.649 | +5.938 |
| AtlasInfer int4 (symmetric) | 4 | 378.7 | 22.887 | +7.176 |

**Takeaways**

- **INT8:** AtlasInfer is on par with (slightly ahead of) bitsandbytes'
  LLM.int8() — both effectively lossless at ~half the memory.
- **4-bit:** this old, small model is intrinsically hard to quantize to 4-bit
  (uniform NF4 costs +5.7). The ladder is symmetric int4 (+7.2) -> bnb NF4 (+5.9)
  -> AtlasInfer NF4 (+5.7) -> **GPTQ-NF4 (+4.0)**. GPTQ error compensation helps
  most here in absolute terms but can't fully fix a model this small at 4-bit;
  it's far more effective on the modern Qwen models (Qwen3 +0.45, Qwen2.5 +0.56).
- **Mixed precision** at ~5 bits (+1.48) remains the best sub-8-bit option on this
  model — spending bits only where they're needed.
