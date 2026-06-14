### AtlasInfer vs bitsandbytes — EleutherAI/pythia-410m

WikiText-2 perplexity (20k eval tokens) and resident weight memory (params +
buffers), measured on an RTX 3060. Reproduce with
`python compare_baselines.py --model EleutherAI/pythia-410m`.

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 773.1 | 15.213 | +0.000 |
| AtlasInfer int8 | 8 | 499.0 | 15.239 | +0.026 |
| bnb int8 (LLM.int8) | 8 | 485.1 | 15.339 | +0.126 |
| AtlasInfer mixed-5bit | 5.0 | 408.7 | 16.698 | +1.484 |
| AtlasInfer nf4 | 4 | 378.7 | 20.670 | +5.456 |
| bnb nf4 | 4 | 341.1 | 21.136 | +5.922 |
| AtlasInfer int4 (symmetric) | 4 | 378.7 | 21.898 | +6.685 |

**Takeaways**

- **INT8:** AtlasInfer is on par with (slightly ahead of) bitsandbytes'
  LLM.int8() — both effectively lossless at ~half the memory.
- **4-bit:** AtlasInfer's NF4 codebook closes the gap that symmetric int4 had —
  the 4-bit penalty drops from +6.69 to **+5.46**, now *ahead of* bnb's NF4
  (+5.92) on perplexity. (bnb is ~37 MB smaller because it double-quantizes its
  block scales while AtlasInfer keeps FP32 scales + sparse FP16 outliers — better
  accuracy for slightly more memory. Scale double-quantization is on the roadmap.)
- **Mixed precision is still the differentiator:** at ~5 bits AtlasInfer holds
  the loss to +1.5, vs +5.9 for any uniform 4-bit method — spending bits only on
  the layers that need them, a knob bitsandbytes doesn't expose.
