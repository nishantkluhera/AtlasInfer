### AtlasInfer vs bitsandbytes — Qwen/Qwen2.5-0.5B

WikiText-2 perplexity (40k eval tokens) and resident weight memory (params +
buffers), RTX 3060. Reproduce with
`python compare_baselines.py --model Qwen/Qwen2.5-0.5B`.

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 942.3 | 12.279 | +0.000 |
| AtlasInfer int8 | 8 | 620.5 | 12.286 | +0.007 |
| bnb int8 (LLM.int8) | 8 | 601.0 | 12.349 | +0.070 |
| AtlasInfer mixed-5bit | 5.0 | 515.1 | 12.787 | +0.509 |
| **AtlasInfer gptq-nf4** | 4 | 479.6 | 12.836 | **+0.558** |
| AtlasInfer nf4 | 4 | 479.6 | 13.222 | +0.943 |
| bnb nf4 | 4 | 430.4 | 13.602 | +1.323 |
| AtlasInfer int4 (symmetric) | 4 | 479.6 | 13.846 | +1.567 |

**Takeaways**

- **INT8:** AtlasInfer is on par with (slightly ahead of) bitsandbytes' LLM.int8().
- **4-bit:** the 4-bit ladder is symmetric int4 (+1.57) → bnb NF4 (+1.32) →
  AtlasInfer NF4 (+0.94) → **AtlasInfer GPTQ-NF4 (+0.56)**. GPTQ error
  compensation cuts the 4-bit penalty ~41% below plain NF4 and ~58% below bnb's
  NF4, at the *same* memory — and nearly matches 5-bit mixed precision while using
  only 4 bits.
- **Calibration matters:** GPTQ needs a well-conditioned Hessian. With ~1k
  calibration tokens it *hurt* (+1.1); with ~65k tokens (128x512) it gives the
  +0.56 above. Documented as a real gotcha.
