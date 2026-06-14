### AtlasInfer vs bitsandbytes — Qwen/Qwen2.5-0.5B

WikiText-2 perplexity (40k eval tokens) and resident weight memory (params +
buffers), RTX 3060. Reproduce with
`python compare_baselines.py --model Qwen/Qwen2.5-0.5B`.

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 942.3 | 12.279 | +0.000 |
| AtlasInfer int8 | 8 | 620.5 | 12.286 | +0.007 |
| bnb int8 (LLM.int8) | 8 | 601.0 | 12.349 | +0.070 |
| AtlasInfer mixed-5bit | 5.7 | 515.1 | 12.787 | +0.509 |
| AtlasInfer nf4 | 4 | 479.6 | 13.222 | +0.943 |
| bnb nf4 | 4 | 430.4 | 13.602 | +1.323 |
| AtlasInfer int4 (symmetric) | 4 | 479.6 | 13.846 | +1.567 |

Same conclusions as on the Pythia models, now on a current (late-2024) model:
INT8 matches/beats bnb's LLM.int8(); **NF4 beats bnb's NF4** clearly (+0.94 vs
+1.32); mixed precision is the best sub-8-bit option (+0.51). bnb remains a bit
smaller at 4-bit (scale double-quantization).
