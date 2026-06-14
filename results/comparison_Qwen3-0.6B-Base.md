### AtlasInfer vs bitsandbytes — Qwen/Qwen3-0.6B-Base

WikiText-2 perplexity (40k eval tokens) and resident weight memory (params +
buffers), RTX 3060. Qwen3 (2025) is the most current model tested. Reproduce with
`python compare_baselines.py --model Qwen/Qwen3-0.6B-Base`.

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 1136.9 | 11.804 | +0.000 |
| AtlasInfer int8 | 8 | 738.8 | 11.813 | +0.009 |
| bnb int8 (LLM.int8) | 8 | 716.9 | 11.861 | +0.057 |
| AtlasInfer mixed-5bit | 5.0 | 608.2 | 12.526 | +0.722 |
| AtlasInfer nf4 | 4 | 564.7 | 13.143 | +1.339 |
| bnb nf4 | 4 | 506.9 | 13.638 | +1.834 |
| AtlasInfer int4 (symmetric) | 4 | 564.7 | 13.670 | +1.866 |

On the most current model in the suite the pattern is unchanged: INT8 lossless
and ahead of bnb's LLM.int8(); **NF4 beats bnb's NF4** (+1.34 vs +1.83); mixed
precision is the best sub-8-bit option (+0.72). bnb stays ~10% smaller at 4-bit
(scale double-quantization).
