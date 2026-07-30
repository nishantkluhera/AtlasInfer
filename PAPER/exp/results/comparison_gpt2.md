### gpt2  (WikiText-2, 2000 eval tokens, seed 0)

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 249.4 | 21.265 | +0.000 |
| AtlasInfer int8 | 8 | 172.6 | 21.306 | +0.041 |
| AtlasInfer int4 (sym) | 4 | 139.6 | 23.522 | +2.257 |
| AtlasInfer nf4 | 4 | 139.6 | 22.254 | +0.989 |
| AtlasInfer gptq-nf4 | 4 | 139.6 | 21.799 | +0.534 |
| AtlasInfer mixed-5bit | 5.0 | 147.7 | 21.952 | +0.687 |
| AtlasInfer greedy-5bit | 5.0 | 147.7 | 21.847 | +0.582 |
| AtlasInfer gptq-mixed-5bit | 5.0 | 147.7 | 21.240 | -0.025 |
