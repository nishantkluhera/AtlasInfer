### Qwen/Qwen2.5-0.5B  (WikiText-2, 40000 eval tokens, seed 0)

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 942.3 | 12.279 | +0.000 |
| AtlasInfer int8 | 8 | 620.9 | 12.279 | +0.001 |
| AtlasInfer int4 (sym) | 4 | 484.1 | 13.716 | +1.437 |
| AtlasInfer nf4 | 4 | 484.1 | 13.067 | +0.788 |
| AtlasInfer gptq-nf4 | 4 | 484.1 | 12.759 | +0.480 |
| AtlasInfer mixed-5bit | 5.0 | 518.8 | 12.739 | +0.460 |
| bnb int8 (LLM.int8) | 8 | 601.0 | 12.349 | +0.070 |
| bnb nf4 | 4 | 430.4 | 13.602 | +1.323 |
