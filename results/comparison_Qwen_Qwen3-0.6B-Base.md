### Qwen/Qwen3-0.6B-Base  (WikiText-2, 40000 eval tokens, seed 0)

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 1136.9 | 11.804 | +0.000 |
| AtlasInfer int8 | 8 | 739.0 | 11.812 | +0.008 |
| AtlasInfer int4 (sym) | 4 | 568.7 | 13.478 | +1.675 |
| AtlasInfer nf4 | 4 | 568.7 | 12.977 | +1.173 |
| AtlasInfer gptq-nf4 | 4 | 568.7 | 12.312 | +0.508 |
| AtlasInfer mixed-5bit | 5.0 | 611.7 | 12.455 | +0.651 |
| bnb int8 (LLM.int8) | 8 | 716.9 | 11.861 | +0.057 |
| bnb nf4 | 4 | 506.9 | 13.638 | +1.834 |
