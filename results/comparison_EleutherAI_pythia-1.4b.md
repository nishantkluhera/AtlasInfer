### EleutherAI/pythia-1.4b  (WikiText-2, 40000 eval tokens, seed 0)

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 2698.2 | 11.695 | +0.000 |
| AtlasInfer int8 | 8 | 1596.7 | 11.698 | +0.003 |
| AtlasInfer int4 (sym) | 4 | 1120.4 | 12.600 | +0.905 |
| AtlasInfer nf4 | 4 | 1120.4 | 12.408 | +0.713 |
| AtlasInfer gptq-nf4 | 4 | 1120.4 | 11.962 | +0.267 |
| AtlasInfer mixed-5bit | 5.0 | 1238.0 | 12.119 | +0.424 |
| bnb int8 (LLM.int8) | 8 | 1546.2 | 11.731 | +0.036 |
| bnb nf4 | 4 | 970.2 | 12.454 | +0.760 |
