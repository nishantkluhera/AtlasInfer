### EleutherAI/pythia-410m  (WikiText-2, 40000 eval tokens, seed 0)

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 773.1 | 15.711 | +0.000 |
| AtlasInfer int8 | 8 | 497.9 | 15.734 | +0.024 |
| AtlasInfer int4 (sym) | 4 | 378.9 | 23.567 | +7.856 |
| AtlasInfer nf4 | 4 | 378.9 | 21.176 | +5.466 |
| AtlasInfer gptq-nf4 | 4 | 378.9 | 19.594 | +3.883 |
| AtlasInfer mixed-5bit | 5.0 | 409.0 | 16.629 | +0.919 |
| bnb int8 (LLM.int8) | 8 | 485.1 | 15.832 | +0.122 |
| bnb nf4 | 4 | 341.1 | 21.649 | +5.939 |
