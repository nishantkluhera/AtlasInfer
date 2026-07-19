### Qwen/Qwen3-0.6B-Base

| Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 1136.9 | 11.804 | +0.000 |
| uniform-int8 | 8.0 | 739.0 | 11.812 | +0.008 |
| uniform-nf4 | 4.0 | 568.7 | 12.977 | +1.173 |
| mixed-4.5bit | 4.5 | 590.6 | 12.591 | +0.787 |
| mixed-5bit | 5.0 | 611.7 | 12.455 | +0.651 |
| mixed-6bit | 5.9 | 652.3 | 12.227 | +0.424 |
| mixed-7bit | 7.0 | 701.7 | 12.110 | +0.306 |
