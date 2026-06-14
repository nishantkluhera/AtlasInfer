### Qwen/Qwen3-0.6B-Base

| Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 1136.9 | 11.803 | +0.000 |
| uniform-int8 | 8.0 | 738.8 | 11.813 | +0.010 |
| uniform-nf4 | 4.0 | 564.7 | 13.143 | +1.340 |
| mixed-4.5bit | 4.5 | 587.0 | 12.682 | +0.879 |
| mixed-5bit | 5.0 | 608.2 | 12.535 | +0.732 |
| mixed-6bit | 5.9 | 650.0 | 12.300 | +0.497 |
| mixed-7bit | 7.0 | 699.8 | 12.120 | +0.317 |
