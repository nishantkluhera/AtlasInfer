### EleutherAI/pythia-410m

| Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 773.1 | 15.712 | +0.000 |
| uniform-int8 | 8.0 | 499.0 | 15.740 | +0.029 |
| uniform-nf4 | 4.0 | 378.7 | 21.391 | +5.679 |
| mixed-4.5bit | 4.5 | 392.4 | 17.115 | +1.403 |
| mixed-5bit | 5.0 | 409.1 | 16.769 | +1.057 |
| mixed-6bit | 6.2 | 439.5 | 16.523 | +0.812 |
| mixed-7bit | 7.1 | 471.9 | 16.396 | +0.685 |
