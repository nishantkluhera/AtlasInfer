### EleutherAI/pythia-410m

| Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16.0 | 773.1 | 15.711 | +0.000 |
| uniform-int8 | 8.0 | 497.9 | 15.734 | +0.024 |
| uniform-nf4 | 4.0 | 378.9 | 21.176 | +5.466 |
| mixed-4.5bit | 4.5 | 392.4 | 16.968 | +1.257 |
| mixed-5bit | 5.0 | 409.0 | 16.629 | +0.919 |
| mixed-6bit | 6.0 | 439.5 | 16.418 | +0.707 |
| mixed-7bit | 7.0 | 472.7 | 16.370 | +0.660 |
