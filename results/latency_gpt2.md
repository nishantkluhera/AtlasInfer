### gpt2 - decode latency / peak memory (64 tokens, NVIDIA GeForce RTX 3060 Laptop GPU, best of 3)

| Config | Peak GPU (MB) | vs FP16 | tok/s | rel. speed |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 255.2 | 1.00x | 87.9 | 1.00x |
| uniform-int8 | 194.6 | 0.76x | 52.3 | 0.60x |
| uniform-nf4 | 184.2 | 0.72x | 25.2 | 0.29x |
| mixed-5bit | 193.3 | 0.76x | 29.6 | 0.34x |
