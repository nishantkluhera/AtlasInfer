### Qwen/Qwen2.5-0.5B — double-quantized scales (WikiText-2, 40000 tokens, seed 0)

| Method | Weights (MB) | Perplexity | Δ ppl vs FP16 | vs bnb NF4 mem |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 942.3 | 12.279 | +0.000 | — |
| nf4 | 484.1 | 13.067 | +0.788 | 1.12x |
| nf4+dq | 468.2 | 13.066 | +0.788 | 1.09x |
| gptq-nf4 | 484.0 | 12.758 | +0.479 | 1.12x |
| gptq-nf4+dq | 468.2 | 12.757 | +0.478 | 1.09x |

Double-quant cuts NF4 weight memory by **3.3%** (484.1 -> 468.2 MB) at a perplexity change of -0.000. The gap to bnb NF4 (430.4 MB) narrows from +12.5% to +8.8%.
