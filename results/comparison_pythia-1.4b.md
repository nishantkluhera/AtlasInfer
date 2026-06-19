### AtlasInfer vs bitsandbytes — EleutherAI/pythia-1.4b

WikiText-2 perplexity (15k eval tokens) and resident weight memory (params +
buffers), RTX 3060. Reproduce with
`python compare_baselines.py --model EleutherAI/pythia-1.4b`.

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 2698.2 | 12.136 | +0.000 |
| AtlasInfer int8 | 8 | 1601.0 | 12.131 | -0.005 |
| bnb int8 (LLM.int8) | 8 | 1546.2 | 12.192 | +0.056 |
| **AtlasInfer gptq-nf4** | 4 | 1118.2 | 12.498 | **+0.362** |
| AtlasInfer mixed-5bit | 5.0 | 1236.8 | 12.663 | +0.526 |
| AtlasInfer nf4 | 4 | 1118.2 | 12.890 | +0.754 |
| bnb nf4 | 4 | 970.2 | 13.058 | +0.922 |
| AtlasInfer int4 (symmetric) | 4 | 1118.2 | 13.179 | +1.043 |

The 4x-larger scale check confirms the pattern: INT8 lossless and ahead of bnb;
**GPTQ-NF4 (+0.36)** is the best 4-bit method — beating plain NF4 (+0.75), bnb's
NF4 (+0.92), and even 5-bit mixed precision (+0.53), at the same 4-bit memory.
(GPTQ here ran with memory-bounded chunked Hessians so it fits a 1.4B model on
6 GB; bnb stays ~13% smaller at 4-bit via scale double-quantization.)
