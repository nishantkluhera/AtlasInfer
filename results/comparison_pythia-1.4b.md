### AtlasInfer vs bitsandbytes — EleutherAI/pythia-1.4b

WikiText-2 perplexity (15k eval tokens) and resident weight memory (params +
buffers), RTX 3060. Reproduce with
`python compare_baselines.py --model EleutherAI/pythia-1.4b`.

| Method | ~bits | Weights (MB) | Perplexity | delta vs FP16 |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 16 | 2698.2 | 12.136 | +0.000 |
| AtlasInfer int8 | 8 | 1601.0 | 12.131 | -0.005 |
| bnb int8 (LLM.int8) | 8 | 1546.2 | 12.192 | +0.056 |
| AtlasInfer mixed-5bit | 4.9 | 1236.8 | 12.663 | +0.526 |
| AtlasInfer nf4 | 4 | 1118.2 | 12.890 | +0.754 |
| bnb nf4 | 4 | 970.2 | 13.058 | +0.922 |
| AtlasInfer int4 (symmetric) | 4 | 1118.2 | 13.179 | +1.043 |

**The 410M findings hold at 4x the size:** AtlasInfer's INT8 is lossless and
edges bnb's LLM.int8(); its NF4 beats bnb's NF4 on perplexity (+0.75 vs +0.92);
and mixed precision is best of all at <5 bits (+0.53). Quantization is gentler on
the larger model overall (every method is closer to FP16 than on 410M), which is
the expected trend. bnb's 4-bit remains ~13% smaller (scale double-quantization).
