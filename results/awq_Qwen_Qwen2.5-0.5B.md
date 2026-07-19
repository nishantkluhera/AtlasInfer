### Qwen/Qwen2.5-0.5B — AWQ activation-aware NF4 (WikiText-2, 40000 tokens, seed 0)

AtlasInfer's from-scratch AWQ ([`awq.py`](../atlasinfer/awq.py)) vs plain NF4 and
GPTQ-NF4. AWQ reaches the GPTQ accuracy tier from an independent, Hessian-free
mechanism (per-input-channel scaling + activation division at run time).

| Method | Weights (MB) | Perplexity | Δ vs FP16 |
| --- | ---: | ---: | ---: |
| fp16 | 942.3 | 12.279 | +0.000 |
| nf4 | 484.1 | 13.067 | +0.788 |
| **awq-nf4** | 513.5 | 12.762 | **+0.483** |
| gptq-nf4 | 484.0 | 12.758 | +0.479 |

AWQ (+0.483) lands on GPTQ (+0.479), roughly halving the plain-NF4 penalty (+0.788)
at 4-bit — validating the from-scratch implementation against a completely
different error-reduction mechanism. It carries more weight memory than plain NF4
(513.5 vs 484.1 MB) because the per-channel up-scaling widens per-block variance,
so more weights cross the sparse-outlier threshold; GPTQ is the more
memory-efficient route to the same accuracy in this implementation. The two are
complementary (AWQ reshapes what's quantized; GPTQ compensates the residual).
Reproduce: `python compare_baselines.py --model Qwen/Qwen2.5-0.5B`.
