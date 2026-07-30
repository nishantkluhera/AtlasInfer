# GPU test checklist

CI runs on a CPU-only GitHub runner, so **the Triton kernel correctness tests
never execute there** — they skip themselves via
`@pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()))`. CI now
prints a warning saying so, but a green check on `main` does *not* mean the
kernels have been verified.

This is the manual gate. Run it on a CUDA host (WSL2 on Windows — see
[wsl_triton.md](wsl_triton.md)) before tagging a release, before publishing any
kernel number, and after touching `triton_kernels.py`.

## 1. Kernel correctness (required)

```bash
pytest tests/test_triton_kernels.py -v
```

Three tests only run here:

| Test | Reference | Bound |
| --- | --- | --- |
| `test_matches_fp16_linear` | `F.linear` on the original FP16 weight | rel < 0.05 |
| `test_with_bias` | same, with bias | rel < 0.05 |
| `test_w4a16_matches_reference` | `F.linear` on the **dequantized packed** weight | rel < 0.02 |

The third is the one that isolates kernel *math* from quantization error — it
should measure ~3e-4 against a 0.02 bound. If it regresses toward the bound, the
kernel is wrong, not merely lossy.

## 2. Full suite on GPU (required)

```bash
pytest tests/ -v -rs
```

Confirm the skip list is **empty**. Anything still skipping here is either a
missing optional dependency or a test gated on something you don't have.

## 3. Kernel benchmark (required if quoting speed)

```bash
python bench_triton_kernel.py --peak-bw 336   # 336 GB/s = RTX 3060 Laptop (192-bit @ 14 Gbps)
```

Writes `results/triton_kernel_<gpu>.json`. Check:

- the **fp16 baseline** reaches a sane fraction of the card's peak bandwidth (it
  printed 85-89% on the 3060). A low number means the baseline is broken and every
  speedup is inflated.
- **efficiency columns**, not just speedup. W8A16 should be near 100% of its
  ideal 2×; W4A16 has historically managed only 26–68% of its ideal 4×.

## 4. End-to-end engine smoke (required)

```bash
python -c "
from atlasinfer import AtlasInference
e = AtlasInference('gpt2', kernel='on')
print(e.generate('The future of on-device AI is', max_tokens=20))
"
```

Must produce coherent text. This is the only check that the fused kernels are
actually wired through `AtlasInference`, as opposed to merely passing unit tests.

## 5. Latency (optional, if quoting tok/s)

```bash
python bench_latency.py --model gpt2 --repeats 5
```

Writes `results/latency_gpt2.json` including every timing sample. Check the
reported spread; if it exceeds ~10% the machine is too noisy to quote from.

---

## Known state — last full run 2026-07-30, WSL2 (Ubuntu 24.04, kernel 6.18)

Host: RTX 3060 Laptop 6 GB, driver 581.80, torch 2.5.1+cu124, Triton 3.1.0,
transformers 5.12.0, bitsandbytes 0.49.2, venv at `~/atlasvenv`.

| Step | Result |
| --- | --- |
| 1. Kernel correctness | **pass** |
| 2. Full suite | **155 passed, 0 skipped** |
| 3. Kernel benchmark | **pass** — see below |
| 4. Engine e2e (`kernel="on"`) | **pass** — 48 fused W8A16 layers, coherent text |
| 5. Latency | **pass**, but see the variance warning below |

- **W8A16 is the real result**: 1.96× of an ideal 2.0× (98% efficiency) at
  (1, 4096, 11008), and accuracy costs only +0.019 ppl. Usable.
- **W4A16 is not**: 1.12–2.66× where 4× is available (28–67% efficiency), **and
  the per-channel int4 format costs +14.39 perplexity**. See
  [tech_debt.md](tech_debt.md) #6.
- **FP16 baseline is competent** — 285–300 GB/s against a 336 GB/s peak (85–89%),
  so the speedups are not measured against a straw man.
- **Latency numbers are too noisy to quote from this machine.** The fp16 row
  varied **36% across 3 runs** on a 64-token decode. Use `--repeats 5`+, check the
  reported spread, and treat anything over ~10% as unquotable. This is a laptop
  GPU sharing the desktop compositor; a headless box would be steadier.
- The kernels implement a *different quantizer* from the one every accuracy
  number uses. Do not quote a speed number and an accuracy number as though they
  describe the same model.

### WSL2 vs Windows

Both work for everything except Triton, which is Linux-only. Differences seen:

- Windows skips 3 kernel tests (no Triton); WSL2 runs all 155.
- `bitsandbytes` imports only on WSL2, so the `bnb-*` comparison rows are
  Windows-unavailable and get logged as FAILED-and-continued there.
- Memory accounting differs by `transformers` version, not by OS — see
  [results/README.md](../results/README.md).
