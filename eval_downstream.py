"""
Downstream zero-shot accuracy for AtlasInfer's quantization methods.

Perplexity is a proxy; reviewers rightly discount a quant method evaluated on
perplexity alone. This harness runs the *same* quantized models through
lm-evaluation-harness on standard multiple-choice tasks (ARC, HellaSwag, PIQA,
WinoGrande, ...), so the accuracy claim is measured on the thing people actually
care about - task accuracy - not just held-out NLL.

Every method quantizes a fresh copy of the same HF model in-place and hands the
patched model to lm-eval's HFLM wrapper, so all methods go through one identical
evaluation code path.

Methods:
  * fp16                - dense baseline
  * AtlasInfer int8     - uniform block-wise INT8
  * AtlasInfer nf4      - uniform block-wise NF4
  * AtlasInfer gptq-nf4 - GPTQ error-compensated NF4 (best 4-bit)
  * AtlasInfer mixed    - sensitivity-allocated per-layer precision
  * bnb nf4             - bitsandbytes 4-bit reference (optional, guarded)

Run under Linux/WSL2 + CUDA:
    pip install -e ".[eval]"
    python eval_downstream.py --model Qwen/Qwen2.5-0.5B --limit 500
    python eval_downstream.py --model Qwen/Qwen2.5-7B --device-map --tasks arc_easy piqa
"""
import argparse
import gc
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import torch
import torch.nn as nn

# Windows/CUDA stability: init the CUDA context before importing transformers
# (see benchmark.py for the full note — avoids a 0xC0000005 access violation on
# some Windows torch builds). No-op on CPU.
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from atlasinfer import seed_everything
from atlasinfer.patcher import quantize_model, quantize_model_mixed
from atlasinfer.sensitivity import SensitivityProfiler
from atlasinfer.allocator import allocate_optimal
from atlasinfer.gptq import quantize_model_gptq
from atlasinfer.linear import QuantizedLinear, QuantizedLinear4bit
from atlasinfer.quantizer import (
    dequantize_tensor, dequantize_tensor_nf4, dequantize_tensor_fp4,
)
from benchmark import load_wikitext, model_weight_bytes

hf_logging.set_verbosity_error()

DEFAULT_TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande"]


@torch.no_grad()
def densify_for_eval(model: nn.Module) -> nn.Module:
    """Bake each eager quantized layer's dequantized weights into a dense Linear.

    The eager ``QuantizedLinear`` forward dequantizes its block-wise INT8/NF4
    weights (+ FP16 outliers) to a full FP16 weight and calls ``F.linear`` on
    *every* matmul — correct but ~15x slower than FP16, which makes a full-set
    downstream eval take hours on the eager path (no fused Triton kernel on
    Windows). Since an accuracy eval measures *accuracy, not footprint*, we
    dequantize each layer ONCE into a plain ``nn.Linear``: the output is
    bit-identical (same dequantized weight, same ``F.linear``) but runs at FP16
    speed. Record the true quantized memory BEFORE calling this — densifying
    materializes full FP16 weights and discards the compression.
    """
    repl = []
    for parent in model.modules():
        for attr, child in parent.named_children():
            if isinstance(child, QuantizedLinear):
                w = dequantize_tensor(child.quantized_weights, device=child.q_data.device)
            elif isinstance(child, QuantizedLinear4bit):
                deq = dequantize_tensor_nf4 if child.scheme == "nf4" else dequantize_tensor_fp4
                w = deq(child.quantized_weights, device=child.q_packed.device)
            else:
                continue
            lin = nn.Linear(child.in_features, child.out_features,
                            bias=child.bias is not None)
            lin.weight = nn.Parameter(w.to(torch.float16), requires_grad=False)
            if child.bias is not None:
                lin.bias = nn.Parameter(child.bias.to(torch.float16), requires_grad=False)
            repl.append((parent, attr, lin.to(w.device)))
    for parent, attr, lin in repl:
        setattr(parent, attr, lin)
    return model


def primary_acc(task_result: dict) -> float:
    """Pull the headline accuracy from an lm-eval task result across versions.

    lm-eval renamed metric keys over releases ("acc" -> "acc,none") and some
    tasks report acc_norm. Prefer normalized accuracy when present, else plain
    accuracy, tolerating both the old and new key spellings.
    """
    for key in ("acc_norm,none", "acc_norm", "acc,none", "acc"):
        if key in task_result:
            return float(task_result[key])
    # Last resort: first non-stderr numeric metric.
    for k, v in task_result.items():
        if "stderr" not in k and isinstance(v, (int, float)):
            return float(v)
    return float("nan")


def evaluate(model, tokenizer, tasks, limit, batch_size, device_map):
    """Run lm-eval on an already-quantized HF model; return {task: acc}."""
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    # device_map models are already dispatched; single-GPU models get device="cuda".
    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size,
              device=(None if device_map else "cuda"))
    out = lm_eval.simple_evaluate(model=lm, tasks=list(tasks), limit=limit,
                                  bootstrap_iters=0)
    return {t: primary_acc(r) for t, r in out["results"].items()}


def main():
    ap = argparse.ArgumentParser(description="AtlasInfer downstream-accuracy eval")
    ap.add_argument("--model", "-m", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap examples per task (faster, less precise); None = full set")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--mixed-bits", type=float, default=5.0)
    ap.add_argument("--device-map", action="store_true",
                    help="shard across all GPUs for big models (Kaggle T4x2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip", nargs="*", default=[],
                    help="method keys to skip, e.g. --skip gptq mixed bnb")
    ap.add_argument("--no-fast-eval", action="store_true",
                    help="run the true eager quantized forward (slow) instead of "
                         "baking dequantized weights into dense Linears; identical "
                         "accuracy, ~15x slower. Auto-disabled under --device-map.")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    seed_everything(args.seed)
    assert torch.cuda.is_available(), "needs CUDA"
    print(f"seed={args.seed}  tasks={args.tasks}  limit={args.limit}")

    dmap = "auto" if args.device_map else None
    tok = AutoTokenizer.from_pretrained(args.model)
    calib, _ = load_wikitext()

    def fp16():
        return AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                                                    device_map=dmap)

    def place(model):
        return model if args.device_map else model.to("cuda")

    rows = []

    def record(label, model, bits):
        on_cuda = next(model.parameters()).device.type == "cuda"
        model = model.eval() if on_cuda else model.to("cuda").eval()
        # Footprint is measured on the *quantized* model; then (unless disabled)
        # bake the dequantized weights into dense Linears so the accuracy eval
        # runs at FP16 speed with bit-identical outputs.
        mb = model_weight_bytes(model) / 1024 ** 2
        if not args.no_fast_eval and not args.device_map:
            model = densify_for_eval(model)
        accs = evaluate(model, tok, args.tasks, args.limit, args.batch_size, args.device_map)
        avg = sum(accs.values()) / len(accs)
        cells = "  ".join(f"{t}={accs[t]:.3f}" for t in args.tasks)
        print(f"  {label:<22} avg={avg:.4f}  {cells}  ({mb:.0f} MB, ~{bits}b)")
        rows.append({"method": label, "bits": bits, "mb": mb, "avg": avg, **accs})
        if not args.device_map:
            model.to("cpu")
        del model; gc.collect(); torch.cuda.empty_cache()

    def method(key, label, fn, bits=4):
        if key in args.skip:
            print(f"  {label:<22} SKIPPED (--skip {key})")
            return
        try:
            record(label, fn(), bits)
        except ImportError as exc:
            print(f"  {label:<22} SKIPPED (not installed: {exc})")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label:<22} SKIPPED (failed: {type(exc).__name__}: {exc})")
            gc.collect(); torch.cuda.empty_cache()

    print(f"\nDownstream eval on {args.model}\n")

    record("fp16", fp16(), 16)
    record("AtlasInfer int8", quantize_model(fp16(), precision="int8", verbose=False), 8)
    record("AtlasInfer nf4",
           quantize_model(fp16(), precision="int4", quant_4bit="nf4", verbose=False), 4)

    def gptq_fn():
        gm = place(fp16()).eval()
        quantize_model_gptq(gm, tokenizer=tok, calibration_texts=calib, verbose=False)
        return gm
    method("gptq", "AtlasInfer gptq-nf4", gptq_fn, bits=4)

    def mixed_fn():
        base = place(fp16()).eval()
        profiles = SensitivityProfiler().profile_end_to_end(
            base, tokenizer=tok, calibration_texts=calib)
        if not args.device_map:
            base.to("cpu")
        del base; gc.collect(); torch.cuda.empty_cache()
        n_params = sum(p.param_count for p in profiles.values())
        alloc = allocate_optimal(profiles, budget_bytes=int(n_params * args.mixed_bits / 8))
        return quantize_model_mixed(fp16(), allocation=alloc.allocations, verbose=False)
    method("mixed", f"AtlasInfer mixed-{args.mixed_bits:g}bit", mixed_fn, bits=args.mixed_bits)

    def bnb_fn():
        from transformers import BitsAndBytesConfig
        return AutoModelForCausalLM.from_pretrained(
            args.model, device_map=("auto" if args.device_map else {"": 0}),
            dtype=torch.float16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16))
    method("bnb", "bnb nf4", bnb_fn, bits=4)

    # Table (stdout + file).
    fp16_avg = next(r["avg"] for r in rows if r["method"] == "fp16")
    cols = "| Method | ~bits | MB | " + " | ".join(args.tasks) + " | avg | delta avg vs FP16 |"
    sep = "| --- | ---: | ---: | " + " | ".join("---:" for _ in args.tasks) + " | ---: | ---: |"
    lines = [f"### {args.model} - downstream accuracy (seed {args.seed}, limit {args.limit})",
             "", cols, sep]
    for r in rows:
        task_cells = " | ".join(f"{r[t]:.3f}" for t in args.tasks)
        lines.append(f"| {r['method']} | {r['bits']} | {r['mb']:.0f} | {task_cells} | "
                     f"{r['avg']:.4f} | {r['avg'] - fp16_avg:+.4f} |")
    table = "\n".join(lines) + "\n"
    print("\n" + table)

    os.makedirs(args.out, exist_ok=True)
    safe = args.model.replace("/", "_")
    with open(os.path.join(args.out, f"downstream_{safe}.md"), "w", encoding="utf-8") as f:
        f.write(table)
    with open(os.path.join(args.out, f"downstream_{safe}.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"Wrote results/downstream_{safe}.{{md,json}}")


if __name__ == "__main__":
    main()
