"""
Mixed-precision quantization within a memory budget, step by step.

This shows the full AtlasInfer pipeline without the high-level wrapper:
    profile sensitivities -> solve the budget allocation -> quantize.

    python examples/02_mixed_precision.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atlasinfer import (
    SensitivityProfiler,
    allocate_optimal,
    quantize_model_mixed,
    print_allocation_report,
)

MODEL = "gpt2"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

    # 1) Measure each layer's INT8/INT4 sensitivity.
    #    profile()           -> fast, activation-local output error (used here)
    #    profile_end_to_end()-> slower, measures each layer's effect on the model's
    #                           loss; this is what AtlasInference uses by default
    #                           and gives noticeably better allocations.
    profiler = SensitivityProfiler()
    profiles = profiler.profile(model, tokenizer=tokenizer)

    # 2) Choose a budget = ~6 bits/weight averaged over the linear layers.
    n_params = sum(p.param_count for p in profiles.values())
    budget_bytes = int(n_params * 6.0 / 8)

    # 3) Solve the exact precision assignment and inspect it.
    allocation = allocate_optimal(profiles, budget_bytes=budget_bytes)
    sensitivities = {n: p.sensitivity() for n, p in profiles.items()}
    print_allocation_report(allocation, sensitivities, top_n=10)

    # 4) Apply it.
    model = model.to("cpu")
    quantize_model_mixed(model, allocation=allocation.allocations)
    model = model.to(device).eval()

    prompt = "In the future, neural networks will"
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    out = model.generate(ids, max_new_tokens=40, do_sample=False)
    print("\n" + tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
