"""
Inspect which layers are most sensitive to quantization.

    python examples/03_profile_layers.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from atlasinfer import SensitivityProfiler, print_sensitivity_report

MODEL = "gpt2"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(device).eval()

    profiler = SensitivityProfiler()
    profiles = profiler.profile(model, tokenizer=tokenizer)

    # Sensitivity = relative output error under the most aggressive precision (INT4).
    sensitivities = {name: p.sensitivity("int4") for name, p in profiles.items()}
    print_sensitivity_report(sensitivities, top_n=15)


if __name__ == "__main__":
    main()
