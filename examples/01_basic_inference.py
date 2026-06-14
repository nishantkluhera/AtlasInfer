"""
Basic AtlasInfer usage: load a model, quantize it uniformly to INT8, generate.

    python examples/01_basic_inference.py
"""
from atlasinfer import AtlasInference


def main():
    # Uniform INT8 quantization (no memory budget given).
    engine = AtlasInference("Qwen/Qwen2.5-0.5B", quantize=True)

    prompt = "The key idea behind quantization is"
    print(f"\nPrompt: {prompt}")
    print("-" * 60)
    print(engine.generate(prompt, max_tokens=40))


if __name__ == "__main__":
    main()
