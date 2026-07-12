"""
One command to regenerate every number in the README.

Runs the three harnesses — perplexity/memory (benchmark.py), head-to-head vs
bitsandbytes/GPTQ/AWQ (compare_baselines.py), and downstream zero-shot accuracy
(eval_downstream.py) — for each model, each in its own subprocess so GPU memory
is fully released between runs. Everything lands under results/.

    python reproduce.py                                  # default small-model suite
    python reproduce.py --models Qwen/Qwen2.5-0.5B gpt2  # pick models
    python reproduce.py --models Qwen/Qwen2.5-7B --device-map --stages benchmark
    python reproduce.py --stages benchmark baselines     # skip the slow downstream eval

Each stage is independent: a failure in one model/stage is reported and the run
continues, so a single flaky download doesn't sink the whole suite.
"""
import argparse
import subprocess
import sys
import time

DEFAULT_MODELS = ["gpt2", "Qwen/Qwen2.5-0.5B", "Qwen/Qwen3-0.6B-Base"]
STAGES = {
    "benchmark":  [sys.executable, "benchmark.py"],
    "baselines":  [sys.executable, "compare_baselines.py"],
    "downstream": [sys.executable, "eval_downstream.py"],
}


def main():
    ap = argparse.ArgumentParser(description="Reproduce the full AtlasInfer benchmark suite")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--stages", nargs="+", default=list(STAGES),
                    choices=list(STAGES), help="which harnesses to run")
    ap.add_argument("--eval-tokens", type=int, default=40000)
    ap.add_argument("--limit", type=int, default=1000, help="downstream examples/task cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-map", action="store_true", help="shard big models across GPUs")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="baseline/method keys to skip (passed through), e.g. awq")
    args = ap.parse_args()

    results = []  # (model, stage, ok, seconds)
    for model in args.models:
        for stage in args.stages:
            cmd = list(STAGES[stage]) + ["--model", model, "--seed", str(args.seed)]
            if args.device_map:
                cmd.append("--device-map")
            if stage in ("benchmark", "baselines"):
                cmd += ["--eval-tokens", str(args.eval_tokens)]
            if stage == "downstream":
                cmd += ["--limit", str(args.limit)]
            if args.skip and stage in ("baselines", "downstream"):
                cmd += ["--skip", *args.skip]

            print(f"\n{'='*72}\n[{stage}] {model}\n  $ {' '.join(cmd)}\n{'='*72}")
            t0 = time.time()
            rc = subprocess.run(cmd).returncode
            dt = time.time() - t0
            results.append((model, stage, rc == 0, dt))
            if rc != 0:
                print(f"!! [{stage}] {model} FAILED (exit {rc}) — continuing")

    print(f"\n{'='*72}\nSUMMARY\n{'='*72}")
    for model, stage, ok, dt in results:
        print(f"  {'OK ' if ok else 'FAIL'}  {stage:<11} {model:<28} {dt:6.0f}s")
    n_fail = sum(1 for _m, _s, ok, _dt in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} stages succeeded")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
