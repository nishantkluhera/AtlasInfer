"""
README <-> results consistency guard.

The benchmark tables and the vs-bitsandbytes summary table in README.md are
hand-embedded copies of numbers that live in results/. They drifted once (the
README quoted a pre-regeneration run while results/ held the new one), and a
reader who runs one benchmark and sees it disagree with the README rightly stops
trusting every other number. These tests fail CI if any embedded number no longer
matches its source file, so the README can't silently go stale again.

Sources of truth:
  * benchmark tables      -> results/<safe>.json  (config -> ppl, mb, avg_bits)
  * vs-bitsandbytes table -> results/comparison_<safe>.md (per-method delta)
"""
import json
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")

# Marker name in README  ->  results/<safe>.json basename.
BENCH_TABLES = {
    "Qwen3-0.6B-Base": "Qwen_Qwen3-0.6B-Base",
    "Qwen2.5-0.5B": "Qwen_Qwen2.5-0.5B",
}

# README vs-bnb column label -> config label in the comparison .md.
BNB_COLUMNS = {
    "Atlas int8": "AtlasInfer int8",
    "bnb int8": "bnb int8 (LLM.int8)",
    "bnb nf4": "bnb nf4",
    "Atlas nf4": "AtlasInfer nf4",
    "Atlas gptq-nf4": "AtlasInfer gptq-nf4",
}
# README vs-bnb row label -> results/comparison_<safe>.md basename.
BNB_ROWS = {
    "Qwen3-0.6B": "comparison_Qwen_Qwen3-0.6B-Base",
    "Qwen2.5-0.5B": "comparison_Qwen_Qwen2.5-0.5B",
    "Pythia-1.4B": "comparison_EleutherAI_pythia-1.4b",
    "Pythia-410M": "comparison_EleutherAI_pythia-410m",
}

PPL_TOL = 0.01   # README prints ppl to 3 dp; JSON is the source
MB_TOL = 0.2     # README prints MB to 1 dp
DELTA_TOL = 0.01


def _num(cell: str) -> float:
    """Parse a table cell to float, tolerating +, unicode minus, and bold."""
    s = cell.strip().strip("*").replace("−", "-").replace("+", "")
    return float(s)


def _read_readme() -> str:
    with open(README, encoding="utf-8") as f:
        return f.read()


def _table_rows(block: str):
    """Yield the data-row cell-lists of the first markdown table in `block`."""
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not cells or set("".join(cells)) <= set("-: "):  # header separator
            continue
        yield cells


@pytest.mark.parametrize("marker,safe", list(BENCH_TABLES.items()))
def test_benchmark_table_matches_json(marker, safe):
    """Every row of a RESULTS-marked README table matches results/<safe>.json."""
    readme = _read_readme()
    m = re.search(
        rf"<!-- RESULTS:{re.escape(marker)} -->(.*?)<!-- /RESULTS:{re.escape(marker)} -->",
        readme, re.S,
    )
    assert m, f"RESULTS markers for {marker} not found in README"

    path = os.path.join(ROOT, "results", f"{safe}.json")
    assert os.path.exists(path), f"missing {path}"
    by_config = {r["config"]: r for r in json.load(open(path, encoding="utf-8"))}

    seen = 0
    for cells in _table_rows(m.group(1)):
        # columns: Config | Avg bits | Weights (MB) | Perplexity | delta vs FP16
        if len(cells) != 5 or cells[0] == "Config":
            continue
        config, _bits, mb, ppl, _delta = cells
        assert config in by_config, f"{marker}: README row '{config}' absent from {safe}.json"
        src = by_config[config]
        assert abs(_num(ppl) - src["ppl"]) <= PPL_TOL, (
            f"{marker}/{config}: README ppl {ppl} != json {src['ppl']:.3f}")
        assert abs(_num(mb) - src["mb"]) <= MB_TOL, (
            f"{marker}/{config}: README MB {mb} != json {src['mb']:.1f}")
        seen += 1
    assert seen >= 5, f"{marker}: parsed only {seen} rows — table structure changed?"


def _comparison_deltas(safe: str):
    """method-label -> delta-vs-fp16 from a results/comparison_<safe>.md table."""
    path = os.path.join(ROOT, "results", f"{safe}.md")
    assert os.path.exists(path), f"missing {path}"
    deltas = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) != 5 or cells[0] in ("Method", "") or set("".join(cells)) <= set("-: "):
                continue
            deltas[cells[0]] = _num(cells[4])   # last col = delta vs FP16
    return deltas


def test_vs_bnb_table_matches_comparison_files():
    """Each cell of the README vs-bitsandbytes summary matches the comparison files."""
    readme = _read_readme()
    # Locate the table by its header signature.
    m = re.search(r"\|\s*Model\s*\|\s*Atlas int8\s*\|.*?\n(?:\|.*\n)+", readme)
    assert m, "vs-bitsandbytes summary table not found in README"

    header = None
    checked = 0
    for cells in _table_rows(m.group(0)):
        if cells[0] == "Model":
            header = cells
            continue
        if header is None:
            continue
        row_label = cells[0].split("(")[0].strip()  # drop "(2025)" year suffix
        assert row_label in BNB_ROWS, f"unexpected vs-bnb row '{row_label}'"
        deltas = _comparison_deltas(BNB_ROWS[row_label])
        for col_idx, col_label in enumerate(header[1:], start=1):
            method = BNB_COLUMNS.get(col_label.strip().strip("*"))
            assert method, f"unmapped vs-bnb column '{col_label}'"
            assert method in deltas, f"{row_label}: '{method}' absent from comparison file"
            assert abs(_num(cells[col_idx]) - deltas[method]) <= DELTA_TOL, (
                f"{row_label}/{col_label}: README {cells[col_idx]} != "
                f"comparison {deltas[method]:+.3f}")
            checked += 1
    assert checked >= 16, f"only checked {checked} vs-bnb cells — table structure changed?"
