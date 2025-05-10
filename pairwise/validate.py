#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate NL2SQL predictions against gold SQL queries with an **explicit baseline file**.

The script compares:
1. **Baseline** prediction set (LLM 原版 SQL)
2. One or more **test** prediction sets (A / B / …)

and reports:
- Overall ACC for baseline & each test set
- Per‑test **delta vs baseline** (acc_test – acc_baseline)
- Counts of answers each test set newly gets correct while baseline failed
- Venn/UpSet style overlap of *test* sets (baseline excluded from the Venn so the visual focuses on improvements)
- Per‑sample CSV detailing correctness & improvements

Usage
-----
```bash
python evaluate_sql.py \
    --gold gold.json \
    --baseline baseline.json \
    --preds A.json B.json \
    --names A B \
    --out_dir eval_results \
    --single_timeout 30
```

Outputs
-------
<out_dir>/detail.csv   — per‑sample breakdown
<out_dir>/summary.json — overall metrics & delta
<out_dir>/venn.png     — Venn diagram for ≤3 test sets (baseline not drawn)

Dependencies
------------
- `db_utils` (provided separately)
- `matplotlib` & `matplotlib-venn` (optional, for Venn ≤3)
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import db_utils  # local helper module supplied by user

# Optional plotting imports (lazy – skip when unavailable)
try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib_venn import venn2, venn3  # type: ignore
except ImportError:  # plotting is optional
    plt = None  # type: ignore
    venn2 = venn3 = None  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Helpers: loading JSON / JSONL
# ---------------------------------------------------------------------------

def load_jsonl_or_json(path: Path) -> List[Dict[str, Any]]:
    """Load a JSON array **or** JSON Lines file into list[dict]."""
    with path.open("r", encoding="utf-8") as f:
        ch = f.read(1)
        f.seek(0)
        if ch == "[":
            logging.info(f"{path} is a json file")
            return json.load(f)
        logging.info(f"{path} is a jsonl file")
        return [json.loads(line) for line in f]

# ---------------------------------------------------------------------------
# Evaluation core
# ---------------------------------------------------------------------------

def _compare_pair(db_id: str, gold_sql: str, pred_sql: str, timeout: int) -> Dict[str, Any]:
    result = db_utils.compare_sqls(
        db_path=db_utils.build_db_path(db_id),
        predicted_sql=pred_sql,
        ground_truth_sql=gold_sql,
        meta_time_out=timeout,
    )
    # {'exec_res': 0/1, 'exec_err': 'timeout' / '--' / 'incorrect answer'}
    result["correct"] = bool(result["exec_res"])
    return result


def evaluate_with_baseline(
    gold: List[Dict[str, Any]],
    baseline: List[Dict[str, Any]],
    test_sets: List[List[Dict[str, Any]]],
    test_names: List[str],
    timeout: int = 30,
):
    """Run evaluation returning detailed rows & summary dictionaries."""
    n_tests = len(test_sets)
    total = len(gold)

    # Per‑sample correctness arrays
    baseline_correct: List[bool] = []
    test_correct: List[List[bool]] = [[] for _ in range(n_tests)]

    # Improvement counters (baseline wrong & test correct)
    improved_counts = [0] * n_tests

    # Venn / UpSet counter for test sets (exclude baseline)
    combo_counter: Counter[str] = Counter()

    # Detail CSV rows
    detail_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(gold):
        db_id = item["db_id"]
        gold_sql = item.get("sql", item.get("SQL"))
        # logging.info(f"gold_sql-{idx}:{gold_sql[:20]}".replace('\n', ' '))
        
        # ---------------- baseline -----------------
        base_sql = baseline[idx]["sql"]
        base_info = _compare_pair(db_id, gold_sql, base_sql, timeout)
        base_is_correct = base_info["correct"]
        logging.info(f"{idx}: base sql is {base_is_correct}")
        baseline_correct.append(base_is_correct)

        row: Dict[str, Any] = {
            "index": idx,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "baseline_sql": base_sql,
            "baseline_correct": int(base_is_correct),
            "baseline_err": base_info["exec_err"],
        }

        # ---------------- each test set -------------
        bool_vec: List[bool] = []
        for k, (preds, name) in enumerate(zip(test_sets, test_names)):
            pred_sql = preds[idx]["sql"]
            # logging.info(f"pred_sql-{idx}-{k}:{pred_sql[:20]}".replace('\n', ' '))
            info = _compare_pair(db_id, gold_sql, pred_sql, timeout)
            is_correct = info["correct"]
            logging.info(f"pred_sql-{idx}-{k} is {is_correct}")
            test_correct[k].append(is_correct)
            bool_vec.append(is_correct)

            # improvement?
            improved = (not base_is_correct) and is_correct
            if improved:
                improved_counts[k] += 1

            # record in row
            row[f"pred_sql_{name}"] = pred_sql
            row[f"correct_{name}"] = int(is_correct)
            row[f"err_{name}"] = info["exec_err"]
            row[f"improved_{name}"] = int(improved)

        # combo label for Venn of test sets
        label_parts = [test_names[i] for i, ok in enumerate(bool_vec) if ok]
        combo_label = "&".join(label_parts) if label_parts else "None"
        combo_counter[combo_label] += 1

        detail_rows.append(row)

    # ACC calculations
    baseline_acc = sum(baseline_correct) / total
    acc_dict = {name: sum(t) / total for name, t in zip(test_names, test_correct)}
    delta_dict = {name: acc_dict[name] - baseline_acc for name in test_names}
    improved_dict = {name: improved_counts[i] for i, name in enumerate(test_names)}

    return (
        detail_rows,
        {
            "total": total,
            "acc": {"baseline": baseline_acc, **acc_dict},
            "delta_vs_baseline": delta_dict,
            "improved_counts": improved_dict,
            "venn_counts": dict(combo_counter),
        },
        combo_counter,
    )

# ---------------------------------------------------------------------------
# CSV / JSON / plotting helpers
# ---------------------------------------------------------------------------

def save_csv(rows: List[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logging.info("Detail CSV saved → %s", path)


def save_json(obj: Any, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    logging.info("Summary JSON saved → %s", path)


def plot_venn(combo: Counter[str], names: List[str], out_png: Path):
    if len(names) < 2:
        logging.warning("Venn diagram requires ≥2 test sets, skipped.")
        return
    if len(names) > 3:
        logging.warning(">3 test sets detected, skipping Venn (use UpSet instead).")
        return
    if plt is None or (venn2 is None and len(names) <= 3):
        logging.warning("matplotlib‑venn not installed; skipping Venn plot.")
        return

    if len(names) == 2:
        a, b = names
        subsets = (
            combo.get(a, 0),
            combo.get(b, 0),
            combo.get(f"{a}&{b}", 0),
        )
        plt.figure(figsize=(6, 6), dpi=120)
        venn2(subsets=subsets, set_labels=names)
    else:  # 3
        a, b, c = names
        subsets = (
            combo.get(a, 0),
            combo.get(b, 0),
            combo.get(f"{a}&{b}", 0),
            combo.get(c, 0),
            combo.get(f"{a}&{c}", 0),
            combo.get(f"{b}&{c}", 0),
            combo.get(f"{a}&{b}&{c}", 0),
        )
        plt.figure(figsize=(8, 8), dpi=120)
        venn3(subsets=subsets, set_labels=names)

    plt.title("Test Sets Overlap (Venn)")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logging.info("Venn diagram saved → %s", out_png)

# ---------------------------------------------------------------------------
# CLI parsing & main
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Evaluate NL2SQL predictions with explicit baseline.")
    p.add_argument("--gold", type=Path, required=True, help="Gold JSON/JSONL file")
    p.add_argument("--baseline", type=Path, required=True, help="Baseline prediction file")
    p.add_argument("--preds", type=Path, nargs="+", required=True, help="Test prediction files")
    p.add_argument("--names", type=str, nargs="+", help="Readable names for the test files (same order)")
    p.add_argument("--out_dir", type=Path, default=Path("eval_results"))
    p.add_argument("--single_timeout", type=int, default=30)
    p.add_argument("--no_venn", action="store_true", help="Skip Venn diagram")
    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv)

    # ----- label mapping for test sets -----
    if args.names:
        if len(args.names) != len(args.preds):
            raise ValueError("--names count must match --preds count")
        test_names = list(args.names)
    else:
        test_names = [f"T{i+1}" for i in range(len(args.preds))]
    logging.info("Test sets: %s", ", ".join(test_names))

    # ----- load JSON -----
    gold = load_jsonl_or_json(args.gold)
    baseline = load_jsonl_or_json(args.baseline)
    test_sets = [load_jsonl_or_json(p) for p in args.preds]

    n = len(gold)
    if len(baseline) != n:
        raise ValueError("Baseline length differs from gold length")
    for p, name in zip(test_sets, test_names):
        if len(p) != n:
            raise ValueError(f"Test set '{name}' length differs from gold length")

    # ----- run evaluation -----
    detail_rows, summary_dict, venn_counter = evaluate_with_baseline(
        gold,
        baseline,
        test_sets,
        test_names,
        timeout=args.single_timeout,
    )

    # ----- save outputs -----
    out_dir = args.out_dir
    save_csv(detail_rows, out_dir / "detail.csv")
    save_json(summary_dict, out_dir / "summary.json")

    if not args.no_venn:
        plot_venn(venn_counter, test_names, out_dir / "venn.png")

    logging.info("Done. ACC summary: %s", summary_dict["acc"])


if __name__ == "__main__":  # pragma: no cover
    main()
