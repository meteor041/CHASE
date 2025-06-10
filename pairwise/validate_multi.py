#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate NL2SQL predictions against gold SQL queries with an **explicit baseline file** and optional multithreading.

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
    --single_timeout 30 \
    [--workers 4]
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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple
import multiprocessing
from tqdm import tqdm
import db_utils  # local helper module supplied by user

# Optional plotting imports (lazy – skip when unavailable)
try:
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib_venn import venn2, venn3  # type: ignore
except ImportError:
    plt = None  # type: ignore
    venn2 = venn3 = None  # type: ignore

from concurrent.futures import ThreadPoolExecutor, as_completed

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
            return json.load(f)
        return [json.loads(line) for line in f]

# ---------------------------------------------------------------------------
# Comparison per pair
# ---------------------------------------------------------------------------

def _compare_pair(db_id: str, gold_sql: str, pred_sql: str, timeout: int) -> Dict[str, Any]:
    result = db_utils.compare_sqls(
        db_path=db_utils.build_db_path(db_id),
        predicted_sql=pred_sql,
        ground_truth_sql=gold_sql,
        meta_time_out=timeout,
    )
    result["correct"] = bool(result.get("exec_res", 0))
    return result

# ---------------------------------------------------------------------------
# Evaluation core with multithreading
# ---------------------------------------------------------------------------

def evaluate_with_baseline(
    gold: List[Dict[str, Any]],
    baseline: List[Dict[str, Any]],
    test_sets: List[List[Dict[str, Any]]],
    test_names: List[str],
    timeout: int = 30,
    workers: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Counter[str]]:
    """Run evaluation returning detailed rows, summary dict, and venn counter."""
    n_tests = len(test_sets)
    total = len(gold)

    baseline_correct = [False] * total
    test_correct = [[False] * total for _ in range(n_tests)]
    difficulties = ['simple', 'moderate', 'challenging', 'unknown']
    improved_counts = [{level: 0 for level in difficulties} for _ in range(n_tests)]
    combo_counter: Counter[str] = Counter()

    detail_rows: List[Dict[str, Any]] = [None] * total  # placeholder

    def process(idx: int) -> Tuple[int, Dict[str, Any], bool, List[bool], List[bool], str, str]:
        item = gold[idx]
        db_id = item.get("db_id")
        gold_sql = item.get("sql") or item.get("SQL") or item.get("best_sql") or ""
        difficulty = item.get("difficulty", "unknown")
        # baseline
        base_sql = baseline[idx].get("sql") or baseline[idx].get("SQL") or baseline[idx].get("best_sql") or ""
        base_info = _compare_pair(db_id, gold_sql, base_sql, timeout)
        base_ok = base_info["correct"]
        # logging.info(f"{idx}: base sql is {base_ok}")
        row: Dict[str, Any] = {
            "index": idx,
            "db_id": db_id,
            "gold_sql": gold_sql,
            "baseline_sql": base_sql,
            "baseline_correct": int(base_ok),
            "baseline_err": base_info.get("exec_err", ""),
        }

        bools: List[bool] = []
        improves: List[bool] = []
        # each test
        for k, name in enumerate(test_names):
            pred_sql = test_sets[k][idx].get("sql") or test_sets[k][idx].get("SQL") or test_sets[k][idx].get("best_sql") or ""
            info = _compare_pair(db_id, gold_sql, pred_sql, timeout)
            ok = info["correct"]
            # logging.info(f"pred_sql-{idx}-{k} is {ok}")
            bools.append(ok)
            improve = (not base_ok) and ok
            improves.append(improve)

            row[f"pred_sql_{name}"] = pred_sql
            row[f"correct_{name}"] = int(ok)
            row[f"err_{name}"] = info.get("exec_err", "")
            row[f"improved_{name}"] = int(improve)
        # 组装输出行
        # lines = [f"Sample {idx} results:"]
        # lines.append(f"  baseline: correct={base_ok}, sql={base_sql!r}")
        # for k, name in enumerate(test_names):
        #     sql_k   = row[f"pred_sql_{name}"]
        #     ok_k    = bools[k]
        #     lines.append(f"  {name}: correct={ok_k}")
        # # 一次性打印多行
        # logging.info("\n".join(lines))
        label = "&".join([test_names[i] for i, v in enumerate(bools) if v]) or "None"
        return idx, row, base_ok, bools, improves, label, difficulty

    # ThreadPoolExecutor
    workers = workers or multiprocessing.cpu_count()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process, i): i for i in range(total)}
        for future in tqdm(as_completed(futures), total=total, desc="Evaluating samples"):
            i, row, base_ok, bools, improves, label, difficulty = future.result()
            detail_rows[i] = row
            baseline_correct[i] = base_ok
            for k in range(n_tests):
                test_correct[k][i] = bools[k]
                if improves[k]:
                    improved_counts[k][difficulty] += 1
            combo_counter[label] += 1

    # summary
    baseline_acc = sum(baseline_correct) / total
    acc_dict = {name: sum(test_correct[k]) / total for k, name in enumerate(test_names)}
    delta_dict = {name: acc_dict[name] - baseline_acc for name in test_names}
    improved_dict = {name: improved_counts[k] for k, name in enumerate(test_names)}

    summary = {
        "total": total,
        "acc": {"baseline": baseline_acc, **acc_dict},
        "delta_vs_baseline": delta_dict,
        "improved_counts": improved_dict,
        "venn_counts": dict(combo_counter),
    }
    return detail_rows, summary, combo_counter

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
    if len(names) < 2 or len(names) > 3 or plt is None:
        return
    if len(names) == 2:
        a, b = names
        subsets = (combo.get(a, 0), combo.get(b, 0), combo.get(f"{a}&{b}", 0))
        plt.figure(figsize=(6, 6), dpi=120)
        venn2(subsets=subsets, set_labels=names)
    else:
        a, b, c = names
        subsets = (
            combo.get(a, 0), combo.get(b, 0), combo.get(f"{a}&{b}", 0),
            combo.get(c, 0), combo.get(f"{a}&{c}", 0), combo.get(f"{b}&{c}", 0),
            combo.get(f"{a}&{b}&{c}", 0)
        )
        plt.figure(figsize=(8, 8), dpi=120)
        venn3(subsets=subsets, set_labels=names)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()
    logging.info("Venn diagram saved → %s", out_png)

# ---------------------------------------------------------------------------
# CLI parsing & main
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Evaluate NL2SQL predictions with explicit baseline and multithreading.")
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--preds", type=Path, nargs="+", required=True)
    p.add_argument("--names", type=str, nargs="+", help="Readable names for the test files")
    p.add_argument("--out_dir", type=Path, default=Path("eval_results"))
    p.add_argument("--single_timeout", type=int, default=30)
    p.add_argument("--workers", type=int, default=1, help="Number of threads to use; 0 for auto")
    p.add_argument("--no_venn", action="store_true")
    return p.parse_args(argv)


def main(argv: List[str] | None = None):
    args = parse_args(argv)
    if args.names and len(args.names) != len(args.preds):
        raise ValueError("--names count must match --preds count")
    test_names = args.names or [f"T{i+1}" for i in range(len(args.preds))]
    logging.info("Test sets: %s", ", ".join(test_names))

    gold = load_jsonl_or_json(args.gold)
    baseline = load_jsonl_or_json(args.baseline)
    test_sets = [load_jsonl_or_json(p) for p in args.preds]
    n = len(gold)
    if len(baseline) != n or any(len(ts) != n for ts in test_sets):
        raise ValueError("All input files must have the same number of entries as gold.")

    detail_rows, summary, venn_counter = evaluate_with_baseline(
        gold, baseline, test_sets, test_names,
        timeout=args.single_timeout,
        workers=args.workers
    )
    out_dir = args.out_dir
    save_csv(detail_rows, out_dir / "detail.csv")
    save_json(summary, out_dir / "summary.json")
    if not args.no_venn:
        plot_venn(venn_counter, test_names, out_dir / "venn.png")
    logging.info("Done. ACC summary: %s", summary["acc"])

if __name__ == "__main__":
    main()
