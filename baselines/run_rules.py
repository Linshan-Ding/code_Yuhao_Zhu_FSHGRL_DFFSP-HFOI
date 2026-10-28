"""Dispatching-rule baselines on the benchmark instances, and the reference solutions (new code).

    python baselines/run_rules.py                         # all rules, all instances
    python baselines/run_rules.py --rules fifo spt --instances DDT500_num50 --runs-random 3

Outputs
- result/rules/<rule>/eval_results.csv   one row per (instance, run) with the columns of eval.py
  (method = rule name, decode = "rule"); deterministic rules have run1 only, random / rrc have
  --runs-random runs with seeds --seed + k;
- result/rules.csv                        all rules merged;
- data/instances/reference.csv            best deterministic rule per instance (obj_ref = 1 - fulfilment,
  source = rule name, time_limit_s = 0, status = ok), the reference of every gap in the experiment contract
  (``--reference-out none`` leaves it untouched).

The simulator settings (order_top_k, discard_penalty, ...) come from the YAML configuration; the fluid
features are never computed for rules. Timing columns are wall-clock milliseconds of one episode with
``policy_ms`` = time spent in the rule.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from agent.baselines.rules import DETERMINISTIC_RULES, RULES, RulePolicy, run_rule_episode  # noqa: E402
from data.dataset import (REFERENCE_PATH, benchmark_instance_files, instance_files_by_ids, load_index,  # noqa: E402
                          tier_directories, tier_instance_files)
from data.instance_io import parse_instance_file  # noqa: E402
from environment.profiled import ProfiledSimulator  # noqa: E402
from eval import COLS, episode_row, parse_instance_names, write_rows  # noqa: E402
from result.metrics import load_reference  # noqa: E402
from train import load_config  # noqa: E402

REFERENCE_COLS = ["instance_id", "obj_ref", "fulfillment_ref", "source", "time_limit_s", "status"]


def run_rules(rules, instance_files, train_paras: dict, runs_random: int, seed: int, rrc_mode: str,
              reference: dict, index: dict) -> dict:
    rows_by_rule = {rule: [] for rule in rules}
    for rule in rules:
        runs = 1 if rule in DETERMINISTIC_RULES else max(int(runs_random), 1)
        for path in instance_files:
            spec = parse_instance_file(path)
            meta = index.get(path.stem, {})
            started = time.perf_counter()
            rates = []
            for run in range(runs):
                policy = RulePolicy(rule, seed=seed + run, rrc_mode=rrc_mode)
                record = run_rule_episode(policy, spec, train_paras, simulator_factory=ProfiledSimulator)
                row = episode_row(record, path, meta, reference, rule, f"run{run + 1}", "rule", 0, 1,
                                  checkpoint="", checkpoint_iteration=None)
                row["fluid_backend"] = "none"
                rows_by_rule[rule].append(row)
                rates.append(record["fulfillment_rate"])
            print(f"[RULES] {rule:6s} {path.stem:15s} eta mean {sum(rates) / len(rates):.4f} "
                  f"({runs} run{'s' if runs > 1 else ''}, {time.perf_counter() - started:.1f}s)", flush=True)
    return rows_by_rule


def write_reference(rows_by_rule: dict, reference_path: Path) -> list:
    """Best deterministic rule per instance -> reference.csv (minimised objective obj = 1 - fulfilment)."""
    best = {}
    for rule in DETERMINISTIC_RULES:
        for row in rows_by_rule.get(rule, []):
            obj = float(row["obj_model"])
            current = best.get(row["instance_id"])
            if current is None or obj < current["obj_ref"] - 1e-12:
                best[row["instance_id"]] = {"instance_id": row["instance_id"], "obj_ref": obj,
                                            "fulfillment_ref": 1.0 - obj, "source": rule, "time_limit_s": 0,
                                            "status": "ok"}
    if not best:
        return []
    rows = [best[key] for key in sorted(best, key=lambda k: (len(k), k))]
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    with reference_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=REFERENCE_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", nargs="+", default=None, help="YAML overlays (default trio) for the simulator")
    parser.add_argument("--rules", nargs="+", default=["all"], help=f"subset of {RULES} or 'all'")
    parser.add_argument("--instances", nargs="*", default=None, help="instance ids (any tier) or 'all' (default)")
    parser.add_argument("--tiers", nargs="*", default=None,
                        help="tiers to run (default: every indexed tier except 'train'); ignored with --instances")
    parser.add_argument("--runs-random", type=int, default=10, help="independent runs of random / rrc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rrc-mode", choices=["episode", "step"], default="episode")
    parser.add_argument("--out-dir", default="result/rules")
    parser.add_argument("--merged-out", default="result/rules.csv")
    parser.add_argument("--reference-out", default=str(REFERENCE_PATH),
                        help="reference.csv target; 'none' keeps the existing file")
    args = parser.parse_args()

    rules = list(RULES) if args.rules == ["all"] else [rule.lower() for rule in args.rules]
    unknown = [rule for rule in rules if rule not in RULES]
    if unknown:
        parser.error(f"unknown rules {unknown}; choose from {RULES}")
    names = parse_instance_names(args.instances)
    if names and names != ["all"]:
        instance_files = instance_files_by_ids(names)
    else:
        tiers = args.tiers or [p.name for p in tier_directories() if not p.name.endswith("train")]
        instance_files = []
        for tier in tiers:
            instance_files.extend(tier_instance_files(tier))
        print(f"[RULES] tiers {tiers}: {len(instance_files)} instances")
    train_paras = dict(load_config(args.config)["train_paras"])
    index = load_index()
    reference = load_reference()

    rows_by_rule = run_rules(rules, instance_files, train_paras, args.runs_random, args.seed, args.rrc_mode,
                             reference, index)
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    all_rows = []
    for rule, rows in rows_by_rule.items():
        write_rows(rows, out_dir / rule / "eval_results.csv", append=False)
        all_rows.extend(rows)
    merged = Path(args.merged_out)
    merged = merged if merged.is_absolute() else ROOT / merged
    write_rows(all_rows, merged, append=False)

    if str(args.reference_out).lower() != "none" and any(rule in DETERMINISTIC_RULES for rule in rules):
        reference_path = Path(args.reference_out)
        reference_path = reference_path if reference_path.is_absolute() else ROOT / reference_path
        reference_rows = write_reference(rows_by_rule, reference_path)
        print(f"[RULES] reference solutions for {len(reference_rows)} instances -> {reference_path}")
        # gaps of this run were computed against the previous reference (if any); recompute against the new one
        new_reference = {row["instance_id"]: row["obj_ref"] for row in reference_rows}
        from result.metrics import gap_pct

        for row in all_rows:
            ref = new_reference.get(row["instance_id"])
            row["obj_ref"] = ref if ref is not None else ""
            row["gap"] = gap_pct(row["obj_model"], ref)
        for rule, rows in rows_by_rule.items():
            write_rows(rows, out_dir / rule / "eval_results.csv", append=False)
        write_rows(all_rows, merged, append=False)

    frame = pd.DataFrame(all_rows, columns=COLS)
    summary = frame.groupby("method")["fulfillment_rate"].agg(["mean", "min", "max", "count"])
    print("[RULES] fulfilment rate by rule over instances x runs:")
    print(summary.round(4).to_string())
    print(f"[RULES] {len(all_rows)} rows -> {merged}")


if __name__ == "__main__":
    main()
