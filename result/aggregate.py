"""Aggregate the evaluation CSVs into the summary tables of the experiment contract (new code).

    python -m result.aggregate [--eval result/eval_results.csv] [--out-dir result] [--reference ours]

Inputs: ``result/eval_results.csv`` (all methods: ours, ablations, learned baselines, rules; one row per
instance x run x episode) and the per-instance run directories ``result/<method>/<instance>/run<k>/curve.csv``.
Point estimates keep the greedy / incumbent / rule rows (sampled episodes are averaged only for methods
without them) and add ``decision_time_ms`` = episode wall-clock / number of decisions, the per-decision time
the paper reports as "time (s)" (``infer_time_ms`` stays the whole-episode wall-clock).
Outputs (all in ``--out-dir``): ``stats_summary.csv`` (mean +- std over runs per method / tier for the
fulfilment rate and the gap), ``paired_tests.csv`` (Table 11 layout against ``--reference`` on the paired
tier, benchmark by default), ``ablation_results.csv``, ``advantage_table.csv``, ``friedman.csv``,
``conv_<method>.csv`` (convergence curves: the validation objective of ``result/<method>/run*/val_curve.csv``
averaged over the validation instances and runs in the generalisation layout, the incumbent objective of
``curve.csv`` in the per-instance layout), and, v2:
``integer_check.csv`` (every (instance, method, decode) group: S x n_episodes x mean eta must be an integer,
the count of on-time orders; a violation aborts the aggregation) and ``generalization.csv`` (zero-shot
results per method and evaluation tier: K runs, mean +- std of eta, completed_mean, the best deterministic
rule of the tier and the per-instance wins / ties / losses against it). Figures are drawn by
``paper_assets/figures/src/*.py``.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result.plot import aggregate_curves  # noqa: E402
from result.stats import (ABLATION_METHODS, DETERMINISTIC_RULE_METHODS, DRL_METHODS, RULE_METHODS,  # noqa: E402
                          ablation_table, advantage_table, friedman, paired_table, summarize)

INTEGER_TOL = 1e-9


def load_eval_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("fulfillment_rate", "gap", "obj_model", "infer_time_ms", "ddt", "scale"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def point_estimates(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (instance, method, run): greedy / incumbent rows only, sampled episodes averaged."""
    data = frame.copy()
    if "decode" in data.columns:
        preferred = data[data["decode"].isin(["greedy", "incumbent", "rule"])]
        others = data[~data["method"].isin(preferred["method"].unique())]
        data = pd.concat([preferred, others], ignore_index=True)
    if "infer_time_ms" in data.columns and "decision_count" in data.columns:   # wall-clock per decision
        counts = pd.to_numeric(data["decision_count"], errors="coerce")
        data["decision_time_ms"] = pd.to_numeric(data["infer_time_ms"], errors="coerce") / counts.where(counts > 0)
    keys = ["instance_id", "method", "run_id"]
    numeric = ["fulfillment_rate", "gap", "obj_model", "infer_time_ms", "decision_time_ms", "completed"]
    numeric = [c for c in numeric if c in data.columns]
    meta = data.drop_duplicates(keys).set_index(keys)[[c for c in ("tier", "scale", "ddt", "total_orders")
                                                       if c in data.columns]]
    agg = data.groupby(keys)[numeric].mean()
    return meta.join(agg).reset_index()


def integer_check(frame: pd.DataFrame, tol: float = INTEGER_TOL) -> pd.DataFrame:
    """One row per (instance, method, decode): the mean fulfilment times S times the number of episodes is the
    total number of on-time orders and must be an integer (reviewer 1, comment 14); ``completed`` rows are
    checked against ``fulfillment_rate * total_orders`` as well."""
    required = {"instance_id", "method", "fulfillment_rate", "total_orders"}
    if not required <= set(frame.columns):
        return pd.DataFrame(columns=["instance_id", "method", "decode", "n_episodes", "S", "eta_mean",
                                     "on_time_total", "ok"])
    data = frame.copy()
    data["decode"] = data["decode"] if "decode" in data.columns else "all"
    data["fulfillment_rate"] = pd.to_numeric(data["fulfillment_rate"], errors="coerce")
    data["total_orders"] = pd.to_numeric(data["total_orders"], errors="coerce")
    rows = []
    for (instance, method, decode), group in data.groupby(["instance_id", "method", "decode"], sort=True):
        s_values = group["total_orders"].dropna().unique()
        s = float(s_values[0]) if len(s_values) == 1 else float("nan")
        n = int(len(group))
        eta = float(group["fulfillment_rate"].mean())
        total = eta * s * n
        ok = len(s_values) == 1 and abs(total - round(total)) <= tol * max(1.0, abs(total))
        if ok and "completed" in group.columns:
            completed = pd.to_numeric(group["completed"], errors="coerce")
            if completed.notna().all():
                ok = bool(np.allclose(completed, group["fulfillment_rate"] * s, atol=1e-6))
                ok = ok and abs(float(completed.sum()) - total) <= 1e-6 * max(1.0, abs(total))
        rows.append({"instance_id": instance, "method": method, "decode": decode, "n_episodes": n, "S": s,
                     "eta_mean": eta, "on_time_total": total, "ok": bool(ok)})
    return pd.DataFrame(rows)


def generalization_table(points: pd.DataFrame, rules=DETERMINISTIC_RULE_METHODS) -> pd.DataFrame:
    """Zero-shot summary per (method, tier) from the point estimates: K runs, mean +- std of the run means of
    eta, completed_mean (= S x eta averaged over the instances), the best deterministic rule of the tier (by
    mean eta) and the wins / ties / losses of the method against that rule over the tier's instances."""
    if points.empty or "tier" not in points.columns:
        return pd.DataFrame()
    data = points.copy()
    data["fulfillment_rate"] = pd.to_numeric(data["fulfillment_rate"], errors="coerce")
    per_instance = data.groupby(["tier", "method", "instance_id"])["fulfillment_rate"].mean()
    rows = []
    for tier, tier_points in data.groupby("tier", sort=True):
        rule_means = {rule: tier_points[tier_points["method"] == rule]["fulfillment_rate"].mean()
                      for rule in rules if (tier_points["method"] == rule).any()}
        best_rule = max(rule_means, key=rule_means.get) if rule_means else ""
        for method, method_points in tier_points.groupby("method", sort=True):
            per_run = method_points.groupby("run_id")["fulfillment_rate"].mean()
            row = {"tier": tier, "method": method, "n_instances": int(method_points["instance_id"].nunique()),
                   "K": int(per_run.size), "eta_mean": float(per_run.mean()),
                   "eta_std": float(per_run.std(ddof=1)) if per_run.size > 1 else float("nan"),
                   "completed_mean": float(method_points["completed"].mean()) if "completed" in method_points else float("nan"),
                   "decision_time_ms": float(method_points["decision_time_ms"].mean()) if "decision_time_ms" in method_points else float("nan"),
                   "best_rule": best_rule, "best_rule_eta": rule_means.get(best_rule, float("nan")),
                   "delta_pp": float("nan"), "wins": "", "ties": "", "losses": ""}
            if best_rule and method != best_rule:
                mine = per_instance.loc[(tier, method)]
                theirs = per_instance.loc[(tier, best_rule)]
                mine, theirs = mine.align(theirs, join="inner")
                diff = (mine - theirs).round(9)
                row.update({"delta_pp": float(diff.mean() * 100.0), "wins": int((diff > 0).sum()),
                            "ties": int((diff == 0).sum()), "losses": int((diff < 0).sum())})
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(eval_path: Path, out_dir: Path, reference: str = "ours", paired_tier: str = "benchmark") -> dict:
    raw = load_eval_results(eval_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}

    check = integer_check(raw)
    check.to_csv(out_dir / "integer_check.csv", index=False)
    written["integer_check"] = out_dir / "integer_check.csv"
    if not check.empty and not check["ok"].all():
        bad = check[~check["ok"]]
        raise SystemExit(f"[FAIL] integer consistency: S x n_episodes x mean eta is not the number of on-time "
                         f"orders for {len(bad)} (instance, method, decode) group(s); see {written['integer_check']}\n"
                         f"{bad.head(10).to_string(index=False)}")

    all_points = point_estimates(raw)
    summary = pd.concat([summarize(all_points, metric="fulfillment_rate"), summarize(all_points, metric="gap")],
                        ignore_index=True)
    summary.to_csv(out_dir / "stats_summary.csv", index=False)
    written["stats_summary"] = out_dir / "stats_summary.csv"

    generalization = generalization_table(all_points)
    generalization.to_csv(out_dir / "generalization.csv", index=False)
    written["generalization"] = out_dir / "generalization.csv"

    # paired statistics, ablation and advantage tables on one tier (the 15 benchmark configurations of the paper)
    frame = all_points[all_points["tier"] == paired_tier] if "tier" in all_points.columns else all_points
    if frame.empty:
        frame = all_points
    comparators = [m for m in list(ABLATION_METHODS) + list(DRL_METHODS) + list(RULE_METHODS)
                   if (frame["method"] == m).any()]
    if (frame["method"] == reference).any() and comparators:
        paired = paired_table(frame, reference, comparators, metric="fulfillment_rate", decimals=4)
        paired.to_csv(out_dir / "paired_tests.csv", index=False)
        written["paired_tests"] = out_dir / "paired_tests.csv"
        ablation = ablation_table(frame, methods=(reference,) + tuple(ABLATION_METHODS))
        ablation.to_csv(out_dir / "ablation_results.csv", index=False)
        written["ablation_results"] = out_dir / "ablation_results.csv"
        advantage = advantage_table(frame, reference)
        advantage.to_csv(out_dir / "advantage_table.csv", index=False)
        written["advantage_table"] = out_dir / "advantage_table.csv"
        methods = [reference] + comparators
        matrix = frame.pivot_table(index="instance_id", columns="method", values="fulfillment_rate", aggfunc="mean")
        matrix = matrix[[m for m in methods if m in matrix.columns]].dropna()
        if matrix.shape[1] >= 3 and matrix.shape[0] >= 2:
            result = friedman(-matrix.values)       # ranks: 1 = highest fulfilment
            rows = [{"method": m, "mean_rank": r} for m, r in zip(matrix.columns, result["mean_ranks"])]
            table = pd.DataFrame(rows)
            table["statistic"] = result["statistic"]
            table["p_value"] = result["p_value"]
            table["n_instances"] = result["n_blocks"]
            table.to_csv(out_dir / "friedman.csv", index=False)
            written["friedman"] = out_dir / "friedman.csv"

    for method_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        val_runs = sorted(glob.glob(str(method_dir / "run*" / "val_curve.csv")))      # generalisation layout
        if val_runs:                     # validation objective (mean over the validation instances) per epoch
            conv = aggregate_curves([Path(p).parent for p in val_runs], key="obj", filename="val_curve.csv")
        else:                            # per-instance layout: incumbent objective of the training instance
            run_dirs = sorted(glob.glob(str(method_dir / "*" / "run*" / "curve.csv")))
            if not run_dirs:
                continue
            conv = aggregate_curves([Path(p).parent for p in run_dirs], key="obj_best")
        conv.to_csv(out_dir / f"conv_{method_dir.name}.csv", index=False)
        written[f"conv_{method_dir.name}"] = out_dir / f"conv_{method_dir.name}.csv"
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval", default="result/eval_results.csv")
    parser.add_argument("--out-dir", default="result")
    parser.add_argument("--reference", default="ours")
    parser.add_argument("--paired-tier", default="benchmark",
                        help="tier of the paired tests, ablation and advantage tables (all tiers when absent)")
    args = parser.parse_args()
    eval_path = Path(args.eval)
    eval_path = eval_path if eval_path.is_absolute() else ROOT / eval_path
    out_dir = Path(args.out_dir)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    if not eval_path.exists():
        raise SystemExit(f"[FAIL] {eval_path} not found: run scripts/run_07_eval_main.py first")
    written = aggregate(eval_path, out_dir, args.reference, args.paired_tier)
    for name, path in written.items():
        print(f"[AGGREGATE] {name}: {path}")


if __name__ == "__main__":
    main()
