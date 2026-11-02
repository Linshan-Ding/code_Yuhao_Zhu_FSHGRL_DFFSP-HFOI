"""result/plot.py and result/aggregate.py on synthetic data (Agg backend; files written to tmp_path)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from result.aggregate import aggregate, point_estimates  # noqa: E402
from result.plot import (METHOD_STYLE, aggregate_curves, plot_advantage_heatmap, plot_convergence,  # noqa: E402
                         plot_decision_time, plot_decision_time_methods, plot_method_boxplot, plot_paired_gains,
                         plot_quality_time_tradeoff, plot_sensitivity, plot_training_trace, plot_validation_panels,
                         style_of)
from result.stats import advantage_table  # noqa: E402


def _eval_frame(methods, instances=6, runs=2, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(instances):
        ddt, scale = [500, 900, 1600][i % 3], [50, 100][i // 3]
        for method in methods:
            for run in range(runs if method == "ours" else 1):
                for decode in (["greedy", "sample"] if method in ("ours", "dqn") else ["rule"]):
                    rate = min(1.0, max(0.0, {"ours": 0.6, "nf": 0.4, "dqn": 0.45, "spt": 0.5}[method] + rng.normal(0, 0.05)))
                    rows.append({"instance_id": f"DDT{ddt}_num{scale}", "method": method, "run_id": f"run{run + 1}",
                                 "tier": "benchmark", "scale": scale, "ddt": ddt, "decode": decode, "episode": 0,
                                 "fulfillment_rate": rate, "obj_model": 1 - rate, "obj_ref": 0.5, "gap": (0.5 - rate) * 200,
                                 "infer_time_ms": 500 + 5 * scale + rng.normal(0, 20), "decision_count": 3 * scale})
    return pd.DataFrame(rows)


def _write_curves(root: Path, method: str, instance: str, runs: int):
    for run in range(1, runs + 1):
        run_dir = root / method / instance / f"run{run}"
        run_dir.mkdir(parents=True)
        epochs = np.arange(5)
        obj = 0.9 - 0.05 * epochs + 0.01 * run
        pd.DataFrame({"epoch": epochs, "instance_id": instance, "obj": obj, "obj_best": np.minimum.accumulate(obj),
                      "gap": np.nan, "feasible": True, "wall_time_s": epochs * 10.0}).to_csv(run_dir / "curve.csv", index=False)


def test_point_estimates_prefer_greedy_rows():
    frame = _eval_frame(["ours", "nf", "dqn", "spt"])
    points = point_estimates(frame)
    assert set(points["method"]) == {"ours", "nf", "dqn", "spt"}
    assert "decision_time_ms" in points.columns and points["decision_time_ms"].between(1.0, 20.0).all()
    assert len(points[points["method"] == "ours"]) == 12          # 6 instances x 2 runs, sampled rows dropped


def test_aggregate_curves_and_figures(tmp_path):
    frame = _eval_frame(["ours", "nf", "dqn", "spt"])
    for instance in frame["instance_id"].unique()[:2]:
        _write_curves(tmp_path, "ours", instance, runs=2)
    conv = aggregate_curves(sorted((tmp_path / "ours").glob("*/run*")), key="obj_best")
    assert list(conv.columns) == ["epoch", "mean", "std", "n_instances", "n_runs"] and conv["n_runs"].iloc[0] == 4
    assert conv["std"].notna().all()
    single = aggregate_curves([next((tmp_path / "ours").glob("*/run1"))], key="obj")
    assert single["std"].isna().all()                            # one run -> empty std, not 0

    out = tmp_path / "fig"
    plot_convergence([(conv, "ours")], out / "conv", reference=0.5)
    train_csv = tmp_path / "train_metrics.csv"
    pd.DataFrame({"iteration": range(40), "fulfillment_rate_mean": np.linspace(0.1, 0.6, 40)}).to_csv(train_csv, index=False)
    plot_training_trace(train_csv, out / "trace", smooth_window=7)
    points = point_estimates(frame)
    advantage = advantage_table(points, "ours")
    plot_paired_gains(advantage, ["nf", "dqn"], out / "gains")
    plot_advantage_heatmap(advantage, ["nf", "dqn", "spt"], out / "heat")
    plot_decision_time(points, "ours", out_stem=out / "time")
    plot_quality_time_tradeoff(points, [("rules", ["ours", "spt"]), ("drl", ["ours", "dqn"])], out / "tradeoff")
    plot_method_boxplot(points, ["ours", "nf", "dqn", "spt"], out_stem=out / "box")
    for name in ("conv", "trace", "gains", "heat", "time", "tradeoff", "box"):
        assert (out / f"{name}.pdf").exists() and (out / f"{name}.png").exists()


def test_aggregate_writes_summary_tables(tmp_path):
    frame = _eval_frame(["ours", "nf", "dqn", "spt"])
    eval_path = tmp_path / "eval_results.csv"
    frame.to_csv(eval_path, index=False)
    for instance in frame["instance_id"].unique()[:2]:
        _write_curves(tmp_path, "ours", instance, runs=2)
    written = aggregate(eval_path, tmp_path, "ours")
    for name in ("stats_summary", "paired_tests", "ablation_results", "advantage_table", "friedman", "conv_ours"):
        assert name in written and written[name].exists(), name
    summary = pd.read_csv(written["stats_summary"])
    assert set(summary["metric"]) == {"fulfillment_rate", "gap"}
    paired = pd.read_csv(written["paired_tests"])
    assert list(paired["comparator"]) == ["nf", "dqn", "spt"] and paired["p_holm"].between(0, 1).all()


def test_revision_figures_from_synthetic_frames(tmp_path):
    """The v2 revision figures: validation figure with and without the comparator panel, decision time per method,
    single-panel log-time trade-off with labels or a legend, annotated paired gains, heat map with a mean column and
    the sensitivity figure."""
    frame = _eval_frame(["ours", "nf", "dqn", "spt"])
    points = point_estimates(frame)
    out = tmp_path / "fig"
    rows = []
    for epoch in range(0, 100, 10):
        for instance, base in (("DDT500_num50", 0.2), ("DDT1200_num100", 0.6)):
            rows.append({"epoch": epoch, "instance_id": instance, "obj": 1 - base - 0.001 * epoch,
                         "fulfillment_rate": base + 0.001 * epoch, "completed": 10, "total_orders": 50, "wall_time_s": 1.0})
    val_curve = tmp_path / "val_curve.csv"
    pd.DataFrame(rows).to_csv(val_curve, index=False)
    conv = pd.DataFrame({"epoch": range(0, 100, 10), "mean": np.linspace(0.8, 0.5, 10), "std": np.nan,
                         "n_instances": 2, "n_runs": 1})
    fig = plot_validation_panels(val_curve, [(conv, "ours"), (conv, "dqn")], out / "val", checkpoint=90)
    assert len(fig.axes) == 3
    fig2 = plot_validation_panels(val_curve, [], out / "val2", checkpoint=90)          # the manuscript's two panels
    assert len(fig2.axes) == 2 and fig2.axes[0].get_title(loc="left").startswith("(a)")
    low, high = fig2.axes[0].get_ylim()
    assert 0.0 < low < 0.2 and 0.29 < high < 0.5                                       # scaled to the panel's own data
    plot_decision_time_methods(points, ["ours", "dqn"], out / "timem")
    plot_quality_time_tradeoff(points, [("", ["ours", "nf", "dqn", "spt"])], out / "tradeoff_log", log_x=True, unit="ms",
                               label_fontsize=7)
    fig3 = plot_quality_time_tradeoff(points, [("", ["ours", "nf", "dqn", "spt"])], out / "tradeoff_legend", log_x=True,
                                      unit="ms", annotate=False, legend="right")
    assert fig3.axes[0].get_legend() is not None and not fig3.axes[0].texts
    advantage = advantage_table(points, "ours")
    plot_paired_gains(advantage, ["nf", "dqn"], out / "gains_annotated", annotations={"nf": "6/0/0\np = 0.03"})
    plot_advantage_heatmap(advantage, ["nf", "dqn", "spt"], out / "heat_mean", mean_column=True, fontsize=7)
    table = pd.DataFrame({"threshold": ["off", "1e-5", "0.2"], "greedy_eta": [0.78, 0.78, 0.78],
                          "sampled_eta": [0.73, 0.74, 0.75], "pruned_pct": [0.0, 0.1, 0.3]})
    plot_sensitivity(table, out / "sens")
    for name in ("val", "val2", "timem", "tradeoff_log", "tradeoff_legend", "gains_annotated", "heat_mean", "sens"):
        assert (out / f"{name}.pdf").exists() and (out / f"{name}.png").exists()
    assert style_of("ours") == METHOD_STYLE["ours"] and style_of("unknown", 1) != METHOD_STYLE["ours"]
