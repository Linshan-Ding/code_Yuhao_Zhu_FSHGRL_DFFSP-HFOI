"""Mask-threshold sensitivity (scripts/run_10_sensitivity.py) and the validation-curve products of the
generalisation layout: evaluation overlays, the merged sensitivity table, the policy column of the MILP table,
conv_<method>.csv from val_curve.csv and the training-trace figure."""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "paper_assets" / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "paper_assets" / "scripts"))

from result.aggregate import aggregate  # noqa: E402
from result.plot import plot_training_trace  # noqa: E402
from tests.test_aggregate import _frame  # noqa: E402
from train import load_config  # noqa: E402

TRIO = ["configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", "configs/paper.yaml"]
OVERLAYS = {"off": (False, None), "1e-5": (True, 1e-5), "0.05": (True, 0.05), "0.1": (True, 0.1), "0.2": (True, 0.2)}


def test_sensitivity_overlays_change_only_the_mask():
    base = load_config(TRIO)["train_paras"]
    assert base["max_iterations"] == 1000 and base["val_every"] == 10 and base["instance_selection"] == "round_robin"
    assert base["val_instances"] == ["DDT500_num50.txt", "DDT1200_num100.txt"]
    for value, (mask_on, threshold) in OVERLAYS.items():
        paras = load_config(TRIO + [f"configs/sensitivity/eps_{value}.yaml"])["train_paras"]
        assert paras["fluid_mask"] is mask_on
        if threshold is not None:
            assert paras["fluid_mask_threshold"] == pytest.approx(threshold)
        assert str(paras["fluid_features"]).lower() == "on" and paras["max_iterations"] == 1000
        changed = {k for k in base if paras.get(k) != base[k]}
        assert changed <= {"fluid_mask", "fluid_mask_threshold"}


def test_run_10_declares_every_overlay():
    tree = ast.parse((ROOT / "scripts" / "run_10_sensitivity.py").read_text(encoding="utf-8"))
    thresholds = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                      if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "THRESHOLDS" for t in node.targets))
    assert thresholds == list(OVERLAYS)
    assert all((ROOT / "configs" / "sensitivity" / f"eps_{v}.yaml").exists() for v in thresholds)


def _sensitivity_rows(thresholds, instances):
    rows = []
    for threshold in thresholds:
        for instance, s, greedy in instances:
            for decode, completed in (("greedy", greedy), ("sample", max(greedy - 1, 0))):
                rows.append({"threshold": threshold, "instance_id": instance, "method": "ours", "run_id": f"eps_{threshold}",
                             "tier": "small", "scale": s, "ddt": 500, "decode": decode, "episode": 0,
                             "fulfillment_rate": completed / s, "completed": completed, "total_orders": s,
                             "infer_time_ms": 100.0, "decision_count": 5 * s, "candidate_pairs_mean": 4.0,
                             "masked_pairs_mean": 4.0 if threshold == "off" else 2.5, "mask_fallback_rate": 0.0})
    return pd.DataFrame(rows)


def test_sensitivity_table_and_policy_column_of_the_milp_table(tmp_path, monkeypatch):
    import make_tables as mt

    importlib.reload(mt)
    result, tables = tmp_path / "result", tmp_path / "tables"
    monkeypatch.setattr(mt, "RESULT", result)
    monkeypatch.setattr(mt, "TABLES", tables)
    (result / "milp").mkdir(parents=True)
    instances = [("SML5000_DDT500_num6", 6, 4), ("SML5003_DDT500_num8", 8, 6)]
    _sensitivity_rows(["off", "1e-5"], instances).to_csv(result / "sensitivity.csv", index=False)
    pd.DataFrame([{"instance": i, "tier": "small", "S": s, "n_ops": 5 * s, "n_binary": 10, "n_pairs": 5, "variant": v,
                   "completed_opt": s, "eta_opt": 1.0, "status": "optimal", "gap": 0.0, "solve_s": 0.1, "backend": "highs",
                   "big_L": 1.0, "c_ub": 1.0, "time_limit_s": 60, "verifier_eta": 1.0, "replay_eta": 1.0, "replay_ok": True,
                   "brute_force_eta": "", "brute_force_ok": "", "brute_force_status": "skipped_size",
                   "brute_force_nodes": 0, "brute_force_s": 0.0}
                  for i, s, _ in instances for v in ("A", "B")]).to_csv(result / "milp" / "milp_results.csv", index=False)
    eval_rows = _sensitivity_rows(["1e-5"], instances).drop(columns=["threshold"])
    eval_rows.to_csv(result / "eval_results.csv", index=False)
    mt.sensitivity()
    mt.milp_small()
    table = pd.read_csv(tables / "sensitivity.csv")
    assert list(table["threshold"].astype(str)) == ["off", "1e-5"] and (table["n_instances"] == 2).all()
    assert table["greedy_eta"].iloc[0] == pytest.approx((4 / 6 + 6 / 8) / 2)
    assert table["gap_pp"].iloc[0] == pytest.approx(((1 - 4 / 6) + (1 - 6 / 8)) / 2 * 100)
    assert table["masked_pairs_mean"].tolist() == [4.0, 2.5] and table["decision_time_ms"].iloc[0] == pytest.approx((100 / 30 + 100 / 40) / 2)
    tex = (tables / "sensitivity.tex").read_text(encoding="utf-8")
    assert tex.count("\\\\") == 1 + 2 and "4.00 / 2.50" in tex
    milp = pd.read_csv(tables / "milp_small.csv")
    assert milp["ours_eta"].tolist() == pytest.approx([4 / 6, 6 / 8]) and milp["gap_pp"].tolist() == pytest.approx([100 / 3, 25.0])


def test_convergence_and_training_trace_from_val_curve(tmp_path):
    frame = _frame()
    eval_path = tmp_path / "eval_results.csv"
    frame.to_csv(eval_path, index=False)
    out = tmp_path / "out"
    run_dir = out / "ours" / "run1"
    run_dir.mkdir(parents=True)
    rows = []
    for epoch in (0, 10, 20):
        for instance, obj in (("DDT500_num50", 0.6 - 0.01 * epoch), ("DDT1200_num100", 0.5 - 0.01 * epoch)):
            rows.append({"epoch": epoch, "instance_id": instance, "obj": obj, "fulfillment_rate": 1 - obj,
                         "completed": 10, "total_orders": 50, "wall_time_s": epoch * 1.0})
    pd.DataFrame(rows).to_csv(run_dir / "val_curve.csv", index=False)
    written = aggregate(eval_path, out, "ours")
    conv = pd.read_csv(written["conv_ours"])
    assert conv["epoch"].tolist() == [0, 10, 20] and conv["n_instances"].iloc[0] == 2 and conv["n_runs"].iloc[0] == 1
    assert conv["mean"].tolist() == pytest.approx([0.55, 0.45, 0.35])
    fig = plot_training_trace(run_dir / "val_curve.csv", None, column="fulfillment_rate", x_column="epoch",
                              instance_id="DDT1200_num100", smooth_window=3)
    assert fig is not None
    with pytest.raises(ValueError):
        plot_training_trace(run_dir / "val_curve.csv", None, column="fulfillment_rate", x_column="epoch",
                            instance_id="DDT900_num150")
