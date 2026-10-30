"""result/aggregate.py v2 (integer consistency check, generalization.csv, tier-aware paired tests) and the v2
tables of paper_assets/scripts/make_tables.py on synthetic data; the configuration and instance tables on the
repository's own files."""
from __future__ import annotations

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

from result.aggregate import aggregate, generalization_table, integer_check, point_estimates  # noqa: E402

INSTANCES = {"benchmark": [("DDT500_num50", 50, 500), ("DDT900_num100", 100, 900), ("DDT1600_num150", 150, 1600)],
             "arrival": [("ARR3000_uniform_rho0.8_DDT900_num100", 100, 900), ("ARR3001_uniform_rho1.2_DDT900_num100", 100, 900)]}
COMPLETED = {"ours": 0.8, "nf": 0.6, "hsddqn": 0.5, "spt": 0.7, "fifo": 0.65}       # share of on-time orders


def _frame(sampled: int = 2) -> pd.DataFrame:
    """eval_results-like rows whose fulfilment rates are completed / S by construction."""
    rows = []
    for tier, instances in INSTANCES.items():
        for i, (iid, s, ddt) in enumerate(instances):
            for method, share in COMPLETED.items():
                runs = 2 if method in ("ours", "nf") else 1
                decodes = ["rule"] if method in ("spt", "fifo") else ["greedy"] + ["sample"] * sampled
                for run in range(runs):
                    for episode, decode in enumerate(decodes):
                        completed = int(round(share * s)) - i - run - (1 if decode == "sample" else 0) * episode
                        completed = max(0, min(s, completed))
                        rows.append({"instance_id": iid, "method": method, "run_id": f"run{run + 1}", "tier": tier,
                                     "scale": s, "ddt": ddt, "decode": decode, "episode": episode,
                                     "fulfillment_rate": completed / s, "completed": completed, "discarded": s - completed,
                                     "total_orders": s, "obj_model": 1 - completed / s, "obj_ref": 0.3, "gap": 0.0,
                                     "infer_time_ms": 1000.0, "decision_count": 4 * s})
    return pd.DataFrame(rows)


def test_integer_check_passes_on_consistent_rows_and_flags_a_tampered_mean():
    frame = _frame()
    check = integer_check(frame)
    assert check["ok"].all() and set(check["decode"]) == {"greedy", "sample", "rule"}
    greedy = check[(check["method"] == "ours") & (check["decode"] == "greedy")].iloc[0]
    assert greedy["n_episodes"] == 2 and greedy["on_time_total"] == pytest.approx(round(greedy["on_time_total"]))
    tampered = frame.copy()
    tampered.loc[tampered.index[0], "fulfillment_rate"] = 0.3426                 # 0.3426 x 50 = 17.13 orders
    bad = integer_check(tampered)
    assert not bad["ok"].all() and (~bad["ok"]).sum() == 1
    assert integer_check(pd.DataFrame({"instance_id": ["a"], "method": ["m"]})).empty   # columns missing -> empty


def test_point_estimates_carry_completed_and_total_orders():
    points = point_estimates(_frame())
    assert {"completed", "total_orders", "decision_time_ms"} <= set(points.columns)
    ours = points[(points["method"] == "ours") & (points["instance_id"] == "DDT500_num50")]
    assert len(ours) == 2 and ours["completed"].tolist() == [40, 39]                 # greedy rows only, per run


def test_generalization_table_summarises_every_tier():
    table = generalization_table(point_estimates(_frame()))
    assert set(table["tier"]) == {"benchmark", "arrival"}
    ours = table[(table["tier"] == "arrival") & (table["method"] == "ours")].iloc[0]
    assert ours["K"] == 2 and ours["n_instances"] == 2 and ours["best_rule"] == "spt"
    assert ours["eta_mean"] == pytest.approx((0.8 * 100 - 0.5 + 0.8 * 100 - 1.5) / 2 / 100)
    assert ours["wins"] == 2 and ours["ties"] == 0 and ours["losses"] == 0 and ours["delta_pp"] > 0
    assert ours["completed_mean"] == pytest.approx((80 + 79 + 79 + 78) / 4)
    spt = table[(table["tier"] == "arrival") & (table["method"] == "spt")].iloc[0]
    assert spt["wins"] == "" and pd.isna(spt["delta_pp"])                            # the reference rule itself
    nf = table[(table["tier"] == "benchmark") & (table["method"] == "nf")].iloc[0]
    assert nf["losses"] == 3 and nf["delta_pp"] < 0


def test_aggregate_writes_the_v2_products_and_pairs_on_the_benchmark_tier(tmp_path):
    frame = _frame()
    eval_path = tmp_path / "eval_results.csv"
    frame.to_csv(eval_path, index=False)
    written = aggregate(eval_path, tmp_path / "out", "ours")
    assert {"integer_check", "generalization", "stats_summary", "paired_tests", "ablation_results"} <= set(written)
    paired = pd.read_csv(written["paired_tests"])
    assert (paired["n"] == 3).all()                                                   # benchmark instances only
    assert set(paired["comparator"]) == {"nf", "hsddqn", "spt", "fifo"}
    generalization = pd.read_csv(written["generalization"])
    assert len(generalization) == 2 * len(COMPLETED)
    frame.loc[frame.index[0], "fulfillment_rate"] = 0.3426
    frame.to_csv(eval_path, index=False)
    with pytest.raises(SystemExit):
        aggregate(eval_path, tmp_path / "out2", "ours")
    assert (tmp_path / "out2" / "integer_check.csv").exists()


def test_make_tables_v2_outputs(tmp_path, monkeypatch):
    import make_tables as mt

    importlib.reload(mt)
    result = tmp_path / "result"
    tables = tmp_path / "tables"
    monkeypatch.setattr(mt, "RESULT", result)
    monkeypatch.setattr(mt, "TABLES", tables)
    frame = _frame()
    result.mkdir()
    frame.to_csv(result / "eval_results.csv", index=False)
    aggregate(result / "eval_results.csv", result, "ours")
    frame[frame["method"].isin(["spt", "fifo"])].to_csv(result / "rules.csv", index=False)
    (result / "milp").mkdir()
    pd.DataFrame([
        {"instance": "DDT500_num50", "tier": "tiny", "S": 4, "n_ops": 12, "n_binary": 64, "n_pairs": 36, "variant": "A",
         "completed_opt": 3, "eta_opt": 0.75, "status": "optimal", "gap": 0.0, "solve_s": 0.1, "backend": "highs",
         "big_L": 3650.0, "c_ub": 3210.0, "time_limit_s": 600, "verifier_eta": 0.75, "replay_eta": 0.75,
         "replay_ok": True, "brute_force_eta": 0.75, "brute_force_ok": True, "brute_force_nodes": 100, "brute_force_s": 0.1},
        {"instance": "DDT500_num50", "tier": "tiny", "S": 4, "n_ops": 12, "n_binary": 64, "n_pairs": 36, "variant": "B",
         "completed_opt": 3, "eta_opt": 0.75, "status": "optimal", "gap": 0.0, "solve_s": 0.1, "backend": "highs",
         "big_L": 3650.0, "c_ub": 3210.0, "time_limit_s": 600, "verifier_eta": 0.75, "replay_eta": 0.75,
         "replay_ok": True, "brute_force_eta": 0.75, "brute_force_ok": True, "brute_force_nodes": 100, "brute_force_s": 0.1},
    ]).to_csv(result / "milp" / "milp_results.csv", index=False)
    mt.main()
    names = {p.name for p in tables.iterdir()}
    assert {"ablation.tex", "comparison_pdr.tex", "comparison_drl.tex", "wilcoxon.tex", "generalization_arrival.tex",
            "generalization_arrival.csv", "hyperparameters.tex", "hyperparameters.csv", "instance_statistics.tex",
            "instance_statistics.csv", "milp_small.tex", "milp_small.csv"} <= names
    assert "generalization_benchmark.tex" not in names                                # the paired tier has its own tables
    arrival = (tables / "generalization_arrival.tex").read_text(encoding="utf-8")
    assert "FSHGRL & FSHGRL-NF & HSDDQN & FIFO & SPT" in arrival and arrival.count("ARR300") == 2
    comparison = (tables / "comparison_pdr.tex").read_text(encoding="utf-8")
    assert comparison.count("\\\\") == 1 + 3 + 1                                           # header + 3 configurations + mean
    milp = pd.read_csv(tables / "milp_small.csv")
    assert milp.loc[0, "eta_A"] == 0.75 and milp.loc[0, "replay_ok"] == "yes" and milp.loc[0, "brute_force_ok"] == "yes"
    assert milp.loc[0, "best_rule_eta"] == pytest.approx(0.7)
    hyper = pd.read_csv(tables / "hyperparameters.csv")
    values = dict(zip(hyper["parameter"], hyper["value"].astype(str)))
    assert values["Candidate orders per operation type $k$ (EDD)"] == "5"
    assert values["Iterations / episodes per iteration"] == "1000 / 8"
    assert values["Training pool / validation / test"] == "1000 generated / 2 benchmark ((500, 50), (1200, 100)) / 15 benchmark"
    assert values["Training instance per iteration"] == "the next pool instance (each used once)"
    assert values["Other unseen tiers (arrival / small)"] == "6 / 12 instances"
    assert values["Candidate self-attention: heads / feature width / normalisation"] == "1 / 66 / LayerNorm"
    stats = pd.read_csv(tables / "instance_statistics.csv")
    assert len(stats) == 15 and stats["load_ratio"].between(1.0, 3.0).all()


def test_hyperparameter_rows_follow_the_configuration():
    import make_tables as mt

    config = {"env": {"order_top_k": 3, "fluid_backend": "gurobi", "fluid_mask": False},
              "algo": {"model": {"out_size_ope": 8, "out_size_ma": 8, "num_heads": [1, 1, 1]},
                       "attention": {"use_sparse_attention": False},
                       "train": {"max_iterations": 10, "num_envs": 2}}}
    rows = {p: v for _, p, v in mt.hyperparameter_rows(config, {"ours": 1, "eval": {"sampled_episodes": 4}},
                                                          {"train": 7, "val": 2, "benchmark": 15})}
    assert rows["Candidate orders per operation type $k$ (EDD)"] == 3
    assert rows["Fluid LP solver / time limit (s) / threads"].startswith("Gurobi / 2 /")
    assert rows["Fluid node features / candidate mask"] == "on / off"
    assert rows["HGNN layers / attention heads per layer"] == "3 / 1"
    assert rows["Candidate self-attention: heads / feature width / normalisation"] == "off"
    assert rows["Iterations / episodes per iteration"] == "10 / 2"
    assert rows["Training pool / validation / test"] == "7 generated / 2 benchmark (val tier) / 15 benchmark"
    assert rows["Training instance per iteration"] == "random draw from the pool"
    assert rows["Decoding / independent runs $K$"] == "greedy 1 + sampled 4 episodes / 1"
