"""v2 generalisation paradigm: tier index, validation tracker, pilot verdict, evaluation tiers, scripts' job
builders and one two-iteration training run on a generated pool with validation."""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for folder in (ROOT, SCRIPTS):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from configs.config import GeneratorCfg  # noqa: E402
from data.dataset import instance_files_by_ids, load_index, prepare_all, tier_instance_files  # noqa: E402
from data.generator import generate_pool  # noqa: E402
from result.pilot import generalization_verdict  # noqa: E402
from result.validation import ValidationTracker, best_epoch_rows, read_val_curve  # noqa: E402

BENCHMARK = ROOT / "data" / "instances" / "benchmark"


def test_index_covers_every_tier(tmp_path):
    root = tmp_path / "instances"
    shutil.copytree(BENCHMARK / "DDT500_num50", root / "benchmark" / "DDT500_num50")
    generate_pool("val", {"tier": "val", "prefix": "VAL", "seed": 1, "grid": {"n_orders": [50], "ddt": [900], "seeds": 2}},
                  GeneratorCfg(), root)
    rows = prepare_all(root, root / "index.csv")
    assert [r["tier"] for r in rows] == ["benchmark", "val", "val"] and rows[0]["ddt"] == 500 and rows[1]["ddt"] == 900
    params = yaml.safe_load((root / "val" / rows[1]["instance_id"] / "params.yaml").read_text(encoding="utf-8"))
    assert params["seed"] == 1 and params["ddt"] == 900                 # generator record preserved by the index step
    index = load_index(root / "index.csv")
    assert set(index) == {r["instance_id"] for r in rows}
    assert [p.stem for p in tier_instance_files("val", root)] == [rows[1]["instance_id"], rows[2]["instance_id"]]
    assert instance_files_by_ids(["DDT500_num50", rows[1]["instance_id"]], root)[1].parent.parent.name == "val"


def test_repository_pools_are_indexed():
    index = load_index()
    tiers = {row["tier"] for row in index.values()}
    assert {"benchmark", "train", "val"} <= tiers
    assert sum(1 for r in index.values() if r["tier"] == "train") == 200
    val = [r for r in index.values() if r["tier"] == "val"]
    assert len(val) == 30 and {r["ddt"] for r in val} == {500, 600, 900, 1200, 1600} and {r["scale"] for r in val} == {50, 100, 150}
    assert all(50 <= r["scale"] <= 150 and 500 <= r["ddt"] <= 1600 for r in index.values() if r["tier"] == "train")


def test_validation_tracker_selects_saves_stops_and_resumes(tmp_path):
    saved = []
    means = {0: [0.5, 0.7], 1: [0.4, 0.6], 2: [0.45, 0.65], 3: [0.45, 0.65]}

    def evaluate(iteration):
        return lambda files: [{"instance_id": name, "obj": value, "fulfillment_rate": 1 - value, "completed": 1,
                               "total_orders": 2} for name, value in zip(files, means[iteration])]

    tracker = ValidationTracker(tmp_path, ["A", "B"], every=1, patience=2, max_iterations=4)
    assert tracker.enabled and tracker.due(0)
    for iteration in range(4):
        out = tracker.evaluate(iteration, evaluate(iteration), 1.0 * iteration, lambda path: saved.append(path.name))
        assert abs(out["val_obj_mean"] - sum(means[iteration]) / 2) < 1e-12
        if tracker.should_stop(iteration):
            break
    assert saved == ["best_val_checkpoint.pt"] * 2 and tracker.best_iteration == 1 and iteration == 3
    rows = read_val_curve(tmp_path)
    assert len(rows) == 8 and best_epoch_rows(rows)[0] == 1
    resumed = ValidationTracker(tmp_path, ["A", "B"], every=1, patience=0, max_iterations=10, resume=True)
    assert resumed.best_iteration == 1 and abs(resumed.best_obj - 0.5) < 1e-12 and resumed.evaluations == 4
    off = ValidationTracker(tmp_path / "off", [], every=0, patience=0, max_iterations=4)
    assert not off.enabled and not off.due(0) and off.summary()["best_val_obj"] is None


def test_generalization_verdict():
    rows = [{"instance_id": "a", "obj": 0.3}, {"instance_id": "b", "obj": 0.5}, {"instance_id": "c", "obj": 0.2}]
    reference = {"a": 0.4, "b": 0.4, "c": 0.3}
    result = generalization_verdict(rows, reference, 2 / 3)
    assert result["verdict"] == "PASS" and result["wins"] == 2 and abs(result["policy_mean"] - (1 - 1 / 3)) < 1e-12
    assert generalization_verdict(rows, reference, 0.9)["verdict"] == "FAIL"
    assert generalization_verdict(rows, {}, 0.5)["verdict"] == "FAIL"


def test_job_builders_follow_the_paradigm(monkeypatch):
    import _bootstrap

    monkeypatch.setattr(sys, "argv", ["x"])
    jobs = _bootstrap.pool_jobs("ours", ["configs/ablation/nf.yaml"], 2, prefix="pilot/", extra_args=["--max-iterations", "5"])
    assert [j.name for j in jobs] == ["pilot/ours/run1", "pilot/ours/run2"]
    assert "--instances" not in jobs[0].cmd and "configs/ablation/nf.yaml" in jobs[0].cmd and "--max-iterations" in jobs[0].cmd
    assert _bootstrap.paradigm() == "per_instance"
    monkeypatch.setattr(_bootstrap, "paradigm", lambda: "generalization")
    assert [j.name for j in _bootstrap.method_jobs("nf", [], ["DDT500_num50"], 1)] == ["nf/run1"]


def test_generalization_training_run_with_validation(tmp_path):
    """Two iterations on three pool instances with random selection, validation on two instances every
    iteration, curve.csv restricted to the selected instance, then eval.py on the val tier of the run."""
    overlay = tmp_path / "overlay.yaml"
    val_ids = [p.stem for p in tier_instance_files("val")[:2]]
    train_names = [p.name for p in tier_instance_files("train")[:3]]
    overlay.write_text(yaml.safe_dump({
        "instance": {"paradigm": "generalization", "data_dir": "data/instances/train",
                     "train_instances": train_names,
                     "instance_selection": "random", "val_instances": [f"{i}.txt" for i in val_ids],
                     "eval_tiers": ["val"]},
        "env": {"fluid_backend": "scipy"},
        "algo": {"train": {"val_every": 1, "eval_scope": "selected", "early_stop_patience": 0, "max_iterations": 2,
                           "num_envs": 2, "num_workers": 2, "minibatch_size": 128, "resume": "none",
                           "update_device": "cpu", "experiment_name": "test_generalization"},
                 "viz": {"viz": False}},
    }, sort_keys=False), encoding="utf-8")
    run_dir = ROOT / "result" / "test_generalization"
    shutil.rmtree(run_dir, ignore_errors=True)
    cmd = [sys.executable, "train.py", "--config", "configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml",
           str(overlay), "--seed", "3"]
    completed = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
    try:
        with (run_dir / "val_curve.csv").open(newline="", encoding="utf-8") as file:
            val_rows = list(csv.DictReader(file))
        assert len(val_rows) == 4 and {r["instance_id"] for r in val_rows} == set(val_ids)
        with (run_dir / "curve.csv").open(newline="", encoding="utf-8") as file:
            curve_rows = list(csv.DictReader(file))
        assert len(curve_rows) == 2 and all(r["instance_id"].startswith("TRN") for r in curve_rows)
        with (run_dir / "log.csv").open(newline="", encoding="utf-8") as file:
            log_rows = list(csv.DictReader(file))
        assert all(r["val_obj_mean"] != "" for r in log_rows)
        assert (run_dir / "best_val_checkpoint.pt").exists()
        meta = yaml.safe_load((run_dir / "train_meta.yaml").read_text(encoding="utf-8"))
        assert meta["paradigm"] == "generalization" and meta["best_val_iteration"] in (0, 1) and meta["n_val_instances"] == 2
        snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
        assert snapshot["paradigm"] == "generalization" and snapshot["n_train_instances"] == 3 and snapshot["eval_tiers"] == ["val"]
        out = run_dir / "eval_results.csv"
        completed = subprocess.run([sys.executable, "eval.py", "--ckpt", str(run_dir / "best_val_checkpoint.pt"),
                                    "--decode", "greedy", "--out", str(out)], cwd=str(ROOT), capture_output=True, text=True)
        assert completed.returncode == 0, completed.stdout[-2000:] + completed.stderr[-2000:]
        with out.open(newline="", encoding="utf-8") as file:
            eval_rows = list(csv.DictReader(file))
        assert [r["instance_id"] for r in eval_rows] == [p.stem for p in tier_instance_files("val")]   # snapshot eval_tiers
        assert all(r["tier"] == "val" for r in eval_rows)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
