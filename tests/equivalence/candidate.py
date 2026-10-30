"""How the *current* tree is exercised by the harness.

This is the only harness file that changes while the layout migrates: it must
know where the instance lives and how the trainer is started.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import yaml

from .common import INSTANCE_NAME, MAX_ITERATIONS, ROOT, SEED, WORK_DIR, golden_train_overrides, run_subprocess


def instance_path() -> Path:
    return ROOT / "data" / "instances" / "benchmark" / Path(INSTANCE_NAME).stem / INSTANCE_NAME


def overlay_for(model_type: str, experiment_name: str, collector_workers: int = 0) -> dict:
    """The golden train_paras overrides expressed in the YAML layout (plus the collector when requested)."""
    o = golden_train_overrides(model_type)
    env = {"fluid_mode": o["fluid_mode"], "fluid_mask": False, "fluid_backend": "heuristic",
           "fluid_recompute": "event"}               # legacy semantics: the heuristic fallback of the original code (no Gurobi)
    if collector_workers > 0:
        env["collector_workers"] = int(collector_workers)
    return {
        "instance": {"train_instances": o["train_instances"], "instance_selection": o["instance_selection"]},
        "env": env,
        "algo": {
            "model_type": o["model_type"],
            "train": {key: o[key] for key in ("num_envs", "num_workers", "minibatch_size", "max_minibatch_tokens",
                                               "max_iterations", "resume", "rollout_device", "update_device")}
            | {"experiment_name": experiment_name},
            "viz": {"viz": o["viz"]},
        },
    }


def run_candidate(base_config: dict, model_type: str, log_path: Path, collector_workers: int = 0) -> Path:
    """Run the current tree's trainer for the reference configuration.

    Returns the directory holding train_metrics.csv and the checkpoints.
    """
    suffix = f"_cw{collector_workers}" if collector_workers > 0 else ""
    experiment_name = f"eq_{model_type}{suffix}"
    shutil.rmtree(ROOT / "result" / experiment_name, ignore_errors=True)  # the trainer appends to an existing CSV
    overlay_path = WORK_DIR / f"overlay_{model_type}{suffix}.yaml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(yaml.safe_dump(overlay_for(model_type, experiment_name, collector_workers), sort_keys=False),
                            encoding="utf-8")
    cmd = [
        sys.executable, "train.py",
        "--config", "configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", str(overlay_path),
        "--seed", str(SEED),
        "--max-iterations", str(MAX_ITERATIONS),
        "--rollout-device", "cpu",
        "--update-device", "cpu",
        "--resume", "none",
        "--instances", INSTANCE_NAME,
        "--experiment-name", experiment_name,
    ]
    run_subprocess(cmd, ROOT, log_path)
    return ROOT / "result" / experiment_name


def check_contract_files(result_dir: Path) -> tuple[list[str], str]:
    """The per-run files added by the refactor (not compared against golden; existence and row counts)."""
    problems: list[str] = []

    def row_count(name: str) -> int:
        path = result_dir / name
        if not path.exists():
            problems.append(f"{name} missing")
            return -1
        with path.open(newline="", encoding="utf-8") as file:
            return sum(1 for _ in csv.DictReader(file))

    curve_rows = row_count("curve.csv")
    log_rows = row_count("log.csv")
    final_rows = row_count("fixed_set_final.csv")
    if (result_dir / "val_curve.csv").exists():
        val_rows = row_count("val_curve.csv")
        if val_rows <= 0 or val_rows % MAX_ITERATIONS != 0:
            problems.append(f"val_curve.csv has {val_rows} rows, expected a multiple of {MAX_ITERATIONS}")
    if curve_rows != MAX_ITERATIONS:
        problems.append(f"curve.csv has {curve_rows} rows, expected {MAX_ITERATIONS}")
    if log_rows != MAX_ITERATIONS:
        problems.append(f"log.csv has {log_rows} rows, expected {MAX_ITERATIONS}")
    if final_rows != 1:
        problems.append(f"fixed_set_final.csv has {final_rows} rows, expected 1")
    for name in ("config_snapshot.yaml", "train_meta.yaml", "best_greedy_checkpoint.pt"):
        if not (result_dir / name).exists():
            problems.append(f"{name} missing")
    return problems, f"curve.csv {curve_rows} rows, log.csv {log_rows} rows"
