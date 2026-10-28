"""Shared constants and comparison helpers for the equivalence harness."""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_COMMIT = "c030dfa74edd1a3147fd2b92f87791d7dcdca710"
GOLDEN_DIR = ROOT / "tests" / "golden"
WORK_DIR = GOLDEN_DIR / "_work"          # git-ignored: golden checkout, .pt files, logs
LEGACY_CONFIG = GOLDEN_DIR / "legacy_config.json"
MODEL_TYPES = ("hgnn", "mlp")
INSTANCE_NAME = "DDT500_num50.txt"
SEED = 42
MAX_ITERATIONS = 3

# v2 golden (the tree's own frozen commit, recorded in tests/golden_v2/manifest.json by --regenerate-golden)
GOLDEN_V2_DIR = ROOT / "tests" / "golden_v2"
WORK_DIR_V2 = GOLDEN_V2_DIR / "_work"    # git-ignored: golden checkout, .pt files, logs
MANIFEST_V2 = GOLDEN_V2_DIR / "manifest.json"
V2_VAL_INSTANCES = 2                     # validation instances of the reference run (first two of the val tier)
V2_CHECKPOINT_FILES = ("best_checkpoint.pt", "last_checkpoint.pt", "ppo_policy_model.pt",
                       "best_greedy_checkpoint.pt", "best_val_checkpoint.pt")

# Environment pinned for golden and candidate alike (thread count changes
# floating-point reduction order).
PIN_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "CUDA_VISIBLE_DEVICES": "",
}

# Wall-clock / resource columns of train_metrics.csv that legitimately differ.
TIMING_COLUMNS = {
    "episode_seconds_mean", "episode_seconds_std", "steps_per_second",
    "rollout_seconds", "update_seconds", "elapsed_seconds", "fluid_solve_seconds",
    "policy_forward_seconds", "policy_act_seconds", "simulator_step_seconds",
    "obs_build_seconds", "cpu_memory_mb", "gpu_memory_mb", "gpu_temperature",
}
EXACT_COLUMNS = {
    "iteration", "instance_id", "instance_name", "instance_selection", "transitions",
    "episode_count", "discard_count", "completion_count", "fluid_solve_count",
    "fluid_cache_hit_count", "fluid_fallback_count", "policy_forward_count",
    "sparse_attention_enabled", "sparse_attention_scope", "rollout_device", "update_device",
}

REL_TOL = 1e-6
ABS_TOL = 1e-9


def golden_train_overrides(model_type: str) -> dict:
    """train_paras overrides that define the reference run."""
    return {
        "num_envs": 2,
        "num_workers": 2,
        "minibatch_size": 128,
        "max_minibatch_tokens": 0,
        "max_iterations": MAX_ITERATIONS,
        "viz": False,
        "resume": "none",
        "rollout_device": "cpu",
        "update_device": "cpu",
        "train_instances": [INSTANCE_NAME],
        "instance_selection": "fixed",
        "fluid_mode": "cached_lp",
        "model_type": model_type,
    }


def v2_train_overrides(model_type: str) -> dict:
    """train_paras overrides of the v2 reference run (fluid mask through HiGHS, validation every iteration)."""
    overrides = golden_train_overrides(model_type)
    overrides.update({
        "fluid_backend": "scipy", "fluid_mask": True, "fluid_features": "on", "fluid_recompute": "decision",
        "fluid_mask_threshold": 1e-5, "val_every": 1, "eval_scope": "all", "early_stop_patience": 0,
    })
    return overrides


def v2_val_instance_names(tree_root: Path) -> list:
    """File names of the first validation instances (sorted ids) of the val tier of ``tree_root``."""
    val_dir = tree_root / "data" / "instances" / "val"
    files = sorted(p for p in val_dir.glob("*/*.txt") if p.stem == p.parent.name)
    return [p.name for p in files[:V2_VAL_INSTANCES]]


def v2_overlay(model_type: str, experiment_name: str, val_names: list, collector_workers: int = 0) -> dict:
    o = v2_train_overrides(model_type)
    env = {key: o[key] for key in ("fluid_mode", "fluid_backend", "fluid_mask", "fluid_features", "fluid_recompute",
                                   "fluid_mask_threshold")}
    if collector_workers > 0:
        env["collector_workers"] = int(collector_workers)
    return {
        "instance": {"train_instances": o["train_instances"], "instance_selection": o["instance_selection"],
                     "val_instances": list(val_names)},
        "env": env,
        "algo": {
            "model_type": o["model_type"],
            "train": {key: o[key] for key in ("num_envs", "num_workers", "minibatch_size", "max_minibatch_tokens",
                                               "max_iterations", "resume", "rollout_device", "update_device",
                                               "val_every", "eval_scope", "early_stop_patience")}
            | {"experiment_name": experiment_name},
            "viz": {"viz": False},
        },
    }


def run_v2_train(tree_root: Path, model_type: str, experiment_name: str, log_path: Path,
                 collector_workers: int = 0) -> Path:
    """Run ``train.py`` of ``tree_root`` (the current tree or the v2 golden checkout) with the v2 overlay."""
    import shutil

    import yaml

    shutil.rmtree(tree_root / "result" / experiment_name, ignore_errors=True)
    overlay_path = WORK_DIR_V2 / f"overlay_{experiment_name}.yaml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(yaml.safe_dump(v2_overlay(model_type, experiment_name, v2_val_instance_names(tree_root),
                                                      collector_workers), sort_keys=False), encoding="utf-8")
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
    run_subprocess(cmd, tree_root, log_path)
    return tree_root / "result" / experiment_name


def compare_small_csv(golden_csv: Path, candidate_csv: Path, numeric_columns, label: str) -> list[str]:
    """Row-by-row comparison of the given numeric columns (exact) of two small CSV files."""
    mismatches: list[str] = []
    with golden_csv.open(newline="", encoding="utf-8") as g, candidate_csv.open(newline="", encoding="utf-8") as c:
        golden_rows, candidate_rows = list(csv.DictReader(g)), list(csv.DictReader(c))
    if len(golden_rows) != len(candidate_rows):
        mismatches.append(f"{label}: row count differs: golden={len(golden_rows)} candidate={len(candidate_rows)}")
    for index, (grow, crow) in enumerate(zip(golden_rows, candidate_rows)):
        for column in numeric_columns:
            if grow.get(column) != crow.get(column):
                mismatches.append(f"{label} row {index} column {column}: golden={grow.get(column)!r} "
                                  f"candidate={crow.get(column)!r}")
    return mismatches


def make_legacy_config(base_config: dict, model_type: str, experiment_name: str) -> dict:
    config = copy.deepcopy(base_config)
    config["train_paras"].update(golden_train_overrides(model_type))
    config["train_paras"]["experiment_name"] = experiment_name
    return config


def pinned_env() -> dict:
    env = dict(os.environ)
    env.update(PIN_ENV)
    return env


def run_subprocess(cmd: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + f"\n(cwd={cwd})\n\n")
        log.flush()
        result = subprocess.run(cmd, cwd=str(cwd), env=pinned_env(), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(cmd)}\nsee {log_path}")


def run_legacy_train(tree_root: Path, base_config: dict, model_type: str, experiment_name: str,
                     log_path: Path, entry: str = "agent/train.py") -> Path:
    """Run a trainer entry script (legacy CLI) inside `tree_root`.

    Returns the result directory `<tree_root>/result/<experiment_name>`.
    """
    config = make_legacy_config(base_config, model_type, experiment_name)
    config_path = WORK_DIR / f"legacy_config_{experiment_name}.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    cmd = [
        sys.executable, entry,
        "--config", str(config_path),
        "--seed", str(SEED),
        "--max-iterations", str(MAX_ITERATIONS),
        "--rollout-device", "cpu",
        "--update-device", "cpu",
        "--resume", "none",
        "--instances", INSTANCE_NAME,
        "--experiment-name", experiment_name,
    ]
    run_subprocess(cmd, tree_root, log_path)
    return tree_root / "result" / experiment_name


# ---------------------------------------------------------------- comparison

def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def compare_csv(golden_csv: Path, candidate_csv: Path) -> tuple[list[str], int, int]:
    """Compare two train_metrics.csv files.

    Returns (mismatches, float_columns_compared, float_columns_exactly_equal).
    """
    mismatches: list[str] = []
    with golden_csv.open(newline="", encoding="utf-8") as g, candidate_csv.open(newline="", encoding="utf-8") as c:
        golden_rows = list(csv.DictReader(g))
        candidate_rows = list(csv.DictReader(c))
    if not golden_rows:
        return ["golden CSV is empty"], 0, 0
    golden_fields = list(golden_rows[0].keys())
    candidate_fields = list(candidate_rows[0].keys()) if candidate_rows else []
    if golden_fields != candidate_fields:
        mismatches.append(f"column set/order differs:\n  golden={golden_fields}\n  cand  ={candidate_fields}")
    if len(golden_rows) != len(candidate_rows):
        mismatches.append(f"row count differs: golden={len(golden_rows)} candidate={len(candidate_rows)}")
    float_compared = 0
    float_exact = 0
    for row_index, (grow, crow) in enumerate(zip(golden_rows, candidate_rows)):
        for column in golden_fields:
            if column in TIMING_COLUMNS or column not in crow:
                continue
            gval, cval = grow[column], crow[column]
            if column in EXACT_COLUMNS or not _is_number(gval):
                if gval != cval:
                    mismatches.append(f"row {row_index} column {column}: golden={gval!r} candidate={cval!r}")
                continue
            float_compared += 1
            if gval == cval:
                float_exact += 1
                continue
            if not math.isclose(float(gval), float(cval), rel_tol=REL_TOL, abs_tol=ABS_TOL):
                mismatches.append(f"row {row_index} column {column}: golden={gval} candidate={cval}")
    return mismatches, float_compared, float_exact


def compare_tensor_dicts(name: str, golden: dict, candidate: dict, mismatches: list[str], atol: float = 1e-6,
                         rtol: float = 1e-6) -> tuple[int, int]:
    """Compare two state dicts (or any {str: tensor}); key order matters."""
    import torch

    gkeys, ckeys = list(golden.keys()), list(candidate.keys())
    if gkeys != ckeys:
        mismatches.append(f"{name}: key list differs\n  golden={gkeys}\n  cand  ={ckeys}")
    compared = exact = 0
    for key in gkeys:
        if key not in candidate:
            continue
        gt, ct = golden[key], candidate[key]
        if not torch.is_tensor(gt):
            if gt != ct:
                mismatches.append(f"{name}[{key}]: golden={gt!r} candidate={ct!r}")
            continue
        compared += 1
        if gt.shape != ct.shape or gt.dtype != ct.dtype:
            mismatches.append(f"{name}[{key}]: shape/dtype differs {tuple(gt.shape)}/{gt.dtype} vs {tuple(ct.shape)}/{ct.dtype}")
            continue
        if torch.equal(gt, ct):
            exact += 1
        elif not torch.allclose(gt.float(), ct.float(), atol=atol, rtol=rtol):
            diff = (gt.float() - ct.float()).abs().max().item()
            mismatches.append(f"{name}[{key}]: max abs diff {diff:.3e}")
    return compared, exact


def compare_checkpoint(golden_pt: Path, candidate_pt: Path, mismatches: list[str]) -> tuple[int, int]:
    import torch

    g = torch.load(golden_pt, map_location="cpu", weights_only=False)
    c = torch.load(candidate_pt, map_location="cpu", weights_only=False)
    label = golden_pt.name
    if isinstance(g, dict) and "policy" in g:
        for scalar in ("iteration", "best_fulfillment_rate"):
            if g.get(scalar) != c.get(scalar):
                mismatches.append(f"{label}.{scalar}: golden={g.get(scalar)} candidate={c.get(scalar)}")
        compared = exact = 0
        for part in ("policy", "policy_old"):
            n, e = compare_tensor_dicts(f"{label}.{part}", g[part], c[part], mismatches)
            compared += n
            exact += e
        gopt, copt = g.get("optimizer", {}), c.get("optimizer", {})
        gpg, cpg = gopt.get("param_groups", []), copt.get("param_groups", [])
        for field in ("lr", "betas", "eps", "weight_decay"):
            gv = [pg.get(field) for pg in gpg]
            cv = [pg.get(field) for pg in cpg]
            if gv != cv:
                mismatches.append(f"{label}.optimizer.param_groups.{field}: golden={gv} candidate={cv}")
        gstate, cstate = gopt.get("state", {}), copt.get("state", {})
        if list(gstate.keys()) != list(cstate.keys()):
            mismatches.append(f"{label}.optimizer.state keys differ")
        for key in gstate:
            if key not in cstate:
                continue
            flat_g = {f"{key}.{k}": v for k, v in gstate[key].items()}
            flat_c = {f"{key}.{k}": v for k, v in cstate[key].items()}
            n, e = compare_tensor_dicts(f"{label}.optimizer.state", flat_g, flat_c, mismatches)
            compared += n
            exact += e
        return compared, exact
    # plain state dict (ppo_policy_model.pt)
    return compare_tensor_dicts(label, g, c, mismatches)


def compare_nested(name: str, golden, candidate, mismatches: list[str], atol: float = 1e-6, rtol: float = 1e-6):
    """Recursive comparison of probe fixtures (dict / list / tensor / scalar)."""
    import torch

    if torch.is_tensor(golden):
        if not torch.is_tensor(candidate):
            mismatches.append(f"{name}: tensor vs {type(candidate).__name__}")
            return
        if golden.shape != candidate.shape or golden.dtype != candidate.dtype:
            mismatches.append(f"{name}: shape/dtype differs {tuple(golden.shape)}/{golden.dtype} vs "
                              f"{tuple(candidate.shape)}/{candidate.dtype}")
            return
        if not torch.equal(golden, candidate) and not torch.allclose(golden.float(), candidate.float(), atol=atol, rtol=rtol):
            diff = (golden.float() - candidate.float()).abs().max().item()
            mismatches.append(f"{name}: max abs diff {diff:.3e}")
        return
    if isinstance(golden, dict):
        if not isinstance(candidate, dict) or list(golden.keys()) != list(candidate.keys()):
            mismatches.append(f"{name}: dict keys differ\n  golden={list(golden.keys()) if isinstance(golden, dict) else golden}"
                              f"\n  cand  ={list(candidate.keys()) if isinstance(candidate, dict) else candidate}")
            return
        for key in golden:
            compare_nested(f"{name}.{key}", golden[key], candidate[key], mismatches, atol, rtol)
        return
    if isinstance(golden, (list, tuple)):
        if not isinstance(candidate, (list, tuple)) or len(golden) != len(candidate):
            mismatches.append(f"{name}: sequence length differs {len(golden) if hasattr(golden, '__len__') else golden} vs "
                              f"{len(candidate) if hasattr(candidate, '__len__') else candidate}")
            return
        for index, (g, c) in enumerate(zip(golden, candidate)):
            compare_nested(f"{name}[{index}]", g, c, mismatches, atol, rtol)
        return
    if isinstance(golden, float) or isinstance(candidate, float):
        if not math.isclose(float(golden), float(candidate), rel_tol=rtol, abs_tol=atol):
            mismatches.append(f"{name}: golden={golden} candidate={candidate}")
        return
    if golden != candidate:
        mismatches.append(f"{name}: golden={golden!r} candidate={candidate!r}")
