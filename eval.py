"""Evaluation entry point (new code; the original repository had no evaluation script).

Model mode -- load a checkpoint and run complete episodes on benchmark instances::

    python eval.py --ckpt result/<run>/best_greedy_checkpoint.pt --decode both --episodes 10 \\
                   --method ours --run-id run1 --out result/<run>/eval_results.csv

Collect mode -- gather the end-of-training incumbents of per-instance runs (fixed_set_final.csv)::

    python eval.py --collect "result/ours/*/run*/fixed_set_final.csv" --method ours --out result/eval_ours.csv

Decoding: ``greedy`` runs one deterministic episode per instance (argmax of both action distributions);
``sample`` draws ``--episodes`` episodes from the policy (seeded with ``--seed``), ``--batch-episodes`` of
them in lockstep per forward pass. The policy normalises due dates and processing rates per forward batch,
so lockstep episodes of the same instance are not independent of the batch composition; the default
``--batch-episodes 1`` keeps every episode self-contained (slower, but the numbers of one episode do not
depend on the other episodes). Timing columns are wall-clock milliseconds of one complete episode on the
machine that ran it (``infer_time_ms``), split into policy forward passes (``policy_ms``, the share of one
episode when batched), observation building including the fluid features (``obs_ms``, ``fluid_ms``) and
simulator stepping (``sim_ms``). Pin the BLAS threads (OMP_NUM_THREADS=1) and use ``--threads 1`` for
comparable decision times.

Columns of eval_results.csv: the contract columns ``instance_id, method, run_id, tier, scale, obj_model,
obj_ref, gap, feasible, infer_time_ms`` (``obj = 1 - fulfilment rate``, see result/metrics.py) followed by
``decode, episode, eval_batch, fulfillment_rate, completed, discarded, total_orders, decision_count,
policy_ms, obs_ms, fluid_ms, sim_ms, fluid_backend, candidate_pairs_mean, masked_pairs_mean,
mask_fallback_rate, ddt, checkpoint, checkpoint_iteration`` and, for
collected per-instance runs, ``epochs, wall_time_s, obj_best`` (``decode = incumbent``).
"""
from __future__ import annotations

import argparse
import csv
import glob
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.checkpoint import load_policy  # noqa: E402
from data.dataset import benchmark_instance_files, instance_files_by_ids, load_index, parse_ddt, tier_instance_files  # noqa: E402
from data.instance_io import parse_instance_file  # noqa: E402
from environment.profiled import ProfiledSimulator  # noqa: E402
from result.metrics import gap_pct, load_reference, objective_from_fulfillment  # noqa: E402
from rollout import run_episodes  # noqa: E402
from train import load_config as load_train_config, resolve_torch_device  # noqa: E402

COLS = [
    "instance_id", "method", "run_id", "tier", "scale", "obj_model", "obj_ref", "gap", "feasible", "infer_time_ms",
    "decode", "episode", "eval_batch", "fulfillment_rate", "completed", "discarded", "total_orders",
    "decision_count", "policy_ms", "obs_ms", "fluid_ms", "sim_ms", "fluid_backend", "candidate_pairs_mean",
    "masked_pairs_mean", "mask_fallback_rate", "ddt", "checkpoint", "checkpoint_iteration", "epochs",
    "wall_time_s", "obj_best",
]


def parse_instance_names(values) -> list:
    names = []
    for value in values or []:
        names.extend(item.strip() for item in str(value).split(",") if item.strip())
    return names


def read_snapshot(checkpoint_path: Path) -> dict:
    """config_snapshot.yaml next to a checkpoint (empty when absent or unreadable)."""
    snapshot_path = Path(checkpoint_path).resolve().parent / "config_snapshot.yaml"
    if not snapshot_path.exists():
        return {}
    try:
        return yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def trained_instances(checkpoint_path: Path) -> list:
    """Instances a run trained on, from the config_snapshot.yaml next to the checkpoint (the checkpoint's own
    train_paras keep the YAML value even when --instances overrode it on the command line)."""
    return [str(name) for name in (read_snapshot(checkpoint_path).get("instances") or [])]


def tier_files(tiers) -> list:
    files = []
    for tier in tiers:
        files.extend(tier_instance_files(str(tier)))
    return files


def resolve_eval_instances(requested, train_paras: dict, checkpoint_path: Path | None = None, tiers=None) -> list:
    """Evaluation instances: ``--tiers`` (whole tiers), ``--instances`` (ids of any tier, or ``all`` = the
    benchmark tier); default: the evaluation tiers of a generalisation checkpoint (its snapshot's
    ``eval_tiers``), else the instances the checkpoint was trained on (per-instance paradigm)."""
    if tiers:
        return tier_files(tiers)
    names = parse_instance_names(requested)
    if names == ["all"]:
        return benchmark_instance_files(None)
    if names:
        return instance_files_by_ids(names)
    snapshot = read_snapshot(checkpoint_path) if checkpoint_path is not None else {}
    paradigm = str(snapshot.get("paradigm") or train_paras.get("paradigm") or "").lower()
    if paradigm == "generalization":
        eval_tiers = snapshot.get("eval_tiers") or train_paras.get("eval_tiers") or ["benchmark"]
        return tier_files(eval_tiers)
    configured = [str(name) for name in (snapshot.get("instances") or [])]
    if not configured:
        configured = train_paras.get("train_instances") or []
    if isinstance(configured, str):
        configured = [configured]
    if configured:
        return instance_files_by_ids([Path(name).stem for name in configured])
    return benchmark_instance_files(None)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def relative_to_root(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def episode_row(record: dict, path: Path, meta: dict, reference: dict, method: str, run_id: str, decode: str,
                episode: int, eval_batch: int, checkpoint: str, checkpoint_iteration) -> dict:
    instance_id = path.stem
    obj = objective_from_fulfillment(record["fulfillment_rate"])
    ref = reference.get(instance_id)
    ddt = meta.get("ddt")
    if ddt is None:
        ddt = parse_ddt(instance_id)
    return {
        "instance_id": instance_id,
        "method": method,
        "run_id": run_id,
        "tier": meta.get("tier", "benchmark"),
        "scale": int(meta.get("scale") or record["total_orders"]),
        "obj_model": obj,
        "obj_ref": ref if ref is not None else "",
        "gap": gap_pct(obj, ref),
        "feasible": True,
        "infer_time_ms": record["wall_seconds"] * 1000.0,
        "decode": decode,
        "episode": episode,
        "eval_batch": eval_batch,
        "fulfillment_rate": record["fulfillment_rate"],
        "completed": record["completed"],
        "discarded": record["discarded"],
        "total_orders": record["total_orders"],
        "decision_count": record["decision_count"],
        "policy_ms": record["policy_forward_seconds"] * 1000.0 / max(eval_batch, 1),
        "obs_ms": record["obs_build_seconds"] * 1000.0,
        "fluid_ms": record["fluid_feature_seconds"] * 1000.0,
        "sim_ms": record["simulator_step_seconds"] * 1000.0,
        "fluid_backend": record["fluid_backend"],
        "candidate_pairs_mean": record.get("candidate_pairs_mean", ""),
        "masked_pairs_mean": record.get("masked_pairs_mean", ""),
        "mask_fallback_rate": record.get("mask_fallback_rate", ""),
        "ddt": ddt if ddt is not None else "",
        "checkpoint": checkpoint,
        "checkpoint_iteration": checkpoint_iteration if checkpoint_iteration is not None else "",
        "epochs": "",
        "wall_time_s": "",
        "obj_best": "",
    }


def evaluate_checkpoint(args) -> list:
    device = resolve_torch_device(args.device, fallback="cpu")
    overrides = load_train_config(args.config) if args.config else None
    policy, config, train_paras, bundle = load_policy(
        Path(args.ckpt), device, config=overrides,
        train_paras=overrides["train_paras"] if overrides else None,
    )
    if args.threads:
        torch.set_num_threads(int(args.threads))
    seed_everything(args.seed)
    instance_files = resolve_eval_instances(args.instances, train_paras, Path(args.ckpt), args.tiers)
    index = load_index()
    reference = load_reference()
    checkpoint = relative_to_root(Path(args.ckpt))
    rows = []
    for path in instance_files:
        started = time.perf_counter()
        spec = parse_instance_file(path)
        meta = index.get(path.stem, {})
        summary = []
        if args.decode in ("greedy", "both"):
            record = run_episodes(policy, spec, train_paras, episodes=1, sample=False,
                                  simulator_factory=ProfiledSimulator)[0]
            rows.append(episode_row(record, path, meta, reference, args.method, args.run_id, "greedy", 0, 1,
                                    checkpoint, bundle["iteration"]))
            summary.append(f"greedy eta={record['fulfillment_rate']:.4f}")
        if args.decode in ("sample", "both"):
            finished = 0
            episode_index = 0
            rates = []
            while finished < args.episodes:
                batch = min(int(args.batch_episodes), args.episodes - finished)
                records = run_episodes(policy, spec, train_paras, episodes=batch, sample=True,
                                       simulator_factory=ProfiledSimulator)
                for record in records:
                    rows.append(episode_row(record, path, meta, reference, args.method, args.run_id, "sample",
                                            episode_index, batch, checkpoint, bundle["iteration"]))
                    rates.append(record["fulfillment_rate"])
                    episode_index += 1
                finished += batch
            summary.append(f"sampled eta mean={np.mean(rates):.4f} max={np.max(rates):.4f} (n={len(rates)})")
        print(f"[EVAL] {path.stem}: {'; '.join(summary)}; {time.perf_counter() - started:.1f}s", flush=True)
    return rows


def collect_runs(args) -> list:
    """Per-instance runs: result/<method>/<id>/run<k>/fixed_set_final.csv -> one row per (instance, run)."""
    index = load_index()
    reference = load_reference()
    rows = []
    for final_path in sorted(glob.glob(args.collect, recursive=True)):
        final_path = Path(final_path)
        run_dir = final_path.parent
        run_id = run_dir.name if run_dir.name.startswith("run") else args.run_id
        meta_path = run_dir / "train_meta.yaml"
        train_meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        with final_path.open("r", newline="", encoding="utf-8") as file:
            for item in csv.DictReader(file):
                instance_id = str(item["instance_id"])
                obj_best = float(item["obj_best"])
                ref = reference.get(instance_id)
                meta = index.get(instance_id, {})
                ddt = meta.get("ddt")
                if ddt is None:
                    ddt = parse_ddt(instance_id)
                rows.append({
                    "instance_id": instance_id,
                    "method": args.method,
                    "run_id": run_id,
                    "tier": meta.get("tier", "benchmark"),
                    "scale": int(meta.get("scale") or 0) or "",
                    "obj_model": obj_best,
                    "obj_ref": ref if ref is not None else "",
                    "gap": gap_pct(obj_best, ref),
                    "feasible": str(item.get("feasible", "True")) == "True",
                    "infer_time_ms": "",
                    "decode": "incumbent",
                    "episode": "",
                    "eval_batch": "",
                    "fulfillment_rate": 1.0 - obj_best,
                    "completed": "",
                    "discarded": "",
                    "total_orders": meta.get("n_orders", ""),
                    "decision_count": "",
                    "policy_ms": "",
                    "obs_ms": "",
                    "fluid_ms": "",
                    "sim_ms": "",
                    "fluid_backend": "",
                    "candidate_pairs_mean": "",
                    "masked_pairs_mean": "",
                    "mask_fallback_rate": "",
                    "ddt": ddt if ddt is not None else "",
                    "checkpoint": relative_to_root(run_dir),
                    "checkpoint_iteration": train_meta.get("best_greedy_iteration", ""),
                    "epochs": int(float(item["epochs"])),
                    "wall_time_s": float(item["wall_time_s"]),
                    "obj_best": obj_best,
                })
    return rows


def write_rows(rows: list, out_path: Path, append: bool) -> None:
    frame = pd.DataFrame(rows, columns=COLS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append and out_path.exists() and out_path.stat().st_size > 0:
        frame.to_csv(out_path, mode="a", header=False, index=False)
    else:
        frame.to_csv(out_path, index=False)


def print_summary(rows: list) -> None:
    if not rows:
        print("[EVAL] no rows")
        return
    frame = pd.DataFrame(rows, columns=COLS)
    for decode, group in frame.groupby("decode"):
        rate = pd.to_numeric(group["fulfillment_rate"], errors="coerce")
        timing = pd.to_numeric(group["infer_time_ms"], errors="coerce")
        line = f"[EVAL] {decode}: {len(group)} rows, fulfilment mean {rate.mean():.4f}"
        if timing.notna().any():
            line += f", episode wall mean {timing.mean():.1f} ms"
        print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=None, help="trainer checkpoint (*.pt) to evaluate")
    parser.add_argument("--config", nargs="+", default=None,
                        help="YAML overlays or legacy JSON; overrides the configuration stored in the checkpoint")
    parser.add_argument("--collect", default=None, help="glob of fixed_set_final.csv files of per-instance runs")
    parser.add_argument("--instances", nargs="*", default=None,
                        help="instance ids / file names (any tier), or 'all' (benchmark tier); default: the "
                             "evaluation tiers of a generalisation checkpoint, else its training instances")
    parser.add_argument("--tiers", nargs="*", default=None, help="evaluate every instance of these tiers")
    parser.add_argument("--decode", choices=["greedy", "sample", "both"], default="greedy")
    parser.add_argument("--episodes", type=int, default=10, help="sampled episodes per instance")
    parser.add_argument("--batch-episodes", type=int, default=1, help="sampled episodes per lockstep batch")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads for comparable timings")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--method", default="ours")
    parser.add_argument("--run-id", default="run1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--append", action="store_true", help="append to --out instead of overwriting it")
    args = parser.parse_args()
    if bool(args.ckpt) == bool(args.collect):
        parser.error("pass exactly one of --ckpt or --collect")
    rows = collect_runs(args) if args.collect else evaluate_checkpoint(args)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    write_rows(rows, out_path, args.append)
    print_summary(rows)
    print(f"[EVAL] {len(rows)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
