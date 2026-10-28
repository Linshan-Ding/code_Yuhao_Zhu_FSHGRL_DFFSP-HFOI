"""Validation-set tracking for the generalisation paradigm (v2, new code).

A ``ValidationTracker`` evaluates the current policy greedily on a fixed set of validation instances every
``val_every`` iterations (and at the last iteration), appends one row per instance to ``val_curve.csv``, keeps
the running best validation mean (``obj = 1 - fulfilment``, minimised) and saves ``best_val_checkpoint.pt``
whenever it improves. Model selection by this checkpoint replaces the legacy ``best_checkpoint.pt`` (sampled
training episodes, saved after the update) in every v2 experiment. ``early_stop_patience`` > 0 stops the run
when the best validation mean has not improved for that many iterations.

The tracker draws no random numbers (greedy decoding) and is shared by the PPO loop (``train.py``) and the
off-policy loop (``agent/baselines/offpolicy.py``); the evaluation function and the checkpoint writer are
injected to keep this module free of trainer imports.
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Callable, Sequence


class ValidationTracker:
    COLS = ["epoch", "instance_id", "obj", "fulfillment_rate", "completed", "total_orders", "wall_time_s"]
    CHECKPOINT_NAME = "best_val_checkpoint.pt"
    CURVE_NAME = "val_curve.csv"

    def __init__(self, run_dir, val_files: Sequence, every: int, patience: int, max_iterations: int,
                 resume: bool = False):
        self.run_dir = Path(run_dir)
        self.files = list(val_files)
        self.every = max(int(every), 0)
        self.patience = max(int(patience), 0)
        self.max_iterations = int(max_iterations)
        self.curve_path = self.run_dir / self.CURVE_NAME
        self.checkpoint_path = self.run_dir / self.CHECKPOINT_NAME
        self.best_obj = math.inf
        self.best_iteration = -1
        self.evaluations = 0
        if resume and self.curve_path.exists():
            self._restore()
        elif self.curve_path.exists():
            self.curve_path.unlink()

    # -- state -----------------------------------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.every > 0 and bool(self.files)

    def _restore(self) -> None:
        per_epoch: dict = {}
        with self.curve_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    per_epoch.setdefault(int(row["epoch"]), []).append(float(row["obj"]))
                except (KeyError, ValueError):
                    continue
        for epoch in sorted(per_epoch):
            mean = sum(per_epoch[epoch]) / len(per_epoch[epoch])
            self.evaluations += 1
            if mean < self.best_obj:
                self.best_obj, self.best_iteration = mean, epoch

    def due(self, iteration: int) -> bool:
        if not self.enabled:
            return False
        return iteration % self.every == 0 or iteration == self.max_iterations - 1

    def should_stop(self, iteration: int) -> bool:
        """True when the best validation mean is older than ``patience`` iterations (0 disables)."""
        return self.patience > 0 and self.best_iteration >= 0 and iteration - self.best_iteration >= self.patience

    # -- evaluation ------------------------------------------------------------------------------------------
    def evaluate(self, iteration: int, evaluate_fn: Callable[[Sequence], list], wall_time_s: float,
                 save_checkpoint: Callable[[Path], None]) -> dict:
        """``evaluate_fn(files)`` -> rows with ``instance_id, obj, fulfillment_rate, completed, total_orders``;
        ``save_checkpoint(path)`` writes the current model. Returns the log.csv fields."""
        started = time.perf_counter()
        rows = evaluate_fn(self.files)
        out_rows = []
        for row in rows:
            out_rows.append({
                "epoch": iteration,
                "instance_id": row["instance_id"],
                "obj": float(row["obj"]),
                "fulfillment_rate": float(row.get("fulfillment_rate", 1.0 - float(row["obj"]))),
                "completed": row.get("completed", ""),
                "total_orders": row.get("total_orders", ""),
                "wall_time_s": wall_time_s,
            })
        self._append(out_rows)
        mean_obj = float(sum(r["obj"] for r in out_rows) / max(len(out_rows), 1))
        improved = mean_obj < self.best_obj
        if improved:
            self.best_obj, self.best_iteration = mean_obj, iteration
            save_checkpoint(self.checkpoint_path)
        self.evaluations += 1
        return {
            "val_obj_mean": mean_obj,
            "val_fulfillment_mean": 1.0 - mean_obj,
            "val_seconds": round(time.perf_counter() - started, 3),
            "val_improved": improved,
        }

    def _append(self, rows: list) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_header = (not self.curve_path.exists()) or self.curve_path.stat().st_size == 0
        with self.curve_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.COLS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def summary(self) -> dict:
        """Fields for train_meta.yaml."""
        return {
            "val_enabled": self.enabled,
            "n_val_instances": len(self.files),
            "val_every": self.every,
            "val_evaluations": self.evaluations,
            "best_val_obj": self.best_obj if math.isfinite(self.best_obj) else None,
            "best_val_fulfillment": (1.0 - self.best_obj) if math.isfinite(self.best_obj) else None,
            "best_val_iteration": self.best_iteration if self.best_iteration >= 0 else None,
            "early_stop_patience": self.patience,
        }


def read_val_curve(run_dir) -> list:
    """Rows of ``val_curve.csv`` with numeric fields converted; empty when the file does not exist."""
    path = Path(run_dir) / ValidationTracker.CURVE_NAME
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            rows.append({
                "epoch": int(row["epoch"]), "instance_id": row["instance_id"], "obj": float(row["obj"]),
                "fulfillment_rate": float(row["fulfillment_rate"]),
                "completed": int(float(row["completed"])) if row.get("completed", "") != "" else None,
                "total_orders": int(float(row["total_orders"])) if row.get("total_orders", "") != "" else None,
                "wall_time_s": float(row["wall_time_s"]) if row.get("wall_time_s", "") != "" else None,
            })
    return rows


def best_epoch_rows(rows: list) -> tuple:
    """``(best_epoch, rows_of_that_epoch)`` by the lowest per-epoch mean objective (first epoch wins ties)."""
    per_epoch: dict = {}
    for row in rows:
        per_epoch.setdefault(row["epoch"], []).append(row)
    best_epoch, best_mean = None, math.inf
    for epoch in sorted(per_epoch):
        mean = sum(r["obj"] for r in per_epoch[epoch]) / len(per_epoch[epoch])
        if mean < best_mean:
            best_epoch, best_mean = epoch, mean
    return best_epoch, (per_epoch.get(best_epoch, []) if best_epoch is not None else [])
