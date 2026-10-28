"""Run logging: the original ``train_metrics.csv`` writer and Visdom visualiser (moved verbatim from
``agent/train.py`` / ``agent/visualization.py``) plus ``RunLogger``, which adds the files of the experiment
contract (``log.csv``, ``curve.csv``, ``fixed_set_final.csv``, ``config_snapshot.yaml``, ``train_meta.yaml``).

File roles
----------
- ``train_metrics.csv``  the original 49-column per-iteration record; written exactly as before and
                         regression-tested (``tests/equivalence``);
- ``log.csv``            per-epoch training diagnostics projected from the same row. ``train_obj_mean`` comes
                         from the *sampled* training rollouts and is for diagnosis only: convergence plots
                         must read ``curve.csv``;
- ``curve.csv``          per-epoch greedy objective of the current policy on each fixed instance, plus the
                         running incumbent ``obj_best`` (the only data source of convergence figures);
- ``fixed_set_final.csv`` end-of-run incumbent per instance (collected by ``eval.py --collect``);
- ``config_snapshot.yaml`` / ``train_meta.yaml``  effective configuration and stop record.

``epoch`` in the new files is the trainer's 0-based ``iteration``, so all CSV files share one axis.
Objective convention: ``obj = 1 - fulfilment rate`` (minimised), see ``result/metrics.py``.
"""
from __future__ import annotations

import csv
import math
import subprocess
import time
from html import escape
from numbers import Number
from pathlib import Path

import yaml


# ------------------------------------------------------------ legacy (verbatim)

def write_metrics(metrics_path: Path, row: dict):
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    exists = metrics_path.exists()
    write_header = (not exists) or metrics_path.stat().st_size == 0
    fieldnames = list(row.keys())
    if exists:
        with metrics_path.open("r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            existing_fields = [field for field in (reader.fieldnames or []) if field is not None]
            if existing_fields and existing_fields != fieldnames:
                rows = []
                for old_row in reader:
                    old_row.pop(None, None)
                    rows.append({name: old_row.get(name, "") for name in existing_fields})
                fieldnames = existing_fields + [name for name in fieldnames if name not in existing_fields]
                with metrics_path.open("w", newline="", encoding="utf-8") as rewrite_file:
                    writer = csv.DictWriter(rewrite_file, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                write_header = False
            elif existing_fields:
                fieldnames = existing_fields
    with metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


DEFAULT_METRICS = [
    "reward",
    "mean_return",
    "fulfillment_rate_mean",
    "loss",
    "policy_loss",
    "value_loss",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "learning_rate",
    "steps_per_second",
    "transitions",
    "episode_count",
    "episode_transition_mean",
    "episode_transition_std",
    "rollout_seconds",
    "update_seconds",
    "episode_seconds_mean",
    "episode_seconds_std",
    "elapsed_seconds",
    "fluid_solve_count",
    "fluid_solve_seconds",
    "fluid_cache_hit_count",
    "fluid_fallback_count",
    "sparse_attention_enabled",
    "policy_forward_count",
    "policy_forward_seconds",
    "avg_action_tokens",
    "max_action_tokens",
    "padding_ratio",
    "actor_batch_size_mean",
    "policy_act_seconds",
    "simulator_step_seconds",
    "obs_build_seconds",
    "cpu_memory_mb",
    "gpu_memory_mb",
    "gpu_temperature",
    "completion_count",
    "discard_count",
]


class TrainingVisualizer:
    def __init__(self, train_paras: dict):
        self.enabled = bool(train_paras.get("viz", False))
        self.env = train_paras.get("viz_name", "HGNN_PPO")
        self.server = train_paras.get("visdom_server", "http://localhost")
        self.port = int(train_paras.get("visdom_port", 8097))
        self.update_interval = max(int(train_paras.get("viz_update_interval", 1)), 1)
        self.metrics = self._load_metrics(train_paras.get("viz_metrics"))
        self.viz = None
        self.windows = set()
        self.warned = False
        if self.enabled:
            self._connect()

    def _load_metrics(self, configured_metrics):
        if not configured_metrics:
            return DEFAULT_METRICS
        return [str(metric) for metric in configured_metrics]

    def _connect(self):
        try:
            from visdom import Visdom

            self.viz = Visdom(server=self.server, port=self.port, env=self.env)
            if not self.viz.check_connection(timeout_seconds=1):
                self.viz = None
                self._warn_unavailable()
        except Exception as exc:
            self.viz = None
            self._warn_unavailable(exc)

    def _warn_unavailable(self, exc: Exception | None = None):
        if self.warned:
            return
        detail = f" ({exc})" if exc else ""
        print(
            f"[Visdom] unavailable at {self.server}:{self.port}{detail}; "
            "training will continue with CSV/console logging."
        )
        self.warned = True

    def log(self, step: int, metrics: dict):
        if not self.enabled or step % self.update_interval != 0:
            return
        if self.viz is None:
            return
        for metric_name in self.metrics:
            self._log_metric(step, metric_name, metrics)
        self._log_instance(step, metrics)

    def _log_metric(self, step: int, metric_name: str, metrics: dict):
        value = metrics.get(metric_name)
        if not isinstance(value, Number):
            return

        win = metric_name
        opts = {
            "title": metric_name,
            "xlabel": "iteration",
            "ylabel": "value",
        }
        update = "append" if win in self.windows else None
        self.viz.line([float(value)], [step], win=win, update=update, opts=opts)
        self.windows.add(win)

    def _log_instance(self, step: int, metrics: dict):
        instance_text = (
            f"<b>Iteration:</b> {step}<br>"
            f"<b>Instance ID:</b> {metrics.get('instance_id', '')}<br>"
            f"<b>Instance:</b> {escape(str(metrics.get('instance_name', '')))}<br>"
            f"<b>Selection:</b> {escape(str(metrics.get('instance_selection', '')))}<br>"
            f"<b>Fulfillment Mean:</b> {float(metrics.get('fulfillment_rate_mean', 0.0)):.6f}"
        )
        self.viz.text(instance_text, win="instance", opts={"title": "instance"})
        self.windows.add("instance")

    def close(self):
        self.viz = None


# ----------------------------------------------------------------- new code

def _append_csv_rows(path: Path, columns: list, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def to_plain(value):
    """Recursively convert a configuration object into YAML-safe builtins."""
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_plain(item) for item in value]
    if value is None:
        return value
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):           # also subclasses (numpy ints)
        return int(value)
    if isinstance(value, float):         # also subclasses (numpy floats)
        return float(value)
    if isinstance(value, str):           # also subclasses (torch.__version__ is a TorchVersion)
        return str(value)
    if hasattr(value, "item") and callable(value.item):  # numpy / torch scalars
        try:
            return to_plain(value.item())
        except Exception:
            pass
    return str(value)


def git_commit_hash(root: Path) -> str:
    try:
        output = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return output.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(to_plain(data), file, sort_keys=False, allow_unicode=True)


class RunLogger:
    """Per-run files of the experiment contract on top of the legacy writer.

    ``log_legacy`` performs exactly the three calls the original trainer made per iteration
    (``write_metrics``, visualiser update, console print). ``log`` and ``curve`` append the contract files;
    ``obj_best`` is the running minimum over the greedy evaluations and, when the caller passes one, the best
    sampled training episode of the same instance (``obj_best`` argument).

    A fresh run (``resume=False``) starts ``log.csv`` / ``curve.csv`` from scratch; the legacy
    ``train_metrics.csv`` keeps its append-and-reconcile behaviour, so delete the run directory for a clean
    run. A resumed run restores the incumbents, the environment-step counter and the wall-clock offset.
    """

    LOG_COLS = ["epoch", "env_steps", "wall_time_s", "train_obj_mean", "fixed_obj_mean", "fixed_gap_mean",
                "policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl", "lr", "sps", "gpu_util",
                "gpu_mem_gb", "train_fulfillment_mean", "fixed_fulfillment_mean", "eval_seconds",
                "candidate_pairs_mean", "masked_pairs_mean", "mask_fallback_rate"]
    CURVE_COLS = ["epoch", "instance_id", "obj", "obj_best", "gap", "feasible", "wall_time_s"]
    FINAL_COLS = ["instance_id", "obj_best", "obj_greedy_final", "feasible", "epochs", "wall_time_s"]
    VISDOM_SCALARS = ("train_obj_mean", "fixed_obj_mean", "fixed_gap_mean")

    def __init__(self, run_dir, train_paras: dict, resume: bool = False):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "train_metrics.csv"
        self.log_path = self.run_dir / "log.csv"
        self.curve_path = self.run_dir / "curve.csv"
        self.final_path = self.run_dir / "fixed_set_final.csv"
        self.snapshot_path = self.run_dir / "config_snapshot.yaml"
        self.meta_path = self.run_dir / "train_meta.yaml"
        self.visualizer = TrainingVisualizer(train_paras)
        self.incumbent: dict = {}
        self.last_greedy: dict = {}
        self.env_steps = 0
        self._wall_offset = 0.0
        self._start_time = time.time()
        if resume:
            self._restore()
        else:
            for path in (self.log_path, self.curve_path, self.final_path):
                if path.exists():
                    path.unlink()
            if self.metrics_path.exists() and self.metrics_path.stat().st_size > 0:
                print(f"[RunLogger] {self.metrics_path} exists and will be appended to (original behaviour); "
                      "delete the run directory for a clean run.")

    # -- state -------------------------------------------------------------------------------------------
    def _restore(self) -> None:
        for row in _read_csv_rows(self.curve_path):
            try:
                instance_id = str(row["instance_id"])
                self.incumbent[instance_id] = float(row["obj_best"])
                self.last_greedy[instance_id] = (float(row["obj"]), str(row.get("feasible", "True")) == "True")
            except (KeyError, ValueError):
                continue
        log_rows = _read_csv_rows(self.log_path)
        if log_rows:
            try:
                self.env_steps = int(float(log_rows[-1]["env_steps"]))
                self._wall_offset = float(log_rows[-1]["wall_time_s"])
            except (KeyError, ValueError):
                pass

    def wall_time(self) -> float:
        return self._wall_offset + time.time() - self._start_time

    # -- per-iteration writers ---------------------------------------------------------------------------
    def log_legacy(self, iteration: int, row: dict) -> None:
        write_metrics(self.metrics_path, row)
        self.visualizer.log(iteration, row)
        print(row)

    def log(self, epoch: int, metrics: dict) -> dict:
        row = {column: metrics.get(column, "") for column in self.LOG_COLS}
        row["epoch"] = epoch
        _append_csv_rows(self.log_path, self.LOG_COLS, [row])
        self._visdom_scalars(epoch, {name: row[name] for name in self.VISDOM_SCALARS})
        return row

    def curve(self, epoch: int, rows: list) -> list:
        """``rows``: dicts with ``instance_id, obj, gap, feasible, wall_time_s`` and optionally ``obj_best``
        (an externally known better solution of that instance, e.g. the best sampled training episode)."""
        out_rows = []
        for item in rows:
            instance_id = str(item["instance_id"])
            obj = float(item["obj"])
            best = min(self.incumbent.get(instance_id, math.inf), obj)
            if item.get("obj_best") is not None:
                best = min(best, float(item["obj_best"]))
            self.incumbent[instance_id] = best
            self.last_greedy[instance_id] = (obj, bool(item.get("feasible", True)))
            out = {column: item.get(column, "") for column in self.CURVE_COLS}
            out.update(epoch=epoch, instance_id=instance_id, obj=obj, obj_best=best)
            out_rows.append(out)
        _append_csv_rows(self.curve_path, self.CURVE_COLS, out_rows)
        if out_rows:
            self._visdom_scalars(epoch, {
                "curve_obj_mean": sum(r["obj"] for r in out_rows) / len(out_rows),
                "curve_obj_best_mean": sum(r["obj_best"] for r in out_rows) / len(out_rows),
            })
        return out_rows

    def _visdom_scalars(self, step: int, values: dict) -> None:
        visualizer = self.visualizer
        if not visualizer.enabled or visualizer.viz is None or step % visualizer.update_interval != 0:
            return
        for name, value in values.items():
            if not isinstance(value, Number) or (isinstance(value, float) and math.isnan(value)):
                continue
            win = f"contract/{name}"
            update = "append" if win in visualizer.windows else None
            visualizer.viz.line([float(value)], [step], win=win, update=update,
                                opts={"title": name, "xlabel": "epoch", "ylabel": "value"})
            visualizer.windows.add(win)

    # -- end-of-run writers ------------------------------------------------------------------------------
    def write_final(self, epochs: int, wall_time_s: float) -> None:
        if not self.incumbent:
            return
        rows = []
        for instance_id, obj_best in self.incumbent.items():
            obj_greedy, feasible = self.last_greedy.get(instance_id, (math.nan, True))
            rows.append({"instance_id": instance_id, "obj_best": obj_best, "obj_greedy_final": obj_greedy,
                         "feasible": feasible, "epochs": epochs, "wall_time_s": wall_time_s})
        with self.final_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.FINAL_COLS)
            writer.writeheader()
            writer.writerows(rows)

    def write_snapshot(self, data: dict) -> None:
        write_yaml(self.snapshot_path, data)

    def write_meta(self, data: dict) -> None:
        write_yaml(self.meta_path, data)

    def close(self) -> None:
        self.visualizer.close()
