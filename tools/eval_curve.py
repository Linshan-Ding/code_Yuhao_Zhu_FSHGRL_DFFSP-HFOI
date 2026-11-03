"""Post-hoc convergence curve: re-evaluate a run's checkpoint sequence on a fixed instance set.

Zero-argument tool: edit the constants below. Reads result/<run>/checkpoint_epoch*.pt (written when
algo.train.checkpoint_every > 0), runs the greedy evaluation of train.py on the chosen instances and writes
result/<run>/curve_posthoc_<SET_NAME>.csv with the columns of curve.csv; wall_time_s comes from the run's
log.csv row of the same epoch.
"""
from __future__ import annotations

import glob
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from agent.checkpoint import load_policy  # noqa: E402
from data.dataset import benchmark_instance_files  # noqa: E402
from result.logger import RunLogger  # noqa: E402
from result.metrics import load_reference  # noqa: E402
from train import evaluate_fixed_set  # noqa: E402

RUN = "result/smoke/ours/run1"      # run directory holding checkpoint_epoch*.pt
INSTANCES = None               # None = the run's training instances (config_snapshot.yaml); or ["DDT500_num50"]
SET_NAME = "train"             # suffix of the output file
DEVICE = "cpu"


def main() -> None:
    run_dir = ROOT / RUN
    checkpoints = sorted(glob.glob(str(run_dir / "checkpoint_epoch*.pt")))
    if not checkpoints:
        raise SystemExit(f"[FAIL] no checkpoint_epoch*.pt under {run_dir}; train with algo.train.checkpoint_every > 0")
    if INSTANCES:
        instance_ids = list(INSTANCES)
    else:
        snapshot = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
        instance_ids = [Path(name).stem for name in snapshot["instances"]]
    files = benchmark_instance_files(instance_ids)
    log_path = run_dir / "log.csv"
    wall = pd.read_csv(log_path).set_index("epoch")["wall_time_s"] if log_path.exists() else pd.Series(dtype=float)
    reference = load_reference()
    rows, best = [], {}
    for checkpoint in checkpoints:
        epoch = int(re.search(r"epoch(\d+)", Path(checkpoint).name).group(1))
        policy, _, train_paras, _ = load_policy(Path(checkpoint), DEVICE)
        for item in evaluate_fixed_set(policy, files, train_paras, reference):
            best[item["instance_id"]] = min(best.get(item["instance_id"], math.inf), item["obj"])
            rows.append({"epoch": epoch, "instance_id": item["instance_id"], "obj": item["obj"],
                         "obj_best": best[item["instance_id"]], "gap": item["gap"], "feasible": item["feasible"],
                         "wall_time_s": float(wall.get(epoch, math.nan))})
        print(f"[CURVE] epoch {epoch}: obj mean {sum(r['obj'] for r in rows[-len(files):]) / len(files):.4f}", flush=True)
    out = run_dir / f"curve_posthoc_{SET_NAME}.csv"
    pd.DataFrame(rows, columns=RunLogger.CURVE_COLS).to_csv(out, index=False)
    print(f"[CURVE] {len(rows)} rows ({len(best)} instances x {len(checkpoints)} checkpoints) -> {out}")


if __name__ == "__main__":
    main()
