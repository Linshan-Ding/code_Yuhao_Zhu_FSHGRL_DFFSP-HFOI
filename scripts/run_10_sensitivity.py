"""Mask-threshold sensitivity (reviewer comment 5) without retraining: the trained FSHGRL policy
(result/ours/run1/best_val_checkpoint.pt) is evaluated on the small tier under evaluation-time mask settings
epsilon_f in {off, 1e-5, 0.05, 0.1, 0.2} (configs/sensitivity/eps_<v>.yaml on top of the paper configuration),
greedy + eval.sampled_episodes sampled episodes per instance, CPU single thread. The MILP optimum of
result/milp/milp_results.csv (run_08) is the reference of the sensitivity table (make_tables.py).
Products: result/sensitivity/eps_<v>/eval_results.csv per threshold and result/sensitivity.csv (merged, with a
threshold column; not part of result/eval_results.csv)."""
import sys
import time

import pandas as pd

from _bootstrap import CONFIGS, ROOT, Job, done, resolve_workers, run_jobs, runs_cfg, step

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 3, None
THRESHOLDS = ["off", "1e-5", "0.05", "0.1", "0.2"]     # configs/sensitivity/eps_<value>.yaml
TIERS = ("small",)
METHOD, RUN = "ours", "run1"
CHECKPOINT = ROOT / "result" / METHOD / RUN / "best_val_checkpoint.pt"
EVAL = runs_cfg().get("eval", {}) or {}
EPISODES = int(EVAL.get("sampled_episodes", 10))
BATCH = int(EVAL.get("batch_episodes", 1))
MERGED = ROOT / "result" / "sensitivity.csv"

t0 = time.time()
if not CHECKPOINT.exists():
    sys.exit(f"[FAIL] {CHECKPOINT} missing: train the proposed method first (scripts/run_03_train_ours.py)")
jobs, parts = [], []
for value in THRESHOLDS:
    out = ROOT / "result" / "sensitivity" / f"eps_{value}" / "eval_results.csv"
    parts.append((value, out))
    jobs.append(Job(f"sensitivity/eps_{value}",
                    [sys.executable, "eval.py", "--ckpt", CHECKPOINT.relative_to(ROOT), "--config", *CONFIGS,
                     f"configs/sensitivity/eps_{value}.yaml", "--tiers", *TIERS, "--decode", "both", "--episodes", EPISODES,
                     "--batch-episodes", BATCH, "--threads", "1", "--method", METHOD, "--run-id", f"eps_{value}",
                     "--out", out.relative_to(ROOT)],
                    output=out))
step(f"sensitivity: {METHOD}/{RUN} on tiers {TIERS} under {len(THRESHOLDS)} mask settings (greedy + {EPISODES} sampled)")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))

step("sensitivity: merge")
frames = []
for value, out in parts:
    frame = pd.read_csv(out)
    frame.insert(0, "threshold", value)
    frames.append(frame)
pd.concat(frames, ignore_index=True).to_csv(MERGED, index=False)
done(t0, MERGED)
