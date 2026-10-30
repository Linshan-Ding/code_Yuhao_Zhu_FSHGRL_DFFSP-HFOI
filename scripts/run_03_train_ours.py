"""Train the proposed method (FSHGRL: PPO + fluid features + fluid candidate mask + sparse HGNN policy).
Generalization paradigm (configs/paper.yaml): N runs on the training pool (one pool instance per iteration),
validation on the two benchmark validation instances every val_every iterations, products
result/ours/run<k>/{train_metrics.csv,curve.csv,val_curve.csv,log.csv,best_val_checkpoint.pt,
best_greedy_checkpoint.pt,last_checkpoint.pt,config_snapshot.yaml,train_meta.yaml}. POOLS maps the method id to
its pool overlay (only the manuscript's training pool in this revision).
Per-instance paradigm (legacy): every benchmark instance x N runs -> result/ours/<instance_id>/run<k>/.
N = ours in configs/runs.yaml (1 in the revision: fixed seed 1, recorded). Interrupted runs resume from
last_checkpoint.pt; finished runs (train_meta.yaml present) are skipped."""
from _bootstrap import (instance_ids, method_jobs, n_runs, paradigm, require_train_confirmed, resolve_workers,
                        run_jobs, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
POOLS = {"ours": []}
TIERS = ("benchmark",)
N_RUNS = n_runs("ours", 1)

require_train_confirmed()
jobs = []
for tag, extra in (POOLS.items() if paradigm() == "generalization" else [("ours", [])]):
    jobs += method_jobs(tag, extra, instance_ids(*TIERS), N_RUNS)
step(f"train ours ({paradigm()}): {len(jobs)} jobs ({N_RUNS} runs per pool)")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
