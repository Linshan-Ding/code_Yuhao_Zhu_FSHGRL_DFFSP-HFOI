"""Train the proposed method (FSHGRL: PPO + fluid features + fluid candidate mask + sparse HGNN policy).
Generalization paradigm (configs/paper.yaml): N independent runs on the generated training pool, validation
on the val tier, products result/ours/run<k>/{train_metrics.csv,curve.csv,val_curve.csv,log.csv,
best_val_checkpoint.pt,best_greedy_checkpoint.pt,last_checkpoint.pt,config_snapshot.yaml,train_meta.yaml}.
Per-instance paradigm (legacy): every benchmark instance x N runs -> result/ours/<instance_id>/run<k>/.
N = ours in configs/runs.yaml (a single run gets a fixed seed). Interrupted runs resume from
last_checkpoint.pt; finished runs (train_meta.yaml present) are skipped."""
from _bootstrap import (instance_ids, method_jobs, n_runs, paradigm, require_train_confirmed, resolve_workers,
                        run_jobs, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
TIERS = ("benchmark",)
N_RUNS = n_runs("ours", 3)

require_train_confirmed()
jobs = method_jobs("ours", [], instance_ids(*TIERS), N_RUNS)
step(f"train ours ({paradigm()}): {len(jobs)} jobs ({N_RUNS} runs)")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
