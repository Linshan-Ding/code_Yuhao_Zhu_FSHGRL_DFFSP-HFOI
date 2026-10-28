"""Train the proposed method (FSHGRL: PPO + fluid features + fluid candidate mask + sparse HGNN policy).
Generalization paradigm (configs/paper.yaml): N independent runs per pool, validation on the pool's val tier,
products result/<method>/run<k>/{train_metrics.csv,curve.csv,val_curve.csv,log.csv,best_val_checkpoint.pt,
best_greedy_checkpoint.pt,last_checkpoint.pt,config_snapshot.yaml,train_meta.yaml}. POOLS lists the pools of the
manuscript: ours (training pool -> benchmark / arrival / case tiers), ours_arrival (arrival-diverse pool,
Section 5.x robustness study) and ours_case (industrial-case pool -> the nine case configurations).
Per-instance paradigm (legacy): every benchmark instance x N runs -> result/ours/<instance_id>/run<k>/.
N = ours in configs/runs.yaml (a single run gets a fixed seed). Interrupted runs resume from
last_checkpoint.pt; finished runs (train_meta.yaml present) are skipped."""
from _bootstrap import (instance_ids, method_jobs, n_runs, paradigm, require_train_confirmed, resolve_workers,
                        run_jobs, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
POOLS = {"ours": [], "ours_arrival": ["configs/pools/arrival_train.yaml"], "ours_case": ["configs/pools/case_train.yaml"]}
TIERS = ("benchmark",)
N_RUNS = n_runs("ours", 3)

require_train_confirmed()
jobs = []
for tag, extra in (POOLS.items() if paradigm() == "generalization" else [("ours", [])]):
    jobs += method_jobs(tag, extra, instance_ids(*TIERS), N_RUNS)
step(f"train ours ({paradigm()}): {len(jobs)} jobs ({N_RUNS} runs per pool)")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
