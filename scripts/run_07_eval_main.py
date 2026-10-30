"""Evaluate every trained run and build result/eval_results.csv.

Generalization paradigm: for each result/<method>/run<k>/ the checkpoint named in configs/runs.yaml
(eval.checkpoint; auto = best_val_checkpoint.pt) is evaluated on every instance of the configured evaluation
tiers (instance.eval_tiers) with greedy decoding plus eval.sampled_episodes sampled episodes.
Per-instance paradigm: each result/<method>/<instance_id>/run<k>/ checkpoint (auto = best_greedy_checkpoint.pt)
is evaluated on its own instance. One eval_results.csv per job, merged afterwards together with the rule rows
of result/rules.csv; the training incumbents (fixed_set_final.csv) are collected into result/incumbents.csv."""
import sys
import time

from _bootstrap import (ROOT, Job, done, eval_tiers_cfg, merge_csv, paradigm, resolve_workers, run_jobs, run_py,
                        runs_cfg, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 3, None
METHODS = ["ours", "nfe", "nm", "nf", "ng", "na", "hsddqn", "hgnn_ppo", "a2c", "dqn", "ddqn"]
EVAL = runs_cfg().get("eval", {}) or {}
GENERALIZATION = paradigm() == "generalization"
CHECKPOINT = str(EVAL.get("checkpoint", "auto") or "auto")
if CHECKPOINT == "auto":
    CHECKPOINT = "best_val_checkpoint.pt" if GENERALIZATION else "best_greedy_checkpoint.pt"
EPISODES = int(EVAL.get("sampled_episodes", 10))
BATCH = int(EVAL.get("batch_episodes", 1))
TIERS = EVAL.get("tiers") or "auto"
TIERS = None if TIERS == "auto" else [str(t) for t in TIERS]   # None: each run's snapshot names its eval tiers
MERGED = ROOT / "result" / "eval_results.csv"

t0 = time.time()
jobs, parts = [], []
for method in METHODS:
    pattern = f"run*/{CHECKPOINT}" if GENERALIZATION else f"*/run*/{CHECKPOINT}"
    for ckpt in sorted((ROOT / "result" / method).glob(pattern)):
        run_dir = ckpt.parent
        out = run_dir / "eval_results.csv"
        parts.append(out)
        tier_args = ["--tiers", *TIERS] if (GENERALIZATION and TIERS) else []
        label = f"{method}/{run_dir.name}/eval" if GENERALIZATION else f"{method}/{run_dir.parent.name}/{run_dir.name}/eval"
        jobs.append(Job(label,
                        [sys.executable, "eval.py", "--ckpt", ckpt.relative_to(ROOT), "--decode", "both",
                         "--episodes", EPISODES, "--batch-episodes", BATCH, "--threads", "1", "--method", method,
                         "--run-id", run_dir.name, "--out", out.relative_to(ROOT), *tier_args],
                        output=out))
if not jobs:
    sys.exit("[FAIL] no trained runs with the expected checkpoint under result/<method>/: run scripts 03-05 first")
what = (f"tiers {TIERS}" if TIERS else f"each run's snapshot tiers (default {eval_tiers_cfg()})") if GENERALIZATION else "own instance"
step(f"evaluate {len(jobs)} checkpoints ({CHECKPOINT}; {what}; greedy + {EPISODES} sampled episodes each)")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))

step("merge evaluation rows with the rule results; collect training incumbents")
rules = ROOT / "result" / "rules.csv"
if not rules.exists():
    print("[WARN] result/rules.csv missing (run scripts/run_06_run_rules.py); eval_results.csv has no rule rows")
merge_csv(parts + [rules], MERGED)
incumbent_parts = []
for method in METHODS:
    pattern = "run*/fixed_set_final.csv" if GENERALIZATION else "*/run*/fixed_set_final.csv"
    if list((ROOT / "result" / method).glob(pattern)):
        out = ROOT / "result" / f"{method}_incumbents.csv"
        run_py("eval.py", "--collect", f"result/{method}/{pattern}", "--method", method, "--out", out.relative_to(ROOT))
        incumbent_parts.append(out)
merge_csv(incumbent_parts, ROOT / "result" / "incumbents.csv")
done(t0, MERGED, ROOT / "result" / "incumbents.csv")
