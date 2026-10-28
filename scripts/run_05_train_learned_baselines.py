"""Train the learned baselines (A2C, DQN, DDQN; new implementations on the training simulator, see
docs/migration-map.md) with the same num_envs / max_iterations budget as the proposed method, N =
learned_baselines in configs/runs.yaml, under the configured paradigm (see run_03).
Products: result/<method>/run<k>/ (generalization) or result/<method>/<instance_id>/run<k>/ (per-instance)."""
from _bootstrap import (instance_ids, method_jobs, n_runs, paradigm, require_train_confirmed, resolve_workers,
                        run_jobs, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
BASELINES = {"a2c": ["configs/baselines/a2c.yaml"], "dqn": ["configs/baselines/dqn.yaml"],
             "ddqn": ["configs/baselines/ddqn.yaml"]}
TIERS = ("benchmark",)
N_RUNS = n_runs("learned_baselines", 1)

require_train_confirmed()
jobs = []
for tag, extra in BASELINES.items():
    jobs += method_jobs(tag, extra, instance_ids(*TIERS), N_RUNS)
step(f"train {len(BASELINES)} learned baselines x {N_RUNS} run(s) ({paradigm()}): {len(jobs)} jobs")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
