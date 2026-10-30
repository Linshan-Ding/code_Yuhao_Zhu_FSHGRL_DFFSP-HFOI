"""Train the learned comparators of the manuscript (HSDDQN and HGNN-PPO adapted; N = learned_baselines in
configs/runs.yaml) and the additional baselines A2C, DQN, DDQN (N = learned_baselines_extra), all new
implementations on the training simulator (docs/baselines.md) with the same num_envs / max_iterations budget as
the proposed method and fluid guidance off, under the configured paradigm (see run_03).
Products: result/<method>/run<k>/ (generalization) or result/<method>/<instance_id>/run<k>/ (per-instance)."""
from _bootstrap import (instance_ids, method_jobs, n_runs, paradigm, require_train_confirmed, resolve_workers,
                        run_jobs, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
BASELINES = {"hsddqn": ["configs/baselines/hsddqn.yaml"], "hgnn_ppo": ["configs/baselines/hgnn_ppo.yaml"]}
EXTRA = {"a2c": ["configs/baselines/a2c.yaml"], "dqn": ["configs/baselines/dqn.yaml"],
         "ddqn": ["configs/baselines/ddqn.yaml"]}          # additional baselines; empty the dict to skip them
TIERS = ("benchmark",)
N_RUNS = n_runs("learned_baselines", 1)
N_RUNS_EXTRA = n_runs("learned_baselines_extra", 1)

require_train_confirmed()
jobs = []
for tag, extra in BASELINES.items():
    jobs += method_jobs(tag, extra, instance_ids(*TIERS), N_RUNS)
for tag, extra in EXTRA.items():
    jobs += method_jobs(tag, extra, instance_ids(*TIERS), N_RUNS_EXTRA)
step(f"train {len(BASELINES)} comparators x {N_RUNS} run(s) + {len(EXTRA)} extra baselines x {N_RUNS_EXTRA} run(s) "
     f"({paradigm()}): {len(jobs)} jobs")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
