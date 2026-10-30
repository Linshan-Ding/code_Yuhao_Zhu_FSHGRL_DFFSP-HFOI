"""Train the ablation variants, N = ablations in configs/runs.yaml, under the configured paradigm (see run_03):
nfe (fluid node features off, mask on), nm (features on, mask off), nf (both off) form the 2 x 2 factor design
of the fluid guidance; ng replaces the heterogeneous graph by point-wise MLP encoders; na removes the
self-attention over the candidate pairs.
Products: result/<variant>/run<k>/ (generalization) or result/<variant>/<instance_id>/run<k>/ (per-instance).
Adding a variant = one more line in VARIANTS; the README command does not change."""
from _bootstrap import (instance_ids, method_jobs, n_runs, paradigm, require_train_confirmed, resolve_workers,
                        run_jobs, step)

PARALLEL, RESOURCE, MAX_CONCURRENT, GPU_MEM_PER_JOB_GB = True, "cpu", 2, None
VARIANTS = {"nfe": ["configs/ablation/nf_feature.yaml"], "nm": ["configs/ablation/nf_mask.yaml"],
            "nf": ["configs/ablation/nf.yaml"], "ng": ["configs/ablation/ng.yaml"],
            "na": ["configs/ablation/no_sparse_attn.yaml"]}
TIERS = ("benchmark",)
N_RUNS = n_runs("ablations", 1)

require_train_confirmed()
jobs = []
for tag, extra in VARIANTS.items():
    jobs += method_jobs(tag, extra, instance_ids(*TIERS), N_RUNS)
step(f"train {len(VARIANTS)} ablation variants x {N_RUNS} runs ({paradigm()}): {len(jobs)} jobs")
run_jobs(jobs, resolve_workers(MAX_CONCURRENT, RESOURCE, GPU_MEM_PER_JOB_GB, parallel=PARALLEL))
