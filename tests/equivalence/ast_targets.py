"""Functions whose bodies must stay byte-identical (checked by AST dump).

These paths cannot be exercised in this environment (no Gurobi licence, no
checkpoint resume in the reference run), so the harness compares their AST
against the golden checkout instead. Each entry is
(golden_file, qualname, candidate_file, qualname). Update the candidate side
when a function moves.
"""
# v2: `_solve_cached_lp` and `_fluid_features` were extended (per-pair LP solution for the candidate mask, LP
# backend abstraction); the original Gurobi model is kept verbatim in `_solve_lp_gurobi` and the aggregation
# of its solution in `_solve_cached_lp`. Their equivalence with the golden tree under the heuristic fallback is
# covered by L2 and L6; the Gurobi branch is exercised against the HiGHS backend in tests/test_fluid_mask.py.
TARGETS = [
    ("agent/training_simulator.py", "TrainingSimulator._fluid_cache_key",
     "environment/simulator.py", "TrainingSimulator._fluid_cache_key"),
    ("agent/training_simulator.py", "TrainingSimulator._load_gurobi",
     "environment/simulator.py", "TrainingSimulator._load_gurobi"),
    ("agent/train.py", "apply_lr_schedule", "train.py", "apply_lr_schedule"),
    ("agent/train.py", "load_checkpoint", "agent/checkpoint.py", "load_checkpoint"),
    ("agent/train.py", "prepare_rollout_policy", "train.py", "prepare_rollout_policy"),
    ("agent/sci_rollout.py", "select_instance", "data/dataset.py", "select_instance"),
    ("agent/train.py", "write_metrics", "result/logger.py", "write_metrics"),
    ("agent/visualization.py", "TrainingVisualizer", "result/logger.py", "TrainingVisualizer"),
]
