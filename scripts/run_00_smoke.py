"""Minute-scale end-to-end smoke test of the generalisation flow: train 2 iterations (PPO, hgnn, fluid features
and mask on) on three training-pool instances with validation on two val instances (configs/smoke.yaml after
configs/paper.yaml), evaluate the best validation checkpoint on those two val instances, run the dispatching
rules on them once, aggregate, draw the figures and build the tables, then run the unit tests of the reward
identity and the statistics. --smoke bypasses the training-parameter gate; nothing here is paper data
(everything goes to result/smoke/)."""
import os
import shutil
import time

from _bootstrap import CONFIGS, ROOT, done, merge_csv, read_yaml, run_py, step

SMOKE_CONFIGS = [*CONFIGS, "configs/smoke.yaml"]
RUN = "smoke/ours/run1"                        # generalisation layout result/<method>/run<k>/, as the real matrix
SMOKE_DIR = ROOT / "result" / "smoke"
VAL_IDS = [name.replace(".txt", "") for name in (read_yaml(ROOT / "configs" / "smoke.yaml")["instance"]["val_instances"])]
FIGURE_SCRIPTS = ["paper_assets/figures/src/fig_train_curve.py", "paper_assets/figures/src/fig_convergence.py",
                  "paper_assets/figures/src/fig_ablation_paired_gains.py", "paper_assets/figures/src/fig_decision_time.py"]

t0 = time.time()
step("smoke: remove the previous smoke outputs")
shutil.rmtree(SMOKE_DIR, ignore_errors=True)

step("smoke: train 2 iterations x 2 environments on 3 pool instances, validation on 2 val instances")
run_py("train.py", "--config", *SMOKE_CONFIGS, "--run-name", RUN, "--smoke")

step("smoke: evaluate the best validation checkpoint on the 2 val instances (greedy + 2 sampled episodes)")
run_py("eval.py", "--ckpt", f"result/{RUN}/best_val_checkpoint.pt", "--decode", "both", "--episodes", "2",
       "--threads", "1", "--method", "ours", "--run-id", "run1", "--instances", *VAL_IDS,
       "--out", f"result/{RUN}/eval_results.csv")

step("smoke: dispatching rules on the same instances (one run each)")
run_py("baselines/run_rules.py", "--instances", *VAL_IDS, "--runs-random", "1", "--out-dir", "result/smoke/rules",
       "--merged-out", "result/smoke/rules.csv", "--reference-out", "none")

step("smoke: aggregate, figures and tables")
merge_csv([ROOT / "result" / RUN / "eval_results.csv", SMOKE_DIR / "rules.csv"], SMOKE_DIR / "eval_results.csv")
run_py("result.aggregate", "--eval", "result/smoke/eval_results.csv", "--out-dir", "result/smoke")
env = dict(os.environ, FSHGRL_RESULT_DIR=str(SMOKE_DIR), FSHGRL_FIG_DIR=str(SMOKE_DIR / "figures"),
           FSHGRL_TABLE_DIR=str(SMOKE_DIR / "tables"))
for script in FIGURE_SCRIPTS:
    run_py(script, env=env)
run_py("paper_assets/scripts/make_tables.py", env=env)

step("smoke: unit tests (reward identity, rule baselines, statistics)")
run_py("pytest", "-q", "tests/test_rules.py", "tests/test_stats.py", "tests/test_config_bridge.py")

done(t0, ROOT / "result" / RUN / "curve.csv", ROOT / "result" / RUN / "val_curve.csv",
     ROOT / "result" / RUN / "train_metrics.csv", SMOKE_DIR / "eval_results.csv", SMOKE_DIR / "stats_summary.csv",
     SMOKE_DIR / "figures", SMOKE_DIR / "tables" / "comparison_pdr.tex")
