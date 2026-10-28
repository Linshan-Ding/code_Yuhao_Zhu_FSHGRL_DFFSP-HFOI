"""Dispatching-rule baselines (FIFO, SPT, MOR, MWKR, Random, RRC) on every indexed tier except the training
pool (benchmark, val, arrival, ...), CPU only, and the reference solutions: result/rules/<rule>/eval_results.csv,
result/rules.csv and data/instances/reference.csv (best deterministic rule per instance; the reference of every
gap and of the pilot gate). Random / RRC get rules_random independent runs (configs/runs.yaml).
Independent of the training jobs."""
import time

from _bootstrap import ROOT, done, n_runs, run_py, step

t0 = time.time()
step(f"dispatching rules on every evaluation tier ({n_runs('rules_random', 10)} runs of random / rrc)")
run_py("baselines/run_rules.py", "--runs-random", str(n_runs("rules_random", 10)))
done(t0, ROOT / "result" / "rules.csv", ROOT / "data" / "instances" / "reference.csv")
