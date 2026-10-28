"""Exact MILP solutions on the small-instance tiers (baselines/milp_exact.py): MILP-A (the paper's model) and
MILP-B (abandonment variant, the simulator's convention) on every instance of the tiers tiny (2 products x
3 stages x 2 machines per stage, S = 4 ... 12) and tiny5 (benchmark structure 5 x 5 x 5, S = 6 ... 10), Gurobi
when available, else HiGHS; every solution is re-timed by the verifier, replayed through the simulator and,
for S <= 6, cross-checked against exhaustive enumeration. CPU only, independent of the training jobs.
Product: result/milp/milp_results.csv (one row per instance x variant; the appendix table of the paper)."""
import time

from _bootstrap import ROOT, done, run_py, step

TIERS = ("tiny", "tiny5")
TIME_LIMIT_S = 600
BRUTE_FORCE_MAX_ORDERS = 6

t0 = time.time()
step(f"exact MILP on tiers {TIERS}: variants A and B, {TIME_LIMIT_S}s per solve, brute force up to S={BRUTE_FORCE_MAX_ORDERS}")
run_py("baselines/milp_exact.py", "--tiers", *TIERS, "--time-limit", str(TIME_LIMIT_S),
       "--brute-force-max-orders", str(BRUTE_FORCE_MAX_ORDERS))
done(t0, ROOT / "result" / "milp" / "milp_results.csv")
