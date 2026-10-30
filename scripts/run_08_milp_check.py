"""Exact MILP solutions on the small tier (baselines/milp_exact.py): MILP-A (the paper's model) and MILP-B
(abandonment variant, the simulator's convention) on the 12 instances of the tier small (benchmark structure
5 x 5 x 5, S in {6, 8, 10, 12} x DDT in {500, 900, 1500}), Gurobi when available, else HiGHS; every solution is
re-timed by the verifier and replayed through the simulator (the exhaustive cross-check is skipped by size on this
structure; it runs on micro instances in tests/test_milp_exact.py). The same instances serve the mask-threshold
sensitivity study (scripts/run_10_sensitivity.py). CPU only, independent of the training jobs.
Product: result/milp/milp_results.csv (one row per instance x variant; the appendix table of the paper)."""
import time

from _bootstrap import ROOT, done, run_py, step

TIERS = ("small",)
TIME_LIMIT_S = 600
BRUTE_FORCE_MAX_ORDERS = 6

t0 = time.time()
step(f"exact MILP on tiers {TIERS}: variants A and B, {TIME_LIMIT_S}s per solve, brute force up to S={BRUTE_FORCE_MAX_ORDERS}")
run_py("baselines/milp_exact.py", "--tiers", *TIERS, "--time-limit", str(TIME_LIMIT_S),
       "--brute-force-max-orders", str(BRUTE_FORCE_MAX_ORDERS))
done(t0, ROOT / "result" / "milp" / "milp_results.csv")
