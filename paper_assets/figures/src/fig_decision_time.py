"""Decision time per step of FSHGRL and the learned comparators grouped by DDT and order count (paper figure
``decision_time_configurations``, milliseconds, axis from zero) and the quality-time trade-off of every method of
the comparison set on one logarithmic time axis with a legend (``quality_time_tradeoff``), both on the benchmark tier (the
fifteen configurations of the manuscript; FSHGRL_FIG_TIER overrides, empty = every tier)."""
import os

from _common import OUT_DIR, RESULT, require

from result.aggregate import load_eval_results, point_estimates
from result.plot import plot_decision_time_methods, plot_quality_time_tradeoff
from result.stats import ABLATION_METHODS, DRL_METHODS, METHOD_LABELS, RULE_METHODS

TIME_METHODS = ["ours", "hgnn_ppo", "hsddqn"]


def main() -> None:
    path = RESULT / "eval_results.csv"
    if not require(path):
        return
    frame = point_estimates(load_eval_results(path))
    tier = os.environ.get("FSHGRL_FIG_TIER", "benchmark")
    if tier and "tier" in frame.columns and (frame["tier"] == tier).any():
        frame = frame[frame["tier"] == tier]
    present = [m for m in TIME_METHODS if (frame["method"] == m).any()]
    if present:
        print("[FIG]", plot_decision_time_methods(frame, present, OUT_DIR / "decision_time_configurations", METHOD_LABELS))
    methods = ["ours"] + list(ABLATION_METHODS) + list(DRL_METHODS) + list(RULE_METHODS)
    # the methods are identified by a legend outside the axes, so no label can cross the frame
    print("[FIG]", plot_quality_time_tradeoff(frame, [("", methods)], OUT_DIR / "quality_time_tradeoff", METHOD_LABELS,
                                              log_x=True, unit="ms", label_fontsize=7, width_mm=130, height_mm=66,
                                              annotate=False, legend="right"))


if __name__ == "__main__":
    main()
