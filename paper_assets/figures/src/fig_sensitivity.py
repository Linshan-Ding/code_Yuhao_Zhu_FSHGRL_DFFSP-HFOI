"""Mask-threshold sensitivity figure of Section 5.8 (``sensitivity_threshold``): greedy and sampled fulfilment of the
trained FSHGRL policy on the small tier against the mask threshold, with the share of pruned candidate pairs on a
second axis; the rows come from make_tables.sensitivity_frame() (result/sensitivity.csv of run_10)."""
import sys
from pathlib import Path

from _common import OUT_DIR, RESULT, require

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from make_tables import sensitivity_frame  # noqa: E402

from result.plot import plot_sensitivity  # noqa: E402


def main() -> None:
    if not require(RESULT / "sensitivity.csv"):
        return
    table = sensitivity_frame()
    if table is None or table.empty:
        return
    print("[FIG]", plot_sensitivity(table, OUT_DIR / "sensitivity_threshold"))


if __name__ == "__main__":
    main()
