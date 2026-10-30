"""Convergence curves of the trained methods from result/conv_<method>.csv written by result/aggregate.py:
the validation objective (1 - fulfilment, mean over the validation instances) versus epoch in the generalisation
paradigm, the incumbent objective in the per-instance paradigm; mean +- std over runs."""
import pandas as pd
from _common import OUT_DIR, RESULT, require

from result.plot import plot_convergence
from result.stats import METHOD_LABELS

METHODS = ["ours", "nfe", "nm", "nf", "ng", "na", "hsddqn", "hgnn_ppo", "a2c", "dqn", "ddqn"]


def main() -> None:
    series = []
    for method in METHODS:
        path = RESULT / f"conv_{method}.csv"
        if path.exists():
            series.append((pd.read_csv(path), METHOD_LABELS.get(method, method)))
    if not series:
        require(RESULT / "conv_ours.csv")
        return
    print("[FIG]", plot_convergence(series, OUT_DIR / "convergence", ylabel="Validation objective (1 - fulfilment)"))


if __name__ == "__main__":
    main()
