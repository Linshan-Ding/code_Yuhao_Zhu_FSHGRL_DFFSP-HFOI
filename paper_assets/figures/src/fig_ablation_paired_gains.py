"""Paired fulfilment differences of FSHGRL over its ablations (paper figure ``ablation_paired_gains``, annotated
with the win/tie/loss record and the Holm-adjusted p value of result/paired_tests.csv) and the advantage heat map
over every comparator with a mean column (``baseline_advantage_heatmap``)."""
import pandas as pd
from _common import OUT_DIR, RESULT, require

from result.plot import plot_advantage_heatmap, plot_paired_gains
from result.stats import ABLATION_METHODS, DRL_METHODS, METHOD_LABELS, RULE_METHODS

SHORT = {"nfe": "NFe", "nm": "NM", "nf": "NF", "ng": "NG", "na": "NA"}


def _annotations() -> dict:
    path = RESULT / "paired_tests.csv"
    if not path.exists():
        return {}
    table = pd.read_csv(path)
    return {str(r["comparator"]): f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}\n$p$ = {float(r['p_holm']):.3f}"
            for _, r in table.iterrows()}


def main() -> None:
    path = RESULT / "advantage_table.csv"
    if not require(path):
        return
    advantage = pd.read_csv(path)
    ablations = [m for m in ABLATION_METHODS if m in advantage.columns]
    if ablations:
        print("[FIG]", plot_paired_gains(advantage, ablations, OUT_DIR / "ablation_paired_gains", SHORT,
                                         width_mm=84, height_mm=60, annotations=_annotations()))
    methods = [m for m in list(RULE_METHODS) + list(DRL_METHODS) + list(ABLATION_METHODS) if m in advantage.columns]
    if methods:
        print("[FIG]", plot_advantage_heatmap(advantage, methods, OUT_DIR / "baseline_advantage_heatmap", METHOD_LABELS,
                                              width_mm=181, height_mm=75, mean_column=True, fontsize=6.5))


if __name__ == "__main__":
    main()
