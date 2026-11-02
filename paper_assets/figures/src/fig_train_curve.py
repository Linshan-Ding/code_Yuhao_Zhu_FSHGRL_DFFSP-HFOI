"""Fig. 7 of the revised manuscript (``train_curve``): one panel per validation instance with the validation
fulfilment of the FSHGRL policy (result/ours/run1/val_curve.csv: one greedy episode per validation instance every
val_every iterations; raw trace, centred moving average and the validation-selected checkpoint of train_meta.yaml),
each panel scaled to the range of its own curve. COMPARATORS adds a last panel with the smoothed validation
fulfilment of the listed learned methods (result/conv_<method>.csv written by result/aggregate.py); the manuscript
compares the learned methods on the test configurations instead, so it is empty. Zero-argument: edit INSTANCES /
COMPARATORS / RUN.
"""
import re

import pandas as pd
import yaml
from _common import OUT_DIR, RESULT, require

from result.plot import plot_validation_panels
from result.stats import METHOD_LABELS

INSTANCES = ["DDT500_num50", "DDT1200_num100"]
COMPARATORS: list[str] = []                       # e.g. ["ours", "hsddqn", "hgnn_ppo"] for a third panel
RUN = "run1"


def _instance_label(instance_id: str) -> str:
    match = re.search(r"DDT(\d+)_num(\d+)", instance_id)
    return f"(DDT, S) = ({match.group(1)}, {match.group(2)})" if match else instance_id


def _checkpoint(val_curve: pd.DataFrame) -> int | None:
    meta = RESULT / "ours" / RUN / "train_meta.yaml"
    if meta.exists():
        value = (yaml.safe_load(meta.read_text(encoding="utf-8")) or {}).get("best_val_iteration")
        if value is not None:
            return int(value)
    means = val_curve.groupby("epoch")["fulfillment_rate"].mean()
    return int(means.idxmax()) if not means.empty else None


def main() -> None:
    val_curve = RESULT / "ours" / RUN / "val_curve.csv"
    if not require(val_curve):
        return
    frame = pd.read_csv(val_curve)
    series = [(pd.read_csv(RESULT / f"conv_{m}.csv"), m) for m in COMPARATORS if (RESULT / f"conv_{m}.csv").exists()]
    files = plot_validation_panels(val_curve, series, OUT_DIR / "train_curve", instances=INSTANCES,
                                   checkpoint=_checkpoint(frame), smooth_window=9, labels=METHOD_LABELS,
                                   instance_labels={i: _instance_label(i) for i in INSTANCES},
                                   width_mm=181, height_mm=58)
    print(f"[FIG] {files}")


if __name__ == "__main__":
    main()
