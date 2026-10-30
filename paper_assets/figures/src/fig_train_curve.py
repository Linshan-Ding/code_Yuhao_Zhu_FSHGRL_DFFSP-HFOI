"""Training traces of the two validation instances (paper figure ``train_curve``).

Generalisation paradigm (v2): reads result/ours/run1/val_curve.csv (one greedy episode per validation instance
every val_every iterations, the paper's two training-curve instances) and draws, per instance, the raw trace plus
its moving average. Per-instance paradigm (legacy): result/ours/<instance>/run1/curve.csv. Zero-argument: edit
INSTANCES / RUN.
"""
import pandas as pd
from _common import OUT_DIR, RESULT, require

from result.plot import plot_training_trace

INSTANCES = ["DDT500_num50", "DDT1200_num100"]
RUN = "run1"


def main() -> None:
    val_curve = RESULT / "ours" / RUN / "val_curve.csv"
    available = set(pd.read_csv(val_curve)["instance_id"]) if val_curve.exists() else set()
    for instance in INSTANCES:
        if instance in available:
            files = plot_training_trace(val_curve, OUT_DIR / f"train_curve_{instance}", column="fulfillment_rate",
                                        x_column="epoch", instance_id=instance, smooth_window=9,
                                        title=instance.replace("_", ", "))
        else:
            curve = RESULT / "ours" / instance / RUN / "curve.csv"
            if not require(curve):
                continue
            files = plot_training_trace(curve, OUT_DIR / f"train_curve_{instance}", column="obj", x_column="epoch",
                                        title=instance.replace("_", ", "))
        print(f"[FIG] {files}")


if __name__ == "__main__":
    main()
