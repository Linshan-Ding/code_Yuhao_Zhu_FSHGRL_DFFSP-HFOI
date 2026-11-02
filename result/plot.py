"""Publication-style figures from the CSV contract (new code).

The former seaborn scripts (``agent/boxplot.py``, ``violinplot.py``, ``pointplot_*.py``, ``train_curve.py``)
read CSV files that no longer exist and used absolute paths; their figure types are re-implemented here on
``curve.csv`` / ``eval_results.csv`` / the aggregate tables of ``result/aggregate.py``. The style follows the
scifig contract of the paper skills (Okabe-Ito colour-blind-safe palette, Times, closed axes frame, mm-based
sizes, PDF + 300 dpi PNG); ``plot_training_trace`` keeps the raw-trace-plus-moving-average look of the
former ``train_curve.py``. All functions return the figure and write ``<stem>.pdf`` and ``<stem>.png`` when
``out_stem`` is given.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MM = 1 / 25.4
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermilion": "#D55E00",
      "purple": "#CC79A7", "sky": "#56B4E9", "yellow": "#F0E442", "gray": "#6E6E6E", "black": "#000000"}
CYCLE = [(OI["blue"], "-", "o"), (OI["vermilion"], "--", "s"), (OI["green"], ":", "^"),
         (OI["orange"], "-.", "D"), (OI["purple"], (0, (4, 1.5)), "x"), (OI["gray"], (0, (1, 1)), "+"),
         (OI["sky"], (0, (3, 1, 1, 1)), "v"), (OI["yellow"], (0, (5, 1)), "P")]
# One style per method, shared by every figure (v2 revision): colour, linestyle, marker. The proposed method is
# always the blue solid circle; its ablations, the learned comparators and the rules keep their own entries.
METHOD_STYLE = {
    "ours": (OI["blue"], "-", "o"),
    "nfe": (OI["orange"], "--", "s"),
    "nm": (OI["green"], ":", "^"),
    "nf": (OI["vermilion"], "-.", "D"),
    "ng": (OI["purple"], (0, (4, 1.5)), "x"),
    "na": (OI["sky"], (0, (3, 1, 1, 1)), "v"),
    "hsddqn": (OI["gray"], (0, (1, 1)), "+"),
    "hgnn_ppo": (OI["black"], (0, (5, 1)), "P"),
    "fifo": (OI["orange"], "--", "s"),
    "mor": (OI["green"], ":", "^"),
    "mwkr": (OI["vermilion"], "-.", "D"),
    "random": (OI["purple"], (0, (4, 1.5)), "x"),
    "rrc": (OI["sky"], (0, (3, 1, 1, 1)), "v"),
    "spt": (OI["gray"], (0, (1, 1)), "+"),
}
RULE_LIKE = {"fifo", "mor", "mwkr", "random", "rrc", "spt"}       # drawn with hollow markers in scatter plots


def use_scifig(width_mm: float = 84, height_mm: float = 52) -> None:
    """Single-column default 84 x 52 mm; pass width_mm=181 for a double-column figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 7, "legend.frameon": False,
        "axes.linewidth": 0.5, "lines.linewidth": 0.9, "lines.markersize": 2.6,
        "axes.spines.top": True, "axes.spines.right": True, "axes.spines.left": True, "axes.spines.bottom": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "axes.grid": True, "grid.color": "#BFBFBF", "grid.linewidth": 0.3,
        "figure.figsize": (width_mm * MM, height_mm * MM),
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, stem) -> list:
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    plt.close(fig)
    return [stem.with_suffix(".pdf"), stem.with_suffix(".png")]


def _style(index: int):
    return CYCLE[index % len(CYCLE)]


def style_of(method: str | None, index: int = 0):
    """``(colour, linestyle, marker)`` of ``method`` from METHOD_STYLE, else the positional CYCLE entry."""
    if method in METHOD_STYLE:
        return METHOD_STYLE[method]
    return _style(index)


def smooth(series: pd.Series, window: int = 31) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=max(3, window // 4)).mean()


# ------------------------------------------------------------------ curves

def aggregate_curves(run_dirs: Sequence, key: str = "obj", instance_id: str | None = None,
                     filename: str = "curve.csv") -> pd.DataFrame:
    """Several runs' curve.csv -> (epoch, mean, std, n_instances, n_runs): per run the mean over the fixed
    instances per epoch, then mean / std over runs (std NaN for one run). Never reads log.csv."""
    per_run, n_instances = [], 0
    for index, run_dir in enumerate(run_dirs):
        curve = pd.read_csv(Path(run_dir) / filename)
        if instance_id is not None:
            curve = curve[curve["instance_id"] == instance_id]
        series = curve.groupby("epoch")[key].mean()
        series.name = f"run{index}"
        per_run.append(series)
        n_instances = max(n_instances, curve["instance_id"].nunique())
    matrix = pd.concat(per_run, axis=1)
    return pd.DataFrame({"epoch": matrix.index, "mean": matrix.mean(axis=1).values,
                         "std": matrix.std(axis=1, ddof=1).values if matrix.shape[1] > 1 else np.nan,
                         "n_instances": n_instances, "n_runs": matrix.shape[1]})


def plot_convergence(series: Sequence, out_stem=None, ylabel: str = "Objective (1 - fulfillment)",
                     reference: float | None = None, width_mm: float = 84, height_mm: float = 52):
    """``series``: [(conv DataFrame, label), ...]; mean line with +-1 std band; optional reference line."""
    use_scifig(width_mm, height_mm)
    fig, ax = plt.subplots()
    for index, (conv, label) in enumerate(series):
        colour, linestyle, _ = _style(index)
        ax.plot(conv["epoch"], conv["mean"], color=colour, ls=linestyle, label=label)
        if conv["std"].notna().any():
            ax.fill_between(conv["epoch"], conv["mean"] - conv["std"].fillna(0), conv["mean"] + conv["std"].fillna(0),
                            color=colour, alpha=0.18, lw=0)
    if reference is not None and not (isinstance(reference, float) and math.isnan(reference)):
        ax.axhline(reference, color=OI["gray"], ls="--", lw=0.6, label="Reference (best rule)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend()
    if out_stem:
        save(fig, out_stem)
    return fig


def plot_training_trace(csv_path, out_stem=None, column: str = "fulfillment_rate_mean", x_column: str = "iteration",
                        smooth_window: int = 31, title: str | None = None, percent: bool = True,
                        width_mm: float = 84, height_mm: float = 56, instance_id: str | None = None):
    """Raw per-iteration trace and its moving average (the look of the former train_curve.py). Reads
    train_metrics.csv (sampled training rate), curve.csv (pass column='obj', x_column='epoch') or val_curve.csv
    (pass column='fulfillment_rate' and the ``instance_id`` to keep; v2 generalisation paradigm)."""
    frame = pd.read_csv(csv_path)
    if instance_id is not None and "instance_id" in frame.columns:
        frame = frame[frame["instance_id"] == instance_id]
    if frame.empty:
        raise ValueError(f"{csv_path}: no rows" + (f" for instance {instance_id}" if instance_id else ""))
    frame = frame.sort_values(x_column)
    x = frame[x_column]
    y = frame[column].astype(float)
    if column == "obj":
        y = 1.0 - y
    if percent:
        y = y * 100.0
    use_scifig(width_mm, height_mm)
    fig, ax = plt.subplots()
    ax.plot(x, y, color=OI["sky"], lw=0.7, alpha=0.5, label="Fulfillment rate")
    ax.plot(x, smooth(y, smooth_window), color=OI["vermilion"], lw=1.1, label=f"Moving average ({smooth_window})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Fulfillment rate (%)" if percent else "Fulfillment rate")
    if title:
        ax.set_title(title, fontsize=8)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.legend(loc="lower right")
    if out_stem:
        save(fig, out_stem)
    return fig


# --------------------------------------------------------------- comparisons

def plot_paired_gains(advantage: pd.DataFrame, comparators: Sequence[str], out_stem=None, labels: dict | None = None,
                      width_mm: float = 84, height_mm: float = 56, annotations: dict | None = None):
    """Paired percentage-point differences (reference - comparator) per configuration, one column per
    comparator, dashed line at the mean (the paper's ablation_paired_gains figure). ``annotations`` maps a
    comparator to a short text (e.g. the win/tie/loss record and the adjusted p value) printed above its column."""
    labels = labels or {}
    annotations = annotations or {}
    use_scifig(width_mm, height_mm)
    fig, ax = plt.subplots()
    rng = np.random.default_rng(0)
    top = 0.0
    for index, comparator in enumerate(comparators):
        colour, _, marker = style_of(comparator, index)
        values = advantage[comparator].astype(float).values
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        if marker in ("x", "+"):
            ax.scatter(np.full(values.size, index) + jitter, values, s=9, color=colour, marker=marker, alpha=0.85)
        else:
            ax.scatter(np.full(values.size, index) + jitter, values, s=9, color=colour, marker=marker, alpha=0.85,
                       edgecolor="none")
        ax.hlines(values.mean(), index - 0.3, index + 0.3, color=colour, ls="--", lw=0.9)
        top = max(top, float(np.nanmax(values)))
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xticks(range(len(comparators)), [labels.get(c, c) for c in comparators])
    ax.set_ylabel("Fulfillment difference (pp)")
    if annotations:
        ax.margins(y=0.25)
        low, high = ax.get_ylim()
        for index, comparator in enumerate(comparators):
            text = annotations.get(comparator)
            if text:
                ax.text(index, high - 0.02 * (high - low), text, ha="center", va="top", fontsize=7)
    if out_stem:
        save(fig, out_stem)
    return fig


def plot_decision_time(frame: pd.DataFrame, method: str = "ours", column: str = "decision_time_ms", out_stem=None,
                       width_mm: float = 181, height_mm: float = 55, ylabel: str = "Decision time per step (s)"):
    """Two panels: decision time versus DDT (one line per order count S) and versus S (one line per DDT),
    the former pointplot_DDTS.py / pointplot_S.py; values are run means per configuration in seconds
    (``decision_time_ms`` = per decision, ``infer_time_ms`` = per episode)."""
    data = frame[frame["method"] == method].copy()
    data[column] = pd.to_numeric(data[column], errors="coerce") / 1000.0
    table = data.groupby(["ddt", "scale"])[column].mean().reset_index()
    use_scifig(width_mm, height_mm)
    fig, axes = plt.subplots(1, 2)
    for index, (scale, group) in enumerate(table.groupby("scale")):
        colour, linestyle, marker = _style(index)
        axes[0].plot(group["ddt"], group[column], color=colour, ls=linestyle, marker=marker, label=f"S = {int(scale)}")
    axes[0].set_xlabel("DDT")
    axes[0].set_ylabel(ylabel)
    axes[0].set_xticks(sorted(table["ddt"].unique()))
    axes[0].legend()
    for index, (ddt, group) in enumerate(table.groupby("ddt")):
        colour, linestyle, marker = _style(index)
        axes[1].plot(group["scale"], group[column], color=colour, ls=linestyle, marker=marker, label=f"DDT = {int(ddt)}")
    axes[1].set_xlabel("S (orders)")
    axes[1].set_xticks(sorted(table["scale"].unique()))
    axes[1].legend()
    fig.tight_layout()
    if out_stem:
        save(fig, out_stem)
    return fig


def plot_quality_time_tradeoff(frame: pd.DataFrame, panels: Sequence, out_stem=None, labels: dict | None = None,
                               time_column: str = "decision_time_ms", width_mm: float = 181, height_mm: float = 55,
                               xlabel: str | None = None, log_x: bool = False, unit: str = "s",
                               label_fontsize: float = 6, offsets: dict | None = None):
    """Mean fulfillment versus mean decision time, one marker per method; ``panels`` = [(title, methods), ...].
    ``unit`` is "s" or "ms"; ``log_x`` draws a logarithmic time axis (rules at 0.6 ms and policies at 10-15 ms
    on one panel); rule methods use hollow markers; ``offsets`` moves a method's label by (dx, dy) points."""
    labels = labels or {}
    offsets = offsets or {}
    scale = 1.0 if unit == "ms" else 1.0 / 1000.0
    xlabel = xlabel or f"Mean decision time per step ({unit})"
    use_scifig(width_mm, height_mm)
    fig, axes = plt.subplots(1, len(panels), squeeze=False)
    for axis, (title, methods) in zip(axes[0], panels):
        for index, method in enumerate(methods):
            sub = frame[frame["method"] == method]
            if sub.empty:
                continue
            colour, _, marker = style_of(method, index)
            time_value = pd.to_numeric(sub[time_column], errors="coerce").mean() * scale
            rate = pd.to_numeric(sub["fulfillment_rate"], errors="coerce").mean() * 100.0
            size = 28 if method == "ours" else 20
            if marker in ("x", "+"):                                   # unfilled markers take a single colour
                axis.scatter([time_value], [rate], s=size, color=colour, marker=marker, linewidths=0.9,
                             label=labels.get(method, method), zorder=3)
            else:
                hollow = method in RULE_LIKE
                axis.scatter([time_value], [rate], s=size, facecolors="none" if hollow else colour,
                             edgecolors=colour, marker=marker, linewidths=0.8, label=labels.get(method, method), zorder=3)
            dx, dy = offsets.get(method, (3, 3))
            axis.annotate(labels.get(method, method), (time_value, rate), textcoords="offset points",
                          xytext=(dx, dy), fontsize=label_fontsize)
        if log_x:
            axis.set_xscale("log")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Mean fulfillment rate (%)")
        if title:
            axis.set_title(title, fontsize=8)
    fig.tight_layout()
    if out_stem:
        save(fig, out_stem)
    return fig


def plot_advantage_heatmap(advantage: pd.DataFrame, methods: Sequence[str], out_stem=None, labels: dict | None = None,
                           width_mm: float = 181, height_mm: float = 70, mean_column: bool = False,
                           fontsize: float = 5):
    """Configurations x methods matrix of percentage-point differences with a common diverging scale;
    ``mean_column`` appends the row means as a last column, ``fontsize`` sets the tick and cell text size."""
    labels = labels or {}
    present = [m for m in methods if m in advantage.columns]
    matrix = advantage[present].astype(float).values
    rows = [f"({int(r.ddt)}, {int(r.scale)})" for r in advantage.itertuples()]
    if mean_column and matrix.size:
        matrix = np.vstack([matrix, np.nanmean(matrix, axis=0, keepdims=True)])
        rows = rows + ["Mean"]
    use_scifig(width_mm, height_mm)
    fig, ax = plt.subplots()
    limit = float(np.nanmax(np.abs(matrix))) if matrix.size else 1.0
    image = ax.imshow(matrix.T, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    fig.colorbar(image, ax=ax, label="Advantage (pp)", fraction=0.03, pad=0.02)
    ax.set_xticks(range(len(rows)), rows, rotation=60, ha="right", fontsize=fontsize + 1)
    ax.set_yticks(range(len(present)), [labels.get(m, m) for m in present], fontsize=fontsize + 1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if not math.isnan(value):
                ax.text(i, j, f"{value:.1f}", ha="center", va="center", fontsize=fontsize,
                        color="white" if abs(value) > 0.6 * limit else "black")
    if mean_column and matrix.size:
        ax.axvline(len(rows) - 1.5, color="black", lw=0.6)
    ax.grid(False)
    if out_stem:
        save(fig, out_stem)
    return fig


def plot_validation_panels(val_curve_csv, conv_series: Sequence, out_stem=None,
                           instances: Sequence[str] = ("DDT500_num50", "DDT1200_num100"), checkpoint: int | None = None,
                           smooth_window: int = 9, width_mm: float = 181, height_mm: float = 55,
                           labels: dict | None = None, instance_labels: dict | None = None):
    """The paper's Fig. 7 (v2): panels (a), (b) the validation fulfillment of the proposed policy on each validation
    instance (raw trace from val_curve.csv and its centred moving average; the selected checkpoint marked with an
    open circle and its value), panel (c) the moving average of the validation fulfillment (1 - ``mean`` of a
    conv_<method>.csv frame) of the learned methods in ``conv_series`` = [(frame, method_id), ...]."""
    labels = labels or {}
    instance_labels = instance_labels or {}
    frame = pd.read_csv(val_curve_csv)
    use_scifig(width_mm, height_mm)
    n_panels = len(instances) + (1 if conv_series else 0)
    fig, axes = plt.subplots(1, n_panels, squeeze=False)
    axes = axes[0]
    for axis, instance in zip(axes, instances):
        part = frame[frame["instance_id"] == instance].sort_values("epoch")
        if part.empty:
            raise ValueError(f"{val_curve_csv}: no rows for instance {instance}")
        x = part["epoch"].astype(float).values
        y = part["fulfillment_rate"].astype(float).values
        axis.plot(x, y, color=OI["sky"], lw=0.7, alpha=0.6, label="Validation fulfillment")
        axis.plot(x, smooth(pd.Series(y), smooth_window).values, color=OI["vermilion"], lw=1.1,
                  label=f"Moving average ({smooth_window})")
        if checkpoint is not None:
            hit = part[part["epoch"] == checkpoint]
            if not hit.empty:
                value = float(hit["fulfillment_rate"].iloc[0])
                axis.scatter([checkpoint], [value], s=26, facecolors="none", edgecolors="black", linewidths=0.8,
                             zorder=4, label=f"Selected checkpoint ({checkpoint})")
                axis.annotate(f"{value:.2f}", (checkpoint, value), textcoords="offset points", xytext=(-14, 6),
                              fontsize=7)
        axis.set_xlabel("Iteration")
        axis.set_xlim(float(x.min()), float(x.max()))
        axis.set_ylim(0.0, 1.0)
        axis.text(0.02, 0.96, instance_labels.get(instance, instance), transform=axis.transAxes, ha="left", va="top",
                  fontsize=7)
    axes[0].set_ylabel("Fulfillment rate")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.56, 0.90))
    if conv_series:
        axis = axes[-1]
        for index, (conv, method) in enumerate(conv_series):
            colour, linestyle, _ = style_of(method, index)
            values = 1.0 - conv["mean"].astype(float)
            axis.plot(conv["epoch"], smooth(values.reset_index(drop=True), smooth_window).values, color=colour,
                      ls=linestyle, lw=1.0, label=labels.get(method, method))
        axis.set_xlabel("Iteration")
        axis.set_ylim(0.0, 1.0)
        axis.legend(loc="upper left")
    for letter, axis in zip("abc", axes):
        axis.text(-0.08, 1.03, f"({letter})", transform=axis.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    if out_stem:
        save(fig, out_stem)
    return fig


def plot_decision_time_methods(frame: pd.DataFrame, methods: Sequence[str], out_stem=None, labels: dict | None = None,
                               column: str = "decision_time_ms", width_mm: float = 181, height_mm: float = 52,
                               ymax: float | None = None):
    """Decision time per step in milliseconds for several methods: (a) versus DDT (mean over S), (b) versus S
    (mean over DDT); the y axis starts at zero so that the spread between configurations is read at its true size."""
    labels = labels or {}
    use_scifig(width_mm, height_mm)
    fig, axes = plt.subplots(1, 2)
    top = 0.0
    for index, method in enumerate(methods):
        data = frame[frame["method"] == method].copy()
        if data.empty:
            continue
        data[column] = pd.to_numeric(data[column], errors="coerce")
        colour, linestyle, marker = style_of(method, index)
        by_ddt = data.groupby("ddt")[column].mean()
        by_scale = data.groupby("scale")[column].mean()
        axes[0].plot(by_ddt.index, by_ddt.values, color=colour, ls=linestyle, marker=marker, label=labels.get(method, method))
        axes[1].plot(by_scale.index, by_scale.values, color=colour, ls=linestyle, marker=marker, label=labels.get(method, method))
        top = max(top, float(np.nanmax(data[column])))
    axes[0].set_xlabel("DDT")
    axes[0].set_ylabel("Decision time per step (ms)")
    axes[0].set_xticks(sorted(frame["ddt"].unique()))
    axes[1].set_xlabel("S (orders)")
    axes[1].set_xticks(sorted(frame["scale"].unique()))
    limit = ymax if ymax is not None else 1.25 * top
    for letter, axis in zip("ab", axes):
        axis.set_ylim(0.0, limit)
        axis.text(-0.08, 1.03, f"({letter})", transform=axis.transAxes, ha="right", va="bottom", fontsize=8)
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    if out_stem:
        save(fig, out_stem)
    return fig


def plot_sensitivity(table: pd.DataFrame, out_stem=None, width_mm: float = 84, height_mm: float = 52):
    """Mask-threshold sensitivity (the paper's Section 5.8 figure): greedy and sampled fulfillment on the left axis
    and the share of pruned candidate pairs on the right axis against the threshold categories of
    ``table`` (columns threshold, greedy_eta, sampled_eta, pruned_pct)."""
    use_scifig(width_mm, height_mm)
    fig, ax = plt.subplots()
    x = np.arange(len(table))
    ticks = [str(v) for v in table["threshold"]]
    ax.plot(x, table["greedy_eta"].astype(float), color=OI["blue"], ls="-", marker="o", label="Greedy decoding")
    ax.plot(x, table["sampled_eta"].astype(float), color=OI["vermilion"], ls="--", marker="s", label="Sampled decoding (10 episodes)")
    ax.set_xticks(x, ticks)
    ax.set_xlabel(r"Mask threshold $\varepsilon_f$")
    ax.set_ylabel("Fulfillment rate")
    low = float(min(table["greedy_eta"].min(), table["sampled_eta"].min()))
    high = float(max(table["greedy_eta"].max(), table["sampled_eta"].max()))
    ax.set_ylim(low - 0.02, high + 0.07)
    twin = ax.twinx()
    twin.plot(x, table["pruned_pct"].astype(float), color=OI["gray"], ls=":", marker="^", label="Pruned pairs")
    twin.set_ylabel("Pruned candidate pairs (%)")
    twin.set_ylim(0.0, max(1.0, float(table["pruned_pct"].max()) * 2.0))
    twin.grid(False)
    handles, names = ax.get_legend_handles_labels()
    handles2, names2 = twin.get_legend_handles_labels()
    ax.legend(handles + handles2, names + names2, loc="upper left")
    if out_stem:
        save(fig, out_stem)
    return fig


def plot_method_boxplot(frame: pd.DataFrame, methods: Sequence[str], metric: str = "fulfillment_rate", out_stem=None,
                        labels: dict | None = None, ylabel: str = "Fulfillment rate", width_mm: float = 84,
                        height_mm: float = 52):
    """One box per method over the per-instance run means (the former boxplot.py)."""
    labels = labels or {}
    use_scifig(width_mm, height_mm)
    fig, ax = plt.subplots()
    series = []
    for method in methods:
        sub = frame[frame["method"] == method]
        values = pd.to_numeric(sub[metric], errors="coerce").groupby(sub["instance_id"]).mean().dropna().values
        series.append(values)
    boxes = ax.boxplot(series, widths=0.5, patch_artist=True, medianprops=dict(color="black", lw=0.8))
    for index, patch in enumerate(boxes["boxes"]):
        colour, _, _ = _style(index)
        patch.set(facecolor=colour + "33", edgecolor=colour, lw=0.8)
    ax.set_xticks(range(1, len(methods) + 1), [labels.get(m, m) for m in methods])
    ax.set_ylabel(ylabel)
    if out_stem:
        save(fig, out_stem)
    return fig
