"""Statistical tests and summaries over eval_results.csv (new code).

Replaces the former ``agent/wilcoxn_312.py`` (which summed positions instead of ranks). Provided:

- ``wilcoxon_exact``: paired Wilcoxon signed-rank test; zero differences excluded, tied absolute
  differences get average ranks, two-sided p-value by exact enumeration of all sign assignments of the
  non-zero differences (a subset-sum count over the doubled ranks, feasible for the paper's n = 15 and far
  beyond); normal approximation with continuity correction only above ``max_exact_n``.
- ``holm``: Holm's step-down adjustment of a family of p-values.
- ``friedman``: Friedman test over methods on the same instances (scipy), with mean ranks.
- ``paired_table``: the paper's Table 11 layout (n, W/T/L, mean difference in percentage points, R+, R-, p,
  p_Holm) for one reference method against several comparators; ``tests/test_stats.py`` reproduces the
  published numbers (R+ = 120 / 105, p = 6.104e-5 / 1.221e-4, p_Holm = 3.052e-4).
- ``summarize`` / ``ablation_table`` / ``advantage_table``: mean +- std over independent runs per method and
  tier (std empty for a single run), the per-configuration ablation table and the percentage-point
  advantage matrix.

Conventions: ``fulfillment_rate`` is maximised, ``gap`` / ``obj_model`` are minimised; differences are
always oriented so that a positive value means the reference method is better.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

HIGHER_IS_BETTER = {"fulfillment_rate": True, "gap": False, "obj_model": False, "obj_best": False}
METHOD_LABELS = {
    "ours": "FSHGRL",
    "nfe": "FSHGRL-NFe",        # fluid node features off, fluid candidate mask on
    "nm": "FSHGRL-NM",          # fluid node features on, fluid candidate mask off
    "nf": "FSHGRL-NF",          # both off
    "ng": "FSHGRL-NG",          # point-wise MLP encoders instead of the heterogeneous graph
    "na": "FSHGRL-NA",          # no self-attention over the candidate pairs
    "hsddqn": "HSDDQN", "hgnn_ppo": "HGNN-PPO", "a2c": "A2C", "dqn": "DQN", "ddqn": "DDQN",
    "fifo": "FIFO", "spt": "SPT", "mor": "MOR", "mwkr": "MWKR", "random": "Random", "rrc": "RRC",
}
RULE_METHODS = ("fifo", "spt", "mor", "mwkr", "random", "rrc")
DRL_METHODS = ("hsddqn", "hgnn_ppo", "a2c", "dqn", "ddqn")
ABLATION_METHODS = ("nfe", "nm", "nf", "ng", "na")     # 2 x 2 fluid factor design + encoder + attention


def method_label(method: str) -> str:
    return METHOD_LABELS.get(str(method), str(method))


# ------------------------------------------------------------------ tests

def average_ranks(values: Sequence[float]) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(np.asarray(values, dtype=float), method="average")


def _exact_cdf_count(doubled_ranks: Sequence[int], threshold: int) -> int:
    """Number of sign assignments whose positive-rank sum (in doubled units) is <= threshold."""
    total = int(sum(doubled_ranks))
    counts = np.zeros(total + 1, dtype=np.int64)
    counts[0] = 1
    for rank in doubled_ranks:
        rank = int(rank)
        counts[rank:] = counts[rank:] + counts[:total + 1 - rank]
    return int(counts[:threshold + 1].sum())


def wilcoxon_exact(differences: Iterable[float], decimals: int | None = None, max_exact_n: int = 60) -> dict:
    d = np.asarray(list(differences), dtype=float)
    if decimals is not None:
        d = np.round(d, decimals)
    nonzero = d[d != 0]
    n = int(nonzero.size)
    result = {"n_pairs": int(d.size), "n": n, "n_zero": int(d.size - n),
              "wins": int((nonzero > 0).sum()), "losses": int((nonzero < 0).sum()),
              "r_plus": 0.0, "r_minus": 0.0, "statistic": 0.0, "p_value": 1.0, "exact": True}
    if n == 0:
        return result
    ranks = average_ranks(np.abs(nonzero))
    r_plus = float(ranks[nonzero > 0].sum())
    r_minus = float(ranks[nonzero < 0].sum())
    statistic = min(r_plus, r_minus)
    result.update({"r_plus": r_plus, "r_minus": r_minus, "statistic": statistic})
    if n <= max_exact_n:
        doubled = [int(round(2 * r)) for r in ranks]
        count = _exact_cdf_count(doubled, int(math.floor(2 * statistic + 1e-9)))
        result["p_value"] = min(1.0, 2.0 * count / float(2 ** n))
        result["exact"] = True
    else:
        from scipy.stats import norm

        mean = n * (n + 1) / 4.0
        _, tie_counts = np.unique(np.abs(nonzero), return_counts=True)
        variance = n * (n + 1) * (2 * n + 1) / 24.0 - (tie_counts ** 3 - tie_counts).sum() / 48.0
        z = (statistic - mean + 0.5) / math.sqrt(variance)
        result["p_value"] = float(min(1.0, 2.0 * norm.cdf(z)))
        result["exact"] = False
    return result


def holm(p_values: Sequence[float]) -> list:
    """Holm step-down adjusted p-values, returned in the original order."""
    p = np.asarray(list(p_values), dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, (m - position) * p[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def friedman(matrix) -> dict:
    """Friedman test over the columns (methods) of a blocks x methods matrix; mean ranks (1 = best for a
    minimised quantity; pass negated values for a maximised one)."""
    from scipy.stats import friedmanchisquare

    values = np.asarray(matrix, dtype=float)
    n_blocks, k = values.shape
    if k < 3:
        raise ValueError("the Friedman test needs at least three methods")
    statistic, p_value = friedmanchisquare(*[values[:, j] for j in range(k)])
    ranks = np.vstack([average_ranks(row) for row in values])
    return {"statistic": float(statistic), "p_value": float(p_value), "n_blocks": int(n_blocks), "k": int(k),
            "mean_ranks": ranks.mean(axis=0).tolist()}


# ------------------------------------------------------------- summaries

def _per_instance_means(frame: pd.DataFrame, method: str, metric: str) -> pd.Series:
    sub = frame[frame["method"] == method]
    return pd.to_numeric(sub[metric], errors="coerce").groupby(sub["instance_id"]).mean()


def paired_comparison(frame: pd.DataFrame, reference: str, comparator: str, metric: str = "fulfillment_rate",
                      decimals: int | None = 4, scale: float = 100.0, higher_is_better: bool | None = None) -> dict:
    """Per-instance run means of both methods, differences oriented so that > 0 favours ``reference``."""
    if higher_is_better is None:
        higher_is_better = HIGHER_IS_BETTER.get(metric, True)
    ref = _per_instance_means(frame, reference, metric)
    comp = _per_instance_means(frame, comparator, metric)
    ref, comp = ref.align(comp, join="inner")
    ref, comp = ref.dropna(), comp.dropna()
    ref, comp = ref.align(comp, join="inner")
    diff = (ref - comp) if higher_is_better else (comp - ref)
    test = wilcoxon_exact(diff.values, decimals=decimals)
    return {
        "reference": reference, "comparator": comparator, "metric": metric, "n_instances": int(diff.size),
        "n": test["n"], "wins": test["wins"], "ties": test["n_zero"], "losses": test["losses"],
        "mean_delta": float(diff.mean() * scale) if diff.size else float("nan"),
        "r_plus": test["r_plus"], "r_minus": test["r_minus"], "p_value": test["p_value"], "exact": test["exact"],
    }


def paired_table(frame: pd.DataFrame, reference: str, comparators: Sequence[str], metric: str = "fulfillment_rate",
                 decimals: int | None = 4, scale: float = 100.0) -> pd.DataFrame:
    rows = [paired_comparison(frame, reference, comparator, metric, decimals, scale)
            for comparator in comparators if (frame["method"] == comparator).any()]
    table = pd.DataFrame(rows)
    if not table.empty:
        table["p_holm"] = holm(table["p_value"].tolist())
        table["comparator_label"] = table["comparator"].map(method_label)
    return table


def summarize(frame: pd.DataFrame, metric: str = "gap", by: Sequence[str] = ("method", "tier")) -> pd.DataFrame:
    """Mean +- std over independent runs (run-level means over the instances); std is NaN for n_runs = 1."""
    data = frame.copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    keys = list(by)
    per_run = data.groupby(keys + ["run_id"])[metric].mean().reset_index()
    grouped = per_run.groupby(keys)[metric]
    out = grouped.agg(mean="mean", n_runs="count").reset_index()
    out["std"] = grouped.std(ddof=1).values
    out["n_instances"] = data.groupby(keys)["instance_id"].nunique().reindex(
        pd.MultiIndex.from_frame(out[keys]) if len(keys) > 1 else out[keys[0]]).values
    out.insert(len(keys), "metric", metric)
    return out[keys + ["metric", "mean", "std", "n_runs", "n_instances"]]


def ablation_table(frame: pd.DataFrame, methods: Sequence[str] = ("ours",) + ABLATION_METHODS,
                   metric: str = "fulfillment_rate", index_columns: Sequence[str] = ("ddt", "scale")) -> pd.DataFrame:
    """Per-instance run means of each method side by side, plus the differences to the first method."""
    present = [m for m in methods if (frame["method"] == m).any()]
    columns = {m: _per_instance_means(frame, m, metric) for m in present}
    table = pd.DataFrame(columns)
    meta = frame.drop_duplicates("instance_id").set_index("instance_id")[list(index_columns)]
    table = meta.join(table, how="inner")
    if present:
        reference = present[0]
        for m in present[1:]:
            table[f"delta_{m}"] = table[reference] - table[m]
    sort_cols = [c for c in index_columns if c in table.columns]
    table = table.reset_index().rename(columns={"index": "instance_id"})
    if sort_cols:
        table = table.sort_values(sort_cols).reset_index(drop=True)
    return table


def advantage_table(frame: pd.DataFrame, reference: str = "ours", metric: str = "fulfillment_rate",
                    scale: float = 100.0) -> pd.DataFrame:
    """Instance x method matrix of (reference - method) differences in percentage points."""
    ref = _per_instance_means(frame, reference, metric)
    out = {}
    for method in sorted(set(frame["method"]) - {reference}):
        comp = _per_instance_means(frame, method, metric)
        aligned_ref, aligned_comp = ref.align(comp, join="inner")
        out[method] = (aligned_ref - aligned_comp) * scale
    table = pd.DataFrame(out)
    meta = frame.drop_duplicates("instance_id").set_index("instance_id")[["ddt", "scale"]]
    table = meta.join(table, how="inner").reset_index().rename(columns={"index": "instance_id"})
    return table.sort_values(["ddt", "scale"]).reset_index(drop=True)
