"""LaTeX table bodies from the aggregate CSVs (new code). Zero-argument.

Writes paper_assets/tables/{ablation,comparison_pdr,comparison_drl,wilcoxon}.tex from
result/ablation_results.csv, result/eval_results.csv (via result/aggregate.py point estimates) and
result/paired_tests.csv. Each file holds a booktabs tabular body to \\input into the manuscript.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from result.aggregate import load_eval_results, point_estimates  # noqa: E402
from result.stats import DRL_METHODS, METHOD_LABELS, RULE_METHODS  # noqa: E402

RESULT = Path(os.environ.get("FSHGRL_RESULT_DIR", ROOT / "result"))          # redirected by run_00_smoke.py
TABLES = Path(os.environ.get("FSHGRL_TABLE_DIR", ROOT / "paper_assets" / "tables"))


def _fmt(value, digits=4):
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _write(name: str, header: list, rows: list) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.tex"
    spec = "l" + "r" * (len(header) - 1)
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    lines += [" & ".join(str(cell) for cell in row) + r" \\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TABLE] {path}")
    return path


def ablation() -> None:
    path = RESULT / "ablation_results.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
    table = pd.read_csv(path)
    methods = [m for m in ("ours", "nfe", "nm", "nf", "ng", "na") if m in table.columns]
    header = ["Configuration (DDT, S)"] + [METHOD_LABELS[m] for m in methods]
    rows = [[f"({int(r['ddt'])}, {int(r['scale'])})"] + [_fmt(r[m]) for m in methods] for _, r in table.iterrows()]
    rows.append(["Mean"] + [_fmt(table[m].mean()) for m in methods])
    _write("ablation", header, rows)


def comparison(name: str, methods: list) -> None:
    path = RESULT / "eval_results.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
    frame = point_estimates(load_eval_results(path))
    present = [m for m in methods if (frame["method"] == m).any()]
    if not present:
        print(f"[SKIP] none of {methods} in {path}")
        return
    pivot_rate = frame.pivot_table(index=["ddt", "scale"], columns="method", values="fulfillment_rate", aggfunc="mean")
    pivot_time = frame.pivot_table(index=["ddt", "scale"], columns="method", values="decision_time_ms", aggfunc="mean")
    header = ["Configuration (DDT, S)"] + [f"{METHOD_LABELS[m]} $\\eta$" for m in present] + \
             [f"{METHOD_LABELS[m]} time per decision (s)" for m in present]
    rows = []
    for (ddt, scale), row in pivot_rate.iterrows():
        times = pivot_time.loc[(ddt, scale)]
        rows.append([f"({int(ddt)}, {int(scale)})"] + [_fmt(row.get(m)) for m in present]
                    + [_fmt(times.get(m) / 1000.0 if pd.notna(times.get(m)) else None, 3) for m in present])
    rows.append(["Mean"] + [_fmt(pivot_rate[m].mean()) for m in present]
                + [_fmt(pivot_time[m].mean() / 1000.0, 3) for m in present])
    _write(name, header, rows)


def wilcoxon() -> None:
    path = RESULT / "paired_tests.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
    table = pd.read_csv(path)
    header = ["Comparator", "$n$", "W/T/L", "Mean $\\Delta$ (pp)", "$R^+$", "$R^-$", "$p$", "$p_{\\mathrm{Holm}}$"]
    rows = [[r["comparator_label"], int(r["n"]), f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}",
             _fmt(r["mean_delta"], 2), _fmt(r["r_plus"], 1), _fmt(r["r_minus"], 1),
             f"{float(r['p_value']):.3e}", f"{float(r['p_holm']):.3e}"] for _, r in table.iterrows()]
    _write("wilcoxon", header, rows)


def main() -> None:
    ablation()
    comparison("comparison_pdr", ["ours"] + list(RULE_METHODS))
    comparison("comparison_drl", ["ours"] + list(DRL_METHODS))
    wilcoxon()


if __name__ == "__main__":
    main()
