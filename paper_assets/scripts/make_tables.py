"""LaTeX table bodies from the aggregate CSVs, the configuration and the instance parameters (new code).
Zero-argument; every file is a booktabs tabular body to \\input into the manuscript, under
paper_assets/tables/ (FSHGRL_TABLE_DIR) from result/ (FSHGRL_RESULT_DIR):

- ablation.tex            FSHGRL and its five ablations per benchmark configuration (ablation_results.csv);
- comparison_pdr.tex      FSHGRL against the dispatching rules per benchmark configuration: eta and time per
                          decision (eval_results.csv through result/aggregate.py point estimates);
- comparison_drl.tex      the same against the learned comparators (HSDDQN, HGNN-PPO, A2C, DQN, DDQN);
- wilcoxon.tex            paired Wilcoxon tests with Holm correction (paired_tests.csv);
- generalization_<tier>.tex  zero-shot results on every evaluation tier other than benchmark (arrival, case_eval,
                          ...): per instance and method from eval_results.csv, with the tier summary row;
- hyperparameters.tex     Table 6 of the paper generated from configs/{env,algo,paper,runs}.yaml and the
                          instance index, so that the manuscript cannot drift from the code;
- instance_statistics.tex the benchmark instances from data/instances/benchmark/*/params.yaml (S, mean
                          inter-arrival time, CV, last arrival, mean processing time, due slack, load ratio);
- milp_small.tex          the exact MILP check (result/milp/milp_results.csv) with the policy's greedy eta on the
                          same instances (eval_results.csv, tier small), its gap to the optimum and the best rule
                          of result/rules.csv;
- sensitivity.tex         the mask-threshold sensitivity of the trained policy (result/sensitivity.csv written by
                          scripts/run_10_sensitivity.py): per threshold the greedy eta and the mean sampled eta
                          (10 episodes) on the small tier, the share of pruned candidate pairs, candidate pairs
                          before / after the mask, the fallback rate and the time per decision (the CSV also keeps
                          the gap to the MILP optimum);
- sampled_decoding.tex    Appendix B: the sampled-decoding fulfilment (mean of 10 episodes) of every learned policy
                          per benchmark configuration (eval_results.csv, decode = sample);
- generator_settings.tex  Table 5(a): the generator settings of the training pool (configs/instance.yaml);
- milp_small_compact.tex  the nine-column layout of the MILP check used in the manuscript appendix.
Every table is also written as CSV next to the .tex file (same name) for the response letter.
The environment variable FSHGRL_EXCLUDE_METHODS (comma-separated method ids) leaves methods out of every table
(the manuscript compares the five dispatching rules of the paper; the CSVs of result/ keep every method).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from result.aggregate import load_eval_results, point_estimates  # noqa: E402
from result.stats import (ABLATION_METHODS, DETERMINISTIC_RULE_METHODS, DRL_METHODS, METHOD_LABELS,  # noqa: E402
                          RULE_METHODS, method_label)

RESULT = Path(os.environ.get("FSHGRL_RESULT_DIR", ROOT / "result"))          # redirected by run_00_smoke.py
TABLES = Path(os.environ.get("FSHGRL_TABLE_DIR", ROOT / "paper_assets" / "tables"))
INSTANCES = Path(os.environ.get("FSHGRL_INSTANCES_DIR", ROOT / "data" / "instances"))
CONFIG_FILES = ["configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml"]
PAIRED_TIER = "benchmark"


def _fmt(value, digits=4):
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "--"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "--"


def _tex(text) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _sci(value) -> str:
    """``6.104e-05`` -> ``$6.104\\times10^{-5}$`` (p values of the Wilcoxon table)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if number == 0.0:
        return "0"
    mantissa, exponent = f"{number:.3e}".split("e")
    return f"${mantissa}\\times10^{{{int(exponent)}}}$"


def _instance_label(instance_id: str, tier: str | None = None) -> str:
    """Row labels of the zero-shot tables: arrival instances ``ARR3000_uniform_rho0.8_DDT900_num100`` ->
    ``Uniform, $\\rho = 0.8$``; small instances ``SML5000_DDT500_num6`` -> ``SML5000 (500, 6)``; otherwise the
    escaped id."""
    arrival = re.search(r"_(uniform|exponential|bursty)_rho([0-9.]+)_DDT(\d+)_num(\d+)$", str(instance_id))
    if arrival:
        return f"{arrival.group(1).capitalize()}, $\\rho = {arrival.group(2)}$"
    small = re.match(r"^([A-Za-z]+\d+)_DDT(\d+)_num(\d+)$", str(instance_id))
    if small:
        return f"{small.group(1)} ({small.group(2)}, {small.group(3)})"
    return _tex(instance_id)


def _tier_header(instance_ids) -> str:
    """``Instance ($S = 100$, DDT $= 900$)`` when every id of the tier shares S and DDT, else ``Instance``."""
    pairs = {(m.group(1), m.group(2)) for m in (re.search(r"DDT(\d+)_num(\d+)$", str(i)) for i in instance_ids) if m}
    if len(pairs) == 1:
        ddt, scale = next(iter(pairs))
        return f"Instance ($S = {scale}$, DDT $= {ddt}$)"
    return "Instance"


def _write(name: str, header: list, rows: list, align: str | None = None) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.tex"
    spec = align or ("l" + "r" * (len(header) - 1))
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    lines += [" & ".join(str(cell) for cell in row) + r" \\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TABLE] {path}")
    return path


def _write_csv(name: str, frame: pd.DataFrame) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def _points():
    path = RESULT / "eval_results.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return None
    return point_estimates(load_eval_results(path))


# -- benchmark tables (paper Tables 7-11) --------------------------------------------------------------------
def ablation() -> None:
    path = RESULT / "ablation_results.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
    table = pd.read_csv(path)
    methods = [m for m in ("ours",) + tuple(ABLATION_METHODS) if m in table.columns]
    header = ["Configuration (DDT, S)"] + [METHOD_LABELS[m] for m in methods]
    rows = [[f"({int(r['ddt'])}, {int(r['scale'])})"] + [_fmt(r[m]) for m in methods] for _, r in table.iterrows()]
    rows.append(["Mean"] + [_fmt(table[m].mean()) for m in methods])
    _write("ablation", header, rows)
    _write_csv("ablation", table[[c for c in ["instance_id", "ddt", "scale"] if c in table.columns] + methods])


def comparison(name: str, methods: list, tier: str = PAIRED_TIER) -> None:
    frame = _points()
    if frame is None:
        return
    if "tier" in frame.columns and (frame["tier"] == tier).any():
        frame = frame[frame["tier"] == tier]
    present = [m for m in methods if (frame["method"] == m).any()]
    if not present:
        print(f"[SKIP] none of {methods} in {RESULT / 'eval_results.csv'}")
        return
    pivot_rate = frame.pivot_table(index=["ddt", "scale"], columns="method", values="fulfillment_rate", aggfunc="mean")
    pivot_time = frame.pivot_table(index=["ddt", "scale"], columns="method", values="decision_time_ms", aggfunc="mean")
    header = ["Configuration (DDT, S)"] + [f"{METHOD_LABELS[m]} $\\eta$" for m in present] + \
             [f"{METHOD_LABELS[m]} time per decision (ms)" for m in present]
    rows, flat = [], []
    for (ddt, scale), row in pivot_rate.iterrows():
        times = pivot_time.loc[(ddt, scale)] if (ddt, scale) in pivot_time.index else pd.Series(dtype=float)
        rows.append([f"({int(ddt)}, {int(scale)})"] + [_fmt(row.get(m)) for m in present]
                    + [_fmt(times.get(m), 1) for m in present])
        record = {"ddt": int(ddt), "scale": int(scale)}
        record.update({f"{m}_eta": row.get(m) for m in present})
        record.update({f"{m}_time_ms": times.get(m) for m in present})
        flat.append(record)
    rows.append(["Mean"] + [_fmt(pivot_rate[m].mean()) for m in present]
                + [_fmt(pivot_time[m].mean() if m in pivot_time.columns else None, 1) for m in present])
    _write(name, header, rows)
    _write_csv(name, pd.DataFrame(flat))
    _write_grouped(name, present, rows)


def _write_grouped(name: str, present: list, rows: list) -> Path:
    """``<name>_grouped.tex``: the same rows under a two-level header (fulfillment columns, then time columns),
    the layout used inside the manuscript."""
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}_grouped.tex"
    k = len(present)
    spec = "l" + "r" * (2 * k)
    top = (" & \\multicolumn{" + str(k) + "}{c}{Fulfillment rate $\\eta$} & \\multicolumn{" + str(k)
           + "}{c}{Time per decision (ms)} \\\\")
    rules = f"\\cmidrule(lr){{2-{k + 1}}}\\cmidrule(lr){{{k + 2}-{2 * k + 1}}}"
    second = "Configuration (DDT, S) & " + " & ".join(METHOD_LABELS[m] for m in present) + " & " + \
        " & ".join(METHOD_LABELS[m] for m in present) + " \\\\"
    lines = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", top, rules, second, "\\midrule"]
    lines += [" & ".join(str(cell) for cell in row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TABLE] {path}")
    return path


def wilcoxon() -> None:
    path = RESULT / "paired_tests.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
    table = pd.read_csv(path)
    header = ["Comparator", "$n$", "W/T/L", "Mean $\\Delta$ (pp)", "$R^+$", "$R^-$", "$p$", "$p_{\\mathrm{Holm}}$"]
    rows = [[r["comparator_label"], int(r["n"]), f"{int(r['wins'])}/{int(r['ties'])}/{int(r['losses'])}",
             _fmt(r["mean_delta"], 2), _fmt(r["r_plus"], 1), _fmt(r["r_minus"], 1),
             _sci(r["p_value"]), _sci(r["p_holm"])] for _, r in table.iterrows()]
    _write("wilcoxon", header, rows)
    _write_csv("wilcoxon", table)


# -- zero-shot tiers ---------------------------------------------------------------------------------------
def generalization_tables(exclude: tuple = (PAIRED_TIER,)) -> None:
    """One table per evaluation tier other than the paired one: instances x methods (eta, run means), plus the
    tier mean; the methods are ordered FSHGRL, its pool variants, ablations, learned comparators, rules."""
    frame = _points()
    if frame is None or "tier" not in frame.columns:
        return
    order = ["ours"] + list(ABLATION_METHODS) + list(DRL_METHODS) + list(RULE_METHODS)
    for tier, tier_frame in frame.groupby("tier", sort=True):
        if tier in exclude:
            continue
        methods = [m for m in order if (tier_frame["method"] == m).any()]
        methods += sorted(set(tier_frame["method"]) - set(methods))
        pivot = tier_frame.pivot_table(index="instance_id", columns="method", values="fulfillment_rate", aggfunc="mean")
        pivot = pivot.reindex(columns=methods)
        header = [_tier_header(pivot.index)] + [method_label(m) for m in methods]
        rows = [[_instance_label(instance, tier)] + [_fmt(row.get(m)) for m in methods] for instance, row in pivot.iterrows()]
        rows.append(["Mean"] + [_fmt(pivot[m].mean()) for m in methods])
        _write(f"generalization_{tier}", header, rows)
        _write_csv(f"generalization_{tier}", pivot.reset_index())


# -- configuration and data tables -----------------------------------------------------------------------------
def _load_config(files=CONFIG_FILES) -> dict:
    merged: dict = {}

    def merge(base: dict, extra: dict) -> dict:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value
        return base

    for name in files:
        path = ROOT / name
        if path.exists():
            merge(merged, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return merged


def _tier_counts() -> dict:
    index = INSTANCES / "index.csv"
    if not index.exists():
        return {}
    return pd.read_csv(index).groupby("tier").size().to_dict()


def _config_label(instance_id: str) -> str:
    """``DDT500_num50`` -> ``(500, 50)``; other ids are returned with underscores escaped."""
    match = re.search(r"DDT(\d+)_num(\d+)", str(instance_id))
    return f"({match.group(1)}, {match.group(2)})" if match else _tex(instance_id)


def _threshold_text(value) -> str:
    """``1e-05`` -> ``$10^{-5}$``, ``2.5e-03`` -> ``$2.5\\times10^{-3}$`` (the mask threshold of Table 6)."""
    try:
        mantissa, exponent = f"{float(value):.3e}".split("e")
    except (TypeError, ValueError):
        return str(value)
    mantissa = mantissa.rstrip("0").rstrip(".")
    base = f"10^{{{int(exponent)}}}"
    return f"${base}$" if mantissa == "1" else f"${mantissa}\\times{base}$"


def _device_text(device) -> str:
    """Configuration device names as prose: ``cpu`` -> CPU, ``cuda:0`` -> GPU."""
    name = str(device).lower()
    return "GPU" if name.startswith(("cuda", "gpu", "mps")) else "CPU" if name.startswith("cpu") else str(device)


def hyperparameter_rows(config: dict | None = None, runs: dict | None = None, counts: dict | None = None) -> list:
    """``(block, parameter, value)`` triples of the paper's Table 6, read from the configuration and written as
    prose rather than as configuration values (threshold in powers of ten, CPU / GPU, instance sets)."""
    cfg = config if config is not None else _load_config()
    runs = runs if runs is not None else (yaml.safe_load((ROOT / "configs" / "runs.yaml").read_text(encoding="utf-8")) or {})
    counts = counts if counts is not None else _tier_counts()
    env, algo, inst = cfg.get("env", {}), cfg.get("algo", {}), cfg.get("instance", {})
    model, attention, train = algo.get("model", {}), algo.get("attention", {}), algo.get("train", {})
    d_ope, d_ma = int(model.get("out_size_ope", 16)), int(model.get("out_size_ma", 16))
    heads = list(model.get("num_heads", [1, 1]))
    attn_dim = 2 * (d_ope + d_ma) + 2                        # actor feature: (ope, ma, pooled ope, pooled ma, due, rate)
    backend = str(env.get("fluid_backend", "scipy"))
    backend_text = {"scipy": "HiGHS (scipy.optimize.linprog)", "highs": "HiGHS (scipy.optimize.linprog)",
                    "auto": "HiGHS (scipy), heuristic fallback", "gurobi": "Gurobi (unsupported)",
                    "heuristic": "heuristic"}.get(backend, backend)
    eval_cfg = runs.get("eval", {}) or {}
    val_list = [str(v) for v in (inst.get("val_instances") or [])]
    val_count = len(val_list) if val_list else counts.get("val", "--")
    val_names = ", ".join(_config_label(Path(v).stem) for v in val_list) if val_list else "val tier"
    rows = [
        ("Environment and action", "Candidate orders per operation type $k$ (EDD)", env.get("order_top_k", 5)),
        ("Environment and action", "Fluid LP solver / time limit (s)", f"{backend_text} / {env.get('fluid_time_limit', 2)}"),
        ("Environment and action", "Fluid node features / candidate mask",
         f"{env.get('fluid_features', 'on')} / {'on' if env.get('fluid_mask', True) else 'off'}"),
        ("Environment and action", "Mask threshold $\\varepsilon_f$", _threshold_text(env.get("fluid_mask_threshold", 1e-5))),
        ("Environment and action", "LP recomputation",
         {"decision": "at every decision epoch"}.get(str(env.get("fluid_recompute", "decision")), str(env.get("fluid_recompute")))),
        ("Environment and action", "Discard penalty / progress coefficient",
         f"{env.get('discard_penalty', 1.0)} / {env.get('progress_reward_coeff', 0.0)}"),
        ("Network", "Raw feature width (operation type / machine)",
         f"{model.get('in_size_ope', 12)} / {model.get('in_size_ma', 5)}"),
        ("Network", "Embedding width $d$ (operation type / machine)", f"{d_ope} / {d_ma}"),
        ("Network", "HGNN layers / attention heads per layer", f"{len(heads)} / {heads[0] if heads else 1}"),
        ("Network", "Operation MLP hidden width", model.get("hidden_size_ope", 256)),
        ("Network", "Candidate self-attention: heads / feature width / normalisation",
         f"{attention.get('sparse_attention_heads', 1)} / {attn_dim} / layer normalization"
         if attention.get("use_sparse_attention", True) else "off"),
        ("Network", "Actor / order actor / critic (layers $\\times$ width, tanh)",
         f"{model.get('n_hidden_actor', 3)} $\\times$ {model.get('n_latent_actor', 128)} / "
         f"{model.get('n_hidden_actor', 3)} $\\times$ {model.get('n_latent_actor', 128)} / "
         f"{model.get('n_hidden_critic', 3)} $\\times$ {model.get('n_latent_critic', 128)}"),
        ("Network", "Dropout", model.get("dropout", 0.0)),
        ("Training", "Iterations / episodes per iteration", f"{train.get('max_iterations')} / {train.get('num_envs')}"),
        ("Training", "PPO epochs / clip $\\varepsilon_{\\mathrm{clip}}$ / minibatch",
         f"{train.get('K_epochs')} / {train.get('eps_clip')} / {train.get('minibatch_size')}"),
        ("Training", "Optimiser / learning rate / schedule",
         f"Adam {tuple(train.get('betas', [0.9, 0.999]))} / {train.get('lr')} / {train.get('lr_schedule', 'none')}"),
        ("Training", "$\\gamma$ / GAE $\\lambda$", f"{train.get('gamma')} / {train.get('gae_lambda')}"),
        ("Training", "$c_1$ / $c_2$ / $c_3$", f"{train.get('A_coeff')} / {train.get('vf_coeff')} / {train.get('entropy_coeff')}"),
        ("Training", "Gradient clipping / target KL", f"{train.get('max_grad_norm')} / {train.get('target_kl')}"),
        ("Training", "Collection device / update device",
         f"{_device_text(train.get('rollout_device', 'cpu'))} / {_device_text(train.get('update_device', 'cuda:0'))}"),
        ("Training", "Training instance per iteration",
         "the next pool instance (each used once)" if str(inst.get("instance_selection", "random")) in ("round_robin", "fixed")
         else f"{inst.get('instance_selection', 'random')} draw from the pool"),
        ("Data", "Training pool / validation / test",
         f"{counts.get('train', '--')} generated / {val_count} benchmark ({val_names}) / {counts.get('benchmark', '--')} benchmark"),
        ("Data", "Other unseen instance sets (arrival regimes / small instances)",
         f"{counts.get('arrival', '--')} / {counts.get('small', '--')} instances"),
        ("Evaluation", "Model selection",
         f"best validation mean (greedy episode per validation instance every {train.get('val_every', 50)} iterations)"),
        ("Evaluation", "Decoding / independent runs $K$",
         f"greedy 1 + sampled {eval_cfg.get('sampled_episodes', 10)} episodes / {runs.get('ours', 3)}"),
    ]
    return rows


def hyperparameters() -> None:
    rows = hyperparameter_rows()
    _write("hyperparameters", ["Block", "Parameter", "Value"], [[b, p, v] for b, p, v in rows], align="lll")
    _write_csv("hyperparameters", pd.DataFrame(rows, columns=["block", "parameter", "value"]))


def instance_statistics(tier: str = "benchmark") -> None:
    folder = INSTANCES / tier
    params = sorted(folder.glob("*/params.yaml"))
    if not params:
        print(f"[SKIP] no params.yaml under {folder}")
        return
    records = [yaml.safe_load(p.read_text(encoding="utf-8")) for p in params]
    frame = pd.DataFrame(records)
    frame["ddt"] = pd.to_numeric(frame["ddt"], errors="coerce")
    frame = frame.sort_values(["ddt", "n_orders"]).reset_index(drop=True)
    header = ["Instance (DDT, S)", "$S$", "$\\bar{\\Delta t}$ (s)", "CV", "Last arrival (s)", "$\\bar{p}$ (s)",
              "Slack min / mean / max (s)", "$\\rho$", "$\\rho_{\\mathrm{bottleneck}}$"]
    rows = []
    for _, r in frame.iterrows():
        label = f"({int(r['ddt'])}, {int(r['n_orders'])})" if pd.notna(r["ddt"]) else _tex(r["instance_id"])
        rows.append([label, int(r["n_orders"]), _fmt(r["inter_arrival_mean"], 1), _fmt(r["inter_arrival_cv"], 2),
                     _fmt(r["arrival_last"], 0), _fmt(r["process_time_mean"], 1),
                     f"{_fmt(r['due_slack_min'], 0)} / {_fmt(r['due_slack_mean'], 0)} / {_fmt(r['due_slack_max'], 0)}",
                     _fmt(r["load_ratio"], 2), _fmt(r["load_ratio_bottleneck"], 2)])
    _write(f"instance_statistics{'' if tier == 'benchmark' else '_' + tier}", header, rows)
    columns = ["instance_id", "tier", "n_orders", "ddt", "inter_arrival_mean", "inter_arrival_cv", "arrival_last",
               "process_time_mean", "due_slack_min", "due_slack_mean", "due_slack_max", "load_ratio",
               "load_ratio_bottleneck"]
    _write_csv(f"instance_statistics{'' if tier == 'benchmark' else '_' + tier}", frame[[c for c in columns if c in frame]])


def _policy_small_eta(method: str = "ours", tier: str = "small") -> dict:
    """Per-instance greedy eta of ``method`` on ``tier`` from eval_results.csv (empty when absent)."""
    path = RESULT / "eval_results.csv"
    if not path.exists():
        return {}
    frame = point_estimates(load_eval_results(path))
    frame = frame[(frame["method"] == method) & (frame.get("tier", "") == tier)] if "tier" in frame.columns else frame.iloc[0:0]
    return frame.groupby("instance_id")["fulfillment_rate"].mean().to_dict() if not frame.empty else {}


def milp_small() -> None:
    path = RESULT / "milp" / "milp_results.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
    milp = pd.read_csv(path)
    rules_path = RESULT / "rules.csv"
    best_rule = {}
    if rules_path.exists():
        rules = load_eval_results(rules_path)                       # honours FSHGRL_EXCLUDE_METHODS
        rules = rules[rules["method"].isin(DETERMINISTIC_RULE_METHODS)]
        if not rules.empty:
            best_rule = rules.groupby("instance_id")["fulfillment_rate"].max().to_dict()
    policy = _policy_small_eta()
    pivot = milp.pivot_table(index=["tier", "instance", "S", "n_ops", "n_binary"], columns="variant",
                             values=["eta_opt", "status", "solve_s", "replay_ok", "brute_force_ok"], aggfunc="first")
    header = ["Tier", "Instance", "$S$", "$|\\mathcal{K}|$", "Binaries", "$\\eta^*_{A}$", "$\\eta^*_{B}$", "Status (A / B)",
              "Solve time (s, A / B)", "Replay", "Brute force", "Best rule $\\eta$", "FSHGRL $\\eta$", "Gap (pp)"]
    rows, flat, paper = [], [], []
    for (tier, instance, s, n_ops, n_binary), r in pivot.iterrows():
        eta_a, eta_b = r.get(("eta_opt", "A")), r.get(("eta_opt", "B"))
        status = f"{r.get(('status', 'A'), '--')} / {r.get(('status', 'B'), '--')}"
        solve = f"{_fmt(r.get(('solve_s', 'A')), 1)} / {_fmt(r.get(('solve_s', 'B')), 1)}"
        replay = "yes" if all(bool(r.get(("replay_ok", v), False)) for v in ("A", "B") if ("replay_ok", v) in r) else "no"
        brute_values = [r.get(("brute_force_ok", v)) for v in ("A", "B") if ("brute_force_ok", v) in r]
        brute_values = [v for v in brute_values if pd.notna(v) and v != ""]
        brute = "--" if not brute_values else ("yes" if all(bool(v) for v in brute_values) else "no")
        rule_eta = best_rule.get(instance)
        ours_eta = policy.get(instance)
        reference = eta_b if pd.notna(eta_b) else eta_a
        gap_pp = (float(reference) - float(ours_eta)) * 100.0 if ours_eta is not None and pd.notna(reference) else None
        rows.append([_tex(tier), _tex(instance), int(s), int(n_ops), int(n_binary), _fmt(eta_a), _fmt(eta_b), status,
                     solve, replay, brute, _fmt(rule_eta), _fmt(ours_eta), _fmt(gap_pp, 2)])
        flat.append({"tier": tier, "instance": instance, "S": s, "n_ops": n_ops, "n_binary": n_binary, "eta_A": eta_a,
                     "eta_B": eta_b, "status": status, "solve_s": solve, "replay_ok": replay, "brute_force_ok": brute,
                     "best_rule_eta": rule_eta, "ours_eta": ours_eta, "gap_pp": gap_pp})
        paper.append({"instance": instance, "S": s, "n_binary": n_binary, "eta": eta_a, "status": str(r.get(("status", "A"), "--")),
                      "solve_s": r.get(("solve_s", "A")), "replay": "yes" if bool(r.get(("replay_ok", "A"), False)) else "no",
                      "ours_eta": ours_eta})
    _write("milp_small", header, rows, align="llrrrrrllrrrrr")
    _write_csv("milp_small", pd.DataFrame(flat))
    # the manuscript's Table A.1: the model of the paper only (variant A), without the rule column
    all_optimal = all(f["status"] == "optimal" for f in paper)
    compact_header = ["Instance (DDT, $S$)", "$S$", "Binaries", "$\\eta^*$"] + ([] if all_optimal else ["Status"]) + \
        ["Time (s)", "Replay", "FSHGRL $\\eta$"]
    compact_rows = []
    for f in paper:
        compact_rows.append([_instance_label(f["instance"]), int(f["S"]), int(f["n_binary"]), _fmt(f["eta"])]
                            + ([] if all_optimal else [f["status"]])
                            + [_fmt(f["solve_s"], 1), f["replay"], _fmt(f["ours_eta"])])
    _write("milp_small_compact", compact_header, compact_rows, align="lrrr" + ("" if all_optimal else "l") + "rlr")


THRESHOLD_ORDER = ["off", "1e-5", "0.05", "0.1", "0.2"]


def sensitivity_frame() -> pd.DataFrame | None:
    """Per-threshold rows of the mask sensitivity (result/sensitivity.csv of scripts/run_10_sensitivity.py): greedy
    and sampled fulfilment, gap of the sampled fulfilment to the MILP optimum (result/milp/milp_results.csv, variant
    B else A), candidate pairs before / after the mask and the pruned share, fallback rate, time per decision."""
    path = RESULT / "sensitivity.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return None
    frame = load_eval_results(path)
    frame["threshold"] = frame["threshold"].astype(str)
    optimum = {}
    milp_path = RESULT / "milp" / "milp_results.csv"
    if milp_path.exists():
        milp = pd.read_csv(milp_path)
        for variant in ("A", "B"):                       # B overrides A when both exist
            part = milp[(milp["variant"] == variant) & pd.to_numeric(milp["eta_opt"], errors="coerce").notna()]
            optimum.update(dict(zip(part["instance"], pd.to_numeric(part["eta_opt"]))))
    thresholds = sorted(frame["threshold"].unique(),
                        key=lambda v: (THRESHOLD_ORDER.index(v) if v in THRESHOLD_ORDER else len(THRESHOLD_ORDER), v))
    counts = pd.to_numeric(frame.get("decision_count"), errors="coerce") if "decision_count" in frame.columns else None
    if counts is not None:
        frame["decision_time_ms"] = pd.to_numeric(frame["infer_time_ms"], errors="coerce") / counts.where(counts > 0)
    flat = []
    for value in thresholds:
        part = frame[frame["threshold"] == value]
        greedy = part[part["decode"] == "greedy"] if "decode" in part.columns else part
        sampled = part[part["decode"] == "sample"] if "decode" in part.columns else part.iloc[0:0]
        per_instance = greedy.groupby("instance_id")["fulfillment_rate"].mean()
        per_instance_sampled = (sampled if not sampled.empty else greedy).groupby("instance_id")["fulfillment_rate"].mean()
        gaps = [(optimum[i] - eta) * 100.0 for i, eta in per_instance_sampled.items() if i in optimum]
        gap = sum(gaps) / len(gaps) if gaps else None
        before = pd.to_numeric(greedy.get("candidate_pairs_mean"), errors="coerce").mean() if "candidate_pairs_mean" in greedy else None
        after = pd.to_numeric(greedy.get("masked_pairs_mean"), errors="coerce").mean() if "masked_pairs_mean" in greedy else None
        fallback = pd.to_numeric(greedy.get("mask_fallback_rate"), errors="coerce").mean() if "mask_fallback_rate" in greedy else None
        time_ms = greedy["decision_time_ms"].mean() if "decision_time_ms" in greedy.columns else None
        pruned = (before - after) / before * 100.0 if before and after is not None and before > 0 else None
        flat.append({"threshold": value, "greedy_eta": per_instance.mean(), "sampled_eta": per_instance_sampled.mean(),
                     "gap_pp": gap, "candidate_pairs_mean": before, "masked_pairs_mean": after, "pruned_pct": pruned,
                     "mask_fallback_rate": fallback, "decision_time_ms": time_ms, "n_instances": int(per_instance.size)})
    return pd.DataFrame(flat)


def sensitivity() -> None:
    """Mask-threshold sensitivity table: threshold, greedy eta, sampled eta, pruned share, candidate pairs and
    time per decision (the CSV twin also keeps the gap to the MILP optimum and the fallback rate)."""
    table = sensitivity_frame()
    if table is None:
        return
    header = ["$\\varepsilon_f$", "Greedy $\\eta$", "Sampled $\\eta$ (10 episodes)", "Pruned pairs (\\%)",
              "Candidate pairs before / after", "Time per decision (ms)"]
    rows = []
    for _, r in table.iterrows():
        rows.append([r["threshold"], _fmt(r["greedy_eta"]), _fmt(r["sampled_eta"]), _fmt(r["pruned_pct"], 2),
                     f"{_fmt(r['candidate_pairs_mean'], 2)} / {_fmt(r['masked_pairs_mean'], 2)}",
                     _fmt(r["decision_time_ms"], 2)])
    _write("sensitivity", header, rows, align="lrrrrr")
    _write_csv("sensitivity", table)


def sampled_decoding(tier: str = PAIRED_TIER) -> None:
    """Appendix B: mean fulfilment of the ten sampled episodes of every learned policy per benchmark
    configuration (eval_results.csv rows with decode = sample), with the mean over configurations."""
    path = RESULT / "eval_results.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
    frame = load_eval_results(path)
    if "decode" not in frame.columns:
        print("[SKIP] eval_results.csv has no decode column")
        return
    frame = frame[frame["decode"] == "sample"]
    if "tier" in frame.columns and (frame["tier"] == tier).any():
        frame = frame[frame["tier"] == tier]
    order = ["ours"] + list(ABLATION_METHODS) + list(DRL_METHODS)
    present = [m for m in order if (frame["method"] == m).any()]
    if not present:
        print("[SKIP] no sampled-decoding rows")
        return
    pivot = frame.pivot_table(index=["ddt", "scale"], columns="method", values="fulfillment_rate", aggfunc="mean")
    header = ["Configuration (DDT, S)"] + [METHOD_LABELS[m] for m in present]
    rows = [[f"({int(ddt)}, {int(scale)})"] + [_fmt(row.get(m)) for m in present] for (ddt, scale), row in pivot.iterrows()]
    rows.append(["Mean"] + [_fmt(pivot[m].mean()) for m in present])
    _write("sampled_decoding", header, rows)
    _write_csv("sampled_decoding", pivot.reindex(columns=present).reset_index())


def generator_settings() -> None:
    """Table 5(a): the generator settings of the training pool, read from configs/instance.yaml."""
    cfg = _load_config()
    inst = cfg.get("instance", {})
    gen = inst.get("generator", {}) or {}
    pool = ((inst.get("pools") or {}).get("train") or {})
    p_range = gen.get("process_time_range", [25, 450])
    gap = gen.get("inter_arrival_range", [1, 40])
    ddt = gen.get("ddt_range", [500, 1600])
    slack = gen.get("slack_rel_range", [0.6, 1.55])
    orders = gen.get("n_orders_range", [50, 150])
    n_pool, seed = pool.get("n"), pool.get("seed")
    pool_text = (f"{n_pool} instances (seeds {seed} to {int(seed) + int(n_pool) - 1}), each used once"
                 if n_pool is not None and seed is not None else "--")
    rows = [
        ("Product types / stages / machines per stage",
         f"{gen.get('n_products', 5)} / {gen.get('n_stages', 5)} / {gen.get('machines_per_stage', 5)}"),
        ("Processing time $p_{rjm}$ (s)", f"integer uniform in [{p_range[0]}, {p_range[1]}] per (product type, machine)"),
        ("Inter-arrival time $\\Delta t$ (s)", f"integer uniform in [{gap[0]}, {gap[1]}] (mean {(gap[0] + gap[1]) / 2:g})"),
        ("Orders per instance $S$", f"integer uniform in [{orders[0]}, {orders[1]}]"),
        ("Due-date tightness DDT", f"integer uniform in [{ddt[0]}, {ddt[1]}], one value per instance"),
        ("Slack $d_s - a_s$", f"integer uniform in [{slack[0]} DDT, {slack[1]} DDT]"),
        ("Training pool", pool_text),
    ]
    _write("generator_settings", ["Parameter", "Value"], [[k, v] for k, v in rows], align="ll")
    _write_csv("generator_settings", pd.DataFrame(rows, columns=["parameter", "value"]))


def evaluation_statistics(method: str = "ours", tier: str = PAIRED_TIER) -> None:
    """``evaluation_statistics.tex/.csv``: quantities of the greedy benchmark evaluation of ``method`` quoted in the
    manuscript text (candidate pairs before / after the mask, pruned share, fallback rate, decisions per episode,
    time per decision and the shares of the LP, the policy forward pass and the observation construction)."""
    path = RESULT / "eval_results.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
    frame = load_eval_results(path)
    frame = frame[frame["method"] == method]
    if "tier" in frame.columns and (frame["tier"] == tier).any():
        frame = frame[frame["tier"] == tier]
    if "decode" in frame.columns and (frame["decode"] == "greedy").any():
        frame = frame[frame["decode"] == "greedy"]
    if frame.empty:
        print(f"[SKIP] no greedy {tier} rows of {method}")
        return
    num = {c: pd.to_numeric(frame[c], errors="coerce") for c in frame.columns if c in (
        "candidate_pairs_mean", "masked_pairs_mean", "mask_fallback_rate", "decision_count", "infer_time_ms",
        "fluid_ms", "policy_ms", "obs_ms", "sim_ms")}
    required = ("candidate_pairs_mean", "masked_pairs_mean", "mask_fallback_rate", "decision_count", "infer_time_ms")
    if any(c not in num for c in required):
        print("[SKIP] evaluation statistics need the evaluation columns (candidate pairs, decisions, timing)")
        return
    before, after = num["candidate_pairs_mean"].mean(), num["masked_pairs_mean"].mean()
    per_decision = num["infer_time_ms"] / num["decision_count"].where(num["decision_count"] > 0)
    rows = [("Instances (greedy episodes)", int(len(frame))),
            ("Candidate pairs per decision before the mask", round(float(before), 3)),
            ("Candidate pairs per decision after the mask", round(float(after), 3)),
            ("Pruned pairs (\\%)", round(float((before - after) / before * 100.0), 2) if before else None),
            ("Fallback rate", round(float(num["mask_fallback_rate"].mean()), 4)),
            ("Decisions per episode (min / mean / max)",
             f"{int(num['decision_count'].min())} / {int(round(num['decision_count'].mean()))} / {int(num['decision_count'].max())}"),
            ("Time per decision (ms, mean / min / max)",
             f"{per_decision.mean():.1f} / {per_decision.min():.1f} / {per_decision.max():.1f}")]
    total = num["infer_time_ms"].sum()
    for label, column in (("LP share of the decision time (\\%)", "fluid_ms"), ("Policy share (\\%)", "policy_ms"),
                          ("Observation share (\\%)", "obs_ms")):
        if column in num and total > 0:
            rows.append((label, round(float(num[column].sum() / total * 100.0), 1)))
    _write("evaluation_statistics", ["Quantity", "Value"], [[k, v] for k, v in rows], align="lr")
    _write_csv("evaluation_statistics", pd.DataFrame(rows, columns=["quantity", "value"]))


def training_statistics(methods=("ours", "nfe", "nm", "nf", "ng", "na", "hsddqn", "hgnn_ppo")) -> None:
    """``training_statistics.tex/.csv``: per trained method the wall-clock hours, the transitions per update
    (mean / min / max), the mean and maximum pruned share during training and the selected checkpoint
    (result/<method>/run1/log.csv and train_meta.yaml)."""
    rows = []
    for method in methods:
        run = RESULT / method / "run1"
        log_path = run / "log.csv"
        if not log_path.exists():
            continue
        log = pd.read_csv(log_path)
        steps = pd.to_numeric(log["env_steps"], errors="coerce")
        per_iter = steps.diff().fillna(steps.iloc[0])
        before = pd.to_numeric(log.get("candidate_pairs_mean"), errors="coerce")
        after = pd.to_numeric(log.get("masked_pairs_mean"), errors="coerce")
        pruned = ((before - after) / before.where(before > 0) * 100.0)
        meta = {}
        meta_path = run / "train_meta.yaml"
        if meta_path.exists():
            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        hours = float(pd.to_numeric(log["wall_time_s"], errors="coerce").iloc[-1]) / 3600.0
        rows.append({"method": method, "label": METHOD_LABELS.get(method, method), "iterations": int(len(log)),
                     "wall_clock_h": round(hours, 1), "transitions_mean": int(round(per_iter.mean())),
                     "transitions_min": int(per_iter.min()), "transitions_max": int(per_iter.max()),
                     "pruned_pct_mean": round(float(pruned.mean()), 2) if pruned.notna().any() else None,
                     "pruned_pct_max": round(float(pruned.max()), 2) if pruned.notna().any() else None,
                     "best_val_iteration": meta.get("best_val_iteration")})
    if not rows:
        print("[SKIP] no training logs")
        return
    header = ["Method", "Iterations", "Wall-clock (h)", "Transitions per update (mean / min / max)",
              "Pruned pairs in training (\\%, mean / max)", "Selected iteration"]
    body = [[r["label"], r["iterations"], _fmt(r["wall_clock_h"], 1),
             f"{r['transitions_mean']} / {r['transitions_min']} / {r['transitions_max']}",
             f"{_fmt(r['pruned_pct_mean'], 2)} / {_fmt(r['pruned_pct_max'], 2)}",
             r["best_val_iteration"] if r["best_val_iteration"] is not None else "--"] for r in rows]
    _write("training_statistics", header, body, align="lrrrrr")
    _write_csv("training_statistics", pd.DataFrame(rows))


def main() -> None:
    ablation()
    comparison("comparison_pdr", ["ours"] + list(RULE_METHODS))
    comparison("comparison_drl", ["ours"] + list(DRL_METHODS))
    wilcoxon()
    generalization_tables()
    hyperparameters()
    instance_statistics("benchmark")
    milp_small()
    sensitivity()
    sampled_decoding()
    generator_settings()
    evaluation_statistics()
    training_statistics()


if __name__ == "__main__":
    main()
