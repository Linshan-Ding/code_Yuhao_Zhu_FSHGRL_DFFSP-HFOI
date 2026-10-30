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
                          scripts/run_10_sensitivity.py): per threshold the mean greedy / sampled eta on the small
                          tier, the gap to the MILP optimum, candidate pairs before / after the mask, the fallback
                          rate and the time per decision.
Every v2 table is also written as CSV next to the .tex file (same name) for the response letter.
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
CONFIG_FILES = ["configs/instance.yaml", "configs/env.yaml", "configs/algo.yaml", "configs/paper.yaml"]
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
             [f"{METHOD_LABELS[m]} time per decision (s)" for m in present]
    rows = []
    for (ddt, scale), row in pivot_rate.iterrows():
        times = pivot_time.loc[(ddt, scale)] if (ddt, scale) in pivot_time.index else pd.Series(dtype=float)
        rows.append([f"({int(ddt)}, {int(scale)})"] + [_fmt(row.get(m)) for m in present]
                    + [_fmt(times.get(m) / 1000.0 if pd.notna(times.get(m, float("nan"))) else None, 3) for m in present])
    rows.append(["Mean"] + [_fmt(pivot_rate[m].mean()) for m in present]
                + [_fmt(pivot_time[m].mean() / 1000.0 if m in pivot_time.columns else None, 3) for m in present])
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


# -- v2: zero-shot tiers ---------------------------------------------------------------------------------------
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
        header = ["Instance"] + [method_label(m) for m in methods]
        rows = [[_tex(instance)] + [_fmt(row.get(m)) for m in methods] for instance, row in pivot.iterrows()]
        rows.append(["Mean"] + [_fmt(pivot[m].mean()) for m in methods])
        _write(f"generalization_{tier}", header, rows)
        _write_csv(f"generalization_{tier}", pivot.reset_index())


# -- v2: configuration and data tables -----------------------------------------------------------------------------
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


def hyperparameter_rows(config: dict | None = None, runs: dict | None = None, counts: dict | None = None) -> list:
    """``(block, parameter, value)`` triples of the paper's Table 6, read from the configuration."""
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
        ("Environment and action", "Mask threshold $\\varepsilon_f$", env.get("fluid_mask_threshold", 1e-5)),
        ("Environment and action", "LP recomputation", env.get("fluid_recompute", "decision")),
        ("Environment and action", "Discard penalty / progress coefficient",
         f"{env.get('discard_penalty', 1.0)} / {env.get('progress_reward_coeff', 0.0)}"),
        ("Network", "Raw feature width (operation type / machine)",
         f"{model.get('in_size_ope', 12)} / {model.get('in_size_ma', 5)}"),
        ("Network", "Embedding width $d$ (operation type / machine)", f"{d_ope} / {d_ma}"),
        ("Network", "HGNN layers / attention heads per layer", f"{len(heads)} / {heads[0] if heads else 1}"),
        ("Network", "Operation MLP hidden width", model.get("hidden_size_ope", 256)),
        ("Network", "Candidate self-attention: heads / feature width / normalisation",
         f"{attention.get('sparse_attention_heads', 1)} / {attn_dim} / LayerNorm"
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
         f"{train.get('rollout_device', 'cpu')} / {train.get('update_device', 'cuda:0')}"),
        ("Training", "Training instance per iteration",
         "the next pool instance (each used once)" if str(inst.get("instance_selection", "random")) in ("round_robin", "fixed")
         else f"{inst.get('instance_selection', 'random')} draw from the pool"),
        ("Data", "Training pool / validation / test",
         f"{counts.get('train', '--')} generated / {val_count} benchmark ({val_names}) / {counts.get('benchmark', '--')} benchmark"),
        ("Data", "Other unseen tiers (arrival / small)", f"{counts.get('arrival', '--')} / {counts.get('small', '--')} instances"),
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
        rules = pd.read_csv(rules_path)
        rules = rules[rules["method"].isin(DETERMINISTIC_RULE_METHODS)]
        if not rules.empty:
            best_rule = rules.groupby("instance_id")["fulfillment_rate"].max().to_dict()
    policy = _policy_small_eta()
    pivot = milp.pivot_table(index=["tier", "instance", "S", "n_ops", "n_binary"], columns="variant",
                             values=["eta_opt", "status", "solve_s", "replay_ok", "brute_force_ok"], aggfunc="first")
    header = ["Tier", "Instance", "$S$", "$|\\mathcal{K}|$", "Binaries", "$\\eta^*_{A}$", "$\\eta^*_{B}$", "Status (A / B)",
              "Solve time (s, A / B)", "Replay", "Brute force", "Best rule $\\eta$", "FSHGRL $\\eta$", "Gap (pp)"]
    rows, flat = [], []
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
    _write("milp_small", header, rows, align="llrrrrrllrrrrr")
    _write_csv("milp_small", pd.DataFrame(flat))


THRESHOLD_ORDER = ["off", "1e-5", "0.05", "0.1", "0.2"]


def sensitivity() -> None:
    """Mask-threshold sensitivity table from result/sensitivity.csv (scripts/run_10_sensitivity.py) with the MILP
    optimum of result/milp/milp_results.csv (variant B, else A) as reference."""
    path = RESULT / "sensitivity.csv"
    if not path.exists():
        print(f"[SKIP] {path} missing")
        return
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
    header = ["$\\varepsilon_f$", "Greedy $\\eta$", "Sampled $\\eta$", "Gap to $\\eta^*$ (pp)",
              "Candidate pairs before / after", "Fallback rate", "Time per decision (ms)"]
    rows, flat = [], []
    for value in thresholds:
        part = frame[frame["threshold"] == value]
        greedy = part[part["decode"] == "greedy"] if "decode" in part.columns else part
        sampled = part[part["decode"] == "sample"] if "decode" in part.columns else part.iloc[0:0]
        per_instance = greedy.groupby("instance_id")["fulfillment_rate"].mean()
        gaps = [(optimum[i] - eta) * 100.0 for i, eta in per_instance.items() if i in optimum]
        gap = sum(gaps) / len(gaps) if gaps else None
        before = pd.to_numeric(greedy.get("candidate_pairs_mean"), errors="coerce").mean() if "candidate_pairs_mean" in greedy else None
        after = pd.to_numeric(greedy.get("masked_pairs_mean"), errors="coerce").mean() if "masked_pairs_mean" in greedy else None
        fallback = pd.to_numeric(greedy.get("mask_fallback_rate"), errors="coerce").mean() if "mask_fallback_rate" in greedy else None
        time_ms = greedy["decision_time_ms"].mean() if "decision_time_ms" in greedy.columns else None
        label = "off" if value == "off" else value
        rows.append([label, _fmt(per_instance.mean()), _fmt(sampled["fulfillment_rate"].mean()) if not sampled.empty else "--",
                     _fmt(gap, 2), f"{_fmt(before, 2)} / {_fmt(after, 2)}", _fmt(fallback, 3), _fmt(time_ms, 2)])
        flat.append({"threshold": value, "greedy_eta": per_instance.mean(), "sampled_eta": sampled["fulfillment_rate"].mean() if not sampled.empty else None,
                     "gap_pp": gap, "candidate_pairs_mean": before, "masked_pairs_mean": after, "mask_fallback_rate": fallback,
                     "decision_time_ms": time_ms, "n_instances": int(per_instance.size)})
    _write("sensitivity", header, rows, align="lrrrrrr")
    _write_csv("sensitivity", pd.DataFrame(flat))


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


if __name__ == "__main__":
    main()
