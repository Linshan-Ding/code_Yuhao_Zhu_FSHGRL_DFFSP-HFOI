"""Synthetic instance generator in the benchmark file format (new code; the original repository had none).

Distributions (``configs/instance.yaml`` ``instance.generator``, v2 defaults matched to the 15 benchmark files,
see docs/experiment-spec.md): 5 product types, 5 stages, 5 machines per stage, processing times uniform in
[25, 450] s per (product, machine), order inter-arrival gaps uniform in [1, 40] s (mean 20.5 s; the paper's
Table 5 states [1, 200], which the benchmark files do not follow), S orders. Due dates: with
``slack_rule: ddt_uniform`` one due-date tightness DDT is drawn per instance from ``ddt_range`` and every
order's slack is uniform in ``[slack_rel_range[0] * DDT, slack_rel_range[1] * DDT]`` (the benchmark files
have slack in about [0.6, 1.55] x DDT); ``slack_rule: per_order`` is the former behaviour (each order's slack
uniform in ``ddt_range``). Every value is drawn from a private ``numpy`` generator seeded per instance, so an
instance id reproduces the same file.

Arrival models (``arrival`` of a pool spec or ``--arrival``): ``uniform`` integer gaps, ``exponential``
gaps, or ``bursty`` (``burst_size`` orders ``burst_gap`` apart, bursts spaced so that the mean gap is kept);
the mean gap follows from ``mean_gap`` or from a target load ratio ``rho`` = (mean operation time x stages) /
(mean gap x machines). ``structure`` overrides the shop (products / stages / machines per stage) or takes the
industrial case's fixed process table (data/case/process.yaml, pool kind ``case``).

Pools (``instance.pools`` of configs/instance.yaml; written by scripts/run_01_prepare_data.py):

    python -m data.generator --pool train      # 200 instances, seeds 1000-1199, S and DDT drawn
    python -m data.generator --pool val        # S x DDT grid x seeds (30 instances)
    python -m data.generator --pool all

Ad-hoc files:

    python -m data.generator --n 5 --out data/instances/synthetic --tier synthetic
    python -m data.generator --n-orders 80 --ddt 900 1200 --seed 7 --out data/instances/synthetic

File format (what data/instance_io.py parses): line 1 ``R J M``; line 2 the machine count per stage; R lines
with M processing times; then the order ids, product types, arrival times and due dates (one line each).
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import GeneratorCfg, load_config  # noqa: E402

INSTANCES_DIR = ROOT / "data" / "instances"


def mean_gap_for_rho(rho: float, process_times: np.ndarray, machine_division) -> float:
    """Mean inter-arrival time that gives the load ratio ``rho`` = (mean operation time x stages) /
    (mean gap x machines): every order visits every stage once, so the arriving work per time unit is
    stages x p_mean / gap and the capacity is the number of machines."""
    positive = process_times[process_times > 0]
    p_mean = float(positive.mean()) if positive.size else 1.0
    stages = len(machine_division)
    machines = float(sum(machine_division))
    return p_mean * stages / (machines * float(rho))


def sample_arrivals(rng: np.random.Generator, n_orders: int, cfg: GeneratorCfg, arrival: dict | None = None,
                    process_times: np.ndarray | None = None, machine_division=None) -> tuple:
    """Arrival times (the first order arrives at t = 1) and the resolved arrival record.

    ``arrival`` keys: ``model`` (uniform | exponential | bursty; default ``cfg.arrival_model``), ``mean_gap``
    or ``rho`` (target load ratio, converted with ``mean_gap_for_rho``), ``burst_size``, ``burst_gap``,
    ``burst_jitter``. Without ``mean_gap`` / ``rho`` the uniform model draws integer gaps in
    ``cfg.inter_arrival_range``; with one, uniform gaps are drawn in [1, 2 x mean_gap - 1].
    """
    arrival = dict(arrival or {})
    model = str(arrival.get("model", cfg.arrival_model)).lower()
    rho = arrival.get("rho")
    mean_gap = arrival.get("mean_gap")
    if rho is not None:
        if process_times is None or machine_division is None:
            raise ValueError("rho needs the processing times and the machine division")
        mean_gap = mean_gap_for_rho(float(rho), np.asarray(process_times), machine_division)
    record = {"arrival_model": model, "rho_target": float(rho) if rho is not None else None,
              "mean_gap_target": float(mean_gap) if mean_gap is not None else None}
    if model == "uniform":
        if mean_gap is None:
            low, high = int(cfg.inter_arrival_range[0]), int(cfg.inter_arrival_range[1])
        else:
            low, high = 1, max(1, int(round(2.0 * float(mean_gap) - 1.0)))
        gaps = rng.integers(low, high + 1, size=n_orders).astype(np.float64)
        record["inter_arrival_range"] = [low, high]
    elif model == "exponential":
        if mean_gap is None:
            mean_gap = float(cfg.inter_arrival_range[0] + cfg.inter_arrival_range[1]) / 2.0
        gaps = np.maximum(1.0, np.round(rng.exponential(float(mean_gap), size=n_orders)))
        record["mean_gap_target"] = float(mean_gap)
    elif model == "bursty":
        if mean_gap is None:
            mean_gap = float(cfg.inter_arrival_range[0] + cfg.inter_arrival_range[1]) / 2.0
        size = max(int(arrival.get("burst_size", cfg.burst_size)), 1)
        inner = float(arrival.get("burst_gap", cfg.burst_gap))
        jitter = float(arrival.get("burst_jitter", cfg.burst_jitter))
        between = max(1.0, size * float(mean_gap) - (size - 1) * inner)   # keeps the overall mean gap
        gaps = np.empty(n_orders, dtype=np.float64)
        for i in range(n_orders):
            if i % size == 0:
                gaps[i] = max(1.0, np.round(between * (1.0 + jitter * float(rng.uniform(-1.0, 1.0)))))
            else:
                gaps[i] = max(1.0, np.round(inner))
        record.update({"burst_size": size, "burst_gap": inner, "burst_jitter": jitter,
                       "mean_gap_target": float(mean_gap)})
    else:
        raise ValueError(f"arrival model must be uniform, exponential or bursty, got {model!r}")
    arrivals = np.cumsum(gaps) - gaps[0] + 1
    return arrivals.astype(np.int64), record


def sample_slack(rng: np.random.Generator, n_orders: int, cfg: GeneratorCfg, ddt_range, ddt) -> tuple:
    """``(slack per order, instance DDT or None)`` under the configured slack rule."""
    low, high = ddt_range or cfg.ddt_range
    rule = str(cfg.slack_rule).lower()
    if rule == "per_order":
        return rng.integers(int(low), int(high) + 1, size=n_orders), None
    if rule != "ddt_uniform":
        raise ValueError(f"slack_rule must be 'ddt_uniform' or 'per_order', got {cfg.slack_rule!r}")
    instance_ddt = int(ddt) if ddt is not None else int(rng.integers(int(low), int(high) + 1))
    rel_low, rel_high = cfg.slack_rel_range
    slack_low = max(1, int(round(float(rel_low) * instance_ddt)))
    slack_high = max(slack_low, int(round(float(rel_high) * instance_ddt)))
    return rng.integers(slack_low, slack_high + 1, size=n_orders), instance_ddt


def generate_instance(seed: int, n_orders: int, cfg: GeneratorCfg | None = None,
                      ddt_range: Sequence[float] | None = None, ddt: int | None = None,
                      arrival: dict | None = None, structure=None) -> dict:
    """Instance data as arrays (``process_times`` is ``[products, machines]``).

    ``structure``: a ``data.case_structure.ShopStructure`` (fixed machine division and processing times, e.g.
    the industrial case) or a mapping with ``n_products`` / ``n_stages`` / ``machines_per_stage`` overriding
    the generator configuration (times are then sampled as usual).
    """
    cfg = cfg or GeneratorCfg()
    rng = np.random.default_rng(int(seed))
    fixed_times = getattr(structure, "process_times", None)
    if fixed_times is not None:
        n_products = int(structure.n_products)
        machine_division = [int(v) for v in structure.machine_division]
        process_times = np.asarray(fixed_times, dtype=np.int64)
    else:
        overrides = dict(structure or {})
        n_products = int(overrides.get("n_products", cfg.n_products))
        n_stages = int(overrides.get("n_stages", cfg.n_stages))
        per_stage = int(overrides.get("machines_per_stage", cfg.machines_per_stage))
        machine_division = [per_stage] * n_stages
        low, high = cfg.process_time_range
        process_times = rng.integers(int(low), int(high) + 1, size=(n_products, n_stages * per_stage))
    arrivals, arrival_record = sample_arrivals(rng, n_orders, cfg, arrival, process_times, machine_division)
    tightness, instance_ddt = sample_slack(rng, n_orders, cfg, ddt_range, ddt)
    product_types = rng.integers(0, n_products, size=n_orders)
    low, high = ddt_range or cfg.ddt_range
    data = {
        "n_products": n_products, "n_stages": len(machine_division),
        "machine_division": machine_division,
        "process_times": process_times, "order_ids": [str(i + 1) for i in range(n_orders)],
        "product_types": product_types, "arrival_times": arrivals, "due_dates": arrivals + tightness,
        "seed": int(seed), "ddt": instance_ddt, "ddt_range": [int(low), int(high)],
        "slack_rule": str(cfg.slack_rule), "slack_rel_range": [float(cfg.slack_rel_range[0]), float(cfg.slack_rel_range[1])],
        "time_unit": str(getattr(structure, "unit", "second")),
    }
    data.update(arrival_record)
    if fixed_times is not None:
        data["structure"] = "case"
    return data


def format_instance(data: dict) -> str:
    lines = [
        f"{data['n_products']} {data['n_stages']} {sum(data['machine_division'])}",
        " ".join(str(v) for v in data["machine_division"]),
    ]
    lines += [" ".join(str(int(v)) for v in row) for row in data["process_times"]]
    lines.append(" ".join(data["order_ids"]))
    lines.append(" ".join(str(int(v)) for v in data["product_types"]))
    lines.append(" ".join(str(int(v)) for v in data["arrival_times"]))
    lines.append(" ".join(str(int(v)) for v in data["due_dates"]))
    return "\n".join(lines) + "\n"


def instance_id_for(prefix: str, seed: int, n_orders: int, ddt, tag: str = "") -> str:
    middle = f"_{tag}" if tag else ""
    return f"{prefix}{int(seed):04d}{middle}_DDT{int(ddt)}_num{int(n_orders)}" if ddt is not None \
        else f"{prefix}{int(seed):04d}{middle}_num{int(n_orders)}"


def arrival_tag(arrival: dict | None) -> str:
    """Short id component describing a non-default arrival model, e.g. ``uniform_rho0.8``."""
    if not arrival:
        return ""
    parts = [str(arrival.get("model", "uniform"))]
    if arrival.get("rho") is not None:
        parts.append(f"rho{float(arrival['rho']):g}")
    elif arrival.get("mean_gap") is not None:
        parts.append(f"gap{float(arrival['mean_gap']):g}")
    return "_".join(parts)


def write_instance(data: dict, instance_id: str, out_dir: Path, tier: str = "synthetic") -> Path:
    """``<out_dir>/<instance_id>/<instance_id>.txt`` + params.yaml; identical content on re-run."""
    from data.dataset import describe_instance, write_params_yaml

    folder = Path(out_dir) / instance_id
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{instance_id}.txt"
    content = format_instance(data)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    params = describe_instance(path, tier=tier, source="generated")
    params.update({"seed": data["seed"], "ddt": data["ddt"] if data["ddt"] is not None else params.get("ddt"),
                   "ddt_range": data["ddt_range"], "slack_rule": data["slack_rule"],
                   "slack_rel_range": data.get("slack_rel_range"), "time_unit": data.get("time_unit", "second")})
    for key in ("arrival_model", "rho_target", "mean_gap_target", "inter_arrival_range", "burst_size", "burst_gap",
                "burst_jitter", "structure"):
        if key in data:
            params[key] = data[key]
    write_params_yaml(folder, params)
    return path


# ------------------------------------------------------------------------------------------------- pools

def pool_specs(instance_cfg) -> dict:
    """``{pool_name: spec}`` from ``instance.pools`` (plain dicts; missing section -> empty)."""
    pools = getattr(instance_cfg, "pools", None) or {}
    return {str(name): dict(spec or {}) for name, spec in pools.items()}


def resolve_structure(spec: dict):
    """``structure`` of a pool spec: a case process file (``{"file": ...}`` or ``{"case": true}``) or
    generator overrides (``n_products`` / ``n_stages`` / ``machines_per_stage``); None = configuration."""
    structure = spec.get("structure")
    if not structure:
        return None
    if isinstance(structure, dict) and (structure.get("file") or structure.get("case")):
        from data.case_structure import DEFAULT_PROCESS_FILE, load_case_structure

        file = structure.get("file")
        path = ROOT / file if file and not Path(file).is_absolute() else (Path(file) if file else DEFAULT_PROCESS_FILE)
        return load_case_structure(path)
    return dict(structure)


def generate_pool(name: str, spec: dict, cfg: GeneratorCfg, out_root: Path = INSTANCES_DIR) -> list:
    """Write the instances of one pool; returns the written paths (deterministic from the seeds).

    ``spec`` keys: ``tier`` (default = name), ``prefix``, ``seed`` (base), ``structure`` (see
    ``resolve_structure``), ``arrival`` (default arrival model of the pool), and one of
    - ``n`` with optional ``n_orders_range`` / ``ddt_range`` (S and DDT drawn per instance),
    - ``grid`` with ``n_orders`` and ``ddt`` lists and ``seeds`` repetitions per cell,
    - ``cases``: explicit list of ``{n_orders, ddt, arrival, seed?}`` records (one instance each).
    """
    tier = str(spec.get("tier", name))
    prefix = str(spec.get("prefix", name[:3].upper()))
    base_seed = int(spec.get("seed", 1000))
    out_dir = Path(out_root) / tier
    structure = resolve_structure(spec)
    pool_arrival = spec.get("arrival")
    written = []
    cases = spec.get("cases")
    if cases:
        for k, case in enumerate(cases):
            seed = int(case.get("seed", base_seed + k))
            arrival = case.get("arrival", pool_arrival)
            n_orders = int(case["n_orders"])
            data = generate_instance(seed, n_orders, cfg, ddt=int(case["ddt"]), arrival=arrival, structure=structure)
            written.append(write_instance(data, instance_id_for(prefix, seed, n_orders, data["ddt"], arrival_tag(arrival)),
                                          out_dir, tier))
        return written
    grid = spec.get("grid")
    if grid:
        seeds_per_cell = int(grid.get("seeds", 1))
        k = 0
        for n_orders in grid.get("n_orders", []):
            for ddt in grid.get("ddt", []):
                for _ in range(seeds_per_cell):
                    seed = base_seed + k
                    k += 1
                    data = generate_instance(seed, int(n_orders), cfg, ddt=int(ddt), arrival=pool_arrival, structure=structure)
                    written.append(write_instance(data, instance_id_for(prefix, seed, int(n_orders), data["ddt"]),
                                                  out_dir, tier))
        return written
    n = int(spec.get("n", 0))
    orders_range = spec.get("n_orders_range") or cfg.n_orders_range
    ddt_range = spec.get("ddt_range") or cfg.ddt_range
    mix = bool(spec.get("arrival_mix", False))
    rho_range = spec.get("rho_range") or [0.8, 2.6]
    for k in range(n):
        seed = base_seed + k
        rng = np.random.default_rng(seed + 10_000)
        n_orders = int(rng.integers(int(orders_range[0]), int(orders_range[1]) + 1))
        arrival = pool_arrival
        if mix:                                     # arrival-diverse pool: model and load drawn per instance
            mix_rng = np.random.default_rng(seed + 20_000)
            model = ("uniform", "exponential", "bursty")[int(mix_rng.integers(0, 3))]
            arrival = {"model": model, "rho": round(float(mix_rng.uniform(float(rho_range[0]), float(rho_range[1]))), 2)}
            if model == "bursty":
                arrival.update({"burst_size": int(cfg.burst_size), "burst_gap": float(cfg.burst_gap)})
        data = generate_instance(seed, n_orders, cfg, ddt_range=ddt_range, arrival=arrival, structure=structure)
        written.append(write_instance(data, instance_id_for(prefix, seed, n_orders, data["ddt"], arrival_tag(arrival) if mix else ""),
                                      out_dir, tier))
    return written


def generate_pools(names: Sequence[str] | None = None, config=None, out_root: Path = INSTANCES_DIR) -> dict:
    """Generate the configured pools (all, or the named ones); ``{name: [paths]}``."""
    config = config or load_config()
    specs = pool_specs(config.instance)
    selected = list(specs) if not names or list(names) == ["all"] else list(names)
    unknown = [n for n in selected if n not in specs]
    if unknown:
        raise KeyError(f"unknown pool(s) {unknown}; configured: {sorted(specs)}")
    out = {}
    for name in selected:
        out[name] = generate_pool(name, specs[name], config.instance.generator, out_root)
        print(f"[GEN] pool {name}: {len(out[name])} instances -> {Path(out_root) / specs[name].get('tier', name)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", nargs="*", default=None, help="configured pool name(s) or 'all' (configs/instance.yaml)")
    parser.add_argument("--n", type=int, default=1, help="number of ad-hoc instances")
    parser.add_argument("--n-orders", type=int, default=None, help="orders per instance (default: random in the configured range)")
    parser.add_argument("--ddt", type=float, nargs=2, default=None, help="due-date tightness range (default: configured)")
    parser.add_argument("--seed", type=int, default=0, help="base seed; instance k uses seed + k")
    parser.add_argument("--ddt-fixed", type=int, default=None, help="one DDT for every instance (overrides --ddt)")
    parser.add_argument("--slack-rel", type=float, nargs=2, default=None, help="slack range relative to DDT")
    parser.add_argument("--arrival", choices=["uniform", "exponential", "bursty"], default=None)
    parser.add_argument("--rho", type=float, default=None, help="target load ratio (sets the mean inter-arrival gap)")
    parser.add_argument("--mean-gap", type=float, default=None, help="mean inter-arrival gap (time units)")
    parser.add_argument("--burst-size", type=int, default=None)
    parser.add_argument("--burst-gap", type=float, default=None)
    parser.add_argument("--n-products", type=int, default=None)
    parser.add_argument("--n-stages", type=int, default=None)
    parser.add_argument("--machines-per-stage", type=int, default=None)
    parser.add_argument("--case", action="store_true", help="industrial case structure (data/case/process.yaml)")
    parser.add_argument("--out", default="data/instances/synthetic")
    parser.add_argument("--tier", default="synthetic")
    parser.add_argument("--prefix", default="SYN")
    args = parser.parse_args()
    if args.pool is not None:
        generate_pools(args.pool or ["all"])
        return
    cfg = load_config().instance.generator
    if args.slack_rel:
        cfg = replace(cfg, slack_rel_range=[float(args.slack_rel[0]), float(args.slack_rel[1])])
    arrival = {}
    if args.arrival:
        arrival["model"] = args.arrival
    if args.rho is not None:
        arrival["rho"] = args.rho
    if args.mean_gap is not None:
        arrival["mean_gap"] = args.mean_gap
    if args.burst_size is not None:
        arrival["burst_size"] = args.burst_size
    if args.burst_gap is not None:
        arrival["burst_gap"] = args.burst_gap
    structure = None
    if args.case:
        from data.case_structure import load_case_structure

        structure = load_case_structure()
    elif any(v is not None for v in (args.n_products, args.n_stages, args.machines_per_stage)):
        structure = {k: v for k, v in (("n_products", args.n_products), ("n_stages", args.n_stages),
                                       ("machines_per_stage", args.machines_per_stage)) if v is not None}
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    for k in range(args.n):
        seed = args.seed + k
        rng = np.random.default_rng(seed + 10_000)
        n_orders = args.n_orders or int(rng.integers(int(cfg.n_orders_range[0]), int(cfg.n_orders_range[1]) + 1))
        data = generate_instance(seed, n_orders, cfg, args.ddt, ddt=args.ddt_fixed, arrival=arrival or None,
                                 structure=structure)
        path = write_instance(data, instance_id_for(args.prefix, seed, n_orders, data["ddt"], arrival_tag(arrival)),
                              out_dir, args.tier)
        print(f"[GEN] {path}")


if __name__ == "__main__":
    main()
