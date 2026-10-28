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
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import GeneratorCfg, load_config  # noqa: E402

INSTANCES_DIR = ROOT / "data" / "instances"


def sample_arrivals(rng: np.random.Generator, n_orders: int, cfg: GeneratorCfg) -> np.ndarray:
    """Arrival times (the first order arrives at t = 1); uniform integer gaps in ``inter_arrival_range``."""
    low, high = cfg.inter_arrival_range
    gaps = rng.integers(int(low), int(high) + 1, size=n_orders)
    return np.cumsum(gaps) - gaps[0] + 1


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
                      ddt_range: Sequence[float] | None = None, ddt: int | None = None) -> dict:
    """Instance data as arrays (``process_times`` is ``[products, machines]``)."""
    cfg = cfg or GeneratorCfg()
    rng = np.random.default_rng(int(seed))
    machines = int(cfg.n_stages) * int(cfg.machines_per_stage)
    low, high = cfg.process_time_range
    process_times = rng.integers(int(low), int(high) + 1, size=(int(cfg.n_products), machines))
    arrivals = sample_arrivals(rng, n_orders, cfg)
    tightness, instance_ddt = sample_slack(rng, n_orders, cfg, ddt_range, ddt)
    product_types = rng.integers(0, int(cfg.n_products), size=n_orders)
    low, high = ddt_range or cfg.ddt_range
    return {
        "n_products": int(cfg.n_products), "n_stages": int(cfg.n_stages),
        "machine_division": [int(cfg.machines_per_stage)] * int(cfg.n_stages),
        "process_times": process_times, "order_ids": [str(i + 1) for i in range(n_orders)],
        "product_types": product_types, "arrival_times": arrivals, "due_dates": arrivals + tightness,
        "seed": int(seed), "ddt": instance_ddt, "ddt_range": [int(low), int(high)],
        "slack_rule": str(cfg.slack_rule), "arrival_model": "uniform",
        "inter_arrival_range": [int(cfg.inter_arrival_range[0]), int(cfg.inter_arrival_range[1])],
    }


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


def instance_id_for(prefix: str, seed: int, n_orders: int, ddt) -> str:
    return f"{prefix}{int(seed):04d}_DDT{int(ddt)}_num{int(n_orders)}" if ddt is not None \
        else f"{prefix}{int(seed):04d}_num{int(n_orders)}"


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
                   "arrival_model": data.get("arrival_model", "uniform"),
                   "inter_arrival_range": data.get("inter_arrival_range")})
    write_params_yaml(folder, params)
    return path


# ------------------------------------------------------------------------------------------------- pools

def pool_specs(instance_cfg) -> dict:
    """``{pool_name: spec}`` from ``instance.pools`` (plain dicts; missing section -> empty)."""
    pools = getattr(instance_cfg, "pools", None) or {}
    return {str(name): dict(spec or {}) for name, spec in pools.items()}


def generate_pool(name: str, spec: dict, cfg: GeneratorCfg, out_root: Path = INSTANCES_DIR) -> list:
    """Write the instances of one pool; returns the written paths (deterministic from the seeds).

    ``spec`` keys: ``tier`` (default = name), ``prefix``, ``seed`` (base), and either ``n`` with optional
    ``n_orders_range`` / ``ddt_range`` (S and DDT drawn per instance) or a ``grid`` with ``n_orders`` and
    ``ddt`` lists and ``seeds`` repetitions per cell.
    """
    tier = str(spec.get("tier", name))
    prefix = str(spec.get("prefix", name[:3].upper()))
    base_seed = int(spec.get("seed", 1000))
    out_dir = Path(out_root) / tier
    written = []
    grid = spec.get("grid")
    if grid:
        seeds_per_cell = int(grid.get("seeds", 1))
        k = 0
        for n_orders in grid.get("n_orders", []):
            for ddt in grid.get("ddt", []):
                for _ in range(seeds_per_cell):
                    seed = base_seed + k
                    k += 1
                    data = generate_instance(seed, int(n_orders), cfg, ddt=int(ddt))
                    written.append(write_instance(data, instance_id_for(prefix, seed, int(n_orders), data["ddt"]),
                                                  out_dir, tier))
        return written
    n = int(spec.get("n", 0))
    orders_range = spec.get("n_orders_range") or cfg.n_orders_range
    ddt_range = spec.get("ddt_range") or cfg.ddt_range
    for k in range(n):
        seed = base_seed + k
        rng = np.random.default_rng(seed + 10_000)
        n_orders = int(rng.integers(int(orders_range[0]), int(orders_range[1]) + 1))
        data = generate_instance(seed, n_orders, cfg, ddt_range=ddt_range)
        written.append(write_instance(data, instance_id_for(prefix, seed, n_orders, data["ddt"]), out_dir, tier))
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
    parser.add_argument("--out", default="data/instances/synthetic")
    parser.add_argument("--tier", default="synthetic")
    parser.add_argument("--prefix", default="SYN")
    args = parser.parse_args()
    if args.pool is not None:
        generate_pools(args.pool or ["all"])
        return
    cfg = load_config().instance.generator
    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    for k in range(args.n):
        seed = args.seed + k
        rng = np.random.default_rng(seed + 10_000)
        n_orders = args.n_orders or int(rng.integers(int(cfg.n_orders_range[0]), int(cfg.n_orders_range[1]) + 1))
        data = generate_instance(seed, n_orders, cfg, args.ddt)
        path = write_instance(data, instance_id_for(args.prefix, seed, n_orders, data["ddt"]), out_dir, args.tier)
        print(f"[GEN] {path}")


if __name__ == "__main__":
    main()
