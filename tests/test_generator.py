"""data/generator.py: reproducible files in the benchmark format that the parser and simulator accept, with the
v2 distributions (benchmark-like arrivals, one DDT per instance, slack in slack_rel_range x DDT) and pools."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.baselines.rules import RulePolicy, run_rule_episode  # noqa: E402
from configs.config import GeneratorCfg  # noqa: E402
from data.dataset import parse_ddt  # noqa: E402
from data.generator import format_instance, generate_instance, generate_pool, instance_id_for, write_instance  # noqa: E402
from data.instance_io import parse_instance_file  # noqa: E402


def test_generated_instance_is_reproducible_and_parsable(tmp_path):
    first = generate_instance(seed=3, n_orders=30)
    second = generate_instance(seed=3, n_orders=30)
    assert format_instance(first) == format_instance(second) and first["ddt"] == second["ddt"]
    assert 500 <= first["ddt"] <= 1600
    instance_id = instance_id_for("SYN", 3, 30, first["ddt"])
    assert parse_ddt(instance_id) == first["ddt"]
    path = write_instance(first, instance_id, tmp_path, tier="synthetic")
    spec = parse_instance_file(path)
    assert spec.total_orders == 30 and spec.product_count == 5 and spec.machine_count == 25
    slack = spec.due_dates - spec.arrival_times
    assert slack.min() >= round(0.6 * first["ddt"]) and slack.max() <= round(1.55 * first["ddt"])
    gaps = np.diff(np.sort(spec.arrival_times))
    assert gaps.min() >= 1 and gaps.max() <= 40 and spec.arrival_times.min() == 1
    params = yaml.safe_load((path.parent / "params.yaml").read_text(encoding="utf-8"))
    assert params["ddt"] == first["ddt"] and params["seed"] == 3 and params["slack_rule"] == "ddt_uniform"
    record = run_rule_episode(RulePolicy("spt"), spec, {"order_top_k": 5, "fluid_mode": "off"})
    assert record["completed"] + record["discarded"] == 30


def test_legacy_per_order_rule_and_fixed_ddt():
    cfg = GeneratorCfg(slack_rule="per_order", ddt_range=[300, 2500])
    data = generate_instance(seed=5, n_orders=20, cfg=cfg)
    slack = data["due_dates"] - data["arrival_times"]
    assert data["ddt"] is None and slack.min() >= 300 and slack.max() <= 2500
    fixed = generate_instance(seed=5, n_orders=20, ddt=900)
    assert fixed["ddt"] == 900 and (fixed["due_dates"] - fixed["arrival_times"]).max() <= round(1.55 * 900)


def test_pools_are_deterministic(tmp_path):
    cfg = GeneratorCfg()
    grid = {"tier": "val", "prefix": "VAL", "seed": 2000,
            "grid": {"n_orders": [50, 100], "ddt": [500, 1600], "seeds": 2}}
    paths = generate_pool("val", grid, cfg, tmp_path)
    assert len(paths) == 8 and all(p.parent.parent.name == "val" for p in paths)
    assert sorted(parse_ddt(p.stem) for p in paths) == [500] * 4 + [1600] * 4
    again = generate_pool("val", grid, cfg, tmp_path)
    assert [p.read_text() for p in paths] == [p.read_text() for p in again]
    pool = generate_pool("train", {"tier": "train", "prefix": "TRN", "seed": 1000, "n": 3}, cfg, tmp_path)
    assert [p.stem[:7] for p in pool] == ["TRN1000", "TRN1001", "TRN1002"]
    for path in pool:
        spec = parse_instance_file(path)
        assert 50 <= spec.total_orders <= 150


def test_arrival_models_and_load_ratio(tmp_path):
    from data.dataset import describe_instance
    from data.generator import mean_gap_for_rho

    cfg = GeneratorCfg()
    uniform = generate_instance(seed=11, n_orders=400, cfg=cfg, ddt=900, arrival={"model": "uniform", "rho": 1.8})
    expo = generate_instance(seed=11, n_orders=400, cfg=cfg, ddt=900, arrival={"model": "exponential", "rho": 1.8})
    bursty = generate_instance(seed=11, n_orders=400, cfg=cfg, ddt=900,
                               arrival={"model": "bursty", "rho": 1.8, "burst_size": 10, "burst_gap": 2})
    target = mean_gap_for_rho(1.8, uniform["process_times"], uniform["machine_division"])
    for data in (uniform, expo, bursty):
        gaps = np.diff(data["arrival_times"])
        assert abs(gaps.mean() - target) / target < 0.15, (data["arrival_model"], gaps.mean(), target)
    assert np.diff(expo["arrival_times"]).std() > np.diff(uniform["arrival_times"]).std() * 1.2
    stats = {}
    for name, data in (("uniform", uniform), ("exponential", expo), ("bursty", bursty)):
        path = write_instance(data, f"ARR_{name}_DDT900_num400", tmp_path, tier="arrival")
        stats[name] = describe_instance(path, tier="arrival")
        assert abs(stats[name]["load_ratio"] - 1.8) < 0.3, (name, stats[name]["load_ratio"])
    assert stats["bursty"]["max_burst"] >= 9 and stats["bursty"]["inter_arrival_cv"] > 1.0
    assert stats["bursty"]["inter_arrival_max"] > 5 * stats["bursty"]["inter_arrival_mean"]
    assert stats["bursty"]["max_burst"] > max(stats["uniform"]["max_burst"], stats["exponential"]["max_burst"])
    assert stats["exponential"]["inter_arrival_cv"] > stats["uniform"]["inter_arrival_cv"]
    assert generate_instance(seed=11, n_orders=5, cfg=cfg, arrival={"model": "uniform", "mean_gap": 30})["inter_arrival_range"] == [1, 59]


def test_structure_overrides_and_case_structure(tmp_path):
    from data.case_structure import load_case_structure

    small = generate_instance(seed=2, n_orders=6, structure={"n_products": 2, "n_stages": 3, "machines_per_stage": 2}, ddt=500)
    assert small["process_times"].shape == (2, 6) and small["machine_division"] == [2, 2, 2]
    path = write_instance(small, "TNY0002_DDT500_num6", tmp_path, tier="tiny")
    spec = parse_instance_file(path)
    assert spec.product_count == 2 and spec.stage_count == 3 and spec.machine_count == 6
    record = run_rule_episode(RulePolicy("fifo"), spec, {"order_top_k": 5, "fluid_mode": "off"})
    assert record["completed"] + record["discarded"] == 6

    structure = load_case_structure()
    assert structure.n_products == 3 and structure.machine_division == [3, 5, 9, 6, 6, 2] and structure.unit == "minute"
    assert structure.process_times[0, structure.machines.index("FDM2")] == 225
    assert structure.process_times[0, structure.machines.index("SLA1")] == 0        # not eligible
    case = generate_instance(seed=9, n_orders=40, ddt=900, arrival={"model": "uniform", "rho": 1.8}, structure=structure)
    assert case["time_unit"] == "minute" and case["structure"] == "case"
    path = write_instance(case, "CEV0009_DDT900_num40", tmp_path, tier="case_eval")
    spec = parse_instance_file(path)
    assert spec.machine_count == 31 and spec.stage_count == 6
    assert not spec.ope_ma_adj[spec.task_to_product_stage.index((0, 2)), structure.machines.index("SLA1")]
    assert spec.ope_ma_adj[spec.task_to_product_stage.index((2, 2)), structure.machines.index("SLS3")]
    record = run_rule_episode(RulePolicy("mor"), spec, {"order_top_k": 5, "fluid_mode": "off"})
    assert record["completed"] + record["discarded"] == 40
    params = yaml.safe_load((path.parent / "params.yaml").read_text(encoding="utf-8"))
    assert params["time_unit"] == "minute" and params["rho_target"] == 1.8 and params["n_machines"] == 31


def test_configured_pools_cover_the_plan(tmp_path):
    from configs.config import load_config
    from data.generator import generate_pool, pool_specs

    cfg = load_config()
    specs = pool_specs(cfg.instance)
    assert {"train", "val", "arrival", "arrival_train", "tiny", "tiny5", "case_train", "case_val", "case_eval"} <= set(specs)
    arrival = generate_pool("arrival", specs["arrival"], cfg.instance.generator, tmp_path)
    assert len(arrival) == 6 and sum("bursty" in p.stem for p in arrival) == 1 and all("DDT900_num100" in p.stem for p in arrival)
    mixed = generate_pool("arrival_train", dict(specs["arrival_train"], n=6), cfg.instance.generator, tmp_path)
    models = {yaml.safe_load((p.parent / "params.yaml").read_text())["arrival_model"] for p in mixed}
    assert len(mixed) == 6 and len(models) >= 2
    tiny = generate_pool("tiny", dict(specs["tiny"], grid={"n_orders": [4], "ddt": [500], "seeds": 1}), cfg.instance.generator, tmp_path)
    assert parse_instance_file(tiny[0]).machine_count == 6
    case = generate_pool("case_eval", specs["case_eval"], cfg.instance.generator, tmp_path)
    assert len(case) == 9 and all(parse_instance_file(p).machine_count == 31 for p in case)
