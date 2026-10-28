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
