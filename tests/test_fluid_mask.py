"""v2: the fluid capacity-allocation LP backends and the fluid-guided candidate mask (environment/simulator.py)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.checkpoint import build_scheduler  # noqa: E402
from data.instance_io import parse_instance_file  # noqa: E402
from environment.profiled import ProfiledSimulator  # noqa: E402
from environment.simulator import TrainingSimulator  # noqa: E402
from rollout import run_episodes  # noqa: E402
from train import load_config  # noqa: E402

INSTANCE = ROOT / "data" / "instances" / "benchmark" / "DDT500_num50" / "DDT500_num50.txt"
BASE = {"order_top_k": 5, "discard_penalty": 1.0, "progress_reward_coeff": 0.0, "fluid_time_limit": 2}


def _paras(**overrides):
    paras = dict(BASE, fluid_mode="cached_lp", fluid_backend="scipy", fluid_features="on", fluid_mask=True,
                 fluid_mask_threshold=1e-5)
    paras.update(overrides)
    return paras


def _drive(simulator, steps=40, pick=0):
    """Step a simulator with a fixed candidate choice; returns the observations seen."""
    seen = []
    for _ in range(steps):
        if simulator.done:
            break
        obs = simulator.to_policy_obs()
        seen.append(obs)
        pairs = obs["valid_pairs"]
        machine, task = int(pairs[min(pick, pairs.size(0) - 1), 0]), int(pairs[min(pick, pairs.size(0) - 1), 1])
        simulator.step((machine * int(obs["num_tasks"]) + task, 0))
    return seen


def test_scipy_lp_matches_a_hand_solved_optimum():
    """Unique optimum: task 0 (1 order) on machine 0 only, task 1 (1 order) on machines 0 and 1, task 2 (1 order)
    on machine 1 only, all rates 1. The max-min normalised rate is 2/3 with
    u = {(0,0): 2/3, (0,1): 1/3, (1,1): 1/3, (1,2): 2/3} (pairs are (machine, task))."""
    spec = parse_instance_file(INSTANCE)
    simulator = TrainingSimulator(spec, _paras())
    for task_index, machine in ((0, 0), (1, 0), (1, 1), (2, 1)):
        simulator.spec.proc_rates[task_index, machine] = 1.0
    grouped = {0: [0], 1: [1], 2: [2]}
    pairs = [(0, 0), (0, 1), (1, 1), (1, 2)]
    expected = {(0, 0): 2 / 3, (0, 1): 1 / 3, (1, 1): 1 / 3, (1, 2): 2 / 3}
    solution = simulator._solve_lp_scipy(simulator._load_scipy_linprog(), pairs, grouped)
    for pair, value in expected.items():
        assert abs(solution[pair] - value) < 1e-6, (pair, solution)
    allocations, rates, u_star = simulator._solve_cached_lp(grouped, pairs)
    assert u_star == solution and abs(float(allocations[1]) - 2 / 3) < 1e-6 and abs(float(rates[0]) - 2 / 3) < 1e-6
    gurobi = TrainingSimulator._load_gurobi()
    if gurobi is not None:                                   # same optimum with the original Gurobi model
        reference = simulator._solve_lp_gurobi(gurobi, pairs, grouped)
        for pair, value in expected.items():
            assert abs(reference[pair] - value) < 1e-5, pair


def test_backend_resolution_and_counters():
    spec = parse_instance_file(INSTANCE)
    auto = TrainingSimulator(spec, _paras(fluid_backend="auto"))
    expected = "gurobi" if TrainingSimulator._load_gurobi() is not None else "heuristic"
    assert auto.resolve_fluid_backend() == expected
    scipy_sim = TrainingSimulator(spec, _paras())
    assert scipy_sim.resolve_fluid_backend() == "scipy" and scipy_sim.fluid_backend_used == "scipy"
    _drive(scipy_sim, steps=10)
    assert scipy_sim.fluid_solve_count > 0 and scipy_sim.fluid_fallback_count == 0
    assert scipy_sim.fluid_mask_unavailable_count == 0
    heuristic = TrainingSimulator(spec, _paras(fluid_backend="heuristic"))
    assert heuristic.fluid_backend_used == "heuristic_fallback"
    _drive(heuristic, steps=10)
    assert heuristic.fluid_solve_count == 0 and heuristic.fluid_fallback_count > 0
    assert heuristic.fluid_mask_unavailable_count == heuristic.mask_decision_count > 0   # mask needs an LP solution
    assert heuristic.candidate_pairs_after == heuristic.candidate_pairs_before
    with pytest.raises(ValueError):
        TrainingSimulator(spec, _paras(fluid_backend="cplex"))


def test_mask_keeps_only_positively_allocated_pairs_and_falls_back():
    spec = parse_instance_file(INSTANCE)
    masked = TrainingSimulator(spec, _paras())
    unmasked = TrainingSimulator(spec, _paras(fluid_mask=False))
    assert masked.fluid_mask_on and not unmasked.fluid_mask_on
    narrowed = 0
    for _ in range(30):
        if masked.done:
            break
        obs = masked.to_policy_obs()
        grouped = masked._waiting_orders_by_task()
        eligible = masked._eligible_pairs(grouped)
        _, _, u_star = masked._last_fluid_features
        candidates = {(int(m), int(t)) for m, t in obs["valid_pairs"].tolist()}
        assert candidates <= set(eligible)
        if u_star is not None and candidates != set(eligible):
            narrowed += 1
            assert all(u_star.get(pair, 0.0) > 1e-5 for pair in candidates)
            assert all(u_star.get(pair, 0.0) <= 1e-5 for pair in set(eligible) - candidates)
        # node features describe the unmasked eligibility (the paper's |M_rj^idle|)
        for task_index in range(spec.task_count):
            assert int(obs["raw_opes"][task_index, 1]) == sum(1 for _, t in eligible if t == task_index)
        pairs = obs["valid_pairs"]
        masked.step((int(pairs[0, 0]) * spec.task_count + int(pairs[0, 1]), 0))
    assert narrowed > 0 and masked.candidate_pairs_after < masked.candidate_pairs_before
    assert masked.fluid_mask_fallback_count == 0           # a fresh LP solution serves every waiting task
    assert masked.fluid_solve_count + masked.fluid_cache_hit_count == masked.mask_decision_count
    event_mode = TrainingSimulator(spec, _paras(fluid_recompute="event"))
    _drive(event_mode, steps=30)
    assert 0 < event_mode.fluid_solve_count < masked.fluid_solve_count      # legacy: solved after events only
    with pytest.raises(ValueError):
        TrainingSimulator(spec, _paras(fluid_recompute="sometimes"))
    # a threshold above every allocation retains nothing: fallback to all eligible pairs on every decision
    fallback = TrainingSimulator(spec, _paras(fluid_mask_threshold=1.5))
    _drive(fallback, steps=10)
    assert fallback.fluid_mask_fallback_count == fallback.mask_decision_count > 0
    assert fallback.candidate_pairs_after == fallback.candidate_pairs_before
    _drive(unmasked, steps=10)
    assert unmasked.candidate_pairs_after == unmasked.candidate_pairs_before and unmasked.fluid_mask_fallback_count == 0


def test_feature_switch_and_legacy_off_mode():
    spec = parse_instance_file(INSTANCE)
    features_off = TrainingSimulator(spec, _paras(fluid_features="off", fluid_mask=True))
    obs = _drive(features_off, steps=5)
    assert features_off.fluid_solve_count > 0                        # the LP is still solved for the mask
    assert all(torch.all(o["raw_opes"][:, 8:10] == 0) for o in obs)
    both_off = TrainingSimulator(spec, _paras(fluid_features="off", fluid_mask=False))
    obs = _drive(both_off, steps=5)
    assert both_off.fluid_solve_count == 0 and both_off.fluid_backend_used == "off"
    assert all(torch.all(o["raw_opes"][:, 8:10] == 0) for o in obs)
    legacy_off = TrainingSimulator(spec, _paras(fluid_mode="off", fluid_mask=True))
    assert not legacy_off.fluid_mask_on and not legacy_off.fluid_features_on and legacy_off.fluid_backend_used == "off"
    with_features = TrainingSimulator(spec, _paras(fluid_mask=False))
    obs = _drive(with_features, steps=5)
    assert any(torch.any(o["raw_opes"][:, 8:10] != 0) for o in obs)


def test_masked_episode_runs_end_to_end_with_profiling_and_policy():
    config = load_config(None)
    train_paras = dict(config["train_paras"], **_paras())
    torch.manual_seed(0)
    policy = build_scheduler(config, train_paras, "cpu")
    policy.eval()
    spec = parse_instance_file(INSTANCE)
    record = run_episodes(policy, spec, train_paras, episodes=1, sample=False, simulator_factory=ProfiledSimulator)[0]
    assert record["completed"] + record["discarded"] == record["total_orders"]
    assert record["fluid_backend"] == "scipy" and record["fluid_solve_count"] > 0
    assert 0.0 <= record["mask_fallback_rate"] <= 1.0 and record["masked_pairs_mean"] <= record["candidate_pairs_mean"]
    expected = (record["completed"] - record["completed_at_start"]
                - (record["discarded"] - record["discarded_at_start"])) / record["total_orders"]
    assert abs(record["total_reward"] - expected) < 1e-6
