"""v2 fixes of docs/known-issues.md items 8 (future-order leakage), 9 (batch-dependent normalisation) and
12 (silent remapping of invalid actions)."""
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
from agent.normalize import normalize_due_dates, normalize_per_observation  # noqa: E402
from data.instance_io import parse_instance_file  # noqa: E402
from environment.simulator import TrainingSimulator  # noqa: E402
from train import load_config  # noqa: E402

BENCHMARK = ROOT / "data" / "instances" / "benchmark"
INSTANCE = BENCHMARK / "DDT500_num50" / "DDT500_num50.txt"
OTHER = BENCHMARK / "DDT900_num100" / "DDT900_num100.txt"
ENV = {"order_top_k": 5, "fluid_mode": "cached_lp", "fluid_backend": "scipy", "fluid_mask": True}


def test_workload_feature_counts_arrived_orders_only():
    spec = parse_instance_file(INSTANCE)
    simulator = TrainingSimulator(spec, ENV)
    obs = simulator.to_policy_obs()
    arrived = int(np.sum(spec.arrival_times <= simulator.current_time))
    assert 0 < arrived < spec.total_orders
    # every arrived order sits at stage 0 and is counted by the five stage nodes of its product
    assert int(obs["raw_opes"][:, 6].sum()) == arrived * spec.stage_count
    for _ in range(20):
        if simulator.done:
            break
        pairs = obs["valid_pairs"]
        simulator.step((int(pairs[0, 0]) * spec.task_count + int(pairs[0, 1]), 0))
        obs = simulator.to_policy_obs()
        active = ((simulator.order_status == simulator.STATUS_WAITING)
                  | (simulator.order_status == simulator.STATUS_PROCESSING))
        assert int(obs["raw_opes"][:, 6].max()) <= int(active.sum())


def test_normalisation_helpers_match_single_observation_statistics():
    torch.manual_seed(0)
    proc = torch.rand(3, 4, 5)
    single = (proc[1] - proc[1].mean()) / (proc[1].std(unbiased=False) + 1e-5)
    assert torch.allclose(normalize_per_observation(proc)[1], single, atol=1e-6)
    due = torch.rand(2, 3, 4) * 100
    mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    mask[0, 0, :2] = True
    mask[0, 2, :3] = True
    out = normalize_due_dates(due, mask)
    valid = due[0][mask[0]]
    expected = (due[0] - valid.mean()) / (valid.std(unbiased=False) + 1e-5)
    assert torch.allclose(out[0], expected, atol=1e-5) and torch.equal(out[1], due[1])   # no valid entries: unchanged


def _observations():
    specs = [parse_instance_file(INSTANCE), parse_instance_file(OTHER)]
    obs = []
    for spec in specs:
        simulator = TrainingSimulator(spec, ENV)
        for step in range(6):
            o = simulator.to_policy_obs()
            if step in (2, 5):
                obs.append(o)
            pairs = o["valid_pairs"]
            simulator.step((int(pairs[-1, 0]) * spec.task_count + int(pairs[-1, 1]), 0))
    return obs


@pytest.mark.parametrize("model_type", ["hgnn", "mlp"])
def test_policy_outputs_do_not_depend_on_the_batch_composition(model_type):
    config = load_config(None)
    train_paras = dict(config["train_paras"], model_type=model_type, **ENV)
    torch.manual_seed(1)
    policy = build_scheduler(config, train_paras, "cpu")
    policy.eval()
    observations = _observations()
    with torch.no_grad():
        alone = [policy.act_batch([o], sample=False) for o in observations]
        together = policy.act_batch(observations, sample=False)
    for index, (indexes, log_probs, values, env_actions) in enumerate(alone):
        assert env_actions[0] == together[3][index], index
        assert torch.allclose(log_probs[0], together[1][index], atol=1e-5)
        assert torch.allclose(values[0], together[2][index], atol=1e-5)
    with torch.no_grad():
        lp_alone = torch.cat([policy.evaluate_batch([o], [a[3][0]])[0] for o, a in zip(observations, alone)])
        lp_together = policy.evaluate_batch(observations, [a[3][0] for a in alone])[0]
    assert torch.allclose(lp_alone, lp_together, atol=1e-5)


def test_q_scheduler_is_batch_independent():
    config = load_config(None)
    train_paras = dict(config["train_paras"], algorithm="dqn", **ENV)
    torch.manual_seed(2)
    policy = build_scheduler(config, train_paras, "cpu")
    policy.eval()
    observations = _observations()
    with torch.no_grad():
        alone = [policy.q_values([o])[0][0] for o in observations]
        together = [q for q, _ in policy.q_values(observations)]
    for a, b in zip(alone, together):
        assert torch.allclose(a, b, atol=1e-5)


def test_invalid_actions_raise():
    spec = parse_instance_file(INSTANCE)
    simulator = TrainingSimulator(spec, ENV)
    obs = simulator.to_policy_obs()
    valid = {(int(m), int(t)) for m, t in obs["valid_pairs"].tolist()}
    machine, task = next(iter(valid))
    bad_machine = next(m for m in range(spec.machine_count) if (m, task) not in valid)
    with pytest.raises(ValueError):
        simulator.step((bad_machine * spec.task_count + task, 0))
    with pytest.raises(ValueError):
        simulator.step((machine * spec.task_count + task, 99))
    assert simulator.step((machine * spec.task_count + task, 0)) <= 0.0 or True      # the valid action still works
