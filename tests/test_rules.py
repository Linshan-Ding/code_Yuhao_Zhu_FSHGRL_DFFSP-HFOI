"""Dispatching-rule baselines: scoring, tie-breaking, reproducibility and the reward identity."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.baselines.rules import RRC_POOL, RULES, DETERMINISTIC_RULES, RulePolicy, run_rule_episode, select_candidate  # noqa: E402
from data.instance_io import parse_instance_file  # noqa: E402
from rollout import run_episodes  # noqa: E402

INSTANCE = ROOT / "data" / "instances" / "benchmark" / "DDT500_num50" / "DDT500_num50.txt"
ENV = {"order_top_k": 5, "discard_penalty": 1.0, "progress_reward_coeff": 0.0, "fluid_mode": "off"}


def _candidate(machine, task, rank, arrival, due_remaining, p, rem_ops, rem_work):
    return {"machine": machine, "task": task, "rank": rank, "order": 0, "arrival": arrival,
            "due_remaining": due_remaining, "p": p, "rem_ops": rem_ops, "rem_work": rem_work}


CANDIDATES = [
    _candidate(0, 0, 0, arrival=10.0, due_remaining=300.0, p=50.0, rem_ops=5, rem_work=400.0),
    _candidate(1, 0, 0, arrival=10.0, due_remaining=300.0, p=40.0, rem_ops=5, rem_work=400.0),
    _candidate(2, 3, 0, arrival=5.0, due_remaining=200.0, p=90.0, rem_ops=2, rem_work=150.0),
    _candidate(2, 3, 1, arrival=20.0, due_remaining=100.0, p=90.0, rem_ops=2, rem_work=150.0),
]


def test_rule_scores_follow_the_legacy_directions():
    assert select_candidate(CANDIDATES, "fifo")["arrival"] == 5.0           # earliest arrival
    assert select_candidate(CANDIDATES, "spt")["p"] == 40.0                 # shortest processing time (RRC pool only)
    chosen = select_candidate(CANDIDATES, "mor")                            # most remaining operations ...
    assert chosen["rem_ops"] == 5 and chosen["machine"] == 1                # ... tie -> shorter p
    assert select_candidate(CANDIDATES, "mwkr")["rem_work"] == 400.0


def test_rrc_pool_and_the_comparator_rules():
    """RRC draws from four scores in a fixed order; the shortest-processing-time score is not a comparator."""
    assert RRC_POOL == ("fifo", "spt", "mor", "mwkr")
    assert DETERMINISTIC_RULES == ("fifo", "mor", "mwkr") and "spt" not in RULES
    try:
        RulePolicy("spt")
    except ValueError:
        pass
    else:
        raise AssertionError("spt must not be accepted as a comparator rule")


def test_tie_breaking_is_independent_of_enumeration_order():
    for rule in DETERMINISTIC_RULES:
        forward = select_candidate(CANDIDATES, rule)
        backward = select_candidate(list(reversed(CANDIDATES)), rule)
        assert forward == backward, rule


def _reward_identity(record, env=ENV):
    delta_completed = record["completed"] - record["completed_at_start"]
    delta_discarded = record["discarded"] - record["discarded_at_start"]
    expected = (delta_completed - env["discard_penalty"] * delta_discarded) / record["total_orders"]
    expected += env["progress_reward_coeff"] * (record["fulfillment_rate"] - record["fulfillment_rate_at_start"])
    assert abs(record["total_reward"] - expected) < 1e-6, (record["total_reward"], expected)


def test_rules_complete_an_episode_and_satisfy_the_reward_identity():
    spec = parse_instance_file(INSTANCE)
    for rule in RULES:
        record = run_rule_episode(RulePolicy(rule, seed=3), spec, ENV)
        assert record["completed"] + record["discarded"] == record["total_orders"], rule
        assert 0.0 <= record["fulfillment_rate"] <= 1.0
        assert record["decision_count"] == record["policy_forward_count"] > 0
        _reward_identity(record)


def test_rules_are_reproducible():
    spec = parse_instance_file(INSTANCE)
    for rule in RULES:
        first = run_rule_episode(RulePolicy(rule, seed=11), spec, ENV)
        second = run_rule_episode(RulePolicy(rule, seed=11), spec, ENV)
        assert (first["fulfillment_rate"], first["decision_count"]) == (second["fulfillment_rate"], second["decision_count"]), rule


def test_random_policy_episode_reward_identity():
    """The identity also holds for the learned-policy driver (random weights, greedy decoding)."""
    import torch
    from agent.checkpoint import build_scheduler
    from train import load_config

    config = load_config(None)
    train_paras = dict(config["train_paras"], **ENV)
    torch.manual_seed(0)
    policy = build_scheduler(config, train_paras, "cpu")
    policy.eval()
    spec = parse_instance_file(INSTANCE)
    record = run_episodes(policy, spec, train_paras, episodes=1, sample=False)[0]
    assert record["completed"] + record["discarded"] == record["total_orders"]
    _reward_identity(record)
