"""environment/collector.py: the multi-process collector yields a RolloutBuffer bit-identical to the
in-process path, propagates worker errors, and shuts down cleanly."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.checkpoint import build_scheduler  # noqa: E402
from data.instance_io import parse_instance_file  # noqa: E402
from environment.collector import CollectorError, CollectorPool, decode_obs, encode_obs  # noqa: E402
from environment.simulator import TrainingSimulator  # noqa: E402
from rollout import collect_vectorized_rollouts, collect_vectorized_rollouts_parallel  # noqa: E402
from train import load_config  # noqa: E402

BENCHMARK = ROOT / "data" / "instances" / "benchmark"
INSTANCE = BENCHMARK / "DDT500_num50" / "DDT500_num50.txt"
OTHER = BENCHMARK / "DDT600_num50" / "DDT600_num50.txt"
BUFFER_LISTS = ("action_indexes", "env_actions", "log_probs", "values", "rewards", "dones", "trajectory_ids")
BUFFER_STATS = ("fulfillment_rates", "episode_transition_counts", "discard_count", "completion_count",
                "fluid_solve_count", "fluid_cache_hit_count", "fluid_fallback_count", "policy_forward_count",
                "avg_action_tokens", "max_action_tokens", "padding_ratio", "actor_batch_size_mean",
                "fluid_mask_fallback_count", "fluid_mask_unavailable_count", "candidate_pairs_before",
                "candidate_pairs_after", "mask_decision_count")


def _paras(model_type="hgnn", **overrides):
    config = load_config(None)
    paras = dict(config["train_paras"], fluid_mode="off", num_envs=4, num_workers=4, minibatch_size=64, viz=False,
                 model_type=model_type)
    paras.update(overrides)
    return config, paras


def _policy(config, paras):
    torch.manual_seed(0)
    policy = build_scheduler(config, paras, "cpu")
    policy.eval()
    return policy


def assert_buffers_identical(reference, candidate):
    assert len(reference) == len(candidate)
    for obs_a, obs_b in zip(reference.obs, candidate.obs):
        assert list(obs_a.keys()) == list(obs_b.keys())
        for key in obs_a:
            value_a, value_b = obs_a[key], obs_b[key]
            if torch.is_tensor(value_a):
                assert torch.is_tensor(value_b) and value_a.dtype == value_b.dtype and value_a.shape == value_b.shape, key
                assert torch.equal(value_a, value_b), key
            else:
                assert value_a == value_b, key
    for name in BUFFER_LISTS + BUFFER_STATS:
        assert getattr(reference, name) == getattr(candidate, name), name


def _collect_both(config, paras, policy, pool, instance=INSTANCE, seed=123, base=8):
    torch.manual_seed(seed)
    reference = collect_vectorized_rollouts(policy, instance, paras, base)
    torch.manual_seed(seed)
    candidate = collect_vectorized_rollouts_parallel(policy, instance, paras, base, pool=pool)
    return reference, candidate


def test_codec_round_trip_preserves_values_and_dtypes():
    config, paras = _paras()
    obs = TrainingSimulator(parse_instance_file(INSTANCE), paras).to_policy_obs()
    obs["valid_pairs"] = torch.zeros((0, 2), dtype=torch.long)          # the empty-candidate shape
    back = decode_obs(encode_obs(obs))
    assert list(back.keys()) == list(obs.keys())
    for key, value in obs.items():
        if torch.is_tensor(value):
            assert back[key].dtype == value.dtype and back[key].shape == value.shape and torch.equal(back[key], value), key
        else:
            assert back[key] == value, key


def test_pool_matches_in_process_buffer_fluid_off_and_fallback():
    config, paras = _paras()
    policy = _policy(config, paras)
    with CollectorPool(4, 2) as pool:
        reference, candidate = _collect_both(config, paras, policy, pool)
        assert_buffers_identical(reference, candidate)
        assert len(reference) > 0 and reference.policy_forward_count > 0
        config, paras = _paras(fluid_mode="cached_lp", fluid_backend="heuristic", fluid_mask=False)   # fallback path
        reference, candidate = _collect_both(config, paras, policy, pool, instance=OTHER, seed=7, base=20)
        assert_buffers_identical(reference, candidate)
        assert candidate.fluid_fallback_count > 0 and candidate.trajectory_ids[0] == 20
        config, paras = _paras(fluid_mode="cached_lp", fluid_backend="scipy", fluid_mask=True)       # masked path
        reference, candidate = _collect_both(config, paras, policy, pool, instance=OTHER, seed=9, base=40)
        assert_buffers_identical(reference, candidate)
        assert candidate.fluid_solve_count > 0 and candidate.mask_decision_count > 0
        assert candidate.candidate_pairs_after <= candidate.candidate_pairs_before
    assert not pool.alive


def test_pool_reuse_mlp_and_worker_clamping():
    config, paras = _paras(model_type="mlp", num_envs=2, num_workers=2)
    policy = _policy(config, paras)
    pool = CollectorPool(2, 3)                                      # clamped to 2 workers
    assert pool.workers == 2 and pool.blocks == [[0], [1]]
    try:
        for instance in (INSTANCE, OTHER):
            assert_buffers_identical(*_collect_both(config, paras, policy, pool, instance=instance))
    finally:
        pool.close()
    pool.close()                                                   # idempotent
    with CollectorPool(2, 1) as single:
        assert single.blocks == [[0, 1]]
        assert_buffers_identical(*_collect_both(config, paras, policy, single))


def test_worker_errors_propagate_and_pool_survives():
    config, paras = _paras()
    with CollectorPool(4, 2) as pool:
        with pytest.raises(CollectorError) as info:
            pool.load(BENCHMARK / "no_such" / "no_such.txt", paras)
        assert "FileNotFoundError" in str(info.value) and "remote traceback" in str(info.value)
        reports = pool.load(INSTANCE, paras)
        assert [r.env_index for r in reports] == [0, 1, 2, 3] and all(r.obs is not None for r in reports)
        with pytest.raises(ValueError):
            collect_vectorized_rollouts_parallel(_policy(config, paras), INSTANCE, dict(paras, num_envs=3), pool=pool)
    with pytest.raises(CollectorError):
        pool.load(INSTANCE, paras)                                 # closed pools refuse commands
    assert all(not p.is_alive() for p in pool._processes)
