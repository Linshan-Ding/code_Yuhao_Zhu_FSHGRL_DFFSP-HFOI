"""Synchronous multi-environment episode collection (moved verbatim from agent/sci_rollout.py)."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from agent.buffer import RolloutBuffer, _mark_last_transition_terminal
from data.instance_io import parse_instance_file
from environment.simulator import TrainingSimulator


def collect_vectorized_rollouts(policy, instance_path: Path, train_paras: dict,
                                base_trajectory_id: int = 0) -> RolloutBuffer:
    spec = parse_instance_file(instance_path)
    num_envs = int(train_paras.get("num_envs", train_paras.get("num_workers", 4)))
    buffer = RolloutBuffer(minibatch_size=int(train_paras.get("minibatch_size", 512)))
    buffer.max_minibatch_tokens = int(train_paras.get("max_minibatch_tokens", 0))
    simulators = [TrainingSimulator(spec, train_paras) for _ in range(num_envs)]
    trajectory_ids = [base_trajectory_id + index for index in range(num_envs)]
    episode_start_times = [time.perf_counter() for _ in range(num_envs)]
    finished = [False for _ in range(num_envs)]
    policy_forward_count = 0
    policy_forward_seconds = 0.0
    total_requests = 0
    action_tokens_sum = 0.0
    max_action_tokens = 0.0
    padding_ratio_sum = 0.0

    while not all(finished):
        active_indices = [
            index for index, simulator in enumerate(simulators)
            if not finished[index] and not simulator.done and simulator.has_decision()
        ]
        if not active_indices:
            for index, simulator in enumerate(simulators):
                if not finished[index] and simulator.done:
                    finished[index] = True
                    _mark_last_transition_terminal(buffer, trajectory_ids[index])
                    buffer.fulfillment_rates.append(simulator.fulfillment_rate)
                    buffer.discard_count += simulator.discarded_count
                    buffer.completion_count += simulator.completed_count
                    buffer.fluid_solve_count += simulator.fluid_solve_count
                    buffer.fluid_solve_seconds += simulator.fluid_solve_seconds
                    buffer.fluid_cache_hit_count += simulator.fluid_cache_hit_count
                    buffer.fluid_fallback_count += simulator.fluid_fallback_count
                    buffer.add_mask_stats(simulator)
                    buffer.episode_transition_counts.append(simulator.transition_count)
                    buffer.episode_seconds.append(time.perf_counter() - episode_start_times[index])
                    buffer.obs_build_seconds += simulator.obs_build_seconds
                    buffer.simulator_step_seconds += simulator.step_seconds
            continue

        obs_batch = [simulators[index].to_policy_obs() for index in active_indices]
        forward_start = time.perf_counter()
        with torch.no_grad():
            action_indexes, log_probs, values, env_actions = policy.act_batch(obs_batch, sample=True)
        forward_elapsed = time.perf_counter() - forward_start
        stats = getattr(policy, "last_forward_stats", {}) or {}
        batch_size = int(stats.get("batch_size", len(obs_batch)))
        policy_forward_count += 1
        policy_forward_seconds += float(stats.get("forward_seconds", forward_elapsed))
        total_requests += batch_size
        action_tokens_sum += float(stats.get("avg_action_tokens", 0.0)) * batch_size
        max_action_tokens = max(max_action_tokens, float(stats.get("max_action_tokens", 0.0)))
        padding_ratio_sum += float(stats.get("padding_ratio", 0.0)) * batch_size

        for local_index, env_index in enumerate(active_indices):
            simulator = simulators[env_index]
            reward = simulator.step(env_actions[local_index])
            done = simulator.done
            buffer.add(
                obs_batch[local_index],
                int(action_indexes[local_index].detach().cpu().item()),
                env_actions[local_index],
                float(log_probs[local_index].detach().cpu().item()),
                float(values[local_index].detach().cpu().item()),
                reward,
                done,
                trajectory_ids[env_index],
            )

    request_count = max(total_requests, 1)
    buffer.policy_forward_count = policy_forward_count
    buffer.policy_forward_seconds = policy_forward_seconds
    buffer.avg_action_tokens = action_tokens_sum / request_count
    buffer.max_action_tokens = max_action_tokens
    buffer.padding_ratio = padding_ratio_sum / request_count
    buffer.actor_batch_size_mean = total_requests / max(policy_forward_count, 1)
    return buffer


def _episode_record(simulator: TrainingSimulator, total_reward: float, wall_seconds: float) -> dict:
    return {
        "fulfillment_rate": float(simulator.fulfillment_rate),
        "completed": int(simulator.completed_count),
        "discarded": int(simulator.discarded_count),
        "total_orders": int(simulator.spec.total_orders),
        "decision_count": int(simulator.transition_count),
        "schedule_count": int(simulator.schedule_count),
        "total_reward": float(total_reward),
        "wall_seconds": float(wall_seconds),
        "obs_build_seconds": float(simulator.obs_build_seconds),
        "simulator_step_seconds": float(simulator.step_seconds),
        "fluid_solve_count": int(simulator.fluid_solve_count),
        "fluid_solve_seconds": float(simulator.fluid_solve_seconds),
        "fluid_cache_hit_count": int(simulator.fluid_cache_hit_count),
        "fluid_fallback_count": int(simulator.fluid_fallback_count),
        # v2 mask statistics (per decision of this episode)
        "candidate_pairs_mean": simulator.candidate_pairs_before / max(int(simulator.mask_decision_count), 1),
        "masked_pairs_mean": simulator.candidate_pairs_after / max(int(simulator.mask_decision_count), 1),
        "mask_fallback_rate": simulator.fluid_mask_fallback_count / max(int(simulator.mask_decision_count), 1),
        # provided by environment/profiled.py (eval.py); NaN / empty with the plain simulator
        "fluid_feature_seconds": float(getattr(simulator, "fluid_feature_seconds", float("nan"))),
        "fluid_backend": str(getattr(simulator, "fluid_backend", simulator.fluid_backend_used)),
    }


def run_episodes(policy, spec, train_paras: dict, episodes: int = 1, sample: bool = False,
                 simulator_factory=TrainingSimulator) -> list:
    """Run ``episodes`` complete episodes of one instance in lockstep and return one record per episode.

    New code (no counterpart in the original repository). It drives the simulators exactly like
    ``collect_vectorized_rollouts`` but keeps no buffer. With ``sample=False`` the policy decodes greedily
    (argmax of both action distributions) and draws **no random numbers**, which is what lets ``train.py``
    evaluate the policy after every update without disturbing the training stream (checked by
    ``tests/equivalence``). ``sample=True`` draws actions from the policy and is for ``eval.py`` only.

    With ``episodes > 1`` the observations of all running episodes share one forward pass; the policy's
    per-batch normalisation (due dates, processing rates) then sees the batch as a whole, so numbers that
    must not depend on the batch composition are produced with ``episodes=1``.
    """
    simulators = [simulator_factory(spec, train_paras) for _ in range(episodes)]
    counts_at_start = [(int(s.completed_count), int(s.discarded_count), float(s.fulfillment_rate)) for s in simulators]
    start_times = [time.perf_counter() for _ in range(episodes)]
    finished = [False for _ in range(episodes)]
    total_rewards = [0.0 for _ in range(episodes)]
    records: list = [None for _ in range(episodes)]
    forward_seconds = 0.0
    forward_count = 0

    while not all(finished):
        active_indices = [
            index for index, simulator in enumerate(simulators)
            if not finished[index] and not simulator.done and simulator.has_decision()
        ]
        if not active_indices:
            for index, simulator in enumerate(simulators):
                if not finished[index] and simulator.done:
                    finished[index] = True
                    records[index] = _episode_record(simulator, total_rewards[index],
                                                     time.perf_counter() - start_times[index])
            continue

        obs_batch = [simulators[index].to_policy_obs() for index in active_indices]
        forward_start = time.perf_counter()
        with torch.no_grad():
            _, _, _, env_actions = policy.act_batch(obs_batch, sample=sample)
        forward_seconds += time.perf_counter() - forward_start
        forward_count += 1
        for local_index, env_index in enumerate(active_indices):
            total_rewards[env_index] += simulators[env_index].step(env_actions[local_index])

    for index, record in enumerate(records):
        record["policy_forward_seconds"] = forward_seconds   # shared by the lockstep batch
        record["policy_forward_count"] = forward_count
        record["batch_episodes"] = episodes
        record["completed_at_start"], record["discarded_at_start"], record["fulfillment_rate_at_start"] = counts_at_start[index]
    return records


def collect_vectorized_rollouts_parallel(policy, instance_path: Path, train_paras: dict,
                                         base_trajectory_id: int = 0, *, pool) -> RolloutBuffer:
    """``collect_vectorized_rollouts`` with the simulators living in the worker processes of ``pool``
    (``environment.collector.CollectorPool``); new code.

    Statement-by-statement mirror of the in-process loop: the same active set (from the workers' done /
    has_decision flags), the same observation batch order, the single ``policy.act_batch`` call in this
    process (the only consumer of the torch RNG), ``buffer.add`` in the same order with the workers' rewards
    and done flags, and the same finalisation order, so the returned buffer is bit-identical with the same
    ``num_envs``. Timing fields come from the workers (they are not part of the regression comparison).
    """
    num_envs = int(train_paras.get("num_envs", train_paras.get("num_workers", 4)))
    if pool.num_envs != num_envs:
        raise ValueError(f"the collector pool holds {pool.num_envs} environments but num_envs is {num_envs}")
    buffer = RolloutBuffer(minibatch_size=int(train_paras.get("minibatch_size", 512)))
    buffer.max_minibatch_tokens = int(train_paras.get("max_minibatch_tokens", 0))
    episode_start_times = [time.perf_counter() for _ in range(num_envs)]
    reports = pool.load(instance_path, train_paras)
    done_flags = [report.done for report in reports]
    decision_flags = [report.has_decision for report in reports]
    next_obs = [report.obs for report in reports]
    episode_stats = [report.stats for report in reports]
    trajectory_ids = [base_trajectory_id + index for index in range(num_envs)]
    finished = [False for _ in range(num_envs)]
    policy_forward_count = 0
    policy_forward_seconds = 0.0
    total_requests = 0
    action_tokens_sum = 0.0
    max_action_tokens = 0.0
    padding_ratio_sum = 0.0

    while not all(finished):
        active_indices = [
            index for index in range(num_envs)
            if not finished[index] and not done_flags[index] and decision_flags[index]
        ]
        if not active_indices:
            finalised = 0
            for index in range(num_envs):
                if not finished[index] and done_flags[index]:
                    finished[index] = True
                    finalised += 1
                    _mark_last_transition_terminal(buffer, trajectory_ids[index])
                    episode = episode_stats[index]
                    buffer.fulfillment_rates.append(episode["fulfillment_rate"])
                    buffer.discard_count += episode["discarded_count"]
                    buffer.completion_count += episode["completed_count"]
                    buffer.fluid_solve_count += episode["fluid_solve_count"]
                    buffer.fluid_solve_seconds += episode["fluid_solve_seconds"]
                    buffer.fluid_cache_hit_count += episode["fluid_cache_hit_count"]
                    buffer.fluid_fallback_count += episode["fluid_fallback_count"]
                    buffer.add_mask_stats(episode)
                    buffer.episode_transition_counts.append(episode["transition_count"])
                    buffer.episode_seconds.append(time.perf_counter() - episode_start_times[index])
                    buffer.obs_build_seconds += episode["obs_build_seconds"]
                    buffer.simulator_step_seconds += episode["step_seconds"]
            if finalised == 0:
                raise RuntimeError("collector: an unfinished environment is neither done nor at a decision point")
            continue

        obs_batch = [next_obs[index] for index in active_indices]
        forward_start = time.perf_counter()
        with torch.no_grad():
            action_indexes, log_probs, values, env_actions = policy.act_batch(obs_batch, sample=True)
        forward_elapsed = time.perf_counter() - forward_start
        stats = getattr(policy, "last_forward_stats", {}) or {}
        batch_size = int(stats.get("batch_size", len(obs_batch)))
        policy_forward_count += 1
        policy_forward_seconds += float(stats.get("forward_seconds", forward_elapsed))
        total_requests += batch_size
        action_tokens_sum += float(stats.get("avg_action_tokens", 0.0)) * batch_size
        max_action_tokens = max(max_action_tokens, float(stats.get("max_action_tokens", 0.0)))
        padding_ratio_sum += float(stats.get("padding_ratio", 0.0)) * batch_size

        results = pool.step({env_index: env_actions[local_index] for local_index, env_index in enumerate(active_indices)})
        for local_index, env_index in enumerate(active_indices):
            report = results[env_index]
            buffer.add(
                obs_batch[local_index],
                int(action_indexes[local_index].detach().cpu().item()),
                env_actions[local_index],
                float(log_probs[local_index].detach().cpu().item()),
                float(values[local_index].detach().cpu().item()),
                report.reward,
                report.done,
                trajectory_ids[env_index],
            )
            done_flags[env_index] = report.done
            decision_flags[env_index] = report.has_decision
            next_obs[env_index] = report.obs
            episode_stats[env_index] = report.stats

    request_count = max(total_requests, 1)
    buffer.policy_forward_count = policy_forward_count
    buffer.policy_forward_seconds = policy_forward_seconds
    buffer.avg_action_tokens = action_tokens_sum / request_count
    buffer.max_action_tokens = max_action_tokens
    buffer.padding_ratio = padding_ratio_sum / request_count
    buffer.actor_batch_size_mean = total_requests / max(policy_forward_count, 1)
    return buffer
