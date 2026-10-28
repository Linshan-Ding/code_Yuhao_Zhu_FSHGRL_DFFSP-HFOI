"""Rollout buffer (moved verbatim from agent/sci_rollout.py)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class RolloutBuffer:
    obs: List[dict] = field(default_factory=list)
    action_indexes: List[int] = field(default_factory=list)
    env_actions: List[tuple] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    trajectory_ids: List[int] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    advantages: List[float] = field(default_factory=list)
    minibatch_size: int = 512
    max_minibatch_tokens: int = 0
    discard_count: int = 0
    completion_count: int = 0
    fulfillment_rates: List[float] = field(default_factory=list)
    fluid_solve_count: int = 0
    fluid_solve_seconds: float = 0.0
    fluid_cache_hit_count: int = 0
    fluid_fallback_count: int = 0
    fluid_mask_fallback_count: int = 0        # v2 mask statistics (summed over the collected episodes)
    fluid_mask_unavailable_count: int = 0
    candidate_pairs_before: int = 0
    candidate_pairs_after: int = 0
    mask_decision_count: int = 0
    policy_forward_count: int = 0
    policy_forward_seconds: float = 0.0
    avg_action_tokens: float = 0.0
    max_action_tokens: float = 0.0
    padding_ratio: float = 0.0
    actor_batch_size_mean: float = 0.0
    episode_transition_counts: List[int] = field(default_factory=list)
    episode_seconds: List[float] = field(default_factory=list)
    obs_build_seconds: float = 0.0
    simulator_step_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.rewards)

    def add(self, obs, action_index, env_action, log_prob, value, reward, done, trajectory_id):
        self.obs.append(obs)
        self.action_indexes.append(int(action_index))
        self.env_actions.append((int(env_action[0]), int(env_action[1])))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.trajectory_ids.append(int(trajectory_id))

    def add_mask_stats(self, source) -> None:
        """Accumulate the mask counters of a finished simulator (or of a worker's stats dict)."""
        get = source.get if isinstance(source, dict) else (lambda key, default=0: getattr(source, key, default))
        self.fluid_mask_fallback_count += int(get("fluid_mask_fallback_count", 0))
        self.fluid_mask_unavailable_count += int(get("fluid_mask_unavailable_count", 0))
        self.candidate_pairs_before += int(get("candidate_pairs_before", 0))
        self.candidate_pairs_after += int(get("candidate_pairs_after", 0))
        self.mask_decision_count += int(get("mask_decision_count", 0))

    def mask_summary(self) -> dict:
        """Mean eligible / candidate pairs per decision and the mask fallback rate (log.csv columns)."""
        decisions = max(int(self.mask_decision_count), 1)
        return {
            "candidate_pairs_mean": self.candidate_pairs_before / decisions,
            "masked_pairs_mean": self.candidate_pairs_after / decisions,
            "mask_fallback_rate": self.fluid_mask_fallback_count / decisions,
        }

    def fulfillment_summary(self) -> dict:
        if not self.fulfillment_rates:
            return {
                "fulfillment_rate_mean": 0.0,
                "fulfillment_rate_std": 0.0,
                "fulfillment_rate_min": 0.0,
                "fulfillment_rate_max": 0.0,
            }
        values = torch.tensor(self.fulfillment_rates, dtype=torch.float32)
        return {
            "fulfillment_rate_mean": float(values.mean().item()),
            "fulfillment_rate_std": float(values.std(unbiased=False).item()),
            "fulfillment_rate_min": float(values.min().item()),
            "fulfillment_rate_max": float(values.max().item()),
        }

    def episode_summary(self) -> dict:
        transition_values = torch.tensor(self.episode_transition_counts or [0], dtype=torch.float32)
        second_values = torch.tensor(self.episode_seconds or [0.0], dtype=torch.float32)
        return {
            "episode_count": len(self.episode_transition_counts),
            "episode_transition_mean": float(transition_values.mean().item()),
            "episode_transition_std": float(transition_values.std(unbiased=False).item()),
            "episode_seconds_mean": float(second_values.mean().item()),
            "episode_seconds_std": float(second_values.std(unbiased=False).item()),
        }

    def compute_returns_and_advantages(self, gamma: float, gae_lambda: float):
        self.returns = [0.0 for _ in self.rewards]
        self.advantages = [0.0 for _ in self.rewards]
        by_trajectory = {}
        for index, trajectory_id in enumerate(self.trajectory_ids):
            by_trajectory.setdefault(trajectory_id, []).append(index)

        for indices in by_trajectory.values():
            gae = 0.0
            next_value = 0.0
            for index in reversed(indices):
                non_terminal = 0.0 if self.dones[index] else 1.0
                delta = self.rewards[index] + gamma * next_value * non_terminal - self.values[index]
                gae = delta + gamma * gae_lambda * non_terminal * gae
                self.advantages[index] = gae
                self.returns[index] = gae + self.values[index]
                next_value = self.values[index]

    def iter_minibatches(self, shuffle: bool = True):
        buckets = {}
        for index, obs in enumerate(self.obs):
            key = (
                int(obs.get("num_tasks", obs["raw_opes"].size(0))),
                int(obs.get("num_machines", obs["raw_mas"].size(0))),
                int(obs.get("max_orders", obs["due_dates"].size(1))),
            )
            buckets.setdefault(key, []).append(index)

        bucket_keys = list(buckets.keys())
        if shuffle:
            random.shuffle(bucket_keys)

        for key in bucket_keys:
            indices = buckets[key]
            if shuffle:
                random.shuffle(indices)
            token_per_sample = max(key[0] * key[1] * key[2], 1)
            token_limit = int(self.max_minibatch_tokens or 0)
            if token_limit > 0:
                batch_size = max(1, min(self.minibatch_size, token_limit // token_per_sample))
            else:
                batch_size = self.minibatch_size
            for start in range(0, len(indices), batch_size):
                yield indices[start:start + batch_size]


def _mark_last_transition_terminal(buffer: RolloutBuffer, trajectory_id: int):
    for index in range(len(buffer.dones) - 1, -1, -1):
        if buffer.trajectory_ids[index] == trajectory_id:
            buffer.dones[index] = True
            return
