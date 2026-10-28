"""Replay-based training loop of the DQN / DDQN baselines (new code).

It mirrors the bookkeeping of the PPO loop in ``train.py``: the same ``train_metrics.csv`` columns, the same
checkpoint files (``best_checkpoint.pt`` by the mean fulfilment of the collected episodes,
``last_checkpoint.pt``, ``best_greedy_checkpoint.pt``, optional ``checkpoint_epoch*.pt``) and the contract
files written by ``RunLogger`` (``curve.csv`` from a greedy episode after every iteration, ``log.csv``,
``config_snapshot.yaml``, ``train_meta.yaml``, ``fixed_set_final.csv``). One iteration = epsilon-greedy
collection of ``num_envs`` complete episodes into the replay buffer, then ``updates_per_iteration`` TD steps
on uniformly sampled minibatches of ``replay_batch_size`` transitions (after ``warmup_transitions``).

Observations are stored as the simulator produces them; the four instance-level tensors every observation
carries (processing rates and the three adjacency matrices) are replaced by one shared reference per instance
to keep the replay buffer small. All randomness (exploration, replay sampling) uses a private ``random.Random``
seeded with ``--seed``; the network initialisation uses the global torch seed set by ``train.py``.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from agent.baselines.common import legacy_metrics_row, monte_carlo_returns, prepare_policy
from agent.baselines.dqn import DQNAgent
from agent.buffer import RolloutBuffer, _mark_last_transition_terminal
from agent.checkpoint import load_checkpoint, save_checkpoint
from agent.factory import algorithm_of
from data.dataset import select_instance
from data.instance_io import parse_instance_file
from environment.simulator import TrainingSimulator
from result.logger import RunLogger
from result.metrics import gpu_utilization, load_reference, objective_from_fulfillment

STATIC_OBS_KEYS = ("proc_time", "ope_ma_adj", "ope_pre_adj", "ope_sub_adj")
ZERO_UPDATE = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0,
               "clip_fraction": 0.0, "td_error_abs": 0.0, "q_mean": 0.0, "gradient_steps": 0}


@dataclass
class Transition:
    obs: dict
    token: int
    reward: float
    next_obs: Optional[dict]
    done: bool


class ReplayBuffer:
    """Fixed-capacity ring buffer with uniform sampling."""

    def __init__(self, capacity: int):
        self.capacity = max(int(capacity), 1)
        self._items: List[Transition] = []
        self._next = 0

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: Transition) -> None:
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._next] = transition
            self._next = (self._next + 1) % self.capacity

    def sample(self, rng: random.Random, count: int) -> List[Transition]:
        return rng.sample(self._items, min(int(count), len(self._items)))


def share_static_tensors(obs: dict, spec, cache: dict) -> dict:
    """Replace the per-observation clones of the instance tensors by one shared reference per instance."""
    key = str(spec.path)
    static = cache.get(key)
    if static is None:
        static = {"proc_time": spec.proc_rates, "ope_ma_adj": spec.ope_ma_adj,
                  "ope_pre_adj": spec.ope_pre_adj, "ope_sub_adj": spec.ope_sub_adj}
        cache[key] = static
    for name in STATIC_OBS_KEYS:
        obs[name] = static[name]
    return obs


def epsilon_for(iteration: int, train_paras: dict) -> float:
    """Linear decay of the original implementation: max(eps0 * (N - iteration) / N, eps_end)."""
    start = float(train_paras.get("epsilon_start", 1.0))
    end = float(train_paras.get("epsilon_end", 0.01))
    decay = max(int(train_paras.get("epsilon_decay_iterations", 500)), 1)
    return max(start * (decay - iteration) / decay, end)


def collect_offpolicy(policy, spec, train_paras: dict, num_envs: int, epsilon: float, rng: random.Random,
                      replay: ReplayBuffer, static_cache: dict, base_trajectory_id: int = 0) -> RolloutBuffer:
    """Lockstep collection as in ``rollout.collect_vectorized_rollouts``; the transitions go to the replay
    buffer, the returned ``RolloutBuffer`` carries only rewards / dones / trajectory ids and the statistics
    needed for the log row."""
    buffer = RolloutBuffer()
    simulators = [TrainingSimulator(spec, train_paras) for _ in range(num_envs)]
    trajectory_ids = [base_trajectory_id + index for index in range(num_envs)]
    episode_start_times = [time.perf_counter() for _ in range(num_envs)]
    finished = [False for _ in range(num_envs)]
    pending: List[Optional[tuple]] = [None for _ in range(num_envs)]
    policy_forward_count = 0
    policy_forward_seconds = 0.0
    total_requests = 0
    action_tokens_sum = 0.0
    max_action_tokens = 0.0

    while not all(finished):
        active_indices = [
            index for index, simulator in enumerate(simulators)
            if not finished[index] and not simulator.done and simulator.has_decision()
        ]
        if not active_indices:
            for index, simulator in enumerate(simulators):
                if not finished[index] and simulator.done:
                    finished[index] = True
                    if pending[index] is not None:
                        obs, token, reward = pending[index]
                        replay.add(Transition(obs, token, reward, None, True))
                        pending[index] = None
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

        obs_batch = [share_static_tensors(simulators[index].to_policy_obs(), spec, static_cache)
                     for index in active_indices]
        for local_index, env_index in enumerate(active_indices):
            if pending[env_index] is not None:
                obs, token, reward = pending[env_index]
                replay.add(Transition(obs, token, reward, obs_batch[local_index], False))
                pending[env_index] = None
        forward_start = time.perf_counter()
        with torch.no_grad():
            tokens, _, _, env_actions = policy.act_batch(obs_batch, sample=False, epsilon=epsilon, rng=rng)
        forward_elapsed = time.perf_counter() - forward_start
        stats = getattr(policy, "last_forward_stats", {}) or {}
        batch_size = int(stats.get("batch_size", len(obs_batch)))
        policy_forward_count += 1
        policy_forward_seconds += float(stats.get("forward_seconds", forward_elapsed))
        total_requests += batch_size
        action_tokens_sum += float(stats.get("avg_action_tokens", 0.0)) * batch_size
        max_action_tokens = max(max_action_tokens, float(stats.get("max_action_tokens", 0.0)))

        for local_index, env_index in enumerate(active_indices):
            simulator = simulators[env_index]
            reward = simulator.step(env_actions[local_index])
            buffer.rewards.append(float(reward))
            buffer.dones.append(bool(simulator.done))
            buffer.trajectory_ids.append(trajectory_ids[env_index])
            pending[env_index] = (obs_batch[local_index], int(tokens[local_index].item()), float(reward))

    request_count = max(total_requests, 1)
    buffer.policy_forward_count = policy_forward_count
    buffer.policy_forward_seconds = policy_forward_seconds
    buffer.avg_action_tokens = action_tokens_sum / request_count
    buffer.max_action_tokens = max_action_tokens
    buffer.padding_ratio = 0.0
    buffer.actor_batch_size_mean = total_requests / max(policy_forward_count, 1)
    return buffer


def train_offpolicy(args, config: dict, train_paras: dict, instance_files, rollout_device, update_device) -> None:
    import train as trainer  # the PPO entry point; imported here to avoid an import cycle

    algorithm = algorithm_of(train_paras)
    model_paras = trainer.build_model_paras(config, update_device, train_paras)
    agent = DQNAgent(model_paras, train_paras, double=(algorithm == "ddqn"))
    exp_name = args.experiment_name or train_paras.get("experiment_name", "default")
    result_dir = trainer.ROOT_DIR / "result" / exp_name
    result_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = result_dir / "last_checkpoint.pt"
    best_checkpoint_path = result_dir / "best_checkpoint.pt"
    best_greedy_checkpoint_path = result_dir / "best_greedy_checkpoint.pt"
    best_fulfillment_rate = float("-inf")
    trajectory_base = 0
    start_iteration = 0
    if train_paras.get("resume", "auto") == "auto" and last_checkpoint_path.exists():
        try:
            start_iteration, best_fulfillment_rate = load_checkpoint(last_checkpoint_path, agent, update_device)
            print(f"[Checkpoint] Resumed from iteration {start_iteration}; best fulfillment={best_fulfillment_rate:.6f}")
        except Exception as exc:
            print(f"[Checkpoint] Could not resume from {last_checkpoint_path}: {exc}. Starting fresh.")

    max_iterations = int(train_paras.get("max_iterations", 1000))
    logger = RunLogger(result_dir, train_paras, resume=start_iteration > 0)
    reference = load_reference()
    logger.write_snapshot(trainer.build_snapshot(args, train_paras, model_paras, instance_files, rollout_device,
                                                 update_device, start_iteration, max_iterations))
    eval_every = int(train_paras.get("eval_every", 1))
    checkpoint_every = int(train_paras.get("checkpoint_every", 0))
    num_envs = int(train_paras.get("num_envs", train_paras.get("num_workers", 4)))
    batch_size = int(train_paras.get("replay_batch_size", 128))
    updates_per_iteration = int(train_paras.get("updates_per_iteration", 50))
    warmup = int(train_paras.get("warmup_transitions", 500))
    gamma = float(train_paras.get("gamma", 1.0))
    replay = ReplayBuffer(int(train_paras.get("replay_capacity", 50000)))
    rng = random.Random(int(args.seed))
    static_cache: dict = {}
    best_greedy_obj = float("inf")
    best_greedy_iteration = -1
    last_iteration = start_iteration - 1
    stop_reason = "running"
    print(f"[{algorithm.upper()}] replay capacity {replay.capacity}, batch {batch_size}, "
          f"{updates_per_iteration} updates/iteration, warm-up {warmup} transitions")
    try:
        for iteration in range(start_iteration, max_iterations):
            if update_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(update_device)
            start_time = time.time()
            learning_rate = trainer.apply_lr_schedule(agent, train_paras, iteration, max_iterations)
            instance_selection = train_paras.get("instance_selection", "fixed")
            instance_index, selected_instance = select_instance(instance_files, instance_selection, iteration)
            spec = parse_instance_file(selected_instance)
            epsilon = epsilon_for(iteration, train_paras)
            rollout_start_time = time.time()
            rollout_policy = prepare_policy(agent.policy, rollout_device, update_device)
            buffer = collect_offpolicy(rollout_policy, spec, train_paras, num_envs, epsilon, rng, replay,
                                       static_cache, trajectory_base)
            rollout_seconds = time.time() - rollout_start_time
            trajectory_base += num_envs
            update_start_time = time.time()
            if len(replay) >= warmup:
                update_metrics = agent.update(replay, rng, updates_per_iteration, batch_size)
            else:
                update_metrics = dict(ZERO_UPDATE)
            update_seconds = time.time() - update_start_time
            elapsed = time.time() - start_time
            resources = trainer.resource_snapshot(update_device)
            returns = monte_carlo_returns(buffer.rewards, buffer.dones, buffer.trajectory_ids, gamma)
            mean_return = float(np.mean(returns)) if returns else 0.0
            row = legacy_metrics_row(iteration, instance_index, selected_instance, instance_selection, buffer,
                                     update_metrics, learning_rate, rollout_seconds, update_seconds, elapsed,
                                     resources, rollout_device, update_device, mean_return,
                                     sparse_attention_enabled=False, sparse_attention_scope="none")
            logger.log_legacy(iteration, row)
            print(f"[{algorithm.upper()}] iteration {iteration}: epsilon {epsilon:.3f}, replay {len(replay)}, "
                  f"gradient steps {update_metrics.get('gradient_steps', 0)}, td error "
                  f"{update_metrics.get('td_error_abs', 0.0):.4f}, q mean {update_metrics.get('q_mean', 0.0):.4f}")

            fulfillment_metrics = buffer.fulfillment_summary()
            if fulfillment_metrics["fulfillment_rate_mean"] >= best_fulfillment_rate:
                best_fulfillment_rate = fulfillment_metrics["fulfillment_rate_mean"]
                save_checkpoint(best_checkpoint_path, agent, iteration, best_fulfillment_rate, train_paras, config)
                torch.save(agent.policy.state_dict(), result_dir / "policy_model.pt")
            checkpoint_interval = max(int(train_paras.get("checkpoint_interval", 1)), 1)
            if iteration % checkpoint_interval == 0:
                save_checkpoint(last_checkpoint_path, agent, iteration, best_fulfillment_rate, train_paras, config)

            # ---- contract files (same protocol as the PPO loop: greedy episode after every update) ----
            logger.env_steps += len(buffer)
            fixed_obj_mean = fixed_gap_mean = fixed_fulfillment_mean = eval_seconds = ""
            if eval_every > 0 and iteration % eval_every == 0:
                eval_start_time = time.time()
                eval_policy = prepare_policy(agent.policy, rollout_device, update_device)
                greedy_rows = trainer.evaluate_fixed_set(eval_policy, instance_files, train_paras, reference)
                sampled_best_obj = objective_from_fulfillment(fulfillment_metrics["fulfillment_rate_max"])
                wall_time_s = round(logger.wall_time(), 3)
                for greedy_row in greedy_rows:
                    greedy_row["wall_time_s"] = wall_time_s
                    if greedy_row["instance_name"] == selected_instance.name:
                        greedy_row["obj_best"] = min(greedy_row["obj"], sampled_best_obj)
                logger.curve(iteration, greedy_rows)
                fixed_obj_mean = float(np.mean([greedy_row["obj"] for greedy_row in greedy_rows]))
                fixed_fulfillment_mean = 1.0 - fixed_obj_mean
                gaps = [greedy_row["gap"] for greedy_row in greedy_rows if not math.isnan(greedy_row["gap"])]
                fixed_gap_mean = float(np.mean(gaps)) if gaps else float("nan")
                eval_seconds = round(time.time() - eval_start_time, 3)
                if fixed_obj_mean < best_greedy_obj:
                    best_greedy_obj = fixed_obj_mean
                    best_greedy_iteration = iteration
                    save_checkpoint(best_greedy_checkpoint_path, agent, iteration, best_fulfillment_rate,
                                    train_paras, config)
            logger.log(iteration, {
                "env_steps": logger.env_steps,
                "wall_time_s": round(logger.wall_time(), 3),
                "train_obj_mean": objective_from_fulfillment(fulfillment_metrics["fulfillment_rate_mean"]),
                "fixed_obj_mean": fixed_obj_mean,
                "fixed_gap_mean": fixed_gap_mean,
                "policy_loss": row["policy_loss"],
                "value_loss": row["value_loss"],
                "entropy": row["entropy"],
                "clip_frac": row["clip_fraction"],
                "approx_kl": row["approx_kl"],
                "lr": learning_rate,
                "sps": row["steps_per_second"],
                "gpu_util": gpu_utilization(update_device),
                "gpu_mem_gb": resources["gpu_memory_mb"] / 1024.0 if update_device.type == "cuda" else "",
                "train_fulfillment_mean": fulfillment_metrics["fulfillment_rate_mean"],
                "fixed_fulfillment_mean": fixed_fulfillment_mean,
                "eval_seconds": eval_seconds,
                **buffer.mask_summary(),
            })
            if checkpoint_every > 0 and iteration % checkpoint_every == 0:
                save_checkpoint(result_dir / f"checkpoint_epoch{iteration:04d}.pt", agent, iteration,
                                best_fulfillment_rate, train_paras, config)
            last_iteration = iteration
        stop_reason = "max_iterations"
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        raise
    except Exception:
        stop_reason = "error"
        raise
    finally:
        try:
            trainer.finalize_run(logger, stop_reason, start_iteration, last_iteration, best_fulfillment_rate,
                                 best_greedy_obj, best_greedy_iteration, train_paras, instance_files)
        except Exception as exc:
            print(f"[RunLogger] could not write the run summary: {exc}")
        logger.close()
