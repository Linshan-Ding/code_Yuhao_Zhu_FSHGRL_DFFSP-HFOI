"""Component probe (layers L1-L5) of the equivalence harness.

Usage (from a tree root, i.e. cwd is the repository root of either tree):

    python tests/equivalence/probe.py <out.pt> <legacy_config.json> <instance.txt> [model_type ...]

The script resolves modules through a small adapter so that the very same file
runs in the original tree (legacy module names) and in the refactored tree.
Its output is a nested dict of tensors / scalars saved with torch.save and is
compared by `tests/equivalence/__main__.py`.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def import_first(*candidates):
    last_error: Exception | None = None
    for module_name, attribute in candidates:
        try:
            module = __import__(module_name, fromlist=[attribute])
            return getattr(module, attribute)
        except (ImportError, AttributeError) as exc:  # pragma: no cover - adapter
            last_error = exc
    raise ImportError(f"none of {candidates} could be imported: {last_error}")


parse_instance_file = import_first(("data.instance_io", "parse_instance_file"),
                                   ("agent.training_simulator", "parse_instance_file"))
TrainingSimulator = import_first(("environment.simulator", "TrainingSimulator"),
                                 ("agent.training_simulator", "TrainingSimulator"))
RolloutBuffer = import_first(("agent.buffer", "RolloutBuffer"),
                             ("agent.sci_rollout", "RolloutBuffer"))
PPO = import_first(("agent.ppo", "PPO"), ("agent.PPO_model", "PPO"))
build_model_paras = import_first(("train", "build_model_paras"), ("agent.train", "build_model_paras"))
set_seed = import_first(("train", "set_seed"), ("agent.train", "set_seed"))

# Same overrides as tests/equivalence/common.py (duplicated on purpose: the
# golden tree has no tests/ package).
TRAIN_OVERRIDES = {
    "num_envs": 2, "num_workers": 2, "minibatch_size": 128, "max_minibatch_tokens": 0,
    "max_iterations": 3, "viz": False, "resume": "none", "rollout_device": "cpu",
    "update_device": "cpu", "instance_selection": "fixed", "fluid_mode": "cached_lp",
    "fluid_mask": False, "fluid_backend": "auto", "fluid_recompute": "event",   # v2 keys: legacy semantics
}
OBS_TENSOR_KEYS = ("raw_opes", "raw_mas", "due_dates", "order_mask", "order_counts",
                   "valid_pairs", "eligible", "proc_time", "ope_ma_adj", "ope_pre_adj", "ope_sub_adj")


def probe_instance(spec) -> dict:
    return {
        "product_count": int(spec.product_count),
        "stage_count": int(spec.stage_count),
        "machine_count": int(spec.machine_count),
        "machine_division": [int(v) for v in spec.machine_division],
        "stage_machines": [[int(m) for m in group] for group in spec.stage_machines],
        "process_times": torch.from_numpy(spec.process_times.copy()),
        "order_ids": [str(v) for v in spec.order_ids],
        "product_types": torch.from_numpy(spec.product_types.copy()),
        "arrival_times": torch.from_numpy(spec.arrival_times.copy()),
        "due_dates": torch.from_numpy(spec.due_dates.copy()),
        "task_to_product_stage": [[int(a), int(b)] for a, b in spec.task_to_product_stage],
        "ope_ma_adj": spec.ope_ma_adj.clone(),
        "ope_pre_adj": spec.ope_pre_adj.clone(),
        "ope_sub_adj": spec.ope_sub_adj.clone(),
        "proc_rates": spec.proc_rates.clone(),
    }


def probe_simulator(spec, train_paras: dict, steps: int = 60) -> tuple[list, list]:
    simulator = TrainingSimulator(spec, train_paras)
    trajectory = []
    fixtures = []
    for _ in range(steps):
        if simulator.done:
            break
        obs = simulator.to_policy_obs()
        valid_pairs = obs["valid_pairs"]
        machine, task = int(valid_pairs[0, 0]), int(valid_pairs[0, 1])
        env_action = (machine * int(obs["num_tasks"]) + task, 0)
        reward = simulator.step(env_action)
        record = {key: obs[key].clone() for key in OBS_TENSOR_KEYS}
        record.update({
            "rate": float(obs["current_order_completed_rate"]),
            "max_orders": int(obs["max_orders"]),
            "valid_pair_count": int(obs["valid_pair_count"]),
            "valid_action_tokens": int(obs["valid_action_tokens"]),
            "reward": float(reward),
            "time": float(simulator.current_time),
            "completed": int(simulator.completed_count),
            "discarded": int(simulator.discarded_count),
            "transition": int(simulator.transition_count),
            "fluid_solve_count": int(simulator.fluid_solve_count),
            "fluid_cache_hit_count": int(simulator.fluid_cache_hit_count),
            "fluid_fallback_count": int(simulator.fluid_fallback_count),
        })
        trajectory.append(record)
        if len(fixtures) < 2:
            fixtures.append(obs)
    return trajectory, fixtures


def probe_model(config: dict, train_paras: dict, model_type: str, fixtures: list) -> dict:
    paras = dict(train_paras)
    paras["model_type"] = model_type
    set_seed(42)
    model_paras = build_model_paras(config, torch.device("cpu"), paras)
    model = PPO(model_paras, paras)
    out = {"state_dict": {key: value.detach().clone() for key, value in model.policy.state_dict().items()}}

    policy_old = model.policy_old
    policy_old.eval()
    with torch.no_grad():
        action_indexes, log_probs, values, env_actions = policy_old.act_batch(fixtures, sample=False)
    out["act_greedy"] = {
        "action_indexes": action_indexes.detach().clone(),
        "log_probs": log_probs.detach().clone(),
        "values": values.detach().clone(),
        "env_actions": [[int(a), int(b)] for a, b in env_actions],
    }
    torch.manual_seed(123)
    with torch.no_grad():
        s_indexes, s_log_probs, s_values, s_env_actions = policy_old.act_batch(fixtures, sample=True)
    out["act_sample"] = {
        "action_indexes": s_indexes.detach().clone(),
        "log_probs": s_log_probs.detach().clone(),
        "env_actions": [[int(a), int(b)] for a, b in s_env_actions],
    }
    model.policy.train()
    log_probs2, values2, entropies = model.policy.evaluate_batch(fixtures, env_actions)
    out["evaluate"] = {
        "log_probs": log_probs2.detach().clone(),
        "values": values2.detach().clone(),
        "entropies": entropies.detach().clone(),
    }
    return out


def probe_buffer() -> dict:
    buffer = RolloutBuffer(minibatch_size=8)
    for index in range(37):
        max_orders = [1, 2, 3][index % 3]
        obs = {
            "num_tasks": 25, "num_machines": 25, "max_orders": max_orders,
            "raw_opes": torch.zeros(25, 12), "raw_mas": torch.zeros(25, 5),
            "due_dates": torch.zeros(25, max_orders),
        }
        buffer.add(obs, index, (index, 0), -0.1 * index, math.cos(index), math.sin(index),
                   index in (20, 36), 0 if index <= 20 else 1)
    buffer.fulfillment_rates = [0.3, 0.5]
    buffer.episode_transition_counts = [21, 16]
    buffer.compute_returns_and_advantages(gamma=1.0, gae_lambda=0.95)
    random.seed(7)
    minibatches = [[int(i) for i in batch] for batch in buffer.iter_minibatches(shuffle=True)]
    return {
        "returns": [float(v) for v in buffer.returns],
        "advantages": [float(v) for v in buffer.advantages],
        "minibatches": minibatches,
        "fulfillment_summary": buffer.fulfillment_summary(),
        "episode_summary": {k: v for k, v in buffer.episode_summary().items() if "seconds" not in k},
    }


def main(argv: list[str]) -> None:
    out_path, config_path, instance_path, *model_types = argv
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    train_paras = dict(config["train_paras"])
    train_paras.update(TRAIN_OVERRIDES)

    results: dict = {}
    spec = parse_instance_file(instance_path)
    results["L1_instance"] = probe_instance(spec)
    trajectory, fixtures = probe_simulator(spec, train_paras)
    results["L2_trajectory"] = trajectory
    for model_type in model_types:
        results[f"L3L4_{model_type}"] = probe_model(config, train_paras, model_type, fixtures)
    results["L5_buffer"] = probe_buffer()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, out_path)
    print(f"probe written: {out_path} ({len(trajectory)} trajectory steps)")


if __name__ == "__main__":
    main(sys.argv[1:])
