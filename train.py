import argparse
import copy
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.checkpoint import load_checkpoint, save_checkpoint
from agent.factory import OFF_POLICY, algorithm_of, build_agent
from agent.ppo import PPO
from configs.config import DEFAULT_CONFIG_FILES, load_config as load_yaml_config, to_legacy_config
from data.dataset import resolve_instance_files, select_instance
from data.instance_io import parse_instance_file
from result.logger import RunLogger, TrainingVisualizer, git_commit_hash, write_metrics  # noqa: F401  (re-exported)
from result.metrics import gap_pct, gpu_utilization, load_reference, objective_from_fulfillment
from rollout import collect_vectorized_rollouts, collect_vectorized_rollouts_parallel, run_episodes


def load_config(config_paths) -> dict:
    """Return the legacy-shaped config dict ({"model_paras", "train_paras"}).

    Accepts one legacy JSON file or one or more YAML overlays (default:
    configs/instance.yaml, configs/env.yaml, configs/algo.yaml).
    """
    if config_paths is None:
        return to_legacy_config(load_yaml_config())
    paths = [Path(item) for item in config_paths]
    if len(paths) == 1 and paths[0].suffix.lower() == ".json":
        with paths[0].open("r", encoding="utf-8") as file:
            return json.load(file)
    return to_legacy_config(load_yaml_config(paths))


SMOKE_OVERRIDES = {"max_iterations": 2, "num_envs": 2, "num_workers": 2, "minibatch_size": 128, "viz": False,
                   "updates_per_iteration": 10, "warmup_transitions": 50}   # the last two: DQN / DDQN only


def resolve_data_dir(path_value: str) -> Path:
    data_dir = Path(path_value)
    if not data_dir.is_absolute():
        data_dir = ROOT_DIR / data_dir
    return data_dir


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_instances(values):
    if not values:
        return None
    instances = []
    for value in values:
        instances.extend(item.strip() for item in value.split(",") if item.strip())
    return instances


def build_model_paras(config: dict, device, train_paras: dict | None = None):
    model_paras = dict(config["model_paras"])
    train_paras = train_paras or config.get("train_paras", {})
    model_paras["device"] = device
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
    model_paras["use_sparse_attention"] = bool(train_paras.get("use_sparse_attention", False))
    model_paras["sparse_attention_scope"] = str(train_paras.get("sparse_attention_scope", "pair"))
    model_paras["sparse_attention_heads"] = int(train_paras.get("sparse_attention_heads", 1))
    model_paras["sparse_attention_dropout"] = float(train_paras.get("sparse_attention_dropout", 0.0))
    return model_paras


def resolve_torch_device(name: str | None, fallback: str = "cpu") -> torch.device:
    requested = str(name or fallback)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[Device] CUDA requested as '{requested}' but is unavailable; falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def prepare_rollout_policy(model: PPO, rollout_device: torch.device, update_device: torch.device):
    if rollout_device == update_device:
        model.policy_old.eval()
        return model.policy_old
    policy = copy.deepcopy(model.policy_old).to(rollout_device)
    policy.device = rollout_device
    policy.eval()
    return policy



def gpu_temperature() -> float:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        )
        first_line = output.strip().splitlines()[0]
        return float(first_line)
    except Exception:
        return 0.0


def resource_snapshot(update_device: torch.device) -> dict:
    cpu_memory_mb = 0.0
    try:
        import psutil

        cpu_memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        pass

    gpu_memory_mb = 0.0
    gpu_temp = 0.0
    if update_device.type == "cuda" and torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated(update_device) / (1024 ** 2)
        gpu_temp = gpu_temperature()
    return {
        "cpu_memory_mb": cpu_memory_mb,
        "gpu_memory_mb": gpu_memory_mb,
        "gpu_temperature": gpu_temp,
    }


def apply_lr_schedule(model: PPO, train_paras: dict, iteration: int, max_iterations: int) -> float:
    schedule = str(train_paras.get("lr_schedule", "none")).lower()
    base_lr = float(train_paras.get("lr", 3e-4))
    min_lr = float(train_paras.get("min_lr", base_lr))
    if schedule == "linear" and max_iterations > 1:
        progress = min(max(iteration / (max_iterations - 1), 0.0), 1.0)
        lr = base_lr + progress * (min_lr - base_lr)
    elif schedule in {"none", ""}:
        lr = base_lr
    else:
        raise ValueError("lr_schedule must be 'none' or 'linear'")
    model.set_learning_rate(lr)
    return lr


def evaluate_fixed_set(policy, instance_files, train_paras: dict, reference: dict) -> list:
    """Greedy rollout of the current policy on every training instance (the curve.csv sampling protocol).

    New code: one complete episode per instance with argmax decoding under torch.no_grad(); it draws no
    random numbers, so running it after each PPO update leaves the training stream untouched
    (tests/equivalence runs with eval_every=1 and checks bit-identical training results).
    """
    rows = []
    for path in instance_files:
        spec = parse_instance_file(path)
        episode = run_episodes(policy, spec, train_paras, episodes=1, sample=False)[0]
        obj = objective_from_fulfillment(episode["fulfillment_rate"])
        rows.append({
            "instance_id": path.stem,
            "instance_name": path.name,
            "obj": obj,
            "gap": gap_pct(obj, reference.get(path.stem)),
            "feasible": True,
            "fulfillment_rate": episode["fulfillment_rate"],
            "completed": episode["completed"],
            "discarded": episode["discarded"],
            "decision_count": episode["decision_count"],
            "eval_seconds": episode["wall_seconds"],
        })
    return rows


def fluid_backend_of(train_paras: dict, instance_files) -> str:
    """Name of the fluid backend a simulator of this configuration uses (for the snapshot)."""
    try:
        from environment.simulator import TrainingSimulator

        return TrainingSimulator(parse_instance_file(instance_files[0]), train_paras).fluid_backend_used
    except Exception:
        return "unknown"


def paradigm_name(instance_files) -> str:
    return "per_instance" if len(instance_files) == 1 else "fixed_set"


def build_snapshot(args, train_paras: dict, model_paras: dict, instance_files, rollout_device, update_device,
                   start_iteration: int, max_iterations: int) -> dict:
    """Content of result/<run>/config_snapshot.yaml: the effective configuration and provenance."""
    try:
        import gurobipy  # noqa: F401

        gurobi_available = True
    except Exception:
        gurobi_available = False
    return {
        "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
        "config_files": [str(item) for item in (args.config or DEFAULT_CONFIG_FILES)],
        "seed": args.seed,
        "paradigm": paradigm_name(instance_files),
        "instance_selection": train_paras.get("instance_selection", "fixed"),
        "instances": [path.name for path in instance_files],
        "algorithm": train_paras.get("algorithm", "ppo"),
        "model_type": train_paras.get("model_type", "hgnn"),
        "train_version": train_paras.get("train_version"),
        "start_iteration": start_iteration,
        "max_iterations": max_iterations,
        "rollout_device": str(rollout_device),
        "update_device": str(update_device),
        "git_commit": git_commit_hash(ROOT_DIR),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gurobipy_available": gurobi_available,
        "fluid_backend": fluid_backend_of(train_paras, instance_files),
        "model_paras": model_paras,
        "train_paras": train_paras,
    }


def finalize_run(logger: RunLogger, stop_reason: str, start_iteration: int, last_iteration: int,
                 best_fulfillment_rate: float, best_greedy_obj: float, best_greedy_iteration: int,
                 train_paras: dict, instance_files) -> None:
    """Write fixed_set_final.csv and train_meta.yaml (called from the trainer's finally block)."""
    wall_time_s = round(logger.wall_time(), 3)
    logger.write_final(epochs=last_iteration + 1, wall_time_s=wall_time_s)
    logger.write_meta({
        "stop_reason": stop_reason,
        "epochs_run": max(last_iteration - start_iteration + 1, 0),
        "start_iteration": start_iteration,
        "last_iteration": last_iteration,
        "wall_time_s": wall_time_s,
        "best_sampled_fulfillment_rate": best_fulfillment_rate if math.isfinite(best_fulfillment_rate) else None,
        "best_greedy_obj": best_greedy_obj if math.isfinite(best_greedy_obj) else None,
        "best_greedy_iteration": best_greedy_iteration if best_greedy_iteration >= 0 else None,
        "paradigm": paradigm_name(instance_files),
        "eval_set": "train_set",
        "n_instances": len(instance_files),
        "train_version": train_paras.get("train_version"),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })


def main():
    parser = argparse.ArgumentParser(description="Parallel HGNN-PPO trainer")
    parser.add_argument("--config", nargs="+", default=None,
                        help="YAML overlays (default: configs/instance.yaml env.yaml algo.yaml) or one legacy JSON file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rollout-device", default=None)
    parser.add_argument("--update-device", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--experiment-name", "--run-name", dest="experiment_name", default=None)
    parser.add_argument("--smoke", action="store_true", help="2 iterations, 2 environments, no Visdom")
    parser.add_argument("--instances", nargs="*", default=None)
    parser.add_argument("--instance-selection", choices=["fixed", "round_robin", "random"], default=None)
    parser.add_argument("--resume", choices=["auto", "none"], default=None)
    parser.add_argument("--collector-workers", type=int, default=None,
                        help="simulators in N worker processes during collection (0 = in-process, the original path)")
    args = parser.parse_args()

    config = load_config(args.config)
    train_paras = dict(config["train_paras"])
    if args.smoke:
        train_paras.update(SMOKE_OVERRIDES)
    if args.max_iterations is not None:
        train_paras["max_iterations"] = args.max_iterations
    if args.instance_selection is not None:
        train_paras["instance_selection"] = args.instance_selection
    if args.rollout_device is not None:
        train_paras["rollout_device"] = args.rollout_device
    if args.update_device is not None:
        train_paras["update_device"] = args.update_device
    if args.resume is not None:
        train_paras["resume"] = args.resume
    if args.collector_workers is not None:
        train_paras["collector_workers"] = int(args.collector_workers)
    if not train_paras.get("train_confirmed", True) and not args.smoke:
        raise SystemExit("[FAIL] the training-parameter table is not confirmed: review configs/algo.yaml "
                         "(algo.train_confirmed) or pass --smoke")
    set_seed(args.seed)

    update_device_name = args.device or train_paras.get("update_device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    rollout_device_name = train_paras.get("rollout_device", "cpu")
    update_device = resolve_torch_device(update_device_name, fallback="cpu")
    rollout_device = resolve_torch_device(rollout_device_name, fallback="cpu")
    data_dir = resolve_data_dir(args.data_dir or train_paras.get("train_data_dir", "data/instances/benchmark"))
    patterns = train_paras.get("train_patterns", ["*.txt"])
    if isinstance(patterns, str):
        patterns = [patterns]
    requested_instances = parse_instances(args.instances)
    if requested_instances is None:
        config_instances = train_paras.get("train_instances") or None
        requested_instances = parse_instances([config_instances]) if isinstance(config_instances, str) else config_instances
    instance_files = resolve_instance_files(str(data_dir), patterns, requested_instances)

    if algorithm_of(train_paras) in OFF_POLICY:          # DQN / DDQN: replay-based loop (new code)
        from agent.baselines.offpolicy import train_offpolicy

        if int(train_paras.get("collector_workers", 0) or 0) > 0:
            print("[Collector] dqn / ddqn keep the in-process collector; collector_workers is ignored")
        train_offpolicy(args, config, train_paras, instance_files, rollout_device, update_device)
        return

    model_paras = build_model_paras(config, update_device, train_paras)
    model = build_agent(model_paras, train_paras)        # PPO for algorithm ppo (unchanged object), A2C otherwise
    exp_name = args.experiment_name or train_paras.get("experiment_name", "default")
    result_dir = ROOT_DIR / "result"/ exp_name
    result_dir.mkdir(parents=True, exist_ok=True)
    last_checkpoint_path = result_dir / "last_checkpoint.pt"
    best_checkpoint_path = result_dir / "best_checkpoint.pt"
    best_greedy_checkpoint_path = result_dir / "best_greedy_checkpoint.pt"
    best_fulfillment_rate = float("-inf")
    trajectory_base = 0
    start_iteration = 0
    if train_paras.get("resume", "auto") == "auto" and last_checkpoint_path.exists():
        try:
            start_iteration, best_fulfillment_rate = load_checkpoint(last_checkpoint_path, model, update_device)
            print(f"[Checkpoint] Resumed from iteration {start_iteration}; best fulfillment={best_fulfillment_rate:.6f}")
        except Exception as exc:
            print(f"[Checkpoint] Could not resume from {last_checkpoint_path}: {exc}. Starting fresh.")

    max_iterations = int(train_paras.get("max_iterations", 1000))
    logger = RunLogger(result_dir, train_paras, resume=start_iteration > 0)
    reference = load_reference()
    logger.write_snapshot(build_snapshot(args, train_paras, model_paras, instance_files, rollout_device,
                                         update_device, start_iteration, max_iterations))
    eval_every = int(train_paras.get("eval_every", 1))
    checkpoint_every = int(train_paras.get("checkpoint_every", 0))
    best_greedy_obj = float("inf")
    best_greedy_iteration = -1
    last_iteration = start_iteration - 1
    stop_reason = "running"
    collector = None
    collector_workers = int(train_paras.get("collector_workers", 0) or 0)
    if collector_workers > 0:                            # simulators in worker processes (new code; bit-identical)
        from environment.collector import CollectorPool

        start_method = str(train_paras.get("collector_start_method", "spawn") or "spawn")
        if update_device.type == "cuda" and start_method != "spawn":
            print(f"[Collector] start method '{start_method}' is unsafe after CUDA initialisation; using spawn")
            start_method = "spawn"
        collector = CollectorPool(int(train_paras.get("num_envs", train_paras.get("num_workers", 4))),
                                  collector_workers, start_method)
        print(f"[Collector] {collector.workers} worker process(es) for {collector.num_envs} environments "
              f"({start_method}, start-up {collector.startup_seconds:.1f} s)")
    try:
        for iteration in range(start_iteration, max_iterations):
            if update_device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(update_device)
            start_time = time.time()
            learning_rate = apply_lr_schedule(model, train_paras, iteration, max_iterations)
            instance_selection = train_paras.get("instance_selection", "fixed")
            instance_index, selected_instance = select_instance(instance_files, instance_selection, iteration)
            rollout_start_time = time.time()
            rollout_policy = prepare_rollout_policy(model, rollout_device, update_device)
            if collector is None:
                rollout = collect_vectorized_rollouts(rollout_policy, selected_instance, train_paras, trajectory_base)
            else:
                rollout = collect_vectorized_rollouts_parallel(rollout_policy, selected_instance, train_paras,
                                                               trajectory_base, pool=collector)
            rollout_seconds = time.time() - rollout_start_time
            num_envs = int(train_paras.get("num_envs", train_paras.get("num_workers", 4)))
            trajectory_base += num_envs
            rollout.minibatch_size = int(train_paras.get("minibatch_size", 512))
            update_start_time = time.time()
            rollout.compute_returns_and_advantages(
                gamma=float(train_paras.get("gamma", 1.0)),
                gae_lambda=float(train_paras.get("gae_lambda", 0.95)),
            )
            update_metrics = model.update_rollout(rollout)
            update_seconds = time.time() - update_start_time
            elapsed = time.time() - start_time
            resources = resource_snapshot(update_device)
            total_reward = float(sum(rollout.rewards))
            steps_per_second = len(rollout) / max(elapsed, 1e-8)
            mean_return = float(np.mean(rollout.returns)) if rollout.returns else 0.0
            loss = float(update_metrics.get("loss", 0.0))
            fulfillment_metrics = rollout.fulfillment_summary()
            episode_metrics = rollout.episode_summary()
            row = {
                "iteration": iteration,
                "instance_id": instance_index + 1,
                "instance_name": selected_instance.name,
                "instance_selection": instance_selection,
                "transitions": len(rollout),
                "episode_count": episode_metrics["episode_count"],
                "episode_transition_mean": episode_metrics["episode_transition_mean"],
                "episode_transition_std": episode_metrics["episode_transition_std"],
                "episode_seconds_mean": episode_metrics["episode_seconds_mean"],
                "episode_seconds_std": episode_metrics["episode_seconds_std"],
                "reward": total_reward,
                "mean_return": mean_return,
                "fulfillment_rate_mean": fulfillment_metrics["fulfillment_rate_mean"],
                "fulfillment_rate_std": fulfillment_metrics["fulfillment_rate_std"],
                "fulfillment_rate_min": fulfillment_metrics["fulfillment_rate_min"],
                "fulfillment_rate_max": fulfillment_metrics["fulfillment_rate_max"],
                "loss": loss,
                "policy_loss": float(update_metrics.get("policy_loss", 0.0)),
                "value_loss": float(update_metrics.get("value_loss", 0.0)),
                "entropy": float(update_metrics.get("entropy", 0.0)),
                "approx_kl": float(update_metrics.get("approx_kl", 0.0)),
                "clip_fraction": float(update_metrics.get("clip_fraction", 0.0)),
                "learning_rate": learning_rate,
                "discard_count": rollout.discard_count,
                "completion_count": rollout.completion_count,
                "steps_per_second": steps_per_second,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "elapsed_seconds": elapsed,
                "fluid_solve_count": rollout.fluid_solve_count,
                "fluid_solve_seconds": rollout.fluid_solve_seconds,
                "fluid_cache_hit_count": rollout.fluid_cache_hit_count,
                "fluid_fallback_count": rollout.fluid_fallback_count,
                "sparse_attention_enabled": bool(train_paras.get("use_sparse_attention", False)),
                "sparse_attention_scope": str(train_paras.get("sparse_attention_scope", "pair")),
                "policy_forward_count": rollout.policy_forward_count,
                "policy_forward_seconds": rollout.policy_forward_seconds,
                "avg_action_tokens": rollout.avg_action_tokens,
                "max_action_tokens": rollout.max_action_tokens,
                "padding_ratio": rollout.padding_ratio,
                "actor_batch_size_mean": rollout.actor_batch_size_mean,
                "policy_act_seconds": rollout.policy_forward_seconds,
                "simulator_step_seconds": rollout.simulator_step_seconds,
                "obs_build_seconds": rollout.obs_build_seconds,
                "rollout_device": str(rollout_device),
                "update_device": str(update_device),
                "cpu_memory_mb": resources["cpu_memory_mb"],
                "gpu_memory_mb": resources["gpu_memory_mb"],
                "gpu_temperature": resources["gpu_temperature"],
            }
            logger.log_legacy(iteration, row)

            if fulfillment_metrics["fulfillment_rate_mean"] >= best_fulfillment_rate:
                best_fulfillment_rate = fulfillment_metrics["fulfillment_rate_mean"]
                save_checkpoint(best_checkpoint_path, model, iteration, best_fulfillment_rate, train_paras, config)
                torch.save(model.policy.state_dict(), result_dir / "ppo_policy_model.pt")
            checkpoint_interval = max(int(train_paras.get("checkpoint_interval", 1)), 1)
            if iteration % checkpoint_interval == 0:
                save_checkpoint(last_checkpoint_path, model, iteration, best_fulfillment_rate, train_paras, config)

            # ---- experiment-contract files (new code; the greedy evaluation draws no random numbers) ----
            logger.env_steps += len(rollout)
            fixed_obj_mean = fixed_gap_mean = fixed_fulfillment_mean = eval_seconds = ""
            if eval_every > 0 and iteration % eval_every == 0:
                eval_start_time = time.time()
                eval_policy = prepare_rollout_policy(model, rollout_device, update_device)
                saved_forward_stats = getattr(eval_policy, "last_forward_stats", None)
                greedy_rows = evaluate_fixed_set(eval_policy, instance_files, train_paras, reference)
                if saved_forward_stats is not None:
                    eval_policy.last_forward_stats = saved_forward_stats
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
                    save_checkpoint(best_greedy_checkpoint_path, model, iteration, best_fulfillment_rate,
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
                "sps": steps_per_second,
                "gpu_util": gpu_utilization(update_device),
                "gpu_mem_gb": resources["gpu_memory_mb"] / 1024.0 if update_device.type == "cuda" else "",
                "train_fulfillment_mean": fulfillment_metrics["fulfillment_rate_mean"],
                "fixed_fulfillment_mean": fixed_fulfillment_mean,
                "eval_seconds": eval_seconds,
                **rollout.mask_summary(),
            })
            if checkpoint_every > 0 and iteration % checkpoint_every == 0:
                save_checkpoint(result_dir / f"checkpoint_epoch{iteration:04d}.pt", model, iteration,
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
            finalize_run(logger, stop_reason, start_iteration, last_iteration, best_fulfillment_rate,
                         best_greedy_obj, best_greedy_iteration, train_paras, instance_files)
        except Exception as exc:  # never mask the training error
            print(f"[RunLogger] could not write the run summary: {exc}")
        if collector is not None:
            collector.close()
        logger.close()


if __name__ == "__main__":
    main()
