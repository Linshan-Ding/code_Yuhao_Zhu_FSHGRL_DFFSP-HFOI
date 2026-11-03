"""The YAML configuration and the flat parameter dictionaries that the trainer consumes agree key for key."""
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config, to_legacy_config  # noqa: E402
from train import build_model_paras  # noqa: E402

# every train_paras key the code reads (train.py, rollout.py, environment/simulator.py, agent/ppo.py, result/logger.py)
TRAIN_KEYS = [
    "lr", "min_lr", "lr_schedule", "betas", "gamma", "gae_lambda", "K_epochs", "eps_clip", "A_coeff",
    "vf_coeff", "entropy_coeff", "max_grad_norm", "target_kl", "max_iterations", "minibatch_size",
    "max_minibatch_tokens", "num_envs", "num_workers", "rollout_device", "update_device", "resume",
    "checkpoint_interval", "experiment_name", "train_patterns", "train_instances",
    "instance_selection", "discard_penalty", "progress_reward_coeff", "order_top_k", "fluid_mode",
    "fluid_time_limit", "fluid_threads", "use_sparse_attention", "sparse_attention_scope",
    "sparse_attention_heads", "sparse_attention_dropout", "viz_name", "visdom_server",
    "visdom_port", "viz_update_interval", "viz_metrics",
]
MODEL_KEYS = [
    "in_size_ma", "out_size_ma", "in_size_ope", "out_size_ope", "hidden_size_ope", "num_heads", "dropout",
    "n_latent_actor", "n_latent_critic", "n_hidden_actor", "n_hidden_critic", "action_dim", "epsilon",
]
DERIVED_KEYS = ["actor_in_dim", "critic_in_dim", "use_sparse_attention", "sparse_attention_scope",
                "sparse_attention_heads", "sparse_attention_dropout"]
TRIO = ["instance.yaml", "env.yaml", "algo.yaml"]


def _same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def test_train_paras_follow_the_yaml_values():
    cfg = load_config()
    raw = asdict(cfg)
    bridged = to_legacy_config(cfg)
    train = bridged["train_paras"]
    for key in TRAIN_KEYS:
        assert key in train, f"missing train key {key}"
    for key in ("lr", "gamma", "gae_lambda", "K_epochs", "eps_clip", "max_iterations", "minibatch_size", "num_envs",
                "target_kl", "entropy_coeff"):
        assert _same(train[key], raw["algo"]["train"][key]), key
    for key in ("discard_penalty", "progress_reward_coeff", "order_top_k", "fluid_mode", "fluid_time_limit"):
        assert _same(train[key], raw["env"][key]), key
    assert train["max_iterations"] == 1000 and train["num_envs"] == 8 and train["order_top_k"] == 5
    assert train["instance_selection"] == "round_robin" and train["train_instances"] == []
    assert train["experiment_name"] == "ours" and bridged["train_paras"]["train_data_dir"] == "data/instances/train"


def test_model_paras_follow_the_yaml_values():
    cfg = load_config()
    raw = asdict(cfg)
    bridged = to_legacy_config(cfg)
    for key in MODEL_KEYS:
        assert _same(bridged["model_paras"][key], raw["algo"]["model"][key]), key
    paras = build_model_paras(bridged, torch.device("cpu"), bridged["train_paras"])
    for key in MODEL_KEYS + DERIVED_KEYS:
        assert key in paras, key
    assert paras["in_size_ope"] == 12 and paras["in_size_ma"] == 5 and paras["out_size_ope"] == 16


def test_ablation_overlays_change_only_their_keys():
    base = to_legacy_config(load_config(TRIO))
    nf = to_legacy_config(load_config(TRIO + ["ablation/nf.yaml"]))
    assert nf["train_paras"]["fluid_features"] == "off" and nf["train_paras"]["fluid_mask"] is False
    ng = to_legacy_config(load_config(TRIO + ["ablation/ng.yaml"]))
    assert ng["train_paras"]["model_type"] == "mlp"
    for paras in (nf, ng):
        for key in TRAIN_KEYS:
            if key in ("fluid_mode", "fluid_features", "fluid_mask", "experiment_name"):
                continue
            assert _same(paras["train_paras"][key], base["train_paras"][key]), key


def test_data_dir_resolves_the_training_pool():
    """The training pool holds the 1000 generated instances, every one used once (Section 5.1)."""
    from data.dataset import resolve_instance_files

    bridged = to_legacy_config(load_config())
    files = resolve_instance_files(str(ROOT / bridged["train_paras"]["train_data_dir"]),
                                   bridged["train_paras"]["train_patterns"], None)
    assert len(files) == 1000 and all(path.name.startswith("TRN") for path in files)
