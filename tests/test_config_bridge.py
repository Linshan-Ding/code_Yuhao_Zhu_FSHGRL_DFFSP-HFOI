"""L0: the YAML configuration reproduces the legacy JSON parameters key for key (value and type)."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config import load_config, to_legacy_config  # noqa: E402
from train import build_model_paras  # noqa: E402

LEGACY = json.loads((ROOT / "tests" / "golden" / "legacy_config.json").read_text(encoding="utf-8"))

# every train_paras key the code reads (train.py, rollout.py, environment/simulator.py, agent/ppo.py,
# agent/visualization.py); `model_type` is read with a default in the legacy code. `train_data_dir` is
# compared separately: the instances moved into per-instance folders, so the path differs by design.
TRAIN_KEYS = [
    "lr", "min_lr", "lr_schedule", "betas", "gamma", "gae_lambda", "K_epochs", "eps_clip", "A_coeff",
    "vf_coeff", "entropy_coeff", "max_grad_norm", "target_kl", "max_iterations", "minibatch_size",
    "max_minibatch_tokens", "num_envs", "num_workers", "rollout_device", "update_device", "resume",
    "checkpoint_interval", "experiment_name", "train_patterns", "train_instances",
    "instance_selection", "discard_penalty", "progress_reward_coeff", "order_top_k", "fluid_mode",
    "fluid_time_limit", "fluid_threads", "use_sparse_attention", "sparse_attention_scope",
    "sparse_attention_heads", "sparse_attention_dropout", "viz", "viz_name", "visdom_server",
    "visdom_port", "viz_update_interval", "viz_metrics",
]
MODEL_KEYS = [
    "in_size_ma", "out_size_ma", "in_size_ope", "out_size_ope", "hidden_size_ope", "num_heads", "dropout",
    "n_latent_actor", "n_latent_critic", "n_hidden_actor", "n_hidden_critic", "action_dim", "epsilon",
]
DERIVED_KEYS = ["actor_in_dim", "critic_in_dim", "use_sparse_attention", "sparse_attention_scope",
                "sparse_attention_heads", "sparse_attention_dropout"]


def _same(a, b) -> bool:
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def test_train_paras_match_legacy_json():
    bridged = to_legacy_config(load_config())
    for key in TRAIN_KEYS:
        assert key in bridged["train_paras"], f"missing train key {key}"
        assert _same(bridged["train_paras"][key], LEGACY["train_paras"][key]), (
            key, bridged["train_paras"][key], LEGACY["train_paras"][key])


def test_model_paras_match_legacy_json():
    bridged = to_legacy_config(load_config())
    for key in MODEL_KEYS:
        assert _same(bridged["model_paras"][key], LEGACY["model_paras"][key]), key
    device = torch.device("cpu")
    new_paras = build_model_paras(bridged, device, bridged["train_paras"])
    old_paras = build_model_paras(LEGACY, device, LEGACY["train_paras"])
    for key in MODEL_KEYS + DERIVED_KEYS:
        assert _same(new_paras[key], old_paras[key]), key


def test_ablation_overlays_change_only_their_keys():
    base = to_legacy_config(load_config())
    nf = to_legacy_config(load_config(["instance.yaml", "env.yaml", "algo.yaml", "ablation/nf.yaml"]))
    assert nf["train_paras"]["fluid_features"] == "off" and nf["train_paras"]["fluid_mask"] is False
    ng = to_legacy_config(load_config(["instance.yaml", "env.yaml", "algo.yaml", "ablation/ng.yaml"]))
    assert ng["train_paras"]["model_type"] == "mlp"
    for paras in (nf, ng):
        for key in TRAIN_KEYS:
            if key in ("fluid_mode", "fluid_features", "fluid_mask", "experiment_name"):
                continue
            assert _same(paras["train_paras"][key], base["train_paras"][key]), key


def test_data_dir_resolves_the_benchmark_set():
    """The data directory differs from the legacy JSON by design; the resolved instance set is the same 15 files."""
    from data.dataset import resolve_instance_files

    bridged = to_legacy_config(load_config())
    assert bridged["train_paras"]["train_data_dir"] == "data/instances/benchmark"
    assert LEGACY["train_paras"]["train_data_dir"] == "data/instance/competition"
    files = resolve_instance_files(str(ROOT / bridged["train_paras"]["train_data_dir"]),
                                   bridged["train_paras"]["train_patterns"], None)
    expected = sorted(f"DDT{ddt}_num{n}.txt" for ddt in (500, 600, 900, 1200, 1600) for n in (50, 100, 150))
    assert sorted(path.name for path in files) == expected
