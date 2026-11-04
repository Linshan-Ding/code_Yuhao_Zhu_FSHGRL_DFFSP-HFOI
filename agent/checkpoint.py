"""Checkpoint helpers ."""
from __future__ import annotations

from pathlib import Path

import torch

from agent.ppo import PPO


def save_checkpoint(path: Path, model: PPO, iteration: int, best_fulfillment_rate: float,
                    train_paras: dict, config: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "iteration": int(iteration),
        "best_fulfillment_rate": float(best_fulfillment_rate),
        "policy": model.policy.state_dict(),
        "policy_old": model.policy_old.state_dict(),
        "optimizer": model.optimizer.state_dict(),
        "train_paras": train_paras,
        "config": config,
    }, path)


def load_checkpoint(path: Path, model: PPO, device: torch.device) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.policy.load_state_dict(checkpoint["policy"])
    model.policy_old.load_state_dict(checkpoint.get("policy_old", checkpoint["policy"]))
    if "optimizer" in checkpoint:
        model.optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint.get("iteration", -1)) + 1, float(checkpoint.get("best_fulfillment_rate", float("-inf")))


# ----------------------------------------------------------------- new code (eval.py / tools)

def load_policy_state(path: Path, device) -> dict:
    """Read a trainer checkpoint (dict with ``policy``) or a bare policy state dict (ppo_policy_model.pt).

    Returns ``{"policy", "config", "train_paras", "iteration", "best_fulfillment_rate"}``; the last four are
    None for a bare state dict, in which case the caller must supply the configuration.
    """
    data = torch.load(Path(path), map_location=device, weights_only=False)
    if isinstance(data, dict) and isinstance(data.get("policy"), dict):
        return {
            "policy": data["policy"],
            "config": data.get("config"),
            "train_paras": data.get("train_paras"),
            "iteration": data.get("iteration"),
            "best_fulfillment_rate": data.get("best_fulfillment_rate"),
        }
    return {"policy": data, "config": None, "train_paras": None, "iteration": None, "best_fulfillment_rate": None}


def build_scheduler(config: dict, train_paras: dict, device) -> torch.nn.Module:
    """Construct the (untrained) policy network described by a legacy-shaped config and train_paras."""
    from agent.factory import build_policy_network
    from train import build_model_paras

    model_paras = build_model_paras(config, torch.device(device), train_paras)
    network = build_policy_network(model_paras, train_paras)
    if hasattr(network, "set_device"):
        network.set_device(torch.device(device))
        return network
    return network.to(torch.device(device))


def load_policy(path: Path, device, config: dict | None = None, train_paras: dict | None = None):
    """Policy network with the weights of ``path`` in eval mode, plus the effective config/train_paras.

    ``config``/``train_paras`` override what the checkpoint stores (needed for bare state dicts; otherwise
    pass them only to evaluate under different simulator settings than the ones used in training).
    """
    bundle = load_policy_state(path, device)
    config = config or bundle["config"]
    train_paras = dict(train_paras or bundle["train_paras"] or {})
    if config is None:
        raise ValueError(f"{path} is a bare state dict; pass the configuration (--config) to rebuild the network")
    policy = build_scheduler(config, train_paras, device)
    policy.load_state_dict(bundle["policy"])
    policy.eval()
    return policy, config, train_paras, bundle
