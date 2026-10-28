"""Typed configuration: YAML overlays -> dataclasses -> legacy parameter dictionaries.

Three files make up the default configuration (``configs/instance.yaml``,
``configs/env.yaml``, ``configs/algo.yaml``); further files overlay them in
the order given (``configs/ablation/*.yaml``, ``configs/smoke.yaml`` ...).

The training code historically consumed two flat dictionaries,
``train_paras`` and ``model_paras``.  ``to_legacy_config`` reproduces them key
for key (same values, same types) so that the numerical behaviour of the
trainer is untouched; ``tests/test_config_bridge.py`` checks this against the
frozen ``tests/golden/legacy_config.json``.
"""
from __future__ import annotations

import copy
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
DEFAULT_CONFIG_FILES = ("instance.yaml", "env.yaml", "algo.yaml")


# ----------------------------------------------------------------- dataclasses

@dataclass
class GeneratorCfg:
    """Synthetic-instance generator settings (data/generator.py). v2 defaults follow the statistics of the 15
    benchmark files (docs/experiment-spec.md), not the paper's Table 5 (inter-arrival [1, 200] s there)."""
    n_products: int = 5
    n_stages: int = 5
    machines_per_stage: int = 5
    process_time_range: list = field(default_factory=lambda: [25, 450])
    inter_arrival_range: list = field(default_factory=lambda: [1, 40])
    ddt_range: list = field(default_factory=lambda: [500, 1600])
    slack_rule: str = "ddt_uniform"       # ddt_uniform (one DDT per instance, slack in slack_rel_range x DDT) | per_order
    slack_rel_range: list = field(default_factory=lambda: [0.6, 1.55])
    n_orders_range: list = field(default_factory=lambda: [50, 150])


@dataclass
class InstanceCfg:
    problem: str = "dffsp_hfoi"
    paradigm: str = "per_instance"        # per_instance (train = test instance) | generalization (pool -> unseen tiers)
    data_dir: str = "data/instances/benchmark"
    patterns: list = field(default_factory=lambda: ["DDT*.txt", "num*.txt", "train_num*.txt"])
    train_instances: list = field(default_factory=lambda: ["DDT500_num100.txt"])
    instance_selection: str = "fixed"
    val_data_dir: str = "data/instances/val"   # validation instances (generalization: model selection)
    val_instances: list = field(default_factory=list)   # explicit files; empty = every instance of val_data_dir
    eval_tiers: list = field(default_factory=lambda: ["benchmark"])   # tiers evaluated by run_07 (generalization)
    pools: dict = field(default_factory=dict)  # generated tiers (data/generator.py --pool); see configs/instance.yaml
    generator: GeneratorCfg = field(default_factory=GeneratorCfg)


@dataclass
class EnvCfg:
    discard_penalty: float = 1.0
    progress_reward_coeff: float = 0.0
    order_top_k: int = 5
    fluid_mode: str = "cached_lp"
    fluid_backend: str = "auto"           # auto | gurobi | scipy | heuristic (LP solver of cached_lp)
    fluid_features: str = "on"            # on | off: the two fluid node features
    fluid_mask: bool = True               # candidate pairs restricted to u* > fluid_mask_threshold (with fallback)
    fluid_mask_threshold: float = 1.0e-5
    fluid_recompute: str = "decision"     # decision (every new state) | event (legacy: after arrivals / discards only)
    fluid_time_limit: float = 2
    fluid_threads: int = 0
    collector_workers: int = 0            # 0 = simulators in the trainer process; N > 0 = N worker processes
    collector_start_method: str = "spawn"  # spawn | fork | forkserver (spawn is forced when CUDA is used)


@dataclass
class ModelCfg:
    in_size_ma: int = 5
    out_size_ma: int = 16
    in_size_ope: int = 12
    out_size_ope: int = 16
    hidden_size_ope: int = 256
    num_heads: list = field(default_factory=lambda: [1, 1])
    dropout: float = 0.0
    n_latent_actor: int = 128
    n_latent_critic: int = 128
    n_hidden_actor: int = 3
    n_hidden_critic: int = 3
    action_dim: int = 1
    epsilon: float = 1.0


@dataclass
class AttentionCfg:
    use_sparse_attention: bool = True
    sparse_attention_scope: str = "pair"
    sparse_attention_heads: int = 1
    sparse_attention_dropout: float = 0.0


@dataclass
class TrainCfg:
    lr: float = 0.0003
    min_lr: float = 0.0003
    lr_schedule: str = "none"
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    gamma: float = 1.0
    gae_lambda: float = 0.95
    K_epochs: int = 3
    eps_clip: float = 0.2
    A_coeff: float = 1.0
    vf_coeff: float = 0.5
    entropy_coeff: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    max_iterations: int = 2000
    minibatch_size: int = 512
    max_minibatch_tokens: int = 0
    num_envs: int = 8
    num_workers: int = 8
    rollout_device: str = "cpu"
    update_device: str = "cuda:0"
    resume: str = "auto"
    checkpoint_interval: int = 1
    experiment_name: str = "HGNNPPO-NF"
    eval_every: int = 1
    checkpoint_every: int = 0
    eval_scope: str = "all"               # curve.csv greedy evaluation on all training instances | selected (this iteration's)
    val_every: int = 0                    # validation-set evaluation every N iterations (0 = off); best_val_checkpoint.pt
    early_stop_patience: int = 0          # stop when the best validation mean is older than N iterations (0 = off)


@dataclass
class OffPolicyCfg:
    """Replay / target-network settings of the DQN and DDQN baselines (ignored by PPO and A2C)."""
    replay_capacity: int = 50000
    replay_batch_size: int = 128
    updates_per_iteration: int = 50
    warmup_transitions: int = 500
    target_update_steps: int = 50
    target_tau: float = 0.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_iterations: int = 500
    td_loss: str = "mse"


@dataclass
class VizCfg:
    viz: bool = True
    viz_name: str = "HGNN_PPO_Training"
    visdom_server: str = "http://localhost"
    visdom_port: int = 8097
    viz_update_interval: int = 1
    viz_metrics: list = field(default_factory=list)


@dataclass
class AlgoCfg:
    algorithm: str = "ppo"
    model_type: str = "hgnn"
    train_confirmed: bool = True
    train_version: int = 1
    model: ModelCfg = field(default_factory=ModelCfg)
    attention: AttentionCfg = field(default_factory=AttentionCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    viz: VizCfg = field(default_factory=VizCfg)
    offpolicy: OffPolicyCfg = field(default_factory=OffPolicyCfg)


@dataclass
class Config:
    instance: InstanceCfg = field(default_factory=InstanceCfg)
    env: EnvCfg = field(default_factory=EnvCfg)
    algo: AlgoCfg = field(default_factory=AlgoCfg)


# ------------------------------------------------------------------- loading

def _deep_merge(base: dict, overlay: dict) -> dict:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _from_dict(cls, data: dict, path: str = ""):
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise TypeError(f"config section '{path or cls.__name__}' must be a mapping, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise KeyError(f"unknown config key(s) under '{path or cls.__name__}': {unknown}")
    kwargs = {}
    for name, spec in known.items():
        if name not in data:
            continue
        value = data[name]
        if spec.default_factory is not MISSING:  # type: ignore[comparison-overlap]
            default = spec.default_factory()  # type: ignore[misc]
        else:
            default = spec.default
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value, f"{path}.{name}" if path else name)
        kwargs[name] = value
    return cls(**kwargs)


def resolve_config_paths(paths: Iterable[str | Path] | None) -> list[Path]:
    resolved = []
    for item in (paths or DEFAULT_CONFIG_FILES):
        path = Path(item)
        if not path.is_absolute():
            candidate = CONFIG_DIR / path if (CONFIG_DIR / path).exists() and not path.exists() else ROOT / path
            path = candidate if candidate.exists() else path
        resolved.append(path)
    return resolved


def load_config_dict(paths: Iterable[str | Path] | None = None) -> dict:
    merged: dict = {}
    for path in resolve_config_paths(paths):
        with Path(path).open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}
        unknown = sorted(set(data) - {"instance", "env", "algo"})
        if unknown:
            raise KeyError(f"{path}: top-level keys must be instance/env/algo, got {unknown}")
        merged = _deep_merge(merged, data)
    return merged


def load_config(paths: Iterable[str | Path] | None = None) -> Config:
    """Merge the YAML files in order and build the typed configuration."""
    return _from_dict(Config, load_config_dict(paths))


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


# -------------------------------------------------------------------- bridge

def legacy_train_paras(cfg: Config) -> dict:
    """The flat ``train_paras`` dictionary the trainer, simulator, PPO and visualiser read."""
    paras: dict = {}
    paras.update(asdict(cfg.algo.train))
    paras.update(asdict(cfg.env))
    paras.update(asdict(cfg.algo.attention))
    paras.update(asdict(cfg.algo.viz))
    paras.update(asdict(cfg.algo.offpolicy))
    paras.update({
        "train_data_dir": cfg.instance.data_dir,
        "train_patterns": list(cfg.instance.patterns),
        "train_instances": list(cfg.instance.train_instances),
        "instance_selection": cfg.instance.instance_selection,
        "paradigm": cfg.instance.paradigm,
        "val_data_dir": cfg.instance.val_data_dir,
        "val_instances": list(cfg.instance.val_instances),
        "eval_tiers": list(cfg.instance.eval_tiers),
        "model_type": cfg.algo.model_type,
        "algorithm": cfg.algo.algorithm,
        "train_confirmed": bool(cfg.algo.train_confirmed),
        "train_version": int(cfg.algo.train_version),
    })
    return paras


def to_legacy_config(cfg: Config) -> dict:
    """Legacy-shaped dictionary: ``{"model_paras": ..., "train_paras": ...}``."""
    return {"model_paras": asdict(cfg.algo.model), "train_paras": legacy_train_paras(cfg)}


def load_legacy_config(paths: Iterable[str | Path] | None = None) -> dict:
    return to_legacy_config(load_config(paths))
