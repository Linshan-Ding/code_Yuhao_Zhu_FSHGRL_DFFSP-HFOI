"""Agent and policy-network construction by ``algorithm`` (new code).

``build_agent`` returns the PPO agent for ``algorithm: ppo`` (the same object the original trainer built,
so the PPO path is unchanged) and the learned baselines otherwise; ``build_policy_network`` returns the
bare network for evaluation tools. Algorithms: ppo (default), a2c, dqn, ddqn, hsddqn (hierarchical double DQN,
v2; the HGNN-PPO comparator is ``ppo`` under configs/baselines/hgnn_ppo.yaml).
"""
from __future__ import annotations

ALGORITHMS = ("ppo", "a2c", "dqn", "ddqn", "hsddqn")
OFF_POLICY = ("dqn", "ddqn", "hsddqn")


def algorithm_of(train_paras: dict) -> str:
    algorithm = str(train_paras.get("algorithm", "ppo")).lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}, got '{algorithm}'")
    return algorithm


def build_agent(model_paras: dict, train_paras: dict):
    algorithm = algorithm_of(train_paras)
    if algorithm == "ppo":
        from agent.ppo import PPO

        return PPO(model_paras, train_paras)
    if algorithm == "a2c":
        from agent.baselines.a2c import A2C

        return A2C(model_paras, train_paras)
    if algorithm == "hsddqn":
        from agent.baselines.hsddqn import HSDDQNAgent

        return HSDDQNAgent(model_paras, train_paras)
    from agent.baselines.dqn import DQNAgent

    return DQNAgent(model_paras, train_paras, double=(algorithm == "ddqn"))


def build_policy_network(model_paras: dict, train_paras: dict):
    algorithm = algorithm_of(train_paras)
    if algorithm in ("ppo", "a2c"):
        from agent.ppo import _resolve_scheduler_class

        return _resolve_scheduler_class(train_paras)(model_paras)
    if algorithm == "hsddqn":
        from agent.baselines.hsddqn import HierarchicalQScheduler

        return HierarchicalQScheduler(model_paras)
    from agent.baselines.dqn import QScheduler

    return QScheduler(model_paras)
