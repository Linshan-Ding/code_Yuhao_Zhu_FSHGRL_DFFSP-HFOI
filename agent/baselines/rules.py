"""Dispatching-rule baselines on the training simulator (new code).

The original ``compare/Rule.py`` scored candidates of the removed competition platform; this module keeps its
scoring directions (``argmax``) and runs on ``environment/simulator.py`` instead, so the rules face exactly
the feasible set the learned policy faces.

Candidate set: the simulator's eligible (machine, operation-type) pairs x the ``order_top_k`` earliest-due
waiting orders of that operation type (set ``env.order_top_k: 0`` to score the whole queue). Scores:

    FIFO   -arrival time of the order                      (earliest arrival first)
    MOR    +remaining operations of the order              (stage_count - stage, current stage included)
    MWKR   +remaining work of the order                    (sum over the remaining stages of the mean
                                                            processing time of the product on that stage)
    random uniform choice among the candidates (private RNG)
    rrc    random rule choice: one score of RRC_POOL drawn per episode (``rrc_mode="episode"``, default) or per
           decision (``"step"``); the pool holds FIFO, MOR, MWKR and the shortest-processing-time score
           (-processing time of (product, machine)), which is used only inside RRC

Ties are broken deterministically by (earlier due date, shorter processing time, lower machine index, lower
operation-type index, lower queue rank), so the chosen action does not depend on the enumeration order.
"""
from __future__ import annotations

import random
import time
from typing import Dict, List, Tuple

import numpy as np

from environment.simulator import TrainingSimulator
from rollout import _episode_record

DETERMINISTIC_RULES = ("fifo", "mor", "mwkr")
RRC_POOL = ("fifo", "spt", "mor", "mwkr")          # scores drawn by RRC (order fixed: it determines the draws)
STOCHASTIC_RULES = ("random", "rrc")
RULES = DETERMINISTIC_RULES + STOCHASTIC_RULES


def mean_stage_process_times(spec) -> np.ndarray:
    """``[product, stage]`` mean processing time over the machines of the stage (zero entries ignored)."""
    table = np.zeros((spec.product_count, spec.stage_count), dtype=np.float64)
    for product in range(spec.product_count):
        for stage, machines in enumerate(spec.stage_machines):
            times = [float(spec.process_times[product, machine]) for machine in machines
                     if float(spec.process_times[product, machine]) > 0]
            table[product, stage] = float(np.mean(times)) if times else 0.0
    return table


def enumerate_candidates(simulator: TrainingSimulator, mean_work: np.ndarray) -> List[dict]:
    """One dict per (machine, operation type, queue rank) the simulator would accept in ``step``."""
    spec = simulator.spec
    grouped = simulator._waiting_orders_by_task()
    candidates = []
    for machine, task in simulator._eligible_pairs(grouped):
        product, stage = spec.task_to_product_stage[task]
        process_time = float(spec.process_times[product, machine])
        remaining_ops = spec.stage_count - stage
        remaining_work = float(mean_work[product, stage:].sum())
        for rank, order in enumerate(grouped.get(task, [])):
            candidates.append({
                "machine": int(machine),
                "task": int(task),
                "rank": int(rank),
                "order": int(order),
                "arrival": float(spec.arrival_times[order]),
                "due_remaining": float(spec.due_dates[order] - simulator.current_time),
                "p": process_time,
                "rem_ops": int(remaining_ops),
                "rem_work": remaining_work,
            })
    return candidates


def rule_score(candidate: dict, rule: str) -> float:
    if rule == "fifo":
        return -candidate["arrival"]
    if rule == "spt":                    # shortest processing time, a score of RRC_POOL only
        return -candidate["p"]
    if rule == "mor":
        return float(candidate["rem_ops"])
    if rule == "mwkr":
        return candidate["rem_work"]
    raise ValueError(f"unknown deterministic rule '{rule}'")


def select_candidate(candidates: List[dict], rule: str) -> dict:
    """Highest score; ties by earlier due date, shorter processing time, lower machine / type / rank."""
    return max(candidates, key=lambda c: (rule_score(c, rule), -c["due_remaining"], -c["p"], -c["machine"],
                                          -c["task"], -c["rank"]))


class RulePolicy:
    def __init__(self, rule: str, seed: int = 0, rrc_mode: str = "episode"):
        rule = str(rule).lower()
        if rule not in RULES:
            raise ValueError(f"unknown rule '{rule}'; choose from {RULES}")
        if rrc_mode not in ("episode", "step"):
            raise ValueError("rrc_mode must be 'episode' or 'step'")
        self.rule = rule
        self.seed = int(seed)
        self.rrc_mode = rrc_mode
        self.rng = random.Random(self.seed)
        self.active_rule = rule if rule in DETERMINISTIC_RULES else None
        self._mean_work = None

    def start_episode(self, spec) -> None:
        self._mean_work = mean_stage_process_times(spec)
        if self.rule == "rrc" and self.rrc_mode == "episode":
            self.active_rule = self.rng.choice(RRC_POOL)

    def choose(self, candidates: List[dict]) -> dict:
        if not candidates:
            raise ValueError("no candidates")
        if self.rule == "random":
            return self.rng.choice(candidates)
        if self.rule == "rrc" and self.rrc_mode == "step":
            self.active_rule = self.rng.choice(RRC_POOL)
        return select_candidate(candidates, self.active_rule)

    def act(self, simulator: TrainingSimulator) -> Tuple[int, int] | None:
        """``(machine * task_count + task, queue rank)`` for ``simulator.step``; None without candidates."""
        if self._mean_work is None:
            self.start_episode(simulator.spec)
        candidates = enumerate_candidates(simulator, self._mean_work)
        if not candidates:
            return None
        chosen = self.choose(candidates)
        return chosen["machine"] * simulator.spec.task_count + chosen["task"], chosen["rank"]


def run_rule_episode(policy: RulePolicy, spec, train_paras: dict, simulator_factory=TrainingSimulator) -> dict:
    """One complete episode of a rule on an instance; same record layout as ``rollout.run_episodes``."""
    simulator = simulator_factory(spec, train_paras)
    policy.start_episode(spec)
    started = time.perf_counter()
    completed_at_start = int(simulator.completed_count)
    discarded_at_start = int(simulator.discarded_count)
    rate_at_start = float(simulator.fulfillment_rate)
    total_reward = 0.0
    decision_seconds = 0.0
    decisions = 0
    while not simulator.done:
        decision_start = time.perf_counter()
        action = policy.act(simulator)
        decision_seconds += time.perf_counter() - decision_start
        if action is None:                      # defensive: the simulator advances to a decision or to done
            simulator._advance_until_decision_or_done()
            if not simulator.done and policy.act(simulator) is None:
                break
            continue
        total_reward += simulator.step(action)
        decisions += 1
    record = _episode_record(simulator, total_reward, time.perf_counter() - started)
    record.update({
        "policy_forward_seconds": decision_seconds,
        "policy_forward_count": decisions,
        "batch_episodes": 1,
        "rule": policy.active_rule or policy.rule,
        "completed_at_start": completed_at_start,
        "discarded_at_start": discarded_at_start,
        "fulfillment_rate_at_start": rate_at_start,
    })
    return record
