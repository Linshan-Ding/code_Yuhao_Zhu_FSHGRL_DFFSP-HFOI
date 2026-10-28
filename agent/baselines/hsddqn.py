"""Hierarchical double DQN comparator (HSDDQN, adapted; v2, new code).

The comparator of the manuscript's Table 10 named HSDDQN is a hierarchical strategy-optimised double DQN for a
dynamic re-entrant hybrid flow shop with batch-processing machines. Its code is not public; what is adapted here
is the hierarchical two-level decision structure with double-DQN targets on the DFFSP-HFOI action space:

* level 1 scores every eligible (machine, operation-type) pair with ``Q_pair`` (same per-pair feature as the
  FSHGRL actor: operation-type and machine embeddings of the heterogeneous graph encoder, pooled graph state,
  the most urgent candidate's due date and the fulfilment rate; fluid features and mask are off);
* level 2 scores the queued orders of the chosen pair with ``Q_order`` (order feature with that order's due date);
* the action is argmax of level 1 then argmax of level 2 (epsilon-greedy on both levels during training);
* targets are double-DQN targets at both levels: the online network selects the next pair and order, the target
  network evaluates them; the two TD losses are summed. Replay, warm-up, target copies and the rest of the loop
  are those of ``agent/baselines/offpolicy.py``.

Not adapted (not part of DFFSP-HFOI): batch-processing machines, re-entrant routes and the original
rule-level strategy layer. See docs/baselines.md.
"""
from __future__ import annotations

import copy
import random
import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agent.baselines.dqn import DQNAgent
from agent.baselines.encoders import SparseGraphEncoder
from agent.mlp import MLPActor


class HierarchicalQScheduler(nn.Module):
    def __init__(self, model_paras: dict):
        super().__init__()
        self.device = model_paras["device"]
        self.encoder = SparseGraphEncoder(model_paras)
        self.pair_head = MLPActor(model_paras["n_hidden_actor"], self.encoder.feature_dim,
                                  model_paras["n_latent_actor"], 1).to(self.device)
        self.order_head = MLPActor(model_paras["n_hidden_actor"], self.encoder.feature_dim,
                                   model_paras["n_latent_actor"], 1).to(self.device)
        self.last_forward_stats = {}

    def set_device(self, device) -> None:
        self.device = torch.device(device)
        self.encoder.device = self.device
        self.to(self.device)

    # -- features ------------------------------------------------------------------------------------------
    def _pair_features(self, obs, batch_index, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, current_rate):
        valid_pairs = obs["valid_pairs"].to(self.device).long()
        if valid_pairs.numel() == 0:
            valid_pairs = torch.zeros((1, 2), dtype=torch.long, device=self.device)
        machine_idx = valid_pairs[:, 0].clamp(0, h_mas.size(1) - 1)
        task_idx = valid_pairs[:, 1].clamp(0, h_opes.size(1) - 1)
        count = valid_pairs.size(0)
        features = torch.cat((
            h_opes[batch_index, task_idx],
            h_mas[batch_index, machine_idx],
            h_opes_pooled[batch_index].view(1, -1).expand(count, -1),
            h_mas_pooled[batch_index].view(1, -1).expand(count, -1),
            due_dates[batch_index, task_idx, 0:1],
            current_rate.view(1, 1).expand(count, 1),
        ), dim=-1)
        return features, valid_pairs

    def _order_features(self, obs, batch_index, machine, task, h_opes, h_mas, h_opes_pooled, h_mas_pooled,
                        due_dates, current_rate):
        order_counts = obs["order_counts"].to(self.device).long()
        max_orders = due_dates.size(-1)
        count = int(order_counts[task].item()) if task < order_counts.numel() else 0
        count = max(1, min(count, max_orders))
        features = torch.cat((
            h_opes[batch_index, task].view(1, -1).expand(count, -1),
            h_mas[batch_index, machine].view(1, -1).expand(count, -1),
            h_opes_pooled[batch_index].view(1, -1).expand(count, -1),
            h_mas_pooled[batch_index].view(1, -1).expand(count, -1),
            due_dates[batch_index, task, :count].view(count, 1),
            current_rate.view(1, 1).expand(count, 1),
        ), dim=-1)
        return features, count

    def encode(self, obs_batch):
        return self.encoder.encode(obs_batch)

    def pair_q(self, obs_batch, encoded=None) -> list:
        """``[(q_pair [P], valid_pairs [P, 2]), ...]`` per observation."""
        batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates = encoded or self.encode(obs_batch)
        out = []
        for batch_index, obs in enumerate(obs_batch):
            features, pairs = self._pair_features(obs, batch_index, h_opes, h_mas, h_opes_pooled, h_mas_pooled,
                                                  due_dates, batch["current_rate"][batch_index])
            out.append((self.pair_head(features.float()).squeeze(-1), pairs))
        return out

    def order_q(self, obs, batch_index, machine, task, encoded):
        batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates = encoded
        features, count = self._order_features(obs, batch_index, machine, task, h_opes, h_mas, h_opes_pooled,
                                               h_mas_pooled, due_dates, batch["current_rate"][batch_index])
        return self.order_head(features.float()).squeeze(-1), count

    # -- acting ----------------------------------------------------------------------------------------------
    @staticmethod
    def encode_token(pair_index: int, rank: int, max_orders: int) -> int:
        return int(pair_index) * max(int(max_orders), 1) + int(rank)

    @staticmethod
    def decode_token(token: int, max_orders: int) -> tuple:
        width = max(int(max_orders), 1)
        return int(token) // width, int(token) % width

    def act_batch(self, obs_batch, sample: bool = False, epsilon: float = 0.0, rng: Optional[random.Random] = None):
        """``(tokens, log_probs(0), values(0), env_actions)``; greedy at both levels, epsilon-greedy with ``rng``,
        Boltzmann sampling of both levels with ``sample=True`` (eval.py --decode sample)."""
        start_time = time.perf_counter()
        encoded = self.encode(obs_batch)
        tokens, env_actions, counts = [], [], []
        for batch_index, (obs, (q_pair, pairs)) in enumerate(zip(obs_batch, self.pair_q(obs_batch, encoded))):
            if rng is not None and epsilon > 0.0 and rng.random() < epsilon:
                pair_index = rng.randrange(pairs.size(0))
            elif sample:
                pair_index = int(Categorical(F.softmax(q_pair, dim=0)).sample().item())
            else:
                pair_index = int(torch.argmax(q_pair).item())
            machine, task = int(pairs[pair_index, 0].item()), int(pairs[pair_index, 1].item())
            q_order, count = self.order_q(obs, batch_index, machine, task, encoded)
            if rng is not None and epsilon > 0.0 and rng.random() < epsilon:
                rank = rng.randrange(count)
            elif sample:
                rank = int(Categorical(F.softmax(q_order, dim=0)).sample().item())
            else:
                rank = int(torch.argmax(q_order).item())
            max_orders = int(obs.get("max_orders", 1))
            tokens.append(self.encode_token(pair_index, rank, max_orders))
            local_tasks = int(obs.get("num_tasks", obs["raw_opes"].size(0)))
            env_actions.append((machine * local_tasks + task, rank))
            counts.append(pairs.size(0) * count)
        count_tensor = torch.tensor(counts, dtype=torch.float32)
        self.last_forward_stats = {
            "batch_size": int(len(obs_batch)),
            "avg_action_tokens": float(count_tensor.mean().item()) if count_tensor.numel() else 0.0,
            "max_action_tokens": float(count_tensor.max().item()) if count_tensor.numel() else 0.0,
            "padding_ratio": 0.0,
            "forward_seconds": float(time.perf_counter() - start_time),
        }
        zeros = torch.zeros(len(obs_batch), dtype=torch.float32, device=self.device)
        return torch.tensor(tokens, dtype=torch.long, device=self.device), zeros, zeros, env_actions

    # -- TD helpers ----------------------------------------------------------------------------------------
    def q_sa(self, obs_batch, tokens: List[int]):
        """``(Q_pair(s, pair), Q_order(s, pair, rank))`` of the stored actions."""
        encoded = self.encode(obs_batch)
        q_pairs, q_orders = [], []
        for batch_index, (obs, (q_pair, pairs)) in enumerate(zip(obs_batch, self.pair_q(obs_batch, encoded))):
            pair_index, rank = self.decode_token(tokens[batch_index], int(obs.get("max_orders", 1)))
            pair_index = min(max(pair_index, 0), pairs.size(0) - 1)
            machine, task = int(pairs[pair_index, 0].item()), int(pairs[pair_index, 1].item())
            q_order, count = self.order_q(obs, batch_index, machine, task, encoded)
            q_pairs.append(q_pair[pair_index])
            q_orders.append(q_order[min(max(rank, 0), count - 1)])
        return torch.stack(q_pairs), torch.stack(q_orders)

    def next_values(self, next_obs, online: "HierarchicalQScheduler", double: bool):
        """Double-DQN (or plain) next-state values at both levels: the online network selects, this (target)
        network evaluates."""
        encoded = self.encode(next_obs)
        target_pairs = self.pair_q(next_obs, encoded)
        online_encoded = online.encode(next_obs) if double else None
        online_pairs = online.pair_q(next_obs, online_encoded) if double else target_pairs
        v_pairs, v_orders = [], []
        for batch_index, (obs, (tq, pairs), (oq, _)) in enumerate(zip(next_obs, target_pairs, online_pairs)):
            pair_index = int(torch.argmax(oq).item())                 # level 1: the selecting network's argmax
            machine, task = int(pairs[pair_index, 0].item()), int(pairs[pair_index, 1].item())
            tq_order, count = self.order_q(obs, batch_index, machine, task, encoded)
            if double:                                                # level 2: the online network selects the order
                oq_order, _ = online.order_q(obs, batch_index, machine, task, online_encoded)
                rank = int(torch.argmax(oq_order).item())
            else:
                rank = int(torch.argmax(tq_order).item())
            v_pairs.append(tq[pair_index])
            v_orders.append(tq_order[min(rank, count - 1)])
        return torch.stack(v_pairs), torch.stack(v_orders)


class HSDDQNAgent(DQNAgent):
    """``DQNAgent`` with the hierarchical network and two-level double-DQN targets."""

    def __init__(self, model_paras: dict, train_paras: dict, double: bool = True):
        super().__init__(model_paras, train_paras, double=True)
        self.policy = HierarchicalQScheduler(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)

    def td_update(self, transitions: List) -> dict:
        self.policy.train()
        obs_batch = [t.obs for t in transitions]
        tokens = [int(t.token) for t in transitions]
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device)
        q_pair_sa, q_order_sa = self.policy.q_sa(obs_batch, tokens)
        target_pair = rewards.clone()
        target_order = rewards.clone()
        next_positions = [i for i, t in enumerate(transitions) if not t.done and t.next_obs is not None]
        if next_positions:
            next_obs = [transitions[i].next_obs for i in next_positions]
            with torch.no_grad():
                v_pair, v_order = self.policy_old.next_values(next_obs, self.policy, self.double)
            target_pair[next_positions] = rewards[next_positions] + self.gamma * v_pair
            target_order[next_positions] = rewards[next_positions] + self.gamma * v_order
        loss_fn = F.smooth_l1_loss if self.td_loss == "huber" else F.mse_loss
        loss = loss_fn(q_pair_sa, target_pair) + loss_fn(q_order_sa, target_order)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.update_count += 1
        self._update_target()
        td_error = (q_pair_sa - target_pair).abs().mean() + (q_order_sa - target_order).abs().mean()
        return {"loss": float(loss.item()), "td_error_abs": float(td_error.item()),
                "q_mean": float((q_pair_sa + q_order_sa).mean().item() / 2.0)}
