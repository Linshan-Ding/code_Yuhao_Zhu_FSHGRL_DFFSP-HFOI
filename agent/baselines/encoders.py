"""Heterogeneous-graph encoder for the learned baselines (new code).

The machine GAT layers, the operation-type MLP layers, the feature normalisation, the batch padding and the
sparse embedding loop are copied from ``agent/networks.py::HGNNScheduler`` (copied, not shared, so the
regression-protected policy class stays untouched). The actor / critic heads are not part of the encoder;
``candidate_tokens`` builds one feature vector per candidate action (eligible machine-operation pair x
queued order) with the same layout as the policy's actor input: ``[h_ope, h_machine, pooled h_ope,
pooled h_machine, normalised due date, fulfilment rate]``.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.normalize import normalize_due_dates, normalize_per_observation

from agent.hgnn import GATedge
from agent.networks import MLPs


class SparseGraphEncoder(nn.Module):
    def __init__(self, model_paras: dict):
        super().__init__()
        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]
        self.out_size_ma = model_paras["out_size_ma"]
        self.in_size_ope = model_paras["in_size_ope"]
        self.out_size_ope = model_paras["out_size_ope"]
        self.hidden_size_ope = model_paras["hidden_size_ope"]
        self.num_heads: List[int] = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]
        self.feature_dim = (self.out_size_ope + self.out_size_ma) * 2 + 2

        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma),
                                         self.out_size_ma, self.num_heads[0],
                                         feat_drop=self.dropout,
                                         attn_drop=self.dropout,
                                         activation=F.elu))
        for i in range(1, len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma),
                                             self.out_size_ma, self.num_heads[i],
                                             feat_drop=self.dropout,
                                             attn_drop=self.dropout,
                                             activation=F.elu))
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads) - 1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

    # -- copied from HGNNScheduler ------------------------------------------------------------------------
    def feature_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Per-observation normalisation (v2): statistics over each batch element, not over the batch."""
        return normalize_per_observation(data)

    def get_normalized(self, raw_opes: torch.Tensor, raw_mas: torch.Tensor,
                       proc_time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True, unbiased=False)
        mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
        std_mas = torch.std(raw_mas, dim=-2, keepdim=True, unbiased=False)
        proc_time_norm = self.feature_normalize(proc_time)
        norm_opes = (raw_opes - mean_opes) / (std_opes + 1e-5)
        norm_mas = (raw_mas - mean_mas) / (std_mas + 1e-5)
        return norm_opes, norm_mas, proc_time_norm

    def _pad_obs_batch(self, obs_batch):
        batch_size = len(obs_batch)
        max_opes = max(obs["raw_opes"].size(0) for obs in obs_batch)
        max_mas = max(obs["raw_mas"].size(0) for obs in obs_batch)
        max_orders = max(obs["due_dates"].size(1) for obs in obs_batch)
        ope_dim = obs_batch[0]["raw_opes"].size(-1)
        ma_dim = obs_batch[0]["raw_mas"].size(-1)
        padded = {
            "raw_opes": torch.zeros((batch_size, max_opes, ope_dim), dtype=torch.float32, device=self.device),
            "raw_mas": torch.zeros((batch_size, max_mas, ma_dim), dtype=torch.float32, device=self.device),
            "proc_time": torch.zeros((batch_size, max_opes, max_mas), dtype=torch.float32, device=self.device),
            "ope_ma_adj": torch.zeros((batch_size, max_opes, max_mas), dtype=torch.bool, device=self.device),
            "ope_pre_adj": torch.zeros((batch_size, max_opes, max_opes), dtype=torch.bool, device=self.device),
            "ope_sub_adj": torch.zeros((batch_size, max_opes, max_opes), dtype=torch.bool, device=self.device),
            "eligible": torch.zeros((batch_size, max_opes, max_mas), dtype=torch.bool, device=self.device),
            "due_dates": torch.zeros((batch_size, max_opes, max_orders), dtype=torch.float32, device=self.device),
            "order_mask": torch.zeros((batch_size, max_opes, max_orders), dtype=torch.bool, device=self.device),
            "current_rate": torch.zeros((batch_size,), dtype=torch.float32, device=self.device),
        }
        for batch_index, obs in enumerate(obs_batch):
            num_opes = obs["raw_opes"].size(0)
            num_mas = obs["raw_mas"].size(0)
            num_orders = obs["due_dates"].size(1)
            padded["raw_opes"][batch_index, :num_opes] = obs["raw_opes"].to(self.device).float()
            padded["raw_mas"][batch_index, :num_mas] = obs["raw_mas"].to(self.device).float()
            padded["proc_time"][batch_index, :num_opes, :num_mas] = obs["proc_time"].to(self.device).float()
            padded["ope_ma_adj"][batch_index, :num_opes, :num_mas] = obs["ope_ma_adj"].to(self.device).bool()
            padded["ope_pre_adj"][batch_index, :num_opes, :num_opes] = obs["ope_pre_adj"].to(self.device).bool()
            padded["ope_sub_adj"][batch_index, :num_opes, :num_opes] = obs["ope_sub_adj"].to(self.device).bool()
            padded["eligible"][batch_index, :num_opes, :num_mas] = obs["eligible"].to(self.device).bool()
            padded["due_dates"][batch_index, :num_opes, :num_orders] = obs["due_dates"].to(self.device).float()
            padded["order_mask"][batch_index, :num_opes, :num_orders] = obs["order_mask"].to(self.device).bool()
            padded["current_rate"][batch_index] = float(obs["current_order_completed_rate"])
        return padded

    def encode(self, obs_batch):
        """Padded batch, node embeddings, pooled embeddings and batch-normalised due dates."""
        batch = self._pad_obs_batch(obs_batch)
        raw_opes, raw_mas, proc_time = batch["raw_opes"], batch["raw_mas"], batch["proc_time"]
        ope_ma_adj, ope_pre_adj, ope_sub_adj = batch["ope_ma_adj"], batch["ope_pre_adj"], batch["ope_sub_adj"]
        features = self.get_normalized(raw_opes, raw_mas, proc_time)
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, features)
            features = (h_opes, features[1], features[2])
        h_mas_pooled = torch.mean(h_mas, dim=-2)
        h_opes_pooled = torch.mean(h_opes, dim=-2)
        due_dates = normalize_due_dates(batch["due_dates"], batch["order_mask"])   # v2: per observation
        return batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates

    # -- new ----------------------------------------------------------------------------------------------
    def candidate_tokens(self, obs, batch_index, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates,
                         current_rate):
        """Feature matrix ``[T, feature_dim]`` of the candidate actions of one observation and the token list
        ``[(machine, task, rank), ...]`` in the simulator's enumeration order (pairs x queue ranks)."""
        valid_pairs = obs["valid_pairs"].to(self.device).long()
        if valid_pairs.numel() == 0:
            valid_pairs = torch.zeros((1, 2), dtype=torch.long, device=self.device)
        order_counts = obs["order_counts"].to(self.device).long()
        max_orders = due_dates.size(-1)
        tokens = []
        for machine, task in valid_pairs.tolist():
            machine = min(max(int(machine), 0), h_mas.size(1) - 1)
            task = min(max(int(task), 0), h_opes.size(1) - 1)
            count = int(order_counts[task].item()) if task < order_counts.numel() else 0
            count = max(1, min(count, max_orders))
            for rank in range(count):
                tokens.append((machine, task, rank))
        task_idx = torch.tensor([t[1] for t in tokens], dtype=torch.long, device=self.device)
        machine_idx = torch.tensor([t[0] for t in tokens], dtype=torch.long, device=self.device)
        rank_idx = torch.tensor([t[2] for t in tokens], dtype=torch.long, device=self.device)
        count = len(tokens)
        features = torch.cat((
            h_opes[batch_index, task_idx],
            h_mas[batch_index, machine_idx],
            h_opes_pooled[batch_index].view(1, -1).expand(count, -1),
            h_mas_pooled[batch_index].view(1, -1).expand(count, -1),
            due_dates[batch_index, task_idx, rank_idx].view(count, 1),
            current_rate.view(1, 1).expand(count, 1),
        ), dim=-1)
        return features, tokens
