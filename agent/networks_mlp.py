"""
对比实验：PPO+MLP 对比 PPO+HGNN
MLP编辑器：去掉图结构，只保留逐点MLP编码
支持 act_batch / evaluate_batch 完整接口，可直接替换 HGNNScheduler
"""

from typing import Tuple, List, Optional
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.normalize import normalize_due_dates, normalize_per_observation
from torch.distributions import Categorical
from agent.mlp import MLPCritic, MLPActor


class NodeMLP(nn.Module):
    """通用逐点 MLP 编码器：输入 [..., in_dim] -> 输出 [..., out_dim]"""
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPScheduler(nn.Module):
    """
    与 HGNNScheduler 保持相同接口，仅把图消息传递替换为逐点 MLP 编码。
    其他流程（动作拼接、掩码、Actor/Critic 前向、evaluate 签名）完全一致，确保可直接替换做对比实验。
    """
    def __init__(self, model_paras):
        super().__init__()

        self.device = model_paras["device"]
        self.in_size_ma = model_paras["in_size_ma"]
        self.out_size_ma = model_paras["out_size_ma"]
        self.in_size_ope = model_paras["in_size_ope"]
        self.out_size_ope = model_paras["out_size_ope"]
        self.hidden_size_ope = model_paras["hidden_size_ope"]
        self.dropout = model_paras["dropout"]
        self.epsilon = model_paras.get("epsilon", 0.05)

        # Actor/Critic 的输入/隐藏/输出（保持与项目里一致）
        self.actor_dim = model_paras["actor_in_dim"]
        self.critic_dim = model_paras["critic_in_dim"]
        self.n_hidden_actor = model_paras["n_hidden_actor"]
        self.n_hidden_critic = model_paras["n_hidden_critic"]
        self.n_latent_actor = model_paras["n_latent_actor"]
        self.n_latent_critic = model_paras["n_latent_critic"]
        self.action_dim = model_paras["action_dim"]

        # 逐点编码器（替代 HGNN 图传播）
        self.ope_encoder = NodeMLP(self.in_size_ope, self.out_size_ope, self.hidden_size_ope, self.dropout)
        self.ma_encoder = NodeMLP(self.in_size_ma, self.out_size_ma, self.hidden_size_ope, self.dropout)

        # 构造 actor 与 critic 网络（与原版 HGNNScheduler 一致，包含 order_actor）
        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim + 2, self.n_latent_actor, self.action_dim).to(self.device)
        self.order_actor = MLPActor(self.n_hidden_actor, self.actor_dim + 2, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim + 1, self.n_latent_critic, 1).to(self.device)

        # 稀疏注意力模块（与原版一致）
        self.use_sparse_attention = bool(model_paras.get("use_sparse_attention", False))
        self.sparse_attention_scope = str(model_paras.get("sparse_attention_scope", "pair")).lower()
        if self.sparse_attention_scope not in {"pair", "pair_order"}:
            self.sparse_attention_scope = "pair"
        sparse_feature_dim = self.actor_dim + 2
        sparse_heads = max(int(model_paras.get("sparse_attention_heads", 1)), 1)
        if sparse_feature_dim % sparse_heads != 0:
            sparse_heads = 1
        sparse_dropout = float(model_paras.get("sparse_attention_dropout", 0.0))
        self.sparse_pair_attention = None
        self.sparse_pair_norm = None
        self.sparse_order_attention = None
        self.sparse_order_norm = None
        if self.use_sparse_attention:
            self.sparse_pair_attention = nn.MultiheadAttention(
                sparse_feature_dim,
                sparse_heads,
                dropout=sparse_dropout,
                batch_first=True,
            ).to(self.device)
            self.sparse_pair_norm = nn.LayerNorm(sparse_feature_dim).to(self.device)
            if self.sparse_attention_scope == "pair_order":
                self.sparse_order_attention = nn.MultiheadAttention(
                    sparse_feature_dim,
                    sparse_heads,
                    dropout=sparse_dropout,
                    batch_first=True,
                ).to(self.device)
                self.sparse_order_norm = nn.LayerNorm(sparse_feature_dim).to(self.device)

        # 为"工序-机器-订单三层"动作空间建立映射字典
        self.current_action_dict: Optional[dict] = None
        self.last_forward_stats = {}


    # ============= 归一化辅助函数 =============

    def feature_normalize(self, data: torch.Tensor) -> torch.Tensor:
        """Per-observation normalisation (v2): statistics over each batch element, not over the batch."""
        return normalize_per_observation(data)

    def get_normalized(self,
                       raw_opes: torch.Tensor,
                       raw_mas: torch.Tensor,
                       proc_time: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)
        std_opes = torch.std(raw_opes, dim=-2, keepdim=True, unbiased=False)
        mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
        std_mas = torch.std(raw_mas, dim=-2, keepdim=True, unbiased=False)
        proc_time_norm = self.feature_normalize(proc_time)
        norm_opes = (raw_opes - mean_opes) / (std_opes + 1e-5)
        norm_mas = (raw_mas - mean_mas) / (std_mas + 1e-5)
        return norm_opes, norm_mas, proc_time_norm

    # ============= 核心：MLP 编码（替代图传播） =============

    def _mlp_encode(self, features: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        用逐点 MLP 替代 HGNN 多层图传播。
        输入 features = (norm_opes, norm_mas, norm_proc)
        返回 (h_opes, h_mas)
        """
        norm_opes, norm_mas, _norm_proc = features
        h_opes = self.ope_encoder(norm_opes)
        h_mas = self.ma_encoder(norm_mas)
        return h_opes, h_mas

    # ============= 单实例接口（兼容旧版） =============


    # ============= Batch 基础设施 =============

    def _pad_obs_batch(self, obs_batch):
        """将变长 obs 填充到统一尺寸的 batch 张量"""
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

    # ============= Dense Batch Forward（非稀疏路径） =============


    # ============= Sparse Attention 辅助 =============

    def _has_sparse_candidates(self, obs_batch):
        return bool(obs_batch) and all("valid_pairs" in obs and "order_counts" in obs for obs in obs_batch)

    def _apply_sparse_attention(self, candidate_features, attention_layer, norm_layer):
        if attention_layer is None or norm_layer is None or candidate_features.size(0) <= 1:
            return candidate_features
        attended, _ = attention_layer(
            candidate_features.unsqueeze(0),
            candidate_features.unsqueeze(0),
            candidate_features.unsqueeze(0),
            need_weights=False,
        )
        return norm_layer(candidate_features + attended.squeeze(0))

    def _sparse_embedding_batch(self, obs_batch):
        """Sparse 路径的嵌入：MLP 编码 + 池化"""
        start_time = time.perf_counter()
        batch = self._pad_obs_batch(obs_batch)
        raw_opes, raw_mas, proc_time = batch["raw_opes"], batch["raw_mas"], batch["proc_time"]

        features = self.get_normalized(raw_opes, raw_mas, proc_time)
        h_opes, h_mas = self._mlp_encode(features)

        h_mas_pooled = torch.mean(h_mas, dim=-2)
        h_opes_pooled = torch.mean(h_opes, dim=-2)

        due_dates = normalize_due_dates(batch["due_dates"], batch["order_mask"])   # v2: per observation

        state_values = self.critic(torch.cat((h_opes_pooled, h_mas_pooled,
                                              batch["current_rate"].view(len(obs_batch), 1)), dim=-1)).squeeze(-1)
        return batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, state_values, start_time

    def _sparse_pair_distribution(self, obs, batch_index, h_opes, h_mas,
                                   h_opes_pooled, h_mas_pooled, due_dates, current_rate):
        valid_pairs = obs["valid_pairs"].to(self.device).long()
        if valid_pairs.numel() == 0:
            valid_pairs = torch.zeros((1, 2), dtype=torch.long, device=self.device)
        machine_indexes = valid_pairs[:, 0].clamp(0, h_mas.size(1) - 1)
        task_indexes = valid_pairs[:, 1].clamp(0, h_opes.size(1) - 1)
        pair_due = due_dates[batch_index, task_indexes, 0:1]
        pair_rate = current_rate.view(1, 1).expand(valid_pairs.size(0), 1)
        pair_features = torch.cat((
            h_opes[batch_index, task_indexes],
            h_mas[batch_index, machine_indexes],
            h_opes_pooled[batch_index].view(1, -1).expand(valid_pairs.size(0), -1),
            h_mas_pooled[batch_index].view(1, -1).expand(valid_pairs.size(0), -1),
            pair_due,
            pair_rate,
        ), dim=-1)
        if self.use_sparse_attention:
            pair_features = self._apply_sparse_attention(
                pair_features, self.sparse_pair_attention, self.sparse_pair_norm)
        pair_scores = self.actor(pair_features.float()).squeeze(-1)
        return Categorical(F.softmax(pair_scores, dim=0)), valid_pairs

    def _sparse_order_distribution(self, obs, batch_index, machine_index, task_index,
                                    h_opes, h_mas, h_opes_pooled, h_mas_pooled,
                                    due_dates, current_rate):
        order_counts = obs["order_counts"].to(self.device).long()
        max_orders = due_dates.size(-1)
        order_count = int(order_counts[task_index].detach().cpu().item()) if task_index < order_counts.numel() else 0
        order_count = max(1, min(order_count, max_orders))
        order_due = due_dates[batch_index, task_index, :order_count].view(order_count, 1)
        order_rate = current_rate.view(1, 1).expand(order_count, 1)
        order_features = torch.cat((
            h_opes[batch_index, task_index].view(1, -1).expand(order_count, -1),
            h_mas[batch_index, machine_index].view(1, -1).expand(order_count, -1),
            h_opes_pooled[batch_index].view(1, -1).expand(order_count, -1),
            h_mas_pooled[batch_index].view(1, -1).expand(order_count, -1),
            order_due,
            order_rate,
        ), dim=-1)
        if self.use_sparse_attention and self.sparse_attention_scope == "pair_order":
            order_features = self._apply_sparse_attention(
                order_features, self.sparse_order_attention, self.sparse_order_norm)
        order_scores = self.order_actor(order_features.float()).squeeze(-1)
        return Categorical(F.softmax(order_scores, dim=0)), order_count

    # ============= Sparse Batch Act / Evaluate =============

    def _act_batch_sparse(self, obs_batch, sample=True):
        batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, state_values, start_time = \
            self._sparse_embedding_batch(obs_batch)
        action_indexes = []
        log_probs = []
        env_actions = []
        valid_action_tokens = []
        for batch_index, obs in enumerate(obs_batch):
            current_rate = batch["current_rate"][batch_index]
            pair_dist, valid_pairs = self._sparse_pair_distribution(
                obs, batch_index, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, current_rate)
            pair_local_index = pair_dist.sample() if sample else pair_dist.probs.argmax(dim=0)
            machine_index = int(valid_pairs[pair_local_index, 0].detach().cpu().item())
            task_index = int(valid_pairs[pair_local_index, 1].detach().cpu().item())
            order_dist, order_count = self._sparse_order_distribution(
                obs, batch_index, machine_index, task_index, h_opes, h_mas,
                h_opes_pooled, h_mas_pooled, due_dates, current_rate)
            order_index = order_dist.sample() if sample else order_dist.probs.argmax(dim=0)
            log_probs.append(pair_dist.log_prob(pair_local_index) + order_dist.log_prob(order_index))
            action_indexes.append(pair_local_index * max(int(obs.get("max_orders", 1)), 1) + order_index)
            local_tasks = int(obs.get("num_tasks", obs["raw_opes"].size(0)))
            env_actions.append((machine_index * local_tasks + task_index, int(order_index.detach().cpu().item())))
            valid_action_tokens.append(int(valid_pairs.size(0)) * int(order_count))

        action_indexes = torch.stack([index if torch.is_tensor(index) else torch.tensor(index, device=self.device)
                                       for index in action_indexes]).long()
        log_probs = torch.stack(log_probs)
        valid_tokens = torch.tensor(valid_action_tokens, dtype=torch.float32, device=self.device)
        self.last_forward_stats = {
            "batch_size": int(len(obs_batch)),
            "avg_action_tokens": float(valid_tokens.mean().detach().cpu().item()) if valid_tokens.numel() else 0.0,
            "max_action_tokens": float(valid_tokens.max().detach().cpu().item()) if valid_tokens.numel() else 0.0,
            "padding_ratio": 0.0,
            "forward_seconds": float(time.perf_counter() - start_time),
        }
        return action_indexes, log_probs, state_values, env_actions

    def _evaluate_batch_sparse(self, obs_batch, actions):
        batch, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, state_values, _ = \
            self._sparse_embedding_batch(obs_batch)
        log_probs = []
        entropies = []
        for batch_index, (obs, env_action) in enumerate(zip(obs_batch, actions)):
            action1, order_index = env_action
            local_tasks = int(obs.get("num_tasks", obs["raw_opes"].size(0)))
            task_index = int(action1) % local_tasks
            machine_index = int(action1) // local_tasks
            current_rate = batch["current_rate"][batch_index]
            pair_dist, valid_pairs = self._sparse_pair_distribution(
                obs, batch_index, h_opes, h_mas, h_opes_pooled, h_mas_pooled, due_dates, current_rate)
            matches = torch.nonzero(
                (valid_pairs[:, 0] == machine_index) & (valid_pairs[:, 1] == task_index), as_tuple=False)
            pair_local_index = matches[0, 0] if matches.numel() else torch.tensor(0, dtype=torch.long, device=self.device)
            order_dist, order_count = self._sparse_order_distribution(
                obs, batch_index, machine_index, task_index, h_opes, h_mas,
                h_opes_pooled, h_mas_pooled, due_dates, current_rate)
            order_tensor = torch.tensor(min(max(int(order_index), 0), order_count - 1),
                                         dtype=torch.long, device=self.device)
            log_probs.append(pair_dist.log_prob(pair_local_index) + order_dist.log_prob(order_tensor))
            entropies.append(pair_dist.entropy() + order_dist.entropy())
        return torch.stack(log_probs), state_values, torch.stack(entropies)

    # ============= 顶层 API：act_batch / evaluate_batch =============

    def act_batch(self, obs_batch, sample=True):
        """Batch 动作采样（rollout 调用入口）"""
        if self._has_sparse_candidates(obs_batch):
            return self._act_batch_sparse(obs_batch, sample=sample)
        raise RuntimeError(
            "observations without 'valid_pairs'/'order_counts' are not supported: "
            "the legacy dense batch path was removed in the refactor"
        )

    def evaluate_batch(self, obs_batch, actions):
        """Batch 评估（PPO update 调用入口）"""
        if self._has_sparse_candidates(obs_batch) and not torch.is_tensor(actions):
            return self._evaluate_batch_sparse(obs_batch, actions)
        raise RuntimeError(
            "observations without 'valid_pairs'/'order_counts' are not supported: "
            "the legacy dense batch path was removed in the refactor"
        )

