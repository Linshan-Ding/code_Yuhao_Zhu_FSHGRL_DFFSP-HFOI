# 对比方法与基线（v2）

所有学习型对比方法与 FSHGRL 共用同一仿真器（`environment/simulator.py`）、同一候选集定义（合法 (机器, 工序类型) 对 ×
按交期排序的前 `order_top_k` 个订单）、同一奖励（`(ΔN_c − ΔN_d)/S`）、同一训练预算（`configs/paper.yaml`：2000 轮 × 8 条
episode）与同一评测协议（`eval.py`：贪心 1 条 + 采样 10 条，验证集选检查点）。差异只在网络、动作结构与更新规则。
论文原对比方法（DRLG、AHP-DQN、HSDDQN）的改编实现在原仓库中不存在（`docs/known-issues.md` 第 3 条），表 10/11 在 v2
以下列可复现方法重做。

| 方法 id | 来源 | 保留的要素 | 未改编 / 与原文不同之处 |
|---|---|---|---|
| `ours` | FSHGRL | 流体特征 + 流体掩码 + 异质图编码 + 候选自注意力 + 两级（pair → 订单）PPO 策略 | — |
| `hsddqn` | HSDDQN（动态可重入混合流水车间、批处理机的层级策略优化 DDQN） | **层级决策结构**（第一层在 (机器, 工序类型) 对上取 Q，第二层在该对的候选订单上取 Q）与**双层 Double-DQN 目标**（在线网络选择、目标网络评估）、经验回放、ε-贪心、目标网络硬拷贝；编码器为本仓库的异质图编码器（`agent/baselines/encoders.py`，流体特征关闭） | 批处理机、可重入工艺与原文的"策略层（规则）"未改编：本问题没有这些要素；原文代码不公开，网络规模与超参按本仓库默认（`algo.offpolicy`） |
| `hgnn_ppo` | HGNN-PPO（柔性作业车间的异质图 PPO 调度器，song2022flexible 一系） | 异质图编码器 + (工序类型, 机器) 对动作 + PPO（本仓库编码器即源于此） | 无流体引导、无候选自注意力；订单维由 EDD 固定（`order_top_k: 1`），即原方法的 O–M 对动作空间 |
| `a2c` | 原仓库 `compare/A2C_model.py` 的移植 | 同 FSHGRL 网络，蒙特卡洛回报，单步 actor-critic 更新 | 原文件依赖已删除的旧平台，无旧数值可比 |
| `dqn` / `ddqn` | 原仓库 `compare/DQN_model.py`、`DDQN_model.py` 的移植 | 图编码器 + 候选 token 的标量 Q，ε-贪心 + 回放（DDQN：在线选择、目标评估） | 同上 |
| `fifo` `spt` `mor` `mwkr` `random` `rrc` | 派工规则 | 与策略相同的候选集，确定性平局打破 | 新实现（评分方向沿用原 `compare/Rule.py`） |

论文正文（表 9）应如实写明：DRLG 与 AHP-DQN 的原改编实现不可追溯，v2 以 HGNN-PPO (adapted) 与 DDQN 替代；HSDDQN 为层级
DDQN 结构的改编而非原文复现。各方法的配置覆盖层在 `configs/baselines/*.yaml`，对比方法一律关闭流体引导（`fluid_features: off`、
`fluid_mask: false`）。
