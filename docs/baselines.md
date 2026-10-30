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

## 精确求解检查（`baselines/milp_exact.py`，v2）

论文 3.2 节的 MILP 在小算例档 `small`（基准结构 5 × 5 × 5，S ∈ {6, 8, 10, 12} × DDT ∈ {500, 900, 1500}，12 个）上用精确求解器求解
（`scripts/run_08_milp_check.py`；求解器为 `scipy.optimize.milp` 的 HiGHS，600 s 时限；Gurobi 不再支持），产物 `result/milp/milp_results.csv`，
每算例两行：

| 变体 | 式 (2) | 含义 |
|---|---|---|
| MILP-A | Σ_m x_sjm = 1 | 论文原模型：所有工序都必须加工，迟交订单也占用产能 |
| MILP-B | Σ_m x_sjm = z_s | 可放弃变体：不能按期的订单不占用产能，与仿真器"到期即丢弃"的约定一致 |

两者最优值相等（迟交工序总可以排在全部按期工作之后，故 MILP-B 的最优解可扩展为同值的 MILP-A 解；反之 MILP-A 解去掉迟交订单
即为 MILP-B 可行解），求解器在每个算例上确认这一点；因此 MILP 最优值是仿真器中任何在线策略（含丢弃约定）的离线全知上界。
每个解经过核对：

1. **验证器**：按解的机器分配与机器内顺序从零重新定时（左移规则：开工 = max(就绪, 机器空闲)），核对求解器给出的完工时间
   不早于左移完工、z_s = 1 的订单确实按期，并按仿真器口径（C ≤ d）重数按期订单；
2. **仿真器回放**：`ScheduleFollowingSimulator` 把仿真器的候选集限制为"该机器序列中的下一道工序"，其余规则（到达、到期丢弃、
   开工即迟交则丢弃、机器释放）不变；MILP-B 的回放达成率必须等于最优值，MILP-A 的回放 ≥ 最优值（仿真器丢弃其迟交工序）；
   回放不截断候选订单（`order_top_k: 0`）；
3. **穷举**：按阶段枚举全部机器分配与机器内排列并左移定时（带上界剪枝），对 MILP-B 再遍历被加工订单的子集；在 5 × 5 × 5 结构下
   枚举规模超过门限而跳过（`brute_force_status: skipped_size`），模型本身的穷举核对在测试套件的 2 × 3 × 2 微型算例上进行
   （`tests/test_milp_exact.py`；V8 阶段曾在 80 个 tiny / tiny5 算例上全部一致）。

big-L 取 L = C_UB + max p，C_UB = max_s a_s + Σ_(s,j) max_m p_sjm 为任何左移调度完工时间的上界，并作为 C_sj 的显式上界一起记录
（`big_L`、`c_ub` 列）。`n_binary` = |x| + |y| + |z|，`n_pairs` = 式 (5) 的排序变量数。规则基线在同一批算例上的结果来自
`run_06`（`result/rules.csv`，档位 small），`make_tables.py` 的 `milp_small.tex` 同时给出 FSHGRL 的贪心 η 与到最优值的差距。

## 剪枝阈值敏感性（`scripts/run_10_sensitivity.py`，v2）

不重训：训练好的 FSHGRL（`result/ours/run1/best_val_checkpoint.pt`）在 small 档上以评测期覆盖层 `configs/sensitivity/eps_<v>.yaml`
（ε_f ∈ {off, 1e-5, 0.05, 0.1, 0.2}；off = 掩码关闭、特征保留）各跑贪心 1 条 + 采样 10 条，产物 `result/sensitivity.csv`；
`sensitivity.tex` 报告每个阈值的均值 η、与 MILP 最优值的平均差距、剪枝前/后候选对数、回退率与每决策时间。
