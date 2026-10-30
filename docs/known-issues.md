# 已知缺陷与论文/代码差异

> **v2 状态**：下列条目描述的是 v1（逐位重现原仓库）的状态；v2（分支 `claude/v2-generalization`，新的实验版本）对其中若干条
> 做了修复或变更，见文末 **C. v2 处理状态**。

重构的硬约束是"可运行路径的数值行为逐位不变"，因此下列问题一律**保留并记录**；修复任何一条都会改变训练结果，
应在新的实验版本中进行并重新跑完整矩阵。编号供 README、docs 与审稿回复引用。

## A. 论文与代码的差异

1. **训练超参与论文超参表不一致**。`configs/algo.yaml`（= 原 `data/config.json`）：`max_iterations: 2000`、
   `entropy_coeff: 0.01`、`target_kl: 0.03`、`minibatch_size: 512`、`num_envs: 8`；论文超参表写 1000 个 epoch、
   熵系数 0。论文还描述"前 500 个 epoch 随机动作"，代码中没有对应逻辑（PPO 从第 0 轮即按策略采样）。
   训练参数表 v1 以代码值为准（`train_confirmed: true`），差异在复现时须明示。
2. **训练器没有流体剪枝**。论文描述用 LP 容量分配筛选候选动作；`environment/simulator.py` 的候选集
   `_eligible_pairs` 是"工序阶段内全部空闲机器 × 有等待订单的工序类型"，流体解只作为两列节点特征
   （`raw_opes[:, 8:10]`）进入状态。原仓库中带 `eligible_fluid`（阈值 1e-5）剪枝的只有已删除的旧平台 B。
   因此 FSHGRL-NF 的消融定义为"两列流体特征置零"（`fluid_mode: off`），特征宽度不变。
3. **论文的 DRL 对比方法（DRLG、AHP-DQN、HSDDQN）在原仓库中不存在**。原 `compare/` 下的 A2C/DQN/DDQN 只能依赖
   已删除的旧平台且 A2C 与现网络不兼容；本仓库的 `agent/baselines/{a2c,dqn,ddqn}.py` 是在新仿真器上的
   **新实现**（超参默认沿用原文件），不是论文中经改编的对比方法，也没有旧数值可对照。
4. **规则基线为新实现**。原 `compare/Rule.py` 用 argmax 打分；`compare/Rules.py` 用 argmin 把四条规则全部反向。
   新实现 `agent/baselines/rules.py` 以 `Rule.py` 的方向为准，候选集与策略相同（合法对 × top-k 订单），
   平局按（更早交期、更短加工时间、更小机器号、更小工序号、更小队列序）确定性打破；`order_top_k: 0` 可让规则
   看到整个队列。
5. **决策时间口径**。论文表格的 "Time (s)" 为每次决策的平均时间；本仓库 `eval_results.csv` 记录整条 episode 的墙钟
   `infer_time_ms` 与决策次数，聚合时算 `decision_time_ms`。CPU 上的数值与论文的 GPU 平台不可比，
   且本仓库的评估包含观测构建与流体特征（`obs_ms`、`fluid_ms` 列可拆分）。
6. **达成率分母**。论文正文中部分达成率的分母口径（订单数）与代码口径（`completed / total_orders`，丢弃订单计入分母）
   需逐表核对；代码口径以 `environment/simulator.py` 的 `fulfillment_rate` 为准。
7. **基准算例与论文算例参数表的差异**。基准文件的订单到达间隔均值约 14.5–22 s（`params.yaml` 的
   `inter_arrival_mean`），远小于论文参数表 rand[1, 200] 的均值 100.5 s；每个基准文件的交期松弛集中在文件名的
   DDT 附近（如 DDT500 的 `due_slack` 在 313–786 之间），不是 rand[300, 2500]。`data/generator.py` 按论文参数表
   生成，交期规则取 `due = arrival + tightness`（基准文件的确切规则未记录）。

## B. 代码中的潜在缺陷（保持原样）

8. **特征泄漏未来订单**：`raw_opes[:, 6]` 统计包括尚未到达订单在内的"未完成同产品订单数"
   （`order_status != COMPLETED/DISCARDED` 含 `NOT_ARRIVED`）。
9. **归一化依赖 batch 组成**：策略对交期与加工率按整个前向 batch 做归一化，因此同一观测在不同 batch 中得到不同
   logits。`num_envs` 改变即改变训练信号；`eval.py --batch-episodes` 默认 1 使评测 episode 彼此独立。
10. **best checkpoint 在更新之后保存**：原仓库 `best_checkpoint.pt` 按本轮采样 episode 的达成率均值选择，但保存的是
    更新后的参数（评估的是更新前的策略）。重构新增的 `best_greedy_checkpoint.pt` 按更新后策略的贪心目标值选择，
    §7 评测默认用它。
11. **`instance_selection` 的 `fixed` 与 `round_robin` 行为相同**（都按 `iteration % 算例数` 轮询）；单算例时无差别。
12. **无效动作静默重映射**：`step()` 收到不合法的 (机器, 工序) 对时改用第一个合法对；队列序越界时截断。
    策略与规则的候选集与仿真器一致，训练中不会触发。
13. **多头注意力形状**：`num_heads` 的任何一项不为 1 时 GAT 层形状不匹配；配置固定为 `[1, 1]`。
14. **工业 3D 打印案例不可复现**：案例数据、文件格式与运行脚本都不在原仓库中，本仓库未重建，论文案例研究表/图
    无法由本仓库生成。
15. **原成品图不可再生成**：`paper_assets/figures/legacy/` 的两张训练曲线图由已删除的 CSV 绘制；原绘图脚本读取
    绝对路径且依赖 Python 3.13 已移除的 `mailcap`，已重写为 `result/plot.py`。
16. **`episode_seconds` 虚高**：在同步向量化采样下，每条 episode 的秒数包含等待同批其它 episode 的时间。
17. **`train_metrics.csv` 追加语义**：原训练器对已存在的 `train_metrics.csv` 追加而不清空（`resume: none` 也是），
    新文件 `log.csv`/`curve.csv` 在非续训时重建；跑干净的实验请删除该 run 目录（脚本用独立目录避免此问题）。
18. **`hgnn_no_fluid` 模型类型已移除**：原 `compare/HGNN_model_no_fluid.py` 在训练器下不可运行，`model_type` 只接受
    `hgnn` 与 `mlp`；无流体消融用 `fluid_mode: off`。
19. **Gurobi 路径未在本机验证**：本机无 `gurobipy`，`cached_lp` 在所有验证运行中走启发式回退（`fluid_backend` 列为
    `heuristic_fallback`）；LP 求解代码以 AST 逐字比对保证与原仓库一致（`tests/equivalence/ast_targets.py`）。
    即使安装了 Gurobi，`fluid_threads: 0` 下求解器自身的多线程也可能带来跨机器的微小差异。
    **v2（V12）**：Gurobi 已确认不可用，流体 LP 与 MILP 一律使用 scipy 的 HiGHS（`fluid_backend: scipy`，单线程确定性），
    Gurobi 模型只作为不受支持的显式选项保留。
20. **Windows 上 rollout 设备**：原仓库把 rollout 固定在 CPU 以规避显卡驱动 TDR 重启，重构保留该默认。
21. **多进程采集与 Gurobi**：`env.collector_workers > 0` 时流体 LP 在 worker 进程内求解，每个 worker 各需一个 Gurobi
    环境/许可证；`fluid_threads: 0` 加时间限制本就可能跨运行不确定（第 19 条），只有 `fluid_threads: 1` 且不触发时间
    限制时才能期望逐位一致。无 gurobipy 的启发式回退在进程内与多进程下逐位一致（`tests/test_collector.py`）。
    DQN/DDQN 的采集器与贪心评估仍在训练进程内，该键对它们无效。

## C. v2 处理状态（修订方案 v2 的 A.1 代码冻结清单）

| 条目 | v2 处理 |
|---|---|
| 1 超参不一致 | `configs/paper.yaml` 为论文配置层（1000 轮、熵系数 0.01、泛化型范式、每方法 1 run），`train_version: 2`；论文表 6 由它生成 |
| 2 训练器没有流体剪枝 | **已实现**（V1）：`env.fluid_mask` + `fluid_mask_threshold` 的候选掩码，空集回退；`env.fluid_features` 单独开关两列特征；`fluid_recompute: decision` 每个新状态重解 LP |
| 3 DRL 对比方法不存在 | V7 新增 HSDDQN（层级 DDQN）与 HGNN-PPO (adapted) 作为可复现对比方法，A2C/DQN/DDQN 保留 |
| 5 决策时间口径 | 聚合按 `infer_time_ms / decision_count` 计每决策时间（`result/aggregate.py`），论文表格改用该口径 |
| 6 达成率分母 | 统一为 `completed / total_orders`；`result/aggregate.py` 的整数一致性检查（S·K·η 为整数）防复发（V9） |
| 7 基准算例与表 5 不符 | 生成器默认分布改为基准统计（到达间隔 rand[1,40] s、每算例一个 DDT、松弛 [0.6,1.55]×DDT），论文表 5 改写 |
| 8 特征泄漏未来订单 | **已修复**（V3）：`raw_opes[:, 6]` 只统计已到达的未完成订单 |
| 9 归一化依赖 batch 组成 | **已修复**（V3）：`agent/normalize.py` 按单个观测归一化交期与加工率；`tests/test_known_issue_fixes.py` 验证 batch 无关 |
| 10 best checkpoint 口径 | **已取代**（V2）：`best_val_checkpoint.pt`（验证集贪心均值最优）为 v2 的模型选择；旧文件仍写出 |
| 11 fixed 与 round_robin 相同 | 保持；泛化型用 `random` |
| 12 无效动作静默重映射 | **已修复**（V3）：`step()` 对非法 (机器, 工序类型) 对与越界订单秩抛 `ValueError` |
| 14 工业案例不可复现 | v2 不重做案例实验（2026-10-30 决定）：论文表 13 沿用原稿结果并说明现场使用的就是文中的派工规则，最优规则结果即工业案例的实际运行结果；表 12 的工艺数据作为生成器能力保留（`data/case/process.yaml`），不生成案例档位 |
| 19 Gurobi 路径未验证 | **已变更**（V1/V9/V12）：Gurobi 不再支持；流体 LP 与 MILP 一律用 scipy 的 HiGHS（`fluid_backend: scipy`），`auto` = HiGHS → 启发式；Gurobi 模型原样保留在 `_solve_lp_gurobi` 作为未受支持的显式选项，等价性回归的旧语义覆盖层用 `heuristic` |
| 21 多进程采集与 Gurobi | 已无关：`fluid_backend: scipy` 下各 worker 进程独立求解 HiGHS，无许可证问题 |

等价性回归：V1、V2 仍以 c030dfa 为黄金（旧语义覆盖层）；V3 起以本树冻结提交为黄金（`tests/golden_v2/manifest.json`，
`python -m tests.equivalence --golden v2`）。

