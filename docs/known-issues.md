# 实现说明与已知局限

编号供 README、docs 与测试引用（`tests/test_known_issue_fixes.py` 覆盖第 8、9、12 条）。

## A. 论文与实现的口径

1. **训练超参**：`configs/algo.yaml` 即论文表 6 的来源（`max_iterations: 1000`、`entropy_coeff: 0.01`、`target_kl: 0.03`、
   `minibatch_size: 512`、`num_envs: 8`）；表 6 由 `make_tables.hyperparameters()` 从配置生成，没有随机动作阶段。
2. **流体剪枝**：`environment/simulator.py` 的候选集先取"工序阶段内全部空闲机器 × 有等待订单的工序类型"，再按 `env.fluid_mask`
   保留 u* > `fluid_mask_threshold` 的对（空集回退全部合法对）；流体解同时作为两列工序类型节点特征（`raw_opes[:, 8:10]`，
   `env.fluid_features` 开关）。消融 nfe / nm / nf 分别去掉特征 / 掩码 / 两者。
3. **学习型对比方法**：`agent/baselines/{hsddqn,a2c,dqn,ddqn}.py` 与 `configs/baselines/hgnn_ppo.yaml` 在同一仿真器上实现
   （`docs/baselines.md`）；论文表格用 HSDDQN 与 HGNN-PPO (adapted)，A2C / DQN / DDQN 为附加基线。
4. **规则基线**：`agent/baselines/rules.py`，候选集与策略相同（合法对 × top-k 订单），平局按（更早交期、更短加工时间、更小机器号、
   更小工序号、更小队列序）确定性打破；`order_top_k: 0` 可让规则看到整个队列。
5. **决策时间口径**：论文表格的时间为每次决策的平均时间；`eval_results.csv` 记录整条 episode 的墙钟 `infer_time_ms` 与决策次数，
   聚合时算 `decision_time_ms`，并按 `obs_ms`、`fluid_ms`、`policy_ms` 拆分。
6. **达成率分母**：`completed / total_orders`（丢弃订单计入分母），以 `environment/simulator.py` 的 `fulfillment_rate` 为准；
   `result/aggregate.py` 的整数一致性检查（S·K·η 为整数）防止口径混用。
7. **生成器与基准算例**：基准文件的订单到达间隔均值约 14.5–22 s，交期松弛集中在文件名的 DDT 附近；`data/generator.py` 的默认
   分布按这些统计（到达间隔 rand[1,40] s、每算例一个 DDT、松弛 [0.6,1.55]×DDT），论文表 5 由配置与 `params.yaml` 生成。

## B. 实现细节与已知局限

8. **特征只统计已到达订单**：`raw_opes[:, 6]` 的"未完成同产品订单数"不含尚未到达的订单。
9. **按单个观测归一化**：`agent/normalize.py` 对交期与加工率按观测归一化，同一观测在任何 batch 中得到相同 logits；
   `eval.py --batch-episodes` 默认 1，评测 episode 彼此独立。
10. **检查点口径**：`best_val_checkpoint.pt` 按两个验证算例的贪心均值选择（评测默认用它）；`best_greedy_checkpoint.pt` 按
    更新后策略在训练算例上的贪心目标值选择；`best_checkpoint.pt` 按本轮采样 episode 的达成率均值选择。
11. **`instance_selection` 的 `fixed` 与 `round_robin` 行为相同**（都按 `iteration % 算例数` 轮询）；单算例时无差别。
12. **非法动作抛错**：`step()` 对非法 (机器, 工序类型) 对与越界订单秩抛 `ValueError`；策略与规则的候选集与仿真器一致。
13. **多头注意力形状**：`num_heads` 的任何一项不为 1 时 GAT 层形状不匹配；配置固定为 `[1, 1]`。
14. **工业 3D 打印案例不可由本仓库复现**：案例数据与运行记录不在仓库中；案例工艺（`data/case/process.yaml`）只作为生成器能力保留。
15. **`episode_seconds` 虚高**：在同步向量化采样下，每条 episode 的秒数包含等待同批其它 episode 的时间。
16. **`train_metrics.csv` 追加语义**：训练器对已存在的 `train_metrics.csv` 追加而不清空，`log.csv` / `curve.csv` 在非续训时重建；
    跑干净的实验请删除该 run 目录（脚本用独立目录避免此问题）。
17. **`hgnn_no_fluid` 模型类型不存在**：`model_type` 只接受 `hgnn` 与 `mlp`；无流体消融用 `fluid_mode: off` 或 `fluid_features: off`。
18. **求解器**：流体 LP 与 MILP 一律用 scipy 的 HiGHS（单线程、确定性，跨机器数值一致）；`fluid_backend: gurobi` 只是未受支持的
    显式选项（无 gurobipy 时走启发式回退，`config_snapshot.yaml` 记录实际后端）。
19. **rollout 设备**：采集固定在 CPU（规避 Windows 显卡驱动 TDR 重启），PPO 更新在 `cuda:0`（不可用时 CPU）。
20. **多进程采集**：`env.collector_workers > 0` 时各 worker 进程独立求解 HiGHS，数值与进程内路径逐位一致
    （`tests/test_collector.py`）；DQN / DDQN 的采集器与贪心评估仍在训练进程内，该键对它们无效。
21. **检查点不入库**：`result/` 中的记录不含 `*.pt`，重跑评测（`run_07`、`run_10`）需先重新训练。
