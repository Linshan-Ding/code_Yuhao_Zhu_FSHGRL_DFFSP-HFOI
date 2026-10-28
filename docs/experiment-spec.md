# 实验设计规格（v2，泛化型范式）

> v1（逐算例型、逐位重现原仓库）的规格保留在 `main` 分支的本文件历史中；v2 是修订版实验的规格，与审稿人 1 修改方案 v2 的
> 附录 A 一致。变更记录见 §7。

## 0. 问题、范式与声明

- 问题：DFFSP-HFOI，最大化订单按时达成率 `eta = N_c / S`（N_c 为整条 episode 结束时按期完成的订单数，S 为订单总数，丢弃订单计入分母）；
  契约目标值 `obj = 1 - eta`（最小化），`gap` 以各算例最优确定性规则为参考（`data/instances/reference.csv`）。
- 训练范式：**泛化型（generalization）**。每个方法只训练一个策略（K = 3 个独立 run），训练数据是生成的训练池，验证集是另一批
  生成算例，测试集是训练中从未见过的档位（15 个基准算例、6 个到达模式算例、9 个案例配置）；所有表格报告未见算例上的零样本结果。
- 取点协议：每轮 PPO 更新后，当前策略在本轮训练算例上跑一次贪心 episode（`curve.csv`）；每 `val_every`（50）轮在全部验证算例上
  各跑一条贪心 episode（`val_curve.csv`）。
- 模型选择：`best_val_checkpoint.pt`（验证集贪心均值最优轮）；`best_greedy_checkpoint.pt`、`best_checkpoint.pt` 并列保留。
- 流体引导：`fluid_features on`（两列节点特征）+ `fluid_mask true`（u* > ε_f = 1e-5 的合法 (机器, 工序类型) 对，空集回退），
  `fluid_recompute decision`，后端 Gurobi → HiGHS → 启发式（`config_snapshot.yaml` 记录实际后端）。

## 1. 算例（`docs/reproduce.md` §2 的档位表）

基准 15 个外部文件；生成档位由 `configs/instance.yaml` `instance.pools` 定义，`scripts/run_01_prepare_data.py` 生成，种子入库，
`data/instances/index.csv` 568 行。生成器默认分布按基准统计（到达间隔 rand[1,40] s、每算例一个 DDT、松弛 [0.6,1.55]×DDT），
到达模型 uniform / exponential / bursty 以负荷率 ρ（平均工序时间 × 阶段数 / (平均间隔 × 机器数)）或平均间隔参数化，
结构可覆盖（tiny 档）或取工业案例工艺（`data/case/process.yaml`）。每个算例的 `params.yaml` 记录到达、松弛、负荷统计。

## 2. 方法与 run 矩阵（`docs/run-matrix.md`）

| 方法 id | run 数（`configs/runs.yaml`） | 训练池 → 评测档位 | 配置 |
|---|---|---|---|
| ours | 3 | train → benchmark, arrival, case_eval | `configs/paper.yaml` |
| ours_arrival / ours_case | 各 3 | arrival_train → arrival, benchmark / case_train → case_eval | `configs/pools/*.yaml` |
| nfe、nm、nf、ng、na | 各 3 | train → 同 ours | `configs/ablation/*.yaml` |
| hsddqn、hgnn_ppo | 各 3 | train → 同 ours（流体引导关闭） | `configs/baselines/{hsddqn,hgnn_ppo}.yaml` |
| a2c、dqn、ddqn（附加） | 各 1 | 同上 | `configs/baselines/*.yaml` |
| fifo、spt、mor、mwkr | 1（确定性） | 全部评测档位（含 tiny） | `baselines/run_rules.py` |
| random、rrc | 10 | 同上 | 同上 |
| MILP-A / MILP-B | — | tiny、tiny5 | `baselines/milp_exact.py` |

消融为流体引导的 2×2 因子设计（nfe：特征关、掩码开；nm：特征开、掩码关；nf：都关）加编码器（ng）与候选自注意力（na）。
对比方法的改编范围见 `docs/baselines.md`。

## 3. 训练参数表 v2（`configs/paper.yaml`，`train_version: 2`，`train_confirmed: true`）

| 参数 | 值 |
|---|---|
| 轮数 / 每轮 episode | 2000 / 8（每轮随机取一个训练池算例） |
| 学习率 / 调度 / 优化器 | 3e-4 / 无 / Adam (0.9, 0.999) |
| γ / GAE λ / PPO 轮次 / 截断 | 1.0 / 0.95 / 3 / 0.2 |
| c1 / c2 / c3 / 梯度裁剪 / 目标 KL | 1.0 / 0.5 / 0.01 / 0.5 / 0.03 |
| minibatch / 候选订单 k / 掩码阈值 | 512 / 5 / 1e-5 |
| 网络 | 嵌入 16/16，2 层 HGNN（1 头），隐藏 256，actor / order-actor / critic 3 × 128（tanh），候选自注意力 1 头 × 66 维 + LayerNorm |
| 验证 / 早停 | 每 50 轮，30 个验证算例；早停关闭（`early_stop_patience: 0`） |
| 设备 | 采集 CPU，更新 cuda:0（不可用回退 CPU），`fluid_threads: 1` |

表 6 由 `paper_assets/scripts/make_tables.py` 的 `hyperparameters()` 直接从配置生成。预实验门控（`configs/pilot.yaml`）：ours × 1000 轮，
通过标准见 `docs/pilot-log.md`；未通过时的调整顺序：学习率 3e-4 → 1e-4、熵系数 0.01 → 0.005 / 0.02、归一化、轮数 3000。

## 4. 评测协议（所有方法统一）

`eval.py`：每个 run 的 `best_val_checkpoint.pt` 在快照 `eval_tiers` 的每个算例上跑贪心 1 条 + 采样 10 条（`batch_episodes 1`，逐条独立），
CPU 单线程；每决策时间 = episode 墙钟 / 决策次数（含 LP、观测构建、策略前向）。表格主值为贪心 η 的 K 次均值 ± 标准差，
因此 S · K · η 为整数（`result/integer_check.csv` 机检）；采样均值进附录。规则：确定性规则 1 条，Random / RRC 10 条。

## 5. 统计

每算例取 run 均值；基准档 15 个配置上主方法对每个对比方法做配对 Wilcoxon 符号秩检验（精确枚举、零差剔除、平均秩、两侧），
Holm 校正（家族 = 全部对比方法）；Friedman 检验给平均秩；其它档位报告 `generalization.csv`（均值 ± 标准差、与最优规则的胜/平/负）。

## 6. 落盘契约

数据侧：`data/instances/index.csv`、`<tier>/<id>/{<id>.txt,params.yaml}`、`reference.csv`、`data/case/process.yaml`。
run 侧（`result/<method>/run<k>/`）：`train_metrics.csv`、`curve.csv`、`val_curve.csv`（`epoch, instance_id, fulfillment_rate, obj,
wall_time_s`）、`log.csv`（含掩码统计 `candidate_pairs_mean / masked_pairs_mean / mask_fallback_rate` 与验证列）、`fixed_set_final.csv`、
`config_snapshot.yaml`（范式、训练池大小、验证设置、评测档位、实际流体后端、git 提交）、`train_meta.yaml`、检查点。
汇总侧：`eval_results.csv`（含 `completed, discarded, total_orders, decision_count, fluid_backend` 与掩码统计列）、`incumbents.csv`、
`rules.csv`、`integer_check.csv`、`stats_summary.csv`、`generalization.csv`、`paired_tests.csv`、`ablation_results.csv`、
`advantage_table.csv`、`friedman.csv`、`conv_<method>.csv`、`milp/milp_results.csv`、`pilot/{verdict.txt,pilot-report.md,pilot_results.csv}`；
表：`paper_assets/tables/*.tex`（v2 表同时落 CSV）。`sensitivity.csv`（ε_f 敏感性）不在本版中。

## 7. 变更记录

| 版本 | 变更 | 理由 |
|---|---|---|
| 重构 C3/C4/C9 | 训练参数表 v1 = 原仓库配置值；`curve.csv` 每轮贪心；预实验关闭 | 逐位重现原仓库 |
| v2 V1 | 流体掩码、`fluid_features` 键、HiGHS 后端（默认路径逐位不变） | 论文描述的剪枝在原代码中缺失（known-issues 2） |
| v2 V2 | 泛化型范式：训练池/验证集、随机选例、验证钩子、`best_val_checkpoint.pt`、门控 | 修改方案 0.5 节 |
| v2 V3 | 修复 known-issues 8、9、10、12；等价性黄金改为 v2 冻结提交 | 数值全部重做的新实验版本 |
| v2 V4–V6 | 变体 nfe/nm/na；生成器到达模型、结构与案例；`configs/paper.yaml`、2000 轮、`train_version 2` | 意见 5/7/9/11/13、D3 |
| v2 V7 | HSDDQN（层级 DDQN）与 HGNN-PPO (adapted)；对比方法一律关闭流体引导 | 意见 10/12 与 D2 |
| v2 V8 | `baselines/milp_exact.py`：MILP-A/B、验证器、回放、穷举 | 意见 3 |
| v2 V9 | 整数一致性检查、`generalization.csv`、超参表与算例统计表由配置/参数生成、MILP 表 | 意见 2/10/14 |
| v2 V9b | `fluid_backend: auto` 改为 Gurobi → HiGHS → 启发式 | 无 Gurobi 的机器上掩码不再静默失效 |
