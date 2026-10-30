# FSHGRL for DFFSP-HFOI —— 论文代码仓库（v2，修订版实验）

正式代码仓库：`Linshan-Ding/code_Yuhao_Zhu_FSHGRL_DFFSP-HFOI`（本仓库）。`main` 以原仓库的单提交 c030dfa 为根完成了重构
（训练路径与原仓库逐位一致，`CHANGELOG.md` 的 C0–C12）；分支 `claude/v2-generalization`（作者打标签 `v2.0-revision`）是审稿
修订的**新实验版本**：泛化型训练范式、流体候选掩码、缺陷修复、新的对比方法与精确求解检查（`CHANGELOG.md` 的 V1–V11，
修改方案见论文仓库 `revision/reviewer1_response_plan.md`）。实验规模（2026-10-30 定稿）：每个方法训练 1 次、1000 轮，
验证集为基准算例 `DDT500_num50` 与 `DDT1200_num100`，测试集为 15 个基准算例 + 6 个到达模式算例，12 个小算例做精确求解与阈值敏感性。

论文：*Fluid-Guided Sparse Heterogeneous Graph Reinforcement Learning for the Dynamic Flexible Flow Shop
Scheduling Problem with High-Frequency Order Insertion*（FSHGRL）。本仓库按 `opt-paper-codegen` 技能的五大模块骨架
（configs / data / environment / agent / result）组织，是论文实验的复现手册与代码。

- 修订版完整复现步骤与三台机器的预算分配：`docs/reproduce.md`
- 实验设计规格（范式、档位、run 矩阵、落盘契约、参数表 v2）：`docs/experiment-spec.md`；作业数与代价：`docs/run-matrix.md`
- 对比方法与基线的改编范围、精确求解检查、阈值敏感性：`docs/baselines.md`
- 已知缺陷与 v2 处理状态：`docs/known-issues.md`；论文结论与数据文件的对应：`docs/claim-verdicts.md`
- 与原仓库的文件对应：`docs/migration-map.md`；预实验门控（本版关闭）：`docs/pilot-log.md`

## 1. 问题与假设

**问题**：高频插单柔性流水车间（DFFSP-HFOI）。订单按到达时间动态进入，每个订单有交期；目标是**最大化订单按时达成率**
`eta = N_c / S`（N_c 为整条 episode 结束时按期完成的订单数，S 为订单总数，丢弃订单计入分母）。实验契约文件统一用
**最小化**的目标值 `obj = 1 - eta`，`gap` 以各算例最优确定性规则为参考解（`data/instances/reference.csv`）。

**MDP**（`environment/simulator.py`，NumPy 离散事件仿真）：
- 决策点：出现"空闲机器 + 该机器所属工序有等待订单"时；候选动作 = 合法 (机器, 工序类型) 对 × 该工序类型按交期排序的前
  `order_top_k`（5）个订单；
- 流体引导（`configs/env.yaml`）：每个新状态求解一次容量分配 LP（按状态签名缓存），其解有两种用法——`fluid_features`
  两列工序类型节点特征，`fluid_mask` 候选掩码（只保留 u* > `fluid_mask_threshold` = 1e-5 的合法对，空集回退全部合法对，
  回退次数记入 `log.csv`）；求解后端 `fluid_backend: auto` 依次取 Gurobi → HiGHS（`scipy.optimize.linprog`，精确、无许可证）
  → 启发式，实际后端写入 `config_snapshot.yaml`；
- 状态：工序类型节点（12 维）与机器节点（5 维）的异质图，另有候选订单的剩余交期，交期与加工率**按单个观测**归一化；
- 奖励：`(完成增量 - discard_penalty × 丢弃增量) / S`，整条 episode 的累计奖励等于 `2 eta - 1`（`tests/test_rules.py`）；
  订单一旦无法按期完成即被丢弃；非法动作抛错。

**训练范式**：`generalization`（泛化型，`configs/paper.yaml`）。每个方法只训练**一个**策略（1 run，固定种子 1 并记录）：
1000 轮，第 k 轮用训练池的第 k 个生成算例（1000 个，每个恰用一次），采集 8 条完整 episode 做一次 PPO 更新；每 10 轮在两个
基准验证算例 `DDT500_num50`、`DDT1200_num100`（原稿图 7 的两个算例）上各跑一条贪心 episode（`val_curve.csv`），验证均值
最优的检查点 `best_val_checkpoint.pt` 用于评测；评测档位是 15 个基准算例、6 个到达模式算例与 12 个小算例，所有表格报告
零样本结果（两个验证算例同时在测试集内，论文 5.1 节须写明）。旧的逐算例范式（`per_instance`）仍可用（`configs/instance.yaml`
默认），只为与 v1 对照。

**方法**（方法 id 用于目录名与 `eval_results.csv` 的 `method` 列）：

| id | 方法 | 配置 | 说明 |
|---|---|---|---|
| `ours` | FSHGRL：PPO + 流体特征 + 流体掩码 + 稀疏异质图策略（候选自注意力） | `configs/paper.yaml` | |
| `nfe`、`nm`、`nf` | 流体引导的 2×2 消融：去特征 / 去掩码 / 都去 | `configs/ablation/{nf_feature,nf_mask,nf}.yaml` | |
| `ng`、`na` | 逐节点 MLP 编码器替代图编码 / 去候选自注意力 | `configs/ablation/{ng,no_sparse_attn}.yaml` | |
| `hsddqn`、`hgnn_ppo` | 层级 Double DQN（改编）、HGNN-PPO (adapted) | `configs/baselines/{hsddqn,hgnn_ppo}.yaml` | 流体引导关闭，`docs/baselines.md` |
| `a2c`、`dqn`、`ddqn` | 附加学习型基线 | `configs/baselines/*.yaml` | 可省（`run_05` 的 `EXTRA`） |
| `fifo` `spt` `mor` `mwkr` `random` `rrc` | 调度规则 | `baselines/run_rules.py` | 与策略相同的候选集 |
| MILP-A / MILP-B | 论文 3.2 节的 MILP 及其可放弃变体 | `baselines/milp_exact.py` | small 档精确求解 |
| ours @ ε_f | 剪枝阈值敏感性（评测期，不重训） | `configs/sensitivity/*.yaml` | `scripts/run_10` |

训练参数表 v2 = `configs/paper.yaml`（`train_version: 2`，`train_confirmed: true`），论文表 6 由它生成（§8）。

## 2. 环境配置

Python 3.10 以上（本次验证 3.13）；CPU 即可运行，有 NVIDIA GPU 时 PPO 更新默认放在 `cuda:0`（CUDA 不可用自动回退 CPU）。
可选依赖：`gurobipy`（流体 LP 与 MILP 的首选求解器；没有时两者都用 HiGHS）、`visdom`、`psutil`。

```
pip install -r requirements.txt
```

安装自检（单元测试约 80 个，本机约 4 分钟；数值等价回归以 `tests/golden_v2/manifest.json` 的冻结提交为黄金，约 5 分钟）：

```
python -m pytest tests
python -m tests.equivalence --golden v2
```

实测耗时（本机：4 核 CPU、无 GPU、Python 3.13、torch 2.14 CPU 版、每作业 1 个 BLAS 线程、HiGHS 后端）：

| 步骤 | 实测 |
|---|---|
| §3 冒烟 `run_00` | 约 1 min（HiGHS 掩码下每个决策解一次 LP） |
| 一次 PPO 迭代（8 个环境 + 贪心评估，S=50 到 150） | 20–90 s；每 10 轮两条验证 episode 约 3 s |
| 单个 run，1000 轮 | 作者平台约 7 h（FSHGRL）到 11.5 h（NG），见 `docs/run-matrix.md` |
| §5 规则基线 `run_06`（33 个评测算例） | 约 5 min |
| §5 精确求解 `run_08`（12 个小算例 × MILP-A/B） | 约 10 min（S=12 可能触及 600 s 时限） |
| §6 评测，每个检查点（33 算例 × 贪心 + 10 次采样） | 约 30 min；敏感性 `run_10` 约 10 min |

## 3. 冒烟自检（先跑这条，约 1 分钟）

泛化型布局的整条链路（`configs/smoke.yaml`：3 个训练池算例 × 2 轮 × 2 个环境，两个基准验证算例，评测、规则、聚合、出图出表、
单元测试），产物在 `result/smoke/`，不产生论文数据：

```
python scripts/run_00_smoke.py
```

## 4. 数据准备

```
python scripts/run_01_prepare_data.py
```

生成 `configs/instance.yaml` `instance.pools` 的全部档位（种子固定，重跑逐字节相同），为每个算例写 `params.yaml`
（订单数、到达间隔均值与变异系数、负荷率 ρ、交期松弛、加工时间统计），重建 `data/instances/index.csv`（1033 行）：
benchmark 15（外部文件，不重新生成）、train 1000（种子 1000–1999）、arrival 6、small 12；各档位的生成设定见 `docs/reproduce.md` §2。
验证集不是生成档位，而是 `configs/paper.yaml` 指定的两个基准文件。
单独生成算例：`python -m data.generator --help`（到达模型 uniform / exponential / bursty、负荷率 `--rho`、结构覆盖、案例工艺）。

## 5. 训练、规则与精确求解

预实验在本版关闭（`configs/pilot.yaml` `enabled: false`，`run_02` 打印 SKIP），正式训练直接开始；每个方法 1 个作业，
布局 `result/<method>/run1/`（训练参数表未确认时拒绝启动）：

```
python scripts/run_03_train_ours.py
python scripts/run_04_train_ablations.py
python scripts/run_05_train_learned_baselines.py
```

产物：`result/<method>/run1/`，内含 `train_metrics.csv`（逐迭代记录）、`curve.csv`（逐轮贪心目标值）、
`val_curve.csv`（每 10 轮两个验证算例的贪心达成率，图 7 的数据）、`log.csv`（训练诊断：`approx_kl`、掩码统计、验证均值等）、
`best_val_checkpoint.pt`（§6 评测默认用它）、`best_greedy_checkpoint.pt`、`best_checkpoint.pt`、`last_checkpoint.pt`
（断点续训）、`config_snapshot.yaml`（生效配置、范式、训练池大小、验证设置、评测档位、实际流体后端、种子、git 提交）、
`train_meta.yaml`、`fixed_set_final.csv`。被中断的作业重跑时续训；已完成的作业自动跳过。实时曲线：另开终端
`python -m visdom.server` 并把 `configs/algo.yaml` 的 `viz.viz` 置 true。多进程采集（`configs/env.yaml` 的 `collector_workers`）
与进程内路径逐位一致，开启前用 `tools/profile_collector.py` 实测。

规则基线与精确求解是纯 CPU 作业，`run_all.py` 让它们与训练重叠：

```
python scripts/run_06_run_rules.py
python scripts/run_08_milp_check.py
```

产物：`result/rules/<rule>/eval_results.csv`、`result/rules.csv`（benchmark、arrival、small 三档）与 `data/instances/reference.csv`；
`result/milp/milp_results.csv`（small 档 MILP-A/B 的最优值、状态、求解时间、验证器与仿真器回放核对）。

## 6. 评测、敏感性与实验数据生成

```
python scripts/run_07_eval_main.py
python scripts/run_10_sensitivity.py
```

`run_07` 对 `result/<method>/run1/best_val_checkpoint.pt` 按该 run 快照的 `eval_tiers`（benchmark + arrival + small）各跑贪心 1 条
+ `eval.sampled_episodes`（10）条采样 episode（逐条独立），带决策时间分项，再与规则结果合并。`run_10` 用 `ours/run1` 在 small 档上
以 ε_f ∈ {off, 1e-5, 0.05, 0.1, 0.2} 评测（不重训）。

产物：`result/eval_results.csv`，每行一个 (算例, 方法, run, 解码方式, episode)，契约列 `instance_id, method, run_id, tier,
scale, obj_model, obj_ref, gap, feasible, infer_time_ms`，其后 `decode, episode, eval_batch, fulfillment_rate, completed,
discarded, total_orders, decision_count, policy_ms, obs_ms, fluid_ms, sim_ms, fluid_backend, candidate_pairs_mean,
masked_pairs_mean, mask_fallback_rate, ddt, checkpoint, checkpoint_iteration`；聚合时另算每次决策的时间
`decision_time_ms = infer_time_ms / decision_count`（论文的"time"口径）。另有 `result/incumbents.csv`、
`result/sensitivity/eps_<v>/eval_results.csv` 与 `result/sensitivity.csv`（含 `threshold` 列）。
单个检查点的评测：`python eval.py --help`（`--tiers` 指定档位，`--config` 叠加评测期覆盖层）。

## 7. 统计聚合

```
python scripts/run_09_aggregate_stats.py
```

产物：`result/integer_check.csv`（每 (算例, 方法, 解码) 组 S × episode 数 × 平均 η 必为按期订单数，违反即中止）、
`result/stats_summary.csv`、`result/generalization.csv`（方法 × 档位：η 均值、`completed_mean`、每决策时间、最优确定性规则与
逐算例胜/平/负）、`result/paired_tests.csv`（基准档 15 个配置的配对 Wilcoxon：精确 p 值、Holm 校正）、`result/ablation_results.csv`、
`advantage_table.csv`、`friedman.csv`、`conv_<method>.csv`（验证目标值随轮数）；
图：`paper_assets/figures/generated/`（图 7 的两条训练曲线来自 `val_curve.csv`）；表（`paper_assets/tables/`，v2 表同时落 CSV）：
`ablation.tex`、`comparison_pdr.tex`、`comparison_drl.tex`、`wilcoxon.tex`（基准档）、`generalization_arrival.tex`、
`generalization_small.tex`（零样本档位）、`hyperparameters.tex`（表 6，由配置生成）、`instance_statistics.tex`（基准算例统计，
由 `params.yaml` 生成）、`milp_small.tex`（精确求解检查 + 最优规则 + FSHGRL 差距）、`sensitivity.tex`（阈值敏感性）。

一键跑完 §4–§7（按依赖并发调度：数据 → 训练三组作业 ‖ 规则 ‖ 精确求解 → 评测 ‖ 敏感性 → 聚合）：

```
python scripts/run_all.py
```

## 8. 产物与论文表格的对应

| 论文 | 表体 / 数据 | 生成脚本 |
|---|---|---|
| 表 5（生成设定）、算例统计表 | `configs/instance.yaml` `generator` 段、`instance_statistics.tex` | run_01、run_09 |
| 表 6 | `hyperparameters.tex` | run_09 |
| 表 7、图 8 | `ablation.tex`、`ablation_results.csv`、`advantage_table.csv` | run_09 |
| 表 8、10、11 | `comparison_pdr.tex`、`comparison_drl.tex`、`wilcoxon.tex` | run_09 |
| 图 7、收敛图、决策时间图 | `result/ours/run1/val_curve.csv`、`conv_<method>.csv`、`eval_results.csv` | run_09 |
| 新 5.x 到达模式 | `generalization_arrival.tex`、`generalization.csv` | run_09 |
| 新 5.x 阈值敏感性 | `sensitivity.tex`、`result/sensitivity.csv` | run_10、run_09 |
| 附录精确求解 | `milp_small.tex`、`result/milp/milp_results.csv` | run_08、run_09 |
| 表 13（工业案例） | 沿用原稿结果（不重做；最优派工规则 = 现场做法） | — |

## 9. 启动方式与并发度

### 9.1 命令行启动（默认方式，跨平台）

§3–§7 的命令零参数照抄即可，每条用脚本内置的默认并发度。想改并发路数时在命令后加一个数字（bash 与 PowerShell
写法一致）：

```
python scripts/run_04_train_ablations.py 3
python scripts/run_all.py 1
```

这个数字是**上限而非承诺**：脚本按 CPU 核数、可用内存（有 GPU 时按空闲显存）把它夹到设备撑得住的路数并打印理由，
例如 `[PARALLEL] requested 4, clamped to 3: 4 cores - 0 environment workers - 1`；作业因内存不足失败时并发度自动降
一档并重排该作业，下限 1 路。传 `1` 即全串行，排障时用。每个作业写自己的 `result/<作业名>/stdout.log`，控制台只打印
进度表。一次改全部脚本的默认值就改 `configs/parallel.yaml`（`max_concurrent`，以及每作业的 BLAS 线程数
`threads_per_job`，默认 1 以保证各作业耗时可比）。三台 RTX 5060 台式机的分配建议见 `docs/reproduce.md` §8。

| 步骤 | 可并行 | 资源 | 默认路数 |
|---|---|---|---|
| §4 数据准备 | 否 | — | 1 |
| §5 训练（ours、消融、对比方法） | 是（作业彼此独立） | CPU/GPU | 2 |
| §5 规则基线、精确求解 | 是，且与训练重叠 | CPU | 1（脚本内串行） |
| §6 评测、敏感性 | 是（每检查点 / 每阈值一个作业） | CPU | 3 |
| §7 统计聚合 | 否 | — | 1 |

### 9.2 在 PyCharm 中手动启动

所有 `scripts/run_*.py` 都是零参数脚本，PyCharm 里无需任何配置即可运行。

**方式 A（推荐）**：Project 视图里右键目标脚本（如 `scripts/run_03_train_ours.py`）→ **Run**。长训练建议用这种方式：
Run 窗口可随时停止、看完整输出、加断点。

**方式 B（手动新建运行配置）**：Run → Edit Configurations → `+` → Python，逐字段填：

| 字段 | 取值 |
|---|---|
| Name | run_03_train_ours |
| Script path | 工程根目录下的 `scripts/run_03_train_ours.py` |
| Parameters | **留空**即默认并发度；要改并发就填一个数字（如 `3`） |
| Working directory | **工程根目录**（不是 `scripts/`） |
| Python interpreter | 项目虚拟环境解释器 |
| Environment variables | `PYTHONUNBUFFERED=1`；多卡机器另加 `CUDA_VISIBLE_DEVICES=0` |

visdom 面板单独建一个配置：Script path 下拉切换为 **Module name**，填 `visdom.server`，Parameters 留空。

**常见坑**：Working directory 填成 `scripts/` 会让相对路径全部找不到（脚本内已 `os.chdir` 到工程根兜底，但配置仍应
填对）；PyCharm 内置终端在 Windows 上是 PowerShell，本 README 的命令已避开 bash 专有语法，照抄即可；解释器别选成
系统 Python；工程路径含空格不影响以上命令。

## 10. 产物对照表（终检）

| 实验数据文件 | 生成脚本（节号） | 服务论文哪张表/图 |
|---|---|---|
| `data/instances/benchmark/<id>/<id>.txt` | 随仓库发布（原 `data/instance/competition/`，文件名未改） | 固定测试集；其中两个兼作验证集 |
| `data/instances/{train,arrival,small}/<id>/{<id>.txt,params.yaml}`、`index.csv` | run_01（§4） | 训练池；到达模式档；精确求解 / 敏感性档；算例统计表 |
| `data/instances/reference.csv`、`result/rules.csv` | run_06（§5） | 规则对比表（表 8）、到达 / small 档的规则列、所有 `gap` 的参考解 |
| `result/<method>/run1/{train_metrics,curve,val_curve,log}.csv` | run_03–05（§5） | 图 7（`val_curve.csv`）、收敛曲线、模型选择记录 |
| `result/<method>/run1/*.pt`、`config_snapshot.yaml`、`train_meta.yaml` | run_03–05 | 评测输入；复现记录（配置、后端、种子、提交、停止原因） |
| `result/milp/milp_results.csv` | run_08（§5） | 附录精确求解表（`milp_small.tex`） |
| `result/eval_results.csv`、`incumbents.csv` | run_07（§6） | 表 7–11 的达成率与决策时间列；决策时间图 |
| `result/sensitivity.csv` | run_10（§6） | 阈值敏感性表（`sensitivity.tex`） |
| `result/integer_check.csv`、`stats_summary.csv`、`generalization.csv` | run_09（§7） | 意见 14 的机检；均值；新 5.x 节 |
| `result/paired_tests.csv`、`friedman.csv`、`ablation_results.csv`、`advantage_table.csv`、`conv_<method>.csv` | run_09（§7） | 表 11；表 7；优势热力图；收敛曲线图 |
| `paper_assets/figures/generated/*.pdf`、`paper_assets/tables/*.tex` | run_09（§7） | 论文图表成品（源脚本在 `paper_assets/figures/src/`、`paper_assets/scripts/`） |
| `paper_assets/figures/legacy/*` | 原仓库成品图（不可再生成） | 原投稿的训练曲线图 |

## 附：原仓库命令的对应

旧入口 `python agent/train.py` 仍可用（转发到根目录 `train.py`），旧 import 路径（`agent.HGHH_model`、
`agent.PPO_model`、`agent.sci_rollout`、`agent.training_simulator`、`agent.visualization`、`compare.MLP_model`）
保留为转发模块；旧 JSON 配置可用 `python train.py --config path/to/config.json` 直接加载。完整映射见
`docs/migration-map.md`。
