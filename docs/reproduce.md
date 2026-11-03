# 复现手册（从干净检出到论文全部表格）

所有命令在工程根目录执行，零参数；每条命令的产物与对应论文表格见 §7 与 README §10。`result/` 已含论文实验的记录
（检查点除外），重跑脚本会覆盖它们；论文仓库只复制 `paper_assets/` 的产物。

实验规模：每个方法训练 **1 次**、**1000 轮**（每轮 8 条 episode，每轮一个新的训练池算例）；验证集为基准算例
`DDT500_num50` 与 `DDT1200_num100`（图 7 的两个算例），每 10 轮各一条贪心 episode；测试集为 15 个基准算例 + 6 个到达模式算例；
精确求解与剪枝阈值敏感性共用 12 个基准结构的小算例；不设预实验；工业案例不由本仓库生成。

## 0. 自检

```
python -m pytest tests
```

单元测试约 80 个（含 MILP、聚合、敏感性、对比方法、掩码、泛化范式、配置桥接），本机 4 核 CPU 约 4 分钟。

## 1. 环境

Python ≥ 3.10（验证 3.13），`pip install -r requirements.txt`。流体 LP 与 MILP 的求解器都是 scipy 自带的 HiGHS
（`fluid_backend: scipy`，`scipy.optimize.linprog`；MILP 用 `scipy.optimize.milp`）：精确、无许可证、单线程确定性，跨机器数值一致。
实际后端写入每个 run 的 `config_snapshot.yaml`（`fluid_backend`）与 `eval_results.csv`。
GPU：PPO 更新默认 `cuda:0`，不可用时回退 CPU；采集始终在 CPU。

## 2. 数据（`python scripts/run_01_prepare_data.py`）

生成 `configs/instance.yaml` `instance.pools` 的全部档位（种子固定，重跑得到逐字节相同的文件），为每个算例写 `params.yaml`，
重建 `data/instances/index.csv`（1033 行）：

| 档位 | 个数 | 生成设定 | 用途 |
|---|---|---|---|
| benchmark | 15 | 外部文件，不重新生成 | 未见测试集（表 7–10）；其中 `DDT500_num50`、`DDT1200_num100` 兼作验证集 |
| train | 1000 | 5×5×5，S∈[50,150]，DDT∈[500,1600]，松弛 [0.6,1.55]×DDT，到达 rand[1,40] s，种子 1000–1999 | 训练池：第 k 轮用第 k 个算例，每个算例恰用一次 |
| arrival | 6 | S=100，DDT=900；ρ∈{0.8,1.2,1.8,2.6} 均匀、ρ=1.8 指数、ρ=1.8 突发 | 到达模式零样本评测（表 11） |
| small | 12 | 5×5×5，S∈{6,8,10,12} × DDT∈{500,900,1500}，种子 5000 | 精确求解检查（附录 A）与剪枝阈值敏感性（表 12） |

验证集不是生成档位：`configs/instance.yaml` 的 `val_data_dir: data/instances/benchmark`、`val_instances: [DDT500_num50.txt, DDT1200_num100.txt]`。
这两个算例同时在测试集内，论文 5.1 节写明"模型选择只用这两个算例的贪心达成率"。
工业案例的工艺数据（`data/case/process.yaml`、`data/case_structure.py`）作为生成器能力保留，不生成案例档位。

## 3. 冒烟（`python scripts/run_00_smoke.py`）

泛化型布局的整条链路（3 个训练池算例 × 2 轮、两个基准验证算例、评测、规则、聚合、出图出表、单元测试），产物在 `result/smoke/`（不入库）；
本机约 1 分钟（HiGHS 掩码下每个决策解一次 LP）。

## 4. 训练（`run_03`、`run_04`、`run_05`；无预实验）

```
python scripts/run_03_train_ours.py
python scripts/run_04_train_ablations.py
python scripts/run_05_train_learned_baselines.py
```

每个方法一个作业，布局 `result/<method>/run1/`，固定种子 1（`configs/runs.yaml` 全部为 1 run）。配置：`configs/{instance,env,algo}.yaml`
（1000 轮 × 8 episode，`instance_selection: round_robin`，每 10 轮在两个验证算例上各跑一条贪心 episode → `val_curve.csv`，
验证均值最优轮存 `best_val_checkpoint.pt`），变体覆盖层见 `configs/ablation/`、`configs/baselines/`。
被中断的作业从 `last_checkpoint.pt` 续训，完成的作业（有 `train_meta.yaml`）跳过。`run_02_pilot.py` 关闭（`configs/pilot.yaml`
`enabled: false`），直接开始正式训练。方法清单：

| 脚本 | 方法 id | 覆盖层 |
|---|---|---|
| run_03 | ours | — |
| run_04 | nfe / nm / nf / ng / na | `ablation/{nf_feature,nf_mask,nf,ng,no_sparse_attn}.yaml` |
| run_05 | hsddqn / hgnn_ppo（+ 可选 a2c / dqn / ddqn，脚本顶部 `EXTRA`） | `baselines/*.yaml`（流体引导关闭） |

## 5. 规则与精确求解（CPU，与训练并行）

```
python scripts/run_06_run_rules.py
python scripts/run_08_milp_check.py
```

规则在 benchmark、arrival、small 三个档位上运行（Random/RRC 各 10 次），产物 `result/rules.csv`、`result/rules/<rule>/eval_results.csv`、
`data/instances/reference.csv`。精确求解在 small 档上求论文的 MILP 与其可放弃变体（600 s 时限），每个解经验证器与仿真器回放核对
（穷举交叉核对在该结构下按规模跳过，留在测试套件的微型算例上），产物 `result/milp/milp_results.csv`。

## 6. 评测与敏感性（`run_07`、`run_10`）

```
python scripts/run_07_eval_main.py
python scripts/run_10_sensitivity.py
```

`run_07`：对每个 `result/<method>/run1/best_val_checkpoint.pt`，按快照的 `eval_tiers`（benchmark + arrival + small）在每个算例上跑
贪心 1 条 + 采样 10 条（逐条独立），与规则行合并为 `result/eval_results.csv`；`result/incumbents.csv` 收集训练末在位解。
`run_10`：用 `ours/run1` 在 small 档上以 ε_f ∈ {off, 1e-5, 0.05, 0.1, 0.2}（`configs/sensitivity/`）评测，不重训，
产物 `result/sensitivity/eps_<v>/eval_results.csv` 与 `result/sensitivity.csv`。

## 7. 聚合与出表（`python scripts/run_09_aggregate_stats.py`）

聚合默认用论文的比较集（`configs/runs.yaml` `exclude_methods: [spt]`）；SPT 的记录仍在 `result/`。

| 产物 | 内容 | 论文 |
|---|---|---|
| `result/integer_check.csv` | 每 (算例, 方法, 解码) 组 S × episode 数 × 平均 η 为整数，违反即中止 | 表格值 S·η 为整数的机检 |
| `result/stats_summary.csv` | 方法 × 档位的均值（单 run，标准差留空） | 表 7–9 的均值行 |
| `result/generalization.csv` | 方法 × 档位：η 均值、`completed_mean`、每决策时间、最优规则与胜/平/负 | 表 11（到达模式） |
| `result/paired_tests.csv`、`friedman.csv` | 基准档 15 个配置的配对 Wilcoxon（精确、Holm）与 Friedman | 表 10 |
| `result/ablation_results.csv`、`advantage_table.csv` | 基准档逐配置消融与百分点优势 | 表 7、图 8、图 11 |
| `result/conv_<method>.csv` | 验证目标值随轮数（两个验证算例均值） | 收敛图（仓库产物） |
| `paper_assets/tables/ablation.tex`、`comparison_pdr_grouped.tex`、`comparison_drl_grouped.tex`、`wilcoxon.tex` | 基准档表体 | 表 7、8、9、10 |
| `paper_assets/tables/generalization_arrival.tex`、`generalization_small.tex` | 零样本表体（含 CSV） | 表 11 |
| `paper_assets/tables/hyperparameters.tex` | 由配置生成的超参表 | 表 6 |
| `paper_assets/tables/generator_settings.tex`、`instance_statistics.tex` | 生成设定与基准算例统计（由配置与 params.yaml 生成） | 表 5 |
| `paper_assets/tables/milp_small.tex`、`milp_small_compact.tex` | 全表：两个变体的最优值、状态、求解时间、回放核对、最优规则、FSHGRL η 与差距；紧凑表：只含论文模型的最优值、求解时间、回放与 FSHGRL η | 附录 A |
| `paper_assets/tables/sampled_decoding.tex` | 采样解码的均值 | 附录 C |
| `paper_assets/tables/sensitivity.tex` | 各阈值的贪心/采样 η、剪枝比例、候选对数、每决策时间（CSV 另含与 η* 的差距与回退率） | 表 12 |
| `paper_assets/figures/generated/` | `train_curve`（图 7：两个验证算例的 `val_curve.csv` + 最佳检查点，各面板纵轴按自身曲线范围缩放，图例在图下）、`convergence`（仓库产物，稿件不用）、`ablation_paired_gains`（含 W/T/L 与 Holm p）、`baseline_advantage_heatmap`（含 Mean 列）、`decision_time_configurations`（ms，三方法）、`quality_time_tradeoff`（单面板、对数 ms、右侧图例）、`sensitivity_threshold` | 图 7–12 |

一键顺序：`python scripts/run_all.py`（数据 → 训练 ‖ 规则 ‖ MILP → 评测 ‖ 敏感性 → 聚合）。

## 8. 三台 RTX 5060 台式机的预算与分配

单作业耗时按训练日志（i5-14600KF + RTX 4060 Ti）折算到 1000 轮：FSHGRL 约 7 h，NF 9 h，NG 11.5 h，其余变体 7 h，对比方法 8 h；
每轮耗时由 CPU 侧 Python 主导，RTX 5060 与 4060 Ti 的差异可忽略；每作业显存 0.3–0.4 GB；每 10 轮两条验证 episode 约 +3% 墙钟。

| 块 | 作业 | GPU·h |
|---|---|---|
| ours | 1 | 7 |
| 消融 nfe / nm / nf / ng / na | 5 | 41.5 |
| 对比方法 hsddqn / hgnn_ppo | 2 | 16 |
| 附加基线 a2c / dqn / ddqn（可省） | 3 | 24 |
| 合计 | 8（+3） | ≈ 65（+24） |

建议分配（每台 3 个并发作业）：机器 1 跑 ours、nfe、nm；机器 2 跑 nf、ng、na；机器 3 跑 hsddqn、hgnn_ppo（与可选的附加基线）；
规则、MILP 与敏感性在任一台的 CPU 上随时跑。纯运行墙钟约 12 h（最长作业 ng），评测约 4 CPU·h 可并行，全部实验约 1 天。
正式开跑前每台以 1 / 3 个并发作业各跑 20 轮，读 `log.csv` 的墙钟列校准上表。

## 9. 复现检查清单

- 每个 run 的 `config_snapshot.yaml`：`paradigm: generalization`，`seed: 1`，`fluid_backend` 为 scipy（不是 heuristic_fallback）；
  `train_meta.yaml` 的停止原因为 `max_iterations`；
- `result/integer_check.csv` 全部 `ok`；`result/eval_results.csv` 的 `fluid_backend` 列与训练一致；
- `result/milp/milp_results.csv`：全部 `optimal`（否则报告 gap），`replay_ok` 全真，两个变体的 `eta_opt` 相等；
- `result/sensitivity.csv` 含 5 个阈值 × 12 算例 × 11 episode；
- 表格由 `run_09` 生成后再写入论文，不手改数字。
