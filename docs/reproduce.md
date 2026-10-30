# 复现手册（v2，泛化型范式；实验规模定于 2026-10-30）

本文件是修订版实验（审稿人 1 修改方案 v3 的 C 线）的完整复现步骤：从干净检出到论文全部表格。所有命令在工程根目录执行，
零参数；每条命令的产物与对应论文表格见 §8 与 README §10。代码冻结于分支 `claude/v2-generalization` 的末端提交，作者在其上打标签
`v2.0-revision` 后以该标签为准（`git tag -a v2.0-revision -m "revision code freeze"` 与 `git push origin v2.0-revision`）。

实验规模（用户决定）：每个方法训练 **1 次**、**1000 轮**（每轮 8 条 episode，每轮一个新的训练池算例）；验证集为基准算例
`DDT500_num50` 与 `DDT1200_num100`（原稿训练曲线的两个算例），每 10 轮各一条贪心 episode；测试集为 15 个基准算例 + 6 个到达模式算例；
精确求解与剪枝阈值敏感性共用 12 个基准结构的小算例；不设预实验；不做工业案例实验。

## 0. 版本与自检

```
python -m pytest tests
python -m tests.equivalence --golden v2
```

- 单元测试约 80 个（含 MILP、聚合、敏感性、对比方法、掩码、泛化范式），本机 4 核 CPU 约 4 分钟；
- 数值等价回归以 `tests/golden_v2/manifest.json` 记录的冻结提交为黄金（HiGHS 掩码、两个基准验证算例、3 轮训练，hgnn 与 mlp），
  比对 `train_metrics.csv` 的 57 个浮点列、`val_curve.csv` 与全部检查点张量；只有有意改变数值或参考运行的输入时才
  `--regenerate-golden`，并在 `CHANGELOG.md` 记录。

## 1. 环境

Python ≥ 3.10（验证 3.13），`pip install -r requirements.txt`。流体 LP 的求解器由 `env.fluid_backend` 决定：`auto`
依次取 Gurobi（`gurobipy` 可导入）→ HiGHS（`scipy.optimize.linprog`，精确、无许可证）→ 启发式回退；论文运行用
`fluid_threads: 1` 保证跨机一致，实际后端写入每个 run 的 `config_snapshot.yaml`（`fluid_backend`）与 `eval_results.csv`。
GPU：PPO 更新默认 `cuda:0`，不可用时回退 CPU；采集始终在 CPU。

## 2. 数据（`python scripts/run_01_prepare_data.py`）

生成 `configs/instance.yaml` `instance.pools` 的全部档位（种子固定，重跑得到逐字节相同的文件），为每个算例写 `params.yaml`，
重建 `data/instances/index.csv`（1033 行）：

| 档位 | 个数 | 生成设定 | 用途 |
|---|---|---|---|
| benchmark | 15 | 外部文件，不重新生成 | 未见测试集（表 7–11）；其中 `DDT500_num50`、`DDT1200_num100` 兼作验证集 |
| train | 1000 | 5×5×5，S∈[50,150]，DDT∈[500,1600]，松弛 [0.6,1.55]×DDT，到达 rand[1,40] s，种子 1000–1999 | 训练池：第 k 轮用第 k 个算例，每个算例恰用一次 |
| arrival | 6 | S=100，DDT=900；ρ∈{0.8,1.2,1.8,2.6} 均匀、ρ=1.8 指数、ρ=1.8 突发 | 到达模式零样本评测 |
| small | 12 | 5×5×5，S∈{6,8,10,12} × DDT∈{500,900,1500}，种子 5000 | 精确求解检查（MILP-A/B）与剪枝阈值敏感性 |

验证集不是生成档位：`configs/paper.yaml` 的 `val_data_dir: data/instances/benchmark`、`val_instances: [DDT500_num50.txt, DDT1200_num100.txt]`。
这两个算例同时在测试集内，论文 5.1 节须写明"模型选择只用这两个算例的贪心达成率"。
工业案例的工艺数据（`data/case/process.yaml`、`data/case_structure.py`）作为生成器能力保留，本版不生成案例档位。

## 3. 冒烟（`python scripts/run_00_smoke.py`）

泛化型布局的整条链路（3 个训练池算例 × 2 轮、两个基准验证算例、评测、规则、聚合、出图出表、单元测试），产物在 `result/smoke/`；
本机约 1 分钟（HiGHS 掩码下每个决策解一次 LP）。

## 4. 训练（`run_03`、`run_04`、`run_05`；无预实验）

```
python scripts/run_03_train_ours.py
python scripts/run_04_train_ablations.py
python scripts/run_05_train_learned_baselines.py
```

每个方法一个作业，布局 `result/<method>/run1/`，固定种子 1（`configs/runs.yaml` 全部为 1 run）。配置层：`configs/paper.yaml`
（1000 轮 × 8 episode，`instance_selection: round_robin`，每 10 轮在两个验证算例上各跑一条贪心 episode → `val_curve.csv`，
验证均值最优轮存 `best_val_checkpoint.pt`），变体覆盖层见 `configs/ablation/`、`configs/baselines/`。
被中断的作业从 `last_checkpoint.pt` 续训，完成的作业（有 `train_meta.yaml`）跳过。`run_02_pilot.py` 在本版关闭（`configs/pilot.yaml`
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
`data/instances/reference.csv`。精确求解在 small 档上求 MILP-A/B（600 s 时限），每个解经验证器与仿真器回放核对（穷举交叉核对在该结构下
按规模跳过，留在测试套件的微型算例上），产物 `result/milp/milp_results.csv`。

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

| 产物 | 内容 | 论文 |
|---|---|---|
| `result/integer_check.csv` | 每 (算例, 方法, 解码) 组 S × episode 数 × 平均 η 为整数，违反即中止 | 意见 14 的回复依据（单次训练下表格值 S·η 为整数） |
| `result/stats_summary.csv` | 方法 × 档位的均值（单 run，标准差留空） | 表 7–10 的均值行 |
| `result/generalization.csv` | 方法 × 档位：η 均值、`completed_mean`、每决策时间、最优规则与胜/平/负 | 新 5.x 节（到达模式） |
| `result/paired_tests.csv`、`friedman.csv` | 基准档 15 个配置的配对 Wilcoxon（精确、Holm）与 Friedman | 表 11 |
| `result/ablation_results.csv`、`advantage_table.csv` | 基准档逐配置消融与百分点优势 | 表 7、图 8 |
| `result/conv_<method>.csv` | 验证目标值随轮数（两个验证算例均值） | 收敛图 |
| `paper_assets/tables/ablation.tex`、`comparison_pdr.tex`、`comparison_drl.tex`、`wilcoxon.tex` | 基准档表体 | 表 7、8、10、11 |
| `paper_assets/tables/generalization_arrival.tex`、`generalization_small.tex` | 零样本表体（含 CSV） | 新 5.x |
| `paper_assets/tables/hyperparameters.tex` | 由配置生成的超参表 | 表 6 |
| `paper_assets/tables/instance_statistics.tex` | 基准算例统计（由 params.yaml 生成） | 新表（意见 2） |
| `paper_assets/tables/milp_small.tex` | MILP-A/B 最优值、状态、求解时间、回放核对、最优规则、FSHGRL η 与差距 | 附录（意见 3） |
| `paper_assets/tables/sensitivity.tex` | 各阈值的 η、与 η* 的差距、候选对数、回退率、每决策时间 | 新 5.x（意见 5） |
| `paper_assets/figures/generated/` | 训练曲线（两个验证算例的 `val_curve.csv`，图 7）、收敛、配对差值、决策时间、热力图 | 图 7–10 |

一键顺序：`python scripts/run_all.py`（数据 → 训练 ‖ 规则 ‖ MILP → 评测 ‖ 敏感性 → 聚合）。

## 8. 三台 RTX 5060 台式机的预算与分配

单作业耗时按归档日志（i5-14600KF + RTX 4060 Ti）折算到 1000 轮：FSHGRL 约 7 h，NF 9 h，NG 11.5 h，其余变体 7 h，对比方法 8 h；
每轮耗时由 CPU 侧 Python 主导，RTX 5060 与 4060 Ti 的差异可忽略；每作业显存 0.3–0.4 GB；每 10 轮两条验证 episode 约 +3% 墙钟。

| 块 | 作业 | GPU·h |
|---|---|---|
| ours | 1 | 7 |
| 消融 nfe / nm / nf / ng / na | 5 | 41.5 |
| 对比方法 hsddqn / hgnn_ppo | 2 | 16 |
| 附加基线 a2c / dqn / ddqn（可省） | 3 | 24 |
| 合计 | 8（+3） | ≈ 65（+24） |

建议分配（每台 3 个并发作业）：机器 1 跑 ours、nfe、nm；机器 2 跑 nf、ng、na；机器 3 跑 hsddqn、hgnn_ppo（与可选的附加基线）；
规则、MILP 与敏感性在任一台的 CPU 上随时跑。纯运行墙钟约 12 h（最长作业 ng），评测约 4 CPU·h 可并行，整个 C 线约 1 天。
正式开跑前每台以 1 / 3 个并发作业各跑 20 轮，读 `log.csv` 的墙钟列校准上表。

## 9. 复现检查清单

- 每个 run 的 `config_snapshot.yaml`：`git_commit` 为冻结提交，`train_version: 2`，`paradigm: generalization`，`seed: 1`，
  `fluid_backend` 为 gurobi 或 scipy（不是 heuristic_fallback）；`train_meta.yaml` 的停止原因为 `max_iterations`；
- `result/integer_check.csv` 全部 `ok`；`result/eval_results.csv` 的 `fluid_backend` 列与训练一致；
- `result/milp/milp_results.csv`：全部 `optimal`（否则报告 gap），`replay_ok` 全真，MILP-A 与 MILP-B 的 `eta_opt` 相等；
- `result/sensitivity.csv` 含 5 个阈值 × 12 算例 × 11 episode；
- 表格由 `run_09` 生成后再写入论文，不手改数字。
