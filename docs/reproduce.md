# 复现手册（v2，泛化型范式）

本文件是修订版实验（审稿人 1 修改方案 v2 的 C 线）的完整复现步骤：从干净检出到论文全部表格。所有命令在工程根目录执行，
零参数；每条命令的产物与对应论文表格见 §8 与 README §10。代码冻结于标签 `v2.0-revision`（分支 `claude/v2-generalization`）。

## 0. 版本与自检

```
python -m pytest tests
python -m tests.equivalence --golden v2
```

- 单元测试约 80 个（含 MILP、聚合、对比方法、掩码、泛化范式），本机 4 核 CPU 约 3 分钟；
- 数值等价回归以 `tests/golden_v2/manifest.json` 记录的冻结提交为黄金（HiGHS 掩码、2 个验证算例、3 轮训练，hgnn 与 mlp），
  比对 `train_metrics.csv` 的 57 个浮点列、`val_curve.csv` 与全部检查点张量；只有有意改变数值时才 `--regenerate-golden`，
  并在 `CHANGELOG.md` 记录。

## 1. 环境

Python ≥ 3.10（验证 3.13），`pip install -r requirements.txt`。流体 LP 的求解器由 `env.fluid_backend` 决定：`auto`
依次取 Gurobi（`gurobipy` 可导入）→ HiGHS（`scipy.optimize.linprog`，精确、无许可证）→ 启发式回退；论文运行用
`fluid_threads: 1` 保证跨机一致，实际后端写入每个 run 的 `config_snapshot.yaml`（`fluid_backend`）与 `eval_results.csv`。
GPU：PPO 更新默认 `cuda:0`，不可用时回退 CPU；采集始终在 CPU。

## 2. 数据（`python scripts/run_01_prepare_data.py`）

生成 `configs/instance.yaml` `instance.pools` 的全部档位（种子固定，重跑得到逐字节相同的文件），为每个算例写 `params.yaml`，
重建 `data/instances/index.csv`（568 行）：

| 档位 | 个数 | 生成设定 | 用途 |
|---|---|---|---|
| benchmark | 15 | 外部文件，不重新生成 | 未见测试集（表 7–11） |
| train | 200 | 5×5×5，S∈[50,150]，DDT∈[500,1600]，松弛 [0.6,1.55]×DDT，到达 rand[1,40] s，种子 1000–1199 | 训练池 |
| val | 30 | S∈{50,100,150}×DDT∈{500,600,900,1200,1600}×2 种子（2000 起） | 验证集（模型选择、预实验门控） |
| arrival | 6 | S=100，DDT=900；ρ∈{0.8,1.2,1.8,2.6} 均匀、ρ=1.8 指数、ρ=1.8 突发 | 到达模式零样本评测 |
| arrival_train | 120 | 到达模型与负荷混合 | `ours_arrival` 训练池 |
| tiny / tiny5 | 50 / 30 | 2×3×2 结构 S∈{4,…,12} / 5×5×5 结构 S∈{6,8,10} | 精确求解检查 |
| case_train / case_val / case_eval | 100 / 8 / 9 | 论文表 12 的工艺与设备（分钟），ρ=1.8 | 案例训练池 / 验证 / 9 个评测配置 |

## 3. 冒烟（`python scripts/run_00_smoke.py`）

泛化型布局的整条链路（3 个训练池算例 × 2 轮、2 个验证算例、评测、规则、聚合、出图出表、单元测试），产物在 `result/smoke/`；
本机约 1–2 分钟（HiGHS 掩码下每个决策解一次 LP）。

## 4. 预实验门控（`python scripts/run_06_run_rules.py`，再 `python scripts/run_02_pilot.py`）

先跑规则基线得到验证集的参考值 `data/instances/reference.csv`（每算例最优确定性规则），再训练 FSHGRL 1 run ×
`configs/pilot.yaml` 的 `max_iterations`（1000 轮，每 50 轮验证）。判定（`result/pilot/verdict.txt`、`pilot-report.md`）：
最佳验证轮的贪心均值 ≥ 规则参考均值，且在 ≥ 2/3（`configs/runs.yaml` `pilot.min_win_share`）的验证算例上达到参考值即 PASS。
FAIL 时按 `docs/pilot-log.md` 的顺序调整并复跑（最多 2 轮）；PASS 后 `run_02` 自动跳过。

## 5. 训练（`run_03`、`run_04`、`run_05`）

```
python scripts/run_03_train_ours.py
python scripts/run_04_train_ablations.py
python scripts/run_05_train_learned_baselines.py
```

每个 run 一个作业，布局 `result/<method>/run<k>/`，`configs/runs.yaml` 定次数（ours 与消融各 3，对比方法 3，
附加基线 a2c/dqn/ddqn 各 1）。配置层：`configs/paper.yaml`（2000 轮 × 8 episode，每轮随机取一个训练池算例，每 50 轮在 30 个
验证算例上各跑一条贪心 episode → `val_curve.csv` 与 `best_val_checkpoint.pt`），变体覆盖层见 `configs/ablation/`、
`configs/baselines/`、`configs/pools/`。被中断的作业从 `last_checkpoint.pt` 续训，完成的作业（有 `train_meta.yaml`）跳过。
方法清单：

| 脚本 | 方法 id | 训练池 | 覆盖层 |
|---|---|---|---|
| run_03 | ours / ours_arrival / ours_case | train / arrival_train / case_train | — / `pools/arrival_train.yaml` / `pools/case_train.yaml` |
| run_04 | nfe / nm / nf / ng / na | train | `ablation/{nf_feature,nf_mask,nf,ng,no_sparse_attn}.yaml` |
| run_05 | hsddqn / hgnn_ppo（+ a2c / dqn / ddqn） | train | `baselines/*.yaml`（流体引导关闭） |

## 6. 规则与精确求解（CPU，与训练并行）

```
python scripts/run_06_run_rules.py
python scripts/run_08_milp_check.py
```

规则在除 `*train` 外的全部档位上运行（Random/RRC 各 10 次），产物 `result/rules.csv`、`result/rules/<rule>/eval_results.csv`、
`data/instances/reference.csv`。精确求解在 tiny 与 tiny5 上求 MILP-A/B（600 s 时限），每个解经验证器、仿真器回放与
（S ≤ 6 且结构允许时）穷举三重核对，产物 `result/milp/milp_results.csv`（本机 HiGHS 约 5 分钟，含穷举）。

## 7. 评测（`python scripts/run_07_eval_main.py`）

对每个 `result/<method>/run<k>/best_val_checkpoint.pt`，按该 run 快照的 `eval_tiers`（ours 与变体：benchmark + arrival +
case_eval；ours_arrival：arrival + benchmark；ours_case：case_eval）在每个算例上跑贪心 1 条 + 采样 10 条（逐条独立），
与规则行合并为 `result/eval_results.csv`；`result/incumbents.csv` 收集训练末在位解。

## 8. 聚合与出表（`python scripts/run_09_aggregate_stats.py`）

| 产物 | 内容 | 论文 |
|---|---|---|
| `result/integer_check.csv` | 每 (算例, 方法, 解码) 组 S × episode 数 × 平均 η 为整数，违反即中止 | 意见 14 的回复依据 |
| `result/stats_summary.csv` | 方法 × 档位的均值 ± 标准差（run 级） | 表 7–10 的均值行 |
| `result/generalization.csv` | 方法 × 档位：K、η 均值 ± 标准差、`completed_mean`、每决策时间、最优规则与胜/平/负 | 新 5.x 节、表 13 |
| `result/paired_tests.csv`、`friedman.csv` | 基准档 15 个配置的配对 Wilcoxon（精确、Holm）与 Friedman | 表 11 |
| `result/ablation_results.csv`、`advantage_table.csv` | 基准档逐配置消融与百分点优势 | 表 7、图 8 |
| `paper_assets/tables/ablation.tex`、`comparison_pdr.tex`、`comparison_drl.tex`、`wilcoxon.tex` | 基准档表体 | 表 7、8、10、11 |
| `paper_assets/tables/generalization_arrival.tex`、`generalization_case_eval.tex` | 零样本表体（含 CSV） | 新 5.x、表 13 |
| `paper_assets/tables/hyperparameters.tex` | 由配置生成的超参表 | 表 6 |
| `paper_assets/tables/instance_statistics.tex` | 基准算例统计（由 params.yaml 生成） | 新表（意见 2） |
| `paper_assets/tables/milp_small.tex` | MILP-A/B 最优值、状态、求解时间、回放/穷举核对、最优规则 | 附录（意见 3） |
| `paper_assets/figures/generated/` | 训练曲线、收敛、配对差值、决策时间、热力图 | 图 7–10 |

一键顺序：`python scripts/run_all.py`（数据 → 预实验 → 训练 ‖ 规则 ‖ MILP → 评测 → 聚合）。

## 9. 三台 RTX 5060 台式机的预算与分配

单作业耗时按归档日志（i5-14600KF + RTX 4060 Ti）折算：FSHGRL 约 14 h / 2000 轮，NF 18 h，NG 23 h，对比方法 16 h；
每轮耗时由 CPU 侧 Python 主导，RTX 5060 与 4060 Ti 的差异可忽略；每作业显存 0.3–0.4 GB。

| 块 | 作业数 | 合计 GPU·h |
|---|---|---|
| ours / ours_arrival / ours_case | 9 | 126 |
| 消融 nfe / nm / nf / ng / na | 15 | 246 |
| 对比方法 hsddqn / hgnn_ppo | 6 | 96 |
| 附加基线 a2c / dqn / ddqn（各 1 run，可省） | 3 | 48 |
| 预实验（最多 2 轮） | 1–2 | 7–14 |
| 合计 | 34–35 | ≈ 520 |

建议分配（每台 3 个并发作业，效率 0.9）：机器 1 先跑预实验，通过后跑 ours ×3 与 nfe ×3；机器 2 跑 nf ×3、nm ×3、na ×3；
机器 3 跑 ng ×3、hsddqn ×3、hgnn_ppo ×3；ours_arrival、ours_case 与附加基线填空隙；规则与 MILP 在任一台的 CPU 上随时跑。
纯运行墙钟约 3–4 天（含预实验），评测约 20 CPU·h 可并行。正式开跑前每台以 1 / 3 / 4 个并发作业各跑 20 轮，
读 `log.csv` 的墙钟列校准上表，并用 `tools/profile_collector.py` 决定是否开启 `collector_workers`。

## 10. 复现检查清单

- 每个 run 的 `config_snapshot.yaml`：`git_commit` 为 `v2.0-revision` 对应提交，`train_version: 2`，`paradigm: generalization`，
  `fluid_backend` 为 gurobi 或 scipy（不是 heuristic_fallback）；`train_meta.yaml` 的停止原因为 `max_iterations`（或早停）；
- `result/integer_check.csv` 全部 `ok`；`result/eval_results.csv` 的 `fluid_backend` 列与训练一致；
- `result/pilot/verdict.txt` 以 PASS 开头，轮次记录在 `docs/pilot-log.md`；
- `result/milp/milp_results.csv`：全部 `optimal`，`replay_ok` 全真，穷举行 `brute_force_ok` 全真，MILP-A 与 MILP-B 的 `eta_opt` 相等；
- 表格由 `run_09` 生成后再写入论文，不手改数字。
