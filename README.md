# FSHGRL for DFFSP-HFOI —— 论文代码仓库（重构版）

正式代码仓库：`Linshan-Ding/code_Yuhao_Zhu_FSHGRL_DFFSP-HFOI`（本仓库）。原仓库 `Linshan-Ding/Yuhao_Zhu_code_FSHGRL_DFFSP-HFOI`
保留重构前的代码，其 `main` 即本仓库历史中的初始提交 c030dfa；本仓库的 `main` 在该提交之上完成了全部重构（见 `CHANGELOG.md`）。

论文：*Fluid-Guided Sparse Heterogeneous Graph Reinforcement Learning for the Dynamic Flexible Flow Shop
Scheduling Problem with High-Frequency Order Insertion*（FSHGRL）。本仓库是论文实验的**复现手册与代码**：
按 `opt-paper-codegen` 技能的五大模块骨架（configs / data / environment / agent / result）重排了原仓库，
训练路径（`model_type` 为 `hgnn` 与 `mlp` 的 PPO 训练）的数值行为与原仓库**逐位一致**（回归测试见 §2），
并补齐了原仓库缺失的评测入口、规则基线、学习型基线、统计检验、出版级绘图与零参数实验脚本。

- 与原仓库的文件对应关系、删除清单与弃用接口：`docs/migration-map.md`
- 已知缺陷与论文/代码差异：`docs/known-issues.md`
- 实验设计规格（范式、run 矩阵、落盘契约、训练参数表 v1）：`docs/experiment-spec.md`
- 论文结论与数据文件的对应、复核结论：`docs/claim-verdicts.md`

## 1. 问题与假设

**问题**：高频插单柔性流水车间（DFFSP-HFOI）。5 类产品、5 道工序、每道工序 5 台并行机；订单按到达时间动态
进入，每个订单有交期；目标是**最大化订单按时达成率** `eta = 完成订单数 / 订单总数`。实验契约文件里统一用
**最小化**的目标值 `obj = 1 - eta`（未达成比例），`gap` 以各算例最优确定性规则为参考解（`data/instances/reference.csv`）。

**MDP**（`environment/simulator.py`，NumPy 离散事件仿真）：
- 决策点：出现"空闲机器 + 该机器所属工序有等待订单"时；候选动作 = 合法 (机器, 工序类型) 对 × 该工序类型按
  交期排序的前 `order_top_k`（默认 5）个订单；
- 状态：工序类型节点（12 维，含两列流体特征）与机器节点（5 维）的异质图，另有候选订单的剩余交期；
- 奖励：`(完成增量 - discard_penalty × 丢弃增量) / 订单总数`，整条 episode 的累计奖励等于 `2 eta - 1`
  （`tests/test_rules.py` 用随机策略核验此恒等式）；订单一旦无法按期完成即被丢弃，episode 在全部订单完成或
  丢弃时结束；
- 流体引导：`fluid_mode: cached_lp` 时用 Gurobi 求解容量分配 LP 得到两列流体特征（到达/丢弃事件触发重算并缓存），
  `gurobipy` 不可用、超时或不可行时**自动回退启发式特征**；`off` 置零两列（消融 FSHGRL-NF）。
  **注意**：训练器从不按流体解剪枝候选动作（论文描述了剪枝，见 `docs/known-issues.md` 第 2 条）。

**训练范式**：`per_instance`（逐算例型）。15 个基准算例（DDT ∈ {500, 600, 900, 1200, 1600} × S ∈ {50, 100, 150}）
既是训练集也是评测集：每个 (算例, run) 一个训练作业，每轮在该算例上并行采集 `num_envs` 条完整 episode 做一次
PPO 更新，更新后用当前策略在同一算例上跑一次**贪心** episode 写入 `curve.csv`（收敛曲线的唯一数据源）。
因此论文结论**不涉及跨算例泛化**；`data/generator.py` 可按论文算例参数表生成新算例，但不进任何已报告的实验。

**方法与基线**（方法 id 用于目录名与 `eval_results.csv` 的 `method` 列）：

| id | 方法 | 配置 | 代码来源 |
|---|---|---|---|
| `ours` | FSHGRL：PPO + 流体特征 + 稀疏异质图策略（候选自注意力） | `configs/algo.yaml` | 原仓库，数值逐位一致 |
| `nf` | FSHGRL-NF：流体特征置零 | `configs/ablation/nf.yaml` | 同上 |
| `ng` | FSHGRL-NG：逐节点 MLP 编码器替代图编码 | `configs/ablation/ng.yaml` | 原仓库 `compare/MLP_model.py`，数值逐位一致 |
| `a2c` `dqn` `ddqn` | 学习型基线 | `configs/baselines/*.yaml` | **新实现**（原基线依赖已删除的旧平台，无旧数值可对照） |
| `fifo` `spt` `mor` `mwkr` `random` `rrc` | 调度规则 | `baselines/run_rules.py` | **新实现**（评分方向沿用原 `compare/Rule.py`） |

**无随机种子政策**：复现靠固定算例文件 + 多次独立 run（次数见 `configs/runs.yaml`）；某类只跑 1 次时脚本固定
`--seed 1` 并记录在 `config_snapshot.yaml`。训练参数表 v1 取原仓库配置值（`configs/algo.yaml` 已置
`train_confirmed: true`），与论文超参表的差异见 `docs/known-issues.md` 第 1 条。

## 2. 环境配置

Python 3.10 以上（本次验证 3.13）；CPU 即可运行，有 NVIDIA GPU 时 PPO 更新默认放在 `cuda:0`（`configs/algo.yaml`
的 `update_device`，CUDA 不可用自动回退 CPU）。可选依赖：`gurobipy`（精确流体 LP，无许可证时自动回退启发式）、
`visdom`（训练实时曲线）、`psutil`（内存记录）。

```
pip install -r requirements.txt
```

安装自检（单元测试 35 个，约 50 s；数值等价回归 L1–L7，约 3 min，在首次运行时自动检出原仓库提交生成黄金文件）：

```
python -m pytest tests
python -m tests.equivalence
```

实测耗时（本机：4 核 CPU、无 GPU、Python 3.13、torch 2.14 CPU 版、每作业 1 个 BLAS 线程）：

| 步骤 | 实测 |
|---|---|
| §3 冒烟 `run_00` | 41 s |
| 一次 PPO 迭代（8 个环境 + 贪心评估） | DDT500_num50：21 s（采样 6 s、更新 14 s、评估 1 s）；DDT1600_num150：87 s（22 + 61 + 4） |
| 单个 (算例, run)，2000 次迭代 | 约 12 h（S=50）到 48 h（S=150） |
| §6 规则基线 `run_06`（15 算例 × 6 规则，随机规则 10 次） | 1 min |
| §7 评测，每个检查点（贪心 + 10 次采样） | 约 30 s |
| 完整 run 矩阵（§5–§6，180 个训练作业） | CPU 上数周；论文平台为 i5-14600KF + RTX 4060 Ti（本次未在 GPU 上验证） |
| 多进程采集（§5 的 `collector_workers`，8 环境，DDT500_num50） | 采集墙钟由 5.0 s 降到 3.7 s（3 个 worker，1.36 倍，低于 1.5 倍的开启门槛，默认保持 0）；worker 启动约 1.9 s |

`tools/profile_num_envs.py` 给出本机 `num_envs` 与采样吞吐的关系（1/2/4/8 环境分别约 131/156/157/174 转移每秒，
策略前向与观测构建各占一半），`configs/algo.yaml` 保留原仓库的 `num_envs: 8`。`tools/profile_collector.py` 比较 0 到 3 个
环境 worker 进程的采集吞吐并断言各配置采到相同轨迹：策略前向（约 2.2 s）必须留在训练进程以保证逐位一致，
本机 3 个 worker 只把采集提速 1.36 倍，因此 `configs/env.yaml` 默认 `collector_workers: 0`；换机器后重跑该工具定档。

## 3. 冒烟自检（先跑这条，约 1 分钟）

微型配置（`configs/smoke.yaml`：2 次迭代 × 2 个环境、单算例）跑通"训练 → 评测 → 规则基线 → 聚合 → 出图出表 →
单元测试"整条链路，全部产物在 `result/smoke/`，不产生论文数据：

```
python scripts/run_00_smoke.py
```

## 4. 数据准备

15 个基准算例是外部给定的文件（`data/instances/benchmark/<id>/<id>.txt`，随仓库发布，不重新生成），因此没有
"出表 → 审查 → 签字"三步；本步只为每个算例写 `params.yaml`（订单数、交期松弛、加工时间统计等）并生成档位索引：

```
python scripts/run_01_prepare_data.py
```

产物：`data/instances/index.csv`（列 `instance_id, tier, scale, ddt, n_orders, n_products, n_stages, n_machines,
source, path`，`scale` 即论文的 S）与每个算例文件夹的 `params.yaml`。重跑得到完全相同的文件。
合成新算例（不进论文实验）：`python -m data.generator --n 5`，参数见 `configs/instance.yaml` 的 `generator` 段。

## 5. 预实验与训练主方法

预实验在本项目**关闭**（`configs/pilot.yaml` 的 `enabled: false`）：逐算例范式下训练集就是评测集，单独的 pilot 档
会把 CPU 预算花在不进任何结果的算例上；理由记录于 `docs/experiment-spec.md` 变更记录与 `docs/pilot-log.md`。
脚本在关闭时打印 `[SKIP]` 并以 0 退出，`run_all.py` 继续进入训练阶段：

```
python scripts/run_02_pilot.py
```

训练主方法（15 算例 × `configs/runs.yaml` 的 `ours`，默认 3 次独立 run；训练参数表未确认时拒绝启动）：

```
python scripts/run_03_train_ours.py
```

产物：`result/ours/<instance_id>/run<k>/`，内含
`train_metrics.csv`（原仓库的 49 列逐迭代记录，列与数值与原仓库一致）、`curve.csv`（逐轮贪心目标值 `obj`
与在位最优 `obj_best`，`obj_best` 含训练期采样到的最优解）、`log.csv`（训练诊断：`train_obj_mean`、
`clip_frac`、`approx_kl`、`sps` 等，**不用于画收敛曲线**）、`best_greedy_checkpoint.pt`（按贪心目标值选出，
§7 评测默认用它）、`best_checkpoint.pt` 与 `ppo_policy_model.pt`（原仓库口径：按采样 episode 均值）、
`last_checkpoint.pt`（断点续训）、`config_snapshot.yaml`（生效配置、算例、设备、git 提交）、
`train_meta.yaml`（停止原因、轮数、墙钟、最优值）、`fixed_set_final.csv`（训练末在位解）。
被中断的作业重跑时从 `last_checkpoint.pt` 续训；已完成的作业（存在 `train_meta.yaml`）自动跳过。
想看实时曲线：**另开一个终端标签页**执行 `python -m visdom.server`，并把 `configs/algo.yaml` 的 `viz.viz` 置 true。

多进程采集：`configs/env.yaml` 的 `collector_workers` 置 N，即把仿真器推进与观测构建分到 N 个 worker 进程，策略前向、
动作采样与 PPO 更新仍在训练进程；同一 `num_envs` 下采集到的数据与进程内路径**逐位一致**（`tests/test_collector.py`，
`python -m tests.equivalence --collector-workers 3`）。单次运行可用 `python train.py --collector-workers 3` 覆盖。
是否值得开启看 §2 的实测：本机 4 核 CPU 上更新阶段占大头，收益有限；更新在 GPU 上或 S=150 的算例时收益更明显。
DQN/DDQN 与贪心评估不受该键影响。

## 6. 基线与消融

```
python scripts/run_04_train_ablations.py
python scripts/run_05_train_learned_baselines.py
python scripts/run_06_run_rules.py
```

产物：`result/nf/`、`result/ng/`（消融，目录结构同 §5）；`result/a2c/`、`result/dqn/`、`result/ddqn/`
（学习型基线，与主方法相同的 `num_envs` 与 `max_iterations` 预算，默认各 1 次 run）；
`result/rules/<rule>/eval_results.csv`、`result/rules.csv` 与 `data/instances/reference.csv`
（每算例最优确定性规则的目标值，所有 `gap` 的参考解）。规则基线是纯 CPU 作业，`run_all.py` 让它与训练重叠执行。
变体与基线清单写在脚本顶部常量里，增删不改 README 命令。

## 7. 评测与实验数据生成

```
python scripts/run_07_eval_main.py
```

对 `result/<method>/<instance_id>/run<k>/` 的每个检查点（`configs/runs.yaml` 的 `eval.checkpoint`）各跑一次
贪心 episode 和 `eval.sampled_episodes` 次采样 episode（默认 10 次、逐条独立），并带决策时间分项，再与规则结果合并：

产物：`result/eval_results.csv`，每行一个 (算例, 方法, run, 解码方式, episode)，前 10 列为契约列
`instance_id, method, run_id, tier, scale, obj_model, obj_ref, gap, feasible, infer_time_ms`，其后为
`decode, episode, eval_batch, fulfillment_rate, completed, discarded, total_orders, decision_count, policy_ms,
obs_ms, fluid_ms, sim_ms, fluid_backend, ddt, checkpoint, checkpoint_iteration`；`infer_time_ms` 是整条 episode
的墙钟，聚合时另算**每次决策**的时间 `decision_time_ms = infer_time_ms / decision_count`（论文表格的"time"口径）。
另有 `result/incumbents.csv`（各 run 训练末在位解，`decode = incumbent`，作为对照口径）。
单个检查点的评测与其它解码方式见 `python eval.py --help`。

## 8. 统计聚合

```
python scripts/run_09_aggregate_stats.py
```

产物：`result/stats_summary.csv`（方法 × 档位的均值 ± 标准差，单 run 的标准差留空）、`result/paired_tests.csv`
（主方法对每个对比方法的配对 Wilcoxon 符号秩检验：精确枚举 p 值、零差剔除、平均秩，Holm 校正；
`tests/test_stats.py` 复算了论文 Wilcoxon 表的数值）、`result/ablation_results.csv`、`result/advantage_table.csv`
（逐算例百分点差）、`result/friedman.csv`、`result/conv_<method>.csv`（跨算例跨 run 聚合的收敛曲线）；
图：`paper_assets/figures/generated/`（训练曲线、收敛曲线、配对差值、决策时间、质量-时间权衡、优势热力图，
PDF + 300 dpi PNG）；表：`paper_assets/tables/*.tex`（消融表、规则对比表、DRL 对比表、Wilcoxon 表的 booktabs 表体）。

一键跑完 §4–§8（按依赖并发调度：数据 → 预实验（跳过） → 训练三组作业 ‖ 规则基线 → 评测 → 聚合）：

```
python scripts/run_all.py
```

## 9. 启动方式与并发度

### 9.1 命令行启动（默认方式，跨平台）

§3–§8 的命令零参数照抄即可，每条用脚本内置的默认并发度。想改并发路数时在命令后加一个数字（bash 与 PowerShell
写法一致）：

```
python scripts/run_03_train_ours.py 3
python scripts/run_all.py 1
```

这个数字是**上限而非承诺**：脚本按 CPU 核数、可用内存（有 GPU 时按空闲显存）把它夹到设备撑得住的路数并打印理由，
例如 `[PARALLEL] requested 4, clamped to 3: 4 cores - 0 environment workers - 1`；作业因内存不足失败时并发度自动降
一档并重排该作业，下限 1 路。传 `1` 即全串行，排障时用。每个作业写自己的 `result/<作业名>/stdout.log`，控制台只打印
进度表。一次改全部脚本的默认值就改 `configs/parallel.yaml`（`max_concurrent`，以及每作业的 BLAS 线程数
`threads_per_job`，默认 1 以保证各作业耗时可比）。脚本按 `configs/env.yaml` 的 `collector_workers` 扣减每个作业占用的
环境 worker 进程数（夹取公式：核数 − 环境 worker 数 − 1），开启环境 worker 会相应降低作业并发度；本机 4 核上
作业级并发（2 路）比环境 worker 更划算。

| 步骤 | 可并行 | 资源 | 默认路数 |
|---|---|---|---|
| §4 数据准备 | 否 | — | 1 |
| §5 训练主方法 | 是（算例 × run 彼此独立） | CPU/GPU | 2 |
| §6 消融、学习型基线 | 是 | CPU/GPU | 2 |
| §6 规则基线 | 是，且与训练重叠 | CPU | 1（脚本内串行，约 1 min） |
| §7 评测 | 是（每检查点一个作业） | CPU | 3 |
| §8 统计聚合 | 否 | — | 1 |

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
| `data/instances/benchmark/<id>/<id>.txt` | 随仓库发布（原 `data/instance/competition/`，文件名未改） | 算例设置表；固定基准 |
| `data/instances/benchmark/<id>/params.yaml`、`data/instances/index.csv` | run_01（§4） | 算例设置表；档位/规模列 |
| `data/instances/reference.csv`、`result/rules.csv`、`result/rules/<rule>/eval_results.csv` | run_06（§6） | 规则对比表（`tab:comparison_pdr`）；所有 `gap` 的参考解 |
| `result/ours/<id>/run<k>/train_metrics.csv` | run_03（§5） | 原仓库口径的训练记录（49 列，数值逐位一致） |
| `result/ours/<id>/run<k>/curve.csv`、`log.csv`、`fixed_set_final.csv` | run_03（§5） | 训练曲线图（`fig:training_curve`）；收敛曲线；训练末在位解 |
| `result/ours/<id>/run<k>/*.pt`、`config_snapshot.yaml`、`train_meta.yaml` | run_03（§5） | 评测输入；复现记录（配置、提交、停止原因） |
| `result/nf/`、`result/ng/`（同上结构） | run_04（§6） | 消融表（`tab:ablation`）、配对差值图 |
| `result/a2c/`、`result/dqn/`、`result/ddqn/`（同上结构） | run_05（§6） | DRL 对比表（`tab:comparison_drl`） |
| `result/eval_results.csv`、`result/incumbents.csv` | run_07（§7） | 以上三张对比表的达成率与决策时间列；决策时间图；质量-时间权衡图 |
| `result/stats_summary.csv`、`paired_tests.csv`、`friedman.csv` | run_09（§8） | 均值 ± 标准差；Wilcoxon 表（`tab:wilcoxon`） |
| `result/ablation_results.csv`、`advantage_table.csv`、`conv_<method>.csv` | run_09（§8） | 消融表；优势热力图；收敛曲线图 |
| `paper_assets/figures/generated/*.pdf`、`paper_assets/tables/*.tex` | run_09（§8） | 论文图表成品（源脚本在 `paper_assets/figures/src/`、`paper_assets/scripts/`） |
| `paper_assets/figures/legacy/*` | 原仓库成品图（不可再生成，数据 CSV 已不存在） | 原投稿的训练曲线图 |
| 工业 3D 打印案例（论文案例研究表/图） | **无**：数据与格式不在原仓库中，未重建（`docs/known-issues.md` 第 14 条） | 案例研究 |

## 附：原仓库命令的对应

旧入口 `python agent/train.py` 仍可用（转发到根目录 `train.py`），旧 import 路径（`agent.HGHH_model`、
`agent.PPO_model`、`agent.sci_rollout`、`agent.training_simulator`、`agent.visualization`、`compare.MLP_model`）
保留为转发模块；旧 JSON 配置可用 `python train.py --config path/to/config.json` 直接加载。完整映射见
`docs/migration-map.md`。
