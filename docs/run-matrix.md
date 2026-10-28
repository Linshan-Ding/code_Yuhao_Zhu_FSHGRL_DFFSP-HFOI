# Run 矩阵与代价（v2，泛化型范式）

单作业耗时按归档训练日志（作者平台 i5-14600KF + RTX 4060 Ti，每轮 8 episode）折算到 2000 轮；每轮耗时由 CPU 侧 Python 主导，
GPU 型号影响可忽略，每作业显存 0.3–0.4 GB。本机（4 核 CPU、无 GPU）只验证了冒烟与机制测试，`result/` 下没有论文级数据。

| 脚本 | 作业 | 作业数 | 单作业 | 合计 |
|---|---|---|---|---|
| run_02 预实验 | ours × 1000 轮（最多 2 轮） | 1–2 | 7 h | 7–14 h |
| run_03 ours | ours、ours_arrival、ours_case × 3 run | 9 | 14 h | 126 h |
| run_04 消融 | nfe、nm、nf、ng、na × 3 run | 15 | 14 / 14 / 18 / 23 / 14 h | 246 h |
| run_05 对比方法 | hsddqn、hgnn_ppo × 3 run | 6 | 16 h | 96 h |
| run_05 附加基线 | a2c、dqn、ddqn × 1 run（可从 `BASELINES` 删除） | 3 | 16 h | 48 h |
| run_06 规则 | 6 条规则 × 全部评测档位（Random/RRC 10 次） | 1 | 约 10 min（CPU） | 10 min |
| run_08 MILP | tiny 50 + tiny5 30 算例 × MILP-A/B，含穷举 | 1 | 约 5 min（CPU，HiGHS） | 5 min |
| run_07 评测 | 33 个检查点 × 约 30 算例 × 11 episode | 33 | 约 30 min（CPU） | 约 20 h（可并行） |
| run_09 聚合 | — | 1 | 1 min | 1 min |

训练合计约 520 GPU·h；三台 RTX 5060 台式机各跑 3 个并发作业（效率 0.9）时纯运行墙钟约 3.4 天，关键路径为预实验（7–14 h）
+ 最长单作业（ng 23 h）。3000 轮时按 1.5 倍计。减小预算：`configs/runs.yaml` 降低 run 数；删除附加基线；
`configs/paper.yaml` 的 `max_iterations`。分配建议与校准方法见 `docs/reproduce.md` §9。
