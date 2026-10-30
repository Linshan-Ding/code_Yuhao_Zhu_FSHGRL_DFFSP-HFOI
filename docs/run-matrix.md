# Run 矩阵与代价（v2，泛化型范式；规模定于 2026-10-30）

单作业耗时按归档训练日志（作者平台 i5-14600KF + RTX 4060 Ti，每轮 8 episode）折算到 1000 轮；每轮耗时由 CPU 侧 Python 主导，
GPU 型号影响可忽略，每作业显存 0.3–0.4 GB。本机（4 核 CPU、无 GPU）只验证了冒烟、精确求解与机制测试，`result/` 下没有论文级数据。

| 脚本 | 作业 | 作业数 | 单作业 | 合计 |
|---|---|---|---|---|
| run_03 ours | ours × 1 run（固定种子 1） | 1 | 7 h | 7 h |
| run_04 消融 | nfe、nm、nf、ng、na × 1 run | 5 | 7 / 7 / 9 / 11.5 / 7 h | 41.5 h |
| run_05 对比方法 | hsddqn、hgnn_ppo × 1 run | 2 | 8 h | 16 h |
| run_05 附加基线 | a2c、dqn、ddqn × 1 run（可从 `EXTRA` 删除） | 3 | 8 h | 24 h |
| run_06 规则 | 6 条规则 × 33 个评测算例（Random/RRC 10 次） | 1 | 约 5 min（CPU） | 5 min |
| run_08 MILP | small 12 算例 × MILP-A/B | 1 | 约 1 min（CPU，HiGHS；本机 24 次求解全部最优，单次 ≤ 2.6 s） | 1 min |
| run_07 评测 | 8（+3）个检查点 × 33 算例 × 11 episode | 8–11 | 约 30 min（CPU） | 约 4–6 h（可并行） |
| run_10 敏感性 | ours × 5 个阈值 × 12 算例 × 11 episode | 5 | 约 2 min（CPU） | 10 min |
| run_09 聚合 | — | 1 | 1 min | 1 min |

训练合计约 65 GPU·h（+24 h 附加基线）；三台 RTX 5060 台式机各跑 3 个并发作业时纯运行墙钟约 12 h（最长作业 ng），整个 C 线约 1 天。
减小预算：删除附加基线；`configs/paper.yaml` 的 `max_iterations`。分配建议与校准方法见 `docs/reproduce.md` §8。
