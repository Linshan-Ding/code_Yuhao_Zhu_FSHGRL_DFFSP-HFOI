# 论文结论 ↔ 数据文件 ↔ 复核结论（v2；按 `experiments/2026-11-01/` 裁决）

| 论文结论 | 数据文件 / 脚本 | 复核结论 |
|---|---|---|
| FSHGRL 在 15 个未见基准配置上高于消融变体（表 7、图 8） | `experiments/2026-11-01/paper/result/ablation_results.csv`、`paired_tests.csv` | **成立**（单次训练）：贪心均值 0.480 对 NFe 0.426 / NM 0.431 / NF 0.427 / NG 0.398 / NA 0.439；W/T/L 10/3/2 到 13/1/1，Holm p ≤ 0.0154 |
| FSHGRL 高于论文比较的五条规则 FIFO/MOR/MWKR/Random/RRC（表 8） | `experiments/2026-11-01/paper/tables/comparison_pdr.tex`、`paired_tests.csv` | **成立**：0.480 对 0.385 / 0.347 / 0.284 / 0.266 / 0.416，W/T/L 13/1/1 到 15/0/0，Holm p ≤ 0.0022。事实记录：评估档案中的 SPT（0.531）在 13/15 个配置高于 FSHGRL，未纳入论文比较集（`experiments/2026-11-01/result/rules.csv`） |
| FSHGRL 高于 DRL 对比方法（表 9、10） | `experiments/2026-11-01/paper/tables/comparison_drl.tex` | **成立**：0.480 对 HSDDQN 0.377（15/0/0）、HGNN-PPO 0.416（12/1/2），Holm p ≤ 0.0116；A2C/DQN/DDQN 未运行 |
| 配对 Wilcoxon（表 11） | `experiments/2026-11-01/paper/tables/wilcoxon.tex` | **成立**：12 个比较对象 Holm 校正后全部 p < 0.05（家族 = 12，单策略 × 15 配置） |
| 决策时间（表 7/10 的 time 列、图 9） | `experiments/2026-11-01/paper/result/generalization.csv` | 已测：FSHGRL 13.5 ms/决策（含 LP），DRL 对比 10.3–10.7 ms，规则 0.6 ms（CPU 单线程） |
| 训练曲线（图 7，DDT500_num50 与 DDT1200_num100） | `experiments/2026-11-01/result/ours/run1/val_curve.csv`、`paper/figures/train_curve_*.pdf` | 已生成；最佳验证轮 980 的两值 0.34 / 0.76 与表 7、8 一致 |
| 收敛曲线 | `experiments/2026-11-01/paper/result/conv_<method>.csv`、`figures/convergence.pdf` | 已生成（验证目标值均值，8 个方法）；NM 的验证曲线不比 FSHGRL 慢，正文不作收敛速度比较 |
| 到达模式鲁棒性（新 5.x） | `experiments/2026-11-01/paper/tables/generalization_arrival.tex` | **部分成立**：6 个算例均值 0.508 为最高（最优规则 MWKR 0.442，+6.7 pp，W/T/L 5/0/1）；指数到达（0.32）低于 NF 0.38 与 MWKR 0.37，突发到达（0.43）低于 NF 0.46 |
| 剪枝阈值敏感性（新 5.x，意见 5） | `experiments/2026-11-01/paper/tables/sensitivity.tex` | 已测：采样 η 0.731（off）→ 0.749（0.2）；掩码平均去掉 0.3%（small）/ 1.4%（基准）的合法对，贪心解码不变 |
| 工业案例（表 13） | 原稿结果（不重做） | 不重做：现场使用文中的派工规则，表 13 的最优规则结果即工业案例实际运行结果；表注说明该表沿用原稿评估口径（逐算例训练），其余表格为 v2 泛化型结果（known-issues 14） |
| MILP 模型正确性（3.2 节、附录） | `experiments/2026-11-01/paper/tables/milp_small.tex` | **成立**：12 个算例 MILP-A = MILP-B 全部最优（HiGHS ≤ 4.1 s），回放一致；最优值均值 0.886，FSHGRL 0.784，规则最优 0.801 |
| 表 7 中 S·η 为整数（意见 14） | `experiments/2026-11-01/paper/result/integer_check.csv` | **成立**：726 组全部通过 |
| 流体引导剪枝候选动作（4.5 节） | `environment/simulator.py` `_apply_fluid_mask`，`log.csv` 的掩码统计列 | **v2 已实现**（known-issues 2 已处理） |
| 超参表（表 6） | `paper_assets/tables/hyperparameters.tex`（由配置生成） | 与代码一致（known-issues 1 已处理） |
