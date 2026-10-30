# 论文结论 ↔ 数据文件 ↔ 复核结论（v2，待完整矩阵后填写）

| 论文结论 | 数据文件 / 脚本 | 复核结论 |
|---|---|---|
| FSHGRL 在 15 个未见基准配置上高于消融变体（表 7、图 8） | `result/ablation_results.csv`、`advantage_table.csv`、`paper_assets/tables/ablation.tex`（run_09） | 待跑：完整矩阵未运行；2×2 因子消融（nfe / nm / nf）+ ng + na，各 1 run |
| FSHGRL 高于 FIFO/SPT/MOR/MWKR/Random/RRC（表 8） | `result/eval_results.csv`、`rules.csv`、`comparison_pdr.tex` | 待跑；规则为新实现（known-issues 4） |
| FSHGRL 高于 DRL 对比方法（表 9、10） | `comparison_drl.tex`、`docs/baselines.md` | 待跑；对比方法为 HSDDQN（层级 DDQN 改编）与 HGNN-PPO (adapted)，附加 A2C/DQN/DDQN；原稿的 DRLG/AHP-DQN 不可追溯（known-issues 3） |
| 配对 Wilcoxon（表 11） | `result/paired_tests.csv`、`wilcoxon.tex`；`tests/test_stats.py` | 检验实现复算出原表数字；v2 数值待跑 |
| 决策时间（表 7/10 的 time 列、图 9） | `eval_results.csv` 的 `decision_time_ms` | 待跑；口径为每决策时间（known-issues 5），CPU 单线程 |
| 训练曲线（图 7，DDT500_num50 与 DDT1200_num100） | `result/ours/run1/val_curve.csv`（每 10 轮）、`fig_train_curve.py` | 待跑；两个算例即验证集，曲线为贪心达成率 |
| 收敛曲线 | `result/conv_<method>.csv`（验证目标值均值） | 待跑 |
| 到达模式鲁棒性（新 5.x） | `result/generalization.csv`、`generalization_arrival.tex`（零样本） | 待跑 |
| 剪枝阈值敏感性（新 5.x，意见 5） | `result/sensitivity.csv`、`sensitivity.tex`（small 档，MILP 最优值为参照） | 待跑（评测期阈值扫描，不重训） |
| 工业案例（表 13） | 原稿结果（不重做） | 不重做：现场使用文中的派工规则，表 13 的最优规则结果即工业案例实际运行结果；表注说明该表沿用原稿评估口径（逐算例训练），其余表格为 v2 泛化型结果（known-issues 14） |
| MILP 模型正确性（3.2 节、附录） | `result/milp/milp_results.csv`、`milp_small.tex` | 求解链路已在本机验证（V8：80 个微型算例全部最优、回放一致、穷举一致；MILP-A = MILP-B）；small 档 12 算例的正式结果随 run_08 产出 |
| 表 7 中 S·η 为整数（意见 14） | `result/integer_check.csv` | 机检随 run_09 运行，违反即中止；单次训练下表格值 S·η 为整数 |
| 流体引导剪枝候选动作（4.5 节） | `environment/simulator.py` `_apply_fluid_mask`，`log.csv` 的掩码统计列 | **v2 已实现**（known-issues 2 已处理） |
| 超参表（表 6） | `paper_assets/tables/hyperparameters.tex`（由配置生成） | 与代码一致（known-issues 1 已处理） |
