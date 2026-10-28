# 论文结论 ↔ 数据文件 ↔ 复核结论（v2，待完整矩阵后填写）

| 论文结论 | 数据文件 / 脚本 | 复核结论 |
|---|---|---|
| FSHGRL 在 15 个未见基准配置上高于消融变体（表 7、图 8） | `result/ablation_results.csv`、`advantage_table.csv`、`paper_assets/tables/ablation.tex`（run_09） | 待跑：完整矩阵未运行；2×2 因子消融（nfe / nm / nf）+ ng + na |
| FSHGRL 高于 FIFO/SPT/MOR/MWKR/Random/RRC（表 8） | `result/eval_results.csv`、`rules.csv`、`comparison_pdr.tex` | 待跑；规则为新实现（known-issues 4） |
| FSHGRL 高于 DRL 对比方法（表 9、10） | `comparison_drl.tex`、`docs/baselines.md` | 待跑；对比方法为 HSDDQN（层级 DDQN 改编）与 HGNN-PPO (adapted)，附加 A2C/DQN/DDQN；原稿的 DRLG/AHP-DQN 不可追溯（known-issues 3） |
| 配对 Wilcoxon（表 11） | `result/paired_tests.csv`、`wilcoxon.tex`；`tests/test_stats.py` | 检验实现复算出原表数字；v2 数值待跑 |
| 决策时间（表 7/10 的 time 列、图 9） | `eval_results.csv` 的 `decision_time_ms` | 待跑；口径为每决策时间（known-issues 5），CPU 单线程 |
| 训练曲线与收敛（图 7） | `result/<method>/run<k>/curve.csv`、`val_curve.csv`、`conv_<method>.csv` | 待跑 |
| 到达模式鲁棒性（新 5.x） | `result/generalization.csv`、`generalization_arrival.tex`（ours 零样本 + ours_arrival） | 待跑 |
| 工业案例（表 13） | `generalization_case_eval.tex`（ours_case + 规则） | 待跑；案例结构按论文表 12 重建，真实订单流不公开（known-issues 14） |
| MILP 模型正确性（3.2 节、附录） | `result/milp/milp_results.csv`、`milp_small.tex` | 本机已跑：80 算例 × MILP-A/B 全部最优、回放一致、穷举一致（见 CHANGELOG V8/V10）；MILP-A 与 MILP-B 最优值相等 |
| 表 7 中 S·K·η 为整数（意见 14） | `result/integer_check.csv` | 机检随 run_09 运行，违反即中止 |
| 流体引导剪枝候选动作（4.5 节） | `environment/simulator.py` `_apply_fluid_mask`，`log.csv` 的掩码统计列 | **v2 已实现**（known-issues 2 已处理） |
| 超参表（表 6） | `paper_assets/tables/hyperparameters.tex`（由配置生成） | 与代码一致（known-issues 1 已处理） |
