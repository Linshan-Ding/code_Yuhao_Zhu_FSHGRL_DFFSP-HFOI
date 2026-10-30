# 预实验日志

v2 正式实验**不设预实验**（用户决定，2026-10-30）：`configs/pilot.yaml` `enabled: false`，`python scripts/run_02_pilot.py` 打印 SKIP
并以 0 退出，`run_all.py` 的 pilot 阶段跳过，正式训练（`run_03`–`run_05`）直接开始。每个方法只训练一次（1000 轮），模型选择由
`best_val_checkpoint.pt`（两个基准验证算例的贪心均值最优轮）完成，训练是否收敛由 `val_curve.csv`（每 10 轮）事后检查。

门控机制保留：`enabled: true` 时 `run_02` 训练 ours × `max_iterations` 轮到 `result/pilot/ours/run1`，判定（`result/pilot/verdict.txt`）
为"最佳验证轮的贪心达成率均值 ≥ 最优确定性规则的均值且在 ≥ `pilot.min_win_share` 的验证算例上达到参考值"。如需启用，每轮在下表记录。

| 轮次 | 日期 | 配置变更 | 最佳验证轮 | 策略均值 | 规则均值 | 胜率 | 判定 |
|---|---|---|---|---|---|---|---|
| — | — | 未启用 | | | | | |
