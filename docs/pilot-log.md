# 预实验日志（v2 门控）

预实验**启用**（`configs/pilot.yaml` `enabled: true`，`methods: [ours]`，`max_iterations: 1000`）。流程：
`python scripts/run_06_run_rules.py`（验证集参考值）→ `python scripts/run_02_pilot.py`（训练 1 run，每 50 轮验证）。

判定（`result/pilot/verdict.txt`、`pilot-report.md`、`pilot_results.csv`）：最佳验证轮的贪心达成率均值 ≥ 最优确定性规则的均值，
且在 ≥ 2/3（`configs/runs.yaml` `pilot.min_win_share`）的验证算例上达到参考值 → PASS，`run_02` 以后自动跳过，`run_03` 开始正式训练。
FAIL 时按下列顺序改一项并复跑（最多 2 轮，每轮在本表记录；改动落在 `configs/paper.yaml`，不改默认层）：

1. 学习率 3e-4 → 1e-4；
2. 熵系数 0.01 → 0.005 或 0.02；
3. 归一化方式（`agent/normalize.py` 的按观测归一化已是默认，可试固定尺度常数）；
4. 轮数 1000 → 3000（正式训练同步改 `max_iterations`）。

| 轮次 | 日期 | 配置变更 | 最佳验证轮 | 策略均值 | 规则均值 | 胜率（达到参考值的算例份额） | 判定 |
|---|---|---|---|---|---|---|---|
| 1 | 待跑 | 无（`configs/paper.yaml` 默认） | | | | | |
