# 变更记录（分支 `claude/refactor-opt-paper`，基线提交 `c030dfa`）

每步都以 `python -m tests.equivalence`（hgnn 与 mlp 各 3 轮训练，57 个浮点列与全部检查点张量逐位相等）验收。

- C0 等价性回归测试与黄金文件（`tests/equivalence/`、`tests/golden/`）、`requirements.txt`、`.gitignore`。
- C1 删除旧平台（`env/`、`agent/competitionPlatform.py`、`agent/algorithm_main.py`）、旧版基线、`test_code/`、旧绘图/统计脚本；
  删除策略网络中不可达的单状态/稠密路径；成品图移至 `paper_assets/figures/legacy/`。
- C2 五大模块骨架搬移 + 转发模块（`data/instance_io.py`、`environment/simulator.py`、`agent/{buffer,networks,networks_mlp,ppo,checkpoint}.py`、`rollout.py`、`data/dataset.py`、根 `train.py`）。
- C3 YAML 配置 + dataclass 加载器 + 旧字典桥接（`configs/`）、`tests/test_config_bridge.py`、`train.py --config/--run-name/--smoke`。
- C4 `result/logger.py`（`log.csv`、`curve.csv`、`config_snapshot.yaml`、`train_meta.yaml`、`fixed_set_final.csv`）、`result/metrics.py`、每轮贪心评估与 `best_greedy_checkpoint.pt`。
- C5 算例按文件夹存放 + `index.csv` + `params.yaml`、`eval.py`、`environment/profiled.py`、`agent/checkpoint.py` 加载器、`tools/{eval_curve,profile_num_envs}.py`。
- C6 规则基线 `agent/baselines/rules.py` + `baselines/run_rules.py` + `data/instances/reference.csv`、`tests/test_rules.py`（含奖励恒等式）。
- C7 学习型基线 A2C/DQN/DDQN（`agent/baselines/`）、`agent/factory.py` 分派、`configs/baselines/`。
- C8 `result/stats.py`（精确 Wilcoxon + Holm + Friedman）、`result/plot.py`、`result/aggregate.py`、`paper_assets/`。
- C9 `scripts/run_00..09 + run_all`、`configs/{smoke,runs,parallel,pilot}.yaml`、`data/generator.py`、README 十节、`docs/`。
- C10 收尾验证与汇报。
- C11 多进程轨迹采集（`environment/collector.py`、`rollout.collect_vectorized_rollouts_parallel`、`env.collector_workers`）：
  仿真器与观测构建分到 worker 进程，策略前向、采样与更新留在训练进程，同 `num_envs` 下采集数据逐位一致
  （`tests/test_collector.py`、`python -m tests.equivalence --collector-workers 3`）；`tools/profile_collector.py` 定档；默认关闭。
- C12 仓库迁移：完整历史（含原仓库初始提交 `c030dfa`）推送为 `Linshan-Ding/code_Yuhao_Zhu_FSHGRL_DFFSP-HFOI` 的 `main`，
  以后以该仓库为论文对应的代码仓库；原仓库只保留历史。

# v2（分支 `claude/v2-generalization`，泛化型范式；修订方案 v2 的 A.1 代码冻结清单）

v2 是新的实验版本：V1 保持默认路径与 c030dfa 逐位一致（`python -m tests.equivalence` 以旧语义覆盖层验证），
V3 起数值变化（缺陷修复），等价性回归改以 v2 自身的黄金文件为基准。

- V1 流体引导拆成两种用法并实现候选掩码（`environment/simulator.py`）：`env.fluid_features on|off`（两列流体节点特征）、
  `env.fluid_mask`（保留 u* > `fluid_mask_threshold` 的合法 (机器, 工序类型) 对，空集回退全部合法对，计数 `fluid_mask_fallback_count`）、
  `env.fluid_backend auto|gurobi|scipy|heuristic`（HiGHS 通过 `scipy.optimize.linprog` 求同一 LP，无许可证也能得到精确解与掩码）、
  `env.fluid_recompute decision|event`（每个新状态重解 LP，按状态签名缓存；`event` 为旧行为）；`log.csv` 与 `eval_results.csv` 新增
  `candidate_pairs_mean / masked_pairs_mean / mask_fallback_rate`，`config_snapshot.yaml` 记录实际求解后端；
  消融配置 `configs/ablation/{nf,nf_feature,nf_mask}.yaml`；`tests/test_fluid_mask.py`。
- V2 泛化型范式支撑：`data/generator.py` 按基准统计生成训练池与验证集（`instance.pools`：train 200 个、val 30 个，
  到达间隔 rand[1,40] s、每算例一个 DDT、松弛 [0.6, 1.55]×DDT，种子入库），`data/dataset.py` 对全部档位建索引
  （`prepare_all`、`tier_instance_files`、`instance_files_by_ids`）；`instance.paradigm per_instance|generalization`、
  `val_data_dir/val_instances/eval_tiers`；训练器验证钩子 `result/validation.py`（`algo.train.val_every`、`early_stop_patience`、
  `val_curve.csv`、`best_val_checkpoint.pt`，PPO 与 DQN/DDQN 循环共用）、`eval_scope all|selected`；`eval.py --tiers`
  与泛化检查点默认按快照的 `eval_tiers` 评测；`baselines/run_rules.py --tiers`（默认除 train 外全部档位）；脚本按范式分派
  （`_bootstrap.pool_jobs/method_jobs/paradigm`），`run_02` 泛化型门控（`result/pilot.py`，最佳验证轮对规则参考值）；
  `configs/runs.yaml` 的 `eval.checkpoint/tiers: auto`；`tests/test_generalization.py`。
