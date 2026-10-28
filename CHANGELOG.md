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
- V3 已知缺陷修复（数值变化，v2 实验版本的基线）：第 8 条（`raw_opes[:, 6]` 只统计已到达订单）、第 9 条
  （`agent/normalize.py`：交期与加工率按单个观测归一化，策略输出与 batch 组成无关）、第 12 条（非法动作抛错）、第 10 条
  （`best_val_checkpoint.pt` 取代旧口径）；等价性回归新增 v2 黄金模式（`python -m tests.equivalence --golden v2`，
  `tests/golden_v2/manifest.json` 记录冻结提交；参考运行含 HiGHS 掩码与 2 个验证算例）；`tests/test_known_issue_fixes.py`。
- V4 消融变体 `nfe`（去流体特征）、`nm`（去流体掩码）、`na`（去候选自注意力）进入 `scripts/run_04`（与 `nf`、`ng` 共 5 个变体）、
  `result/stats.py` 的标签与 `ABLATION_METHODS`、`run_07` 的方法清单、消融表与收敛图；DRL 对比方法标签预留 `hsddqn`、`hgnn_ppo`（V7）。
- V5 生成器扩展（`data/generator.py`）：到达模型 uniform / exponential / bursty，`rho`（目标负荷率 = 平均工序时间×阶段数 /
  （平均间隔×机器数））或 `mean_gap`，`--slack-rel`、`--ddt-fixed`、结构覆盖 `--n-products/--n-stages/--machines-per-stage`，
  工业案例结构 `data/case/process.yaml` + `data/case_structure.py`（论文表 12 的工艺数据，分钟）；池规格支持 `cases`、`grid`、
  `n`（含 `arrival_mix`）；新档位 arrival（6 个：ρ 0.8/1.2/1.8/2.6 均匀、指数、突发）、arrival_train（120，混合到达）、
  tiny（2×3×2，50）、tiny5（5×5×5 少订单，30）、case_train（100）、case_val（8）、case_eval（9 个论文配置）；
  `params.yaml` 新增 `inter_arrival_cv`、`max_burst`、`load_ratio`、`load_ratio_bottleneck`。
- V6 论文配置层 `configs/paper.yaml`（泛化型范式、训练池/验证集/评测档位、掩码与 HiGHS/Gurobi 后端、2000 轮、每 50 轮验证、
  逐键标注对应论文表 6 的行），脚本默认加载（`_bootstrap.CONFIGS`）；`configs/pools/{arrival_train,case_train}.yaml`
  （`ours_arrival`、`ours_case`）；`configs/algo.yaml` `train_version: 2`、`viz: false`；`configs/pilot.yaml` 启用泛化型门控
  （ours × 1000 轮）；`run_07` 评测档位按各 run 的快照；冒烟脚本改为泛化型流程（42 s）。
- V7 DRL 对比方法：`hsddqn`（`agent/baselines/hsddqn.py`，层级 Double DQN 的改编：第一层在合法 (机器, 工序类型) 对上取 Q，
  第二层在该对的候选订单上取 Q，两层均为在线选择、目标评估的 Double-DQN 目标，复用 `offpolicy.py` 的回放循环与验证钩子；
  `algo.algorithm: hsddqn`，`agent/factory.py` 建网与建 agent）、`hgnn_ppo`（`configs/baselines/hgnn_ppo.yaml`：同编码器与 PPO，
  无流体引导、无候选自注意力、`order_top_k: 1` 即 (工序类型, 机器) 对动作）；全部对比方法的覆盖层关闭流体引导
  （`fluid_features: off`、`fluid_mask: false`）；`run_05`、`run_02` 纳入两者；`docs/baselines.md` 记录改编范围；
  `tests/test_learned_baselines.py` 新增动作可行性、双层 TD 更新、token 编解码、检查点往返与覆盖层测试。
- V8 精确求解检查 `baselines/milp_exact.py`（论文 3.2 节 MILP 的 MILP-A 原模型与 MILP-B 可放弃变体，Gurobi 优先、HiGHS 备用，
  紧 big-L 并记录；验证器左移重定时、`ScheduleFollowingSimulator` 仿真器回放、S ≤ 6 的穷举交叉核对；
  产物 `result/milp/milp_results.csv`）、`scripts/run_08_milp_check.py`（档位 tiny 与 tiny5，`run_all` 的 milp 阶段）、
  `tests/test_milp_exact.py`；两变体最优值相等的论证见 `docs/baselines.md`。
- V9 聚合与出表：`result/aggregate.py` 新增整数一致性检查（每个 (算例, 方法, 解码) 组 S × episode 数 × 平均 η 必为按期订单数，
  `integer_check.csv`，违反即中止；意见 14 的防复发机检）、点估计携带 `completed` 与 `total_orders`、`generalization.csv`
  （每 (方法, 档位) 的零样本汇总：K、η 均值 ± 标准差、`completed_mean`、每决策时间、该档位最优确定性规则及逐算例胜/平/负）、
  配对检验与消融/优势表限定在基准档（`--paired-tier`）；`paper_assets/scripts/make_tables.py` 新增 `generalization_<tier>.tex`、
  `hyperparameters.tex`（表 6 由配置生成）、`instance_statistics.tex`（基准算例统计，由 params.yaml 生成）、`milp_small.tex`
  （精确求解检查 + 最优规则），基准档对比表按档位过滤，v2 表同时落 CSV；`tests/test_aggregate.py`。
- V9b `env.fluid_backend: auto` 改为 Gurobi → HiGHS（scipy）→ 启发式（此前无 Gurobi 即回退启发式，掩码随之失效）；
  旧行为由 `fluid_backend: gurobi` 保留，等价性回归的旧语义覆盖层改用它；`configs/env.yaml` 注释与 `docs/known-issues.md` 第 19 条同步。
- V10 文档与定稿：`docs/reproduce.md`（修订版复现手册：档位、门控、训练、规则与精确求解、评测、出表、三台 RTX 5060 的预算分配、
  检查清单）、`docs/experiment-spec.md`（v2 规格）、`docs/run-matrix.md`、`docs/pilot-log.md`（门控轮次表）、`docs/claim-verdicts.md`、
  README 十节改写为泛化型流程；`configs/runs.yaml` 对比方法 3 run（`learned_baselines`）、附加基线 1 run（`learned_baselines_extra`）；
  `pytest.ini` 把等价性回归的工作树（`tests/golden*/_work`）排除在测试收集之外；本机验证：`python -m pytest tests`（73 个）、
  `python -m tests.equivalence --golden v2`、冒烟（57 s，HiGHS 掩码）；`run_08` 实跑 80 个小算例 × MILP-A/B
  全部最优、回放一致 160/160、穷举一致 40/40（其余因结构规模跳过）、MILP-A 与 MILP-B 最优值相等；标签 `v2.0-revision`。
