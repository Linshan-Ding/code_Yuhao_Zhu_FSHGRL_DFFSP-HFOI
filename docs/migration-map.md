# 迁移映射：原仓库 → 重构后

原仓库 `https://github.com/Linshan-Ding/Yuhao_Zhu_code_FSHGRL_DFFSP-HFOI`，单提交 `c030dfa`（分支 `main`）。
重构后的正式仓库是 `https://github.com/Linshan-Ding/code_Yuhao_Zhu_FSHGRL_DFFSP-HFOI`，其 `main` 以 `c030dfa` 为根，
逐步提交（见 `CHANGELOG.md`；等价性测试从本仓库自身的历史检出 `c030dfa` 生成黄金基线），
每步都通过 `python -m tests.equivalence`（对 `model_type` 为 `hgnn` 与 `mlp` 的 3 轮训练：`train_metrics.csv`
的 57 个浮点列与全部检查点张量逐位相等）。

## 文件对应

| 原仓库 | 重构后 | 方式 |
|---|---|---|
| `agent/train.py` | `train.py`（训练循环逐字保留；`write_metrics` → `result/logger.py`，`save/load_checkpoint` → `agent/checkpoint.py`） | 拆分；`agent/train.py` 保留为转发入口 |
| `agent/sci_rollout.py` | `rollout.py`（`collect_vectorized_rollouts`）、`agent/buffer.py`（`RolloutBuffer`）、`data/dataset.py`（算例扫描/选择） | 原样拆分 + 转发模块 |
| `agent/training_simulator.py` | `data/instance_io.py`（`InstanceSpec`、`parse_instance_file`）+ `environment/simulator.py`（`TrainingSimulator`，含 Gurobi LP） | 原样拆分 + 转发模块 |
| `agent/PPO_model.py` | `agent/ppo.py` | 移动；`hgnn_no_fluid` 分支改为报错 |
| `agent/HGHH_model.py` | `agent/networks.py` | 移动；删除不可达的单状态/稠密路径与 `Memory` 类，保留方法逐字节不变 |
| `compare/MLP_model.py` | `agent/networks_mlp.py` | 移动；同样只删不可达路径；`compare/MLP_model.py` 保留为转发模块 |
| `agent/hgnn.py`、`agent/mlp.py` | 不变 | — |
| `agent/visualization.py` | `result/logger.py`（`TrainingVisualizer` 原样并入） | 转发模块保留 |
| `data/config.json`、`config_annotation.json` | `configs/{instance,env,algo}.yaml` + `configs/config.py`（dataclass 加载器，`to_legacy_config` 桥接出逐键等值等型的旧字典） | 重写；冻结副本 `tests/golden/legacy_config.json`；`tests/test_config_bridge.py` |
| `data/instance/competition/*.txt` | `data/instances/benchmark/<id>/<id>.txt`（文件名不变）+ `params.yaml` + `index.csv` | `git mv` |
| `agent/train_curve.py`、`boxplot.py`、`violinplot.py`、`pointplot_*.py` | `result/plot.py` + `paper_assets/figures/src/*.py` | 重写（风格改为 scifig 契约；原脚本读取已删除的 CSV） |
| `agent/wilcoxn_312.py` | `result/stats.py` | 重写（原脚本的 R+/R- 按位置而非秩求和） |
| `agent/figures/*` | `paper_assets/figures/legacy/` | `git mv` |
| `compare/Rule.py`、`Rules.py` | `agent/baselines/rules.py` + `baselines/run_rules.py` | 重写（新仿真器） |
| `compare/A2C_model.py`、`DQN_model.py`、`DDQN_model.py` | `agent/baselines/{a2c,dqn,ddqn,encoders,offpolicy,common}.py` + `configs/baselines/*.yaml` | 移植为新实现 |
| `compare/HGNN_model_no_fluid.py` | 删除（`fluid_mode: off` 替代） | — |
| `agent/competitionPlatform.py`、`agent/algorithm_main.py`、`env/` | 删除（不可运行的旧平台 B） | — |
| `test_code/` | 删除 | — |
| `README.md` | 十节复现手册 | 重写；旧"训练机制 / 配置要点"并入 §1 与 `configs/*.yaml` 的注释 |
| （无） | `eval.py`、`environment/profiled.py`、`tools/`、`scripts/`、`result/{aggregate,metrics,stats,plot}.py`、`data/generator.py`、`docs/`、`tests/` | 新增 |
| （无） | `environment/collector.py` + `rollout.collect_vectorized_rollouts_parallel`（多进程采集，默认关闭，数据逐位一致） | 新增（C11） |

## 丢弃的配置键（原 JSON 中无人读取）

`env_paras` 整段、`test_paras` 整段，以及 `train_paras` 中的 `simulator_backend`、`fluid_recompute_interval`、
`rollout_steps`、`target_transitions` 等 12 个键：代码中没有任何读取点，YAML 中不再保留。
全部被读取的键（44 个 `train_paras` 键、13 个 `model_paras` 键）由 `tests/test_config_bridge.py` 逐键比对值与类型。
新增键：`eval_every`、`checkpoint_every`、`train_confirmed`、`train_version`、`algorithm`、`algo.offpolicy` 段，以及
`env.collector_workers`、`env.collector_start_method`（多进程采集；0 即原路径）。

## 弃用接口（仅转发，不再维护）

`python agent/train.py`（转发到 `train.py`）；`from agent.HGHH_model import HGNNScheduler`、
`from agent.PPO_model import PPO`、`from agent.sci_rollout import collect_vectorized_rollouts`、
`from agent.training_simulator import TrainingSimulator`、`from agent.visualization import TrainingVisualizer`、
`from compare.MLP_model import MLPScheduler` 均可用。原 README 列出的 `CompetitionPlatform.run_simulation()`、
`SchedulingAlgorithm.generate_schedule(platform)`、`RolloutAlgorithm.generate_schedule()`、`BatchedActor`
随旧平台一并删除。

## 等价性保障（`tests/equivalence/`）

- L0 配置桥接逐键等值等型；L1 算例解析字段；L2 脚本化动作序列下的仿真轨迹；L3 种子 42 的初始化 `state_dict`；
  L4 固定观测上的 `act_batch`/`evaluate_batch` 输出；L5 回报/优势与 minibatch 索引序列；
  L6 端到端 3 轮训练（`train_metrics.csv` 非计时列与三个检查点逐位比对，且 `eval_every: 1` 下运行，证明新增的
  贪心评估不消耗随机数）；L7 对运行期无法触发的函数（Gurobi LP、`load_checkpoint`、`write_metrics` 等）做 AST 比对。
- 黄金基线由原提交在 `tests/golden/_work/src` 自动检出生成，线程钉死（`OMP_NUM_THREADS=1` 等）。
