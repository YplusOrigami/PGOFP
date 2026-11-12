# PGOFP — Prior-Guided One-Step Flow Policy

用**控制先验**（PID / 采样式 MPC / 冻结 PPO，或它们与高斯的混合）替代标准高斯，作为**单步 MeanFlow 策略的源分布**，并以 FLAME 风格的 Q 重加权最大熵流匹配在线训练。应用目标是托卡马克实时位形控制（HFM 仿真器，尚未接入）。

设计依据：`PGOFP实现方案.md`（实施规格）、`Online-FLAME设计.md`（算法思路）、`开题报告.md`（研究主线）。

## 当前状态（先读这一节）

| 项目 | 状态 |
|---|---|
| MuJoCo 全链路（采样→训练→评估→检查点→恢复→矩阵实验） | ✅ 已实现并测试 |
| 主算法 `prior_flame_m` / 同代码高斯对照 `gaussian_flame_m` | ✅ |
| 控制先验：PID、CEM-MPC、冻结 PPO、混合/多控制器、退火 | ✅ |
| 基线：Online-OFQL、SAC、PPO、controller-only | ✅（TD3、扩散类基线未做） |
| 日志：JSONL + TensorBoard；WandB（online/offline/disabled，可选依赖） | ✅（30k 套件当前默认 disabled；此前已通过在线短验证） |
| 聚变接口 + mock 后端 | ✅ 仅验证接口契约 |
| **真实 HFM 接入** | ❌ 未做。观测 schema / 7→12 线圈矩阵 / 任务参数均为 `verified:false`，真实后端拒绝运行 |
| `flame_reference`（官方 FLAME 复现） | ❌ 未实现，仅占位并拒绝启动 |
| 正式实验结果 | ❌ **没有**。仓库中的运行都是链路验证规模 |
| 测试 | 136 个非 slow 测试全部通过；2 个 slow 测试中 **1 个失败**（见「已知问题」） |

> 本仓库不含任何论文级结论。是否"先验优于高斯"是待做实验，不是已有结果。

## 已知问题与偏离（务必知晓）

1. **Q 重加权幅度不足（未解决）**：低维双峰问题 `tests/numerical/test_weighted_transport.py::test_q_reweighting_moves_mass_to_high_q_mode_and_no_q_does_not`（slow）失败。解析目标正模态占比 0.70，训练后约 0.58–0.60（无 Q 对照约 0.49–0.55）。方向正确、幅度偏小，原因（SNIS 偏差 / 优化不足 / 网络容量）**尚未查清**，也未放宽阈值。已验证：精确速度场积分能还原目标分布（理论无误），所以问题在估计或优化。
2. **`K_local=16` 默认开启**，偏离方案字面写法：在逆向核候选之外，加入围绕当前策略样本的局部候选并做 MIS 修正。目标分布不变，只降方差（t→1 时纯逆向核方差爆炸）。`algorithm.K_local=0` 即方案字面版本，保留为消融。
3. **时间嵌入最高频率 `time_max_freq=4`**（而非常见的 64）：高频嵌入会使 MeanFlow 目标中的 ∂ₜu 项发散。
4. 熵为**训练期代理**（连续流逆向密度），与实际部署的单步映射不严格等价；日志一律称 `entropy/proxy_mean`。
5. 初步验证（InvertedPendulum，15k 步，仅 2 个 seed）：PID 单独 = 1000；PID+噪声源（无 flow 改进）≈ 42 且 100% 失败；Gaussian 与 PID 先验两种 flow 末段均达 1000，AUC 上 Gaussian 更高。样本太少、任务太简单，**不能据此得出任何结论**。

完整偏离清单见 `docs/reproduction_notes.md`。

## 安装

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[mujoco,dev,analysis]"     # 需要 WandB 时再加 ".[wandb]"
python scripts/validate_install.py             # 检查 torch.func.jvp、MuJoCo reset/step、写入器
pytest -m "not slow"                           # 约 1–2 分钟
```

- Python 3.10–3.12；已验证环境：macOS arm64、CPU、torch 2.14、gymnasium 1.4、mujoco 3.16。
- CUDA 预期可用（`device=auto`），**未在 CUDA 上验证**；**不支持 MPS**（JVP/密度数值未验证）。
- 不依赖 WandB 账号、HFM SDK 或显示服务即可训练（渲染不是必须）。

## 方法名（不要混用）

| `algorithm=` | 含义 |
|---|---|
| `prior_flame_m` | 主方法：条件先验源 + MeanFlow + 逆向 Q 重加权 + 训练期密度代理 + 单步动作 |
| `gaussian_flame_m` | **同一个类**，配置合成器强制 `source.kind=gaussian`（冲突覆盖会报错）；主对照 |
| `online_ofql` | FBC + 可微 Q 引导基线（自有系数 `eta`，不使用 `alpha_ent`） |
| `sac` / `ppo` / `controller_only` | 基线；`controller_only` 有 `mode=deterministic` 与 `noisy_source`（零速度 flow，隔离"随机化"与"flow 改进"） |
| `flame_reference` | 官方复现占位，未实现 |

## 快速开始

```bash
# 1. 冒烟（3k 步，smoke=true，不进入正式统计）
python scripts/train.py env=inverted_pendulum prior=pid algorithm=prior_flame_m experiment=smoke

# 2. 主方法 vs 同代码高斯对照
python scripts/train.py env=reacher algorithm=prior_flame_m prior=pid source=constant seed=0
python scripts/train.py env=reacher algorithm=gaussian_flame_m prior=gaussian seed=0

# 3. PID 质量分档（独立校准 seed，不使用测试 seed）
python scripts/calibrate_controller.py env=inverted_pendulum prior=pid trials=30

# 4. PPO 先验：训练 → 分阶段导出 artifact + 校准分档 → 作为源接入
python scripts/train_prior.py env=half_cheetah algorithm=ppo seed=100
python scripts/train.py env=half_cheetah algorithm=prior_flame_m prior=ppo \
    prior.artifact_path=/abs/path/artifacts/priors/half_cheetah_ppo_seed100/stage_1000000 prior.distribution=native

# 5. 实验矩阵（先 dry-run，只写清单不启动进程）
python scripts/launch_experiments.py experiment=prior_ratio --dry-run
python scripts/launch_experiments.py experiment=prior_ratio --execute max_parallel=4 gpus=0,1
python scripts/analyze_results.py matrix=/abs/path/outputs/experiments/prior_ratio/manifest.json thresholds=inverted_pendulum:900

# 6. 对检查点做事后分析
python scripts/evaluate.py        checkpoint=/abs/.../checkpoints/last.pt episodes=20     # 冻结 source spec 与 normalizer
python scripts/inspect_policy.py  checkpoint=...   # 固定条件库，每条件 2048 样本，位移/多峰统计
python scripts/benchmark_latency.py checkpoint=... # 分阶段 p50/p95/p99，batch=1，缓存/未缓存先验
python scripts/export_policy.py   checkpoint=...   # 自包含推理包（flow+先验+归一化+source spec）
```

所有命令接受 Hydra 风格 `key=value` 覆盖：`group=name` 选择配置组（`env= algorithm= prior= source= logging= experiment=`），点路径覆盖叶子值（如 `algorithm.K_train=16`）。**未知键会报错**。

**先验 / 源常用开关**
- `source.beta`（先验占比）、`source.sigma_prior`（潜空间扰动，默认 0.3）、`source=annealed`（线性退火）
- `prior.distribution=native|center_gaussian`（PPO：原生 pre-tanh 分布 / 仅取中心加显式 σ，二者实验中必须分开命名）
- `condition.mode=public_state|prior_features`（网络是否额外看到先验中心；与高斯源配对可分离"输入信息"与"源替换"）
- `algorithm.actor_objective=reverse_qrmf|reverse_no_q|reverse_uniform|endpoint_is`
- `algorithm.temperature.mode=fixed|auto`

**WandB 项目与分组**：`launch_experiments.py` 自动路由——每个环境一个 project（`pgofp-<env>`，如 `pgofp-hopper`；smoke 统一进 `pgofp-smoke`），variant 名写成 `group`（Group by 即得多 seed 均值±带）。用户显式给出的 `logging.wandb.project/group/entity` 优先。`logging` 不参与配置哈希。密钥放在项目根 `.env`（`WANDB_API_KEY=...`，已被忽略，自动加载）；服务端若不是默认站点，需显式 `logging.wandb.entity=<你的entity>`。单次 `train.py` 不做自动路由，请手动传 `logging.wandb.project=...`。

**恢复**：`resume=/abs/run_dir resume_mode=exact_resume|episode_restart|weights_only`。`exact_resume` 只接受带环境快照的环境（toy / fusion mock）；MuJoCo 使用 `episode_restart`（不保证逐步一致）。修改任务/网络/先验 artifact 等会被拒绝恢复。

## 目录结构

```
PGOFP/
├── README.md  pyproject.toml  .gitignore  .env.example
├── PGOFP实现方案.md  Online-FLAME设计.md  Online-OFQL设计.md  开题报告.md   # 设计文档（保留原文）
├── docs/                       # 契约文档
│   ├── algorithm_contract.md       符号、坐标、梯度边界、估计器语义、K_local 说明
│   ├── source_prior_contract.md    控制器→源分布、状态一致性、先验 artifact
│   ├── environment_contract.md     终止语义、计数、RNG、任务×先验支持表
│   ├── experiment_protocol.md      seed/预算/统计/公平性规则、消融矩阵
│   ├── fusion_integration.md       HFM 接入所缺信息与步骤
│   └── reproduction_notes.md       与本地设计文档的逐项偏离
├── configs/
│   ├── train.yaml  prior_train.yaml  evaluate.yaml
│   ├── algorithm/   prior_flame_m, gaussian_flame_m, online_ofql, sac, ppo, controller_only, flame_reference
│   ├── env/         inverted_pendulum, reacher, half_cheetah, hopper, walker2d, toy, fusion_stub
│   ├── prior/       gaussian, pid, mpc, ppo, multi_controller
│   ├── source/      constant, annealed
│   ├── controller/  inverted_pendulum_pid, reacher_pid, cem_mpc
│   ├── logging/     local, wandb_offline, wandb_online
│   ├── experiment/  smoke, validation_small, mujoco_main, source_type, prior_ratio,
│   │                prior_quality, schedule_time, flame_components, robustness
│   └── fusion/      observation_schema, coil_map, tasks     # 全部 verified:false
├── src/pgofp/
│   ├── core/         types, specs(ActionSpec/EnvSpec/SourceSpec), protocols, registry, exceptions
│   ├── math/         action_transform(tanh/仿射/log-Jacobian), time_sampling, reverse_sampling,
│   │                 importance(SNIS), meanflow_target(JVP), divergence, integration(Euler/Heun)
│   ├── priors/       base(混合高斯源), gaussian, mixture(build_source), controller_source,
│   │                 native_policy_source, particle_source, schedules, cache
│   ├── controllers/  base, pid, mpc(CEM), ppo(冻结), manager(逐环境状态), artifacts, factory, calibration
│   ├── networks/     mlp, embeddings, conditioning, mean_flow, critics, gaussian_policy
│   ├── density/      continuous_proxy(默认), local_meanflow_proxy, diagnostics(一步/多步差距)
│   ├── algorithms/   prior_flame(主 agent), objectives(损失), temperature, online_ofql, baselines,
│   │                 flame_reference(占位), factory, base
│   ├── data/         replay_buffer, rollout_buffer(GAE)
│   ├── training/     runner, collector, update_scheduler(UTD), checkpoint, prior_trainer(PPO),
│   │                 bundle(推理包), hooks
│   ├── envs/         adapters(Gymnasium), vector(显式逐槽 reset), toy, factory,
│   │                 mujoco/{tasks,features,planning}, fusion/{protocol,adapter,observation,
│   │                 reference,actuator,safety,mock_backend}
│   ├── evaluation/   evaluator, metrics(bootstrap CI/AUC/…), distributions, latency, robustness
│   ├── experiments/  matrix, launcher, manifest, aggregate
│   ├── logging/      logger, jsonl, tensorboard, wandb
│   ├── config/       schema(结构化), compose(组合+覆盖), validate(训练前检查)
│   ├── utils/        seed(命名 RNG 流), normalization, torch_utils, timing, provenance
│   └── cli.py
├── scripts/          train, train_prior, evaluate, launch_experiments, calibrate_controller,
│                     analyze_results, benchmark_latency, inspect_policy, export_policy, validate_install
├── tests/            unit/(10) integration/(7) numerical/(3)，约 2000 行
├── assets/  artifacts/  outputs/                    # 资源说明；运行生成（不入库）
```
约 7800 行源码 + 2000 行测试。

## 已实现的功能

**算法**
- 单步 MeanFlow 策略 `z = ε − u(ε,0,1;c)`，末层零初始化（初始策略 = tanh(源样本)）；动作链 `z → tanh → 仿射`，熵与 Critic 定义在 `tanh(z)` 坐标。
- Actor：条件逆向采样候选 + 每条件 SNIS 权重 `softmax(Q/α + log|J|)` + MeanFlow JVP 目标（目标全 detach；候选/权重/Critic 不回传梯度）。含端点重要性采样校验目标、无 Q、均匀权重变体。
- Critic：双 Q、软目标 `r + γ·mask·(minQ' − α·logπ)`，目标动作来自同一条件源；`target_tau` EMA。
- 熵代理：连续流逆向增广 ODE（Heun，Hutchinson/精确 trace，记录真实 NFE 与逆向残差）；局部平均速度估计器（可选）；NaN 触发明确异常而非置零。
- 温度：固定 / 自动（方向由测试保证），记录触边比例。

**源分布与先验**
- 统一为对角高斯混合：采样与 `log_prob` 永远指向同一实际分布（含失败回退）。
- 确定性控制器 → pre-tanh 中心 + 潜空间 σ；PPO 原生 `(μ, log_std)`；多控制器混合；粒子集需正带宽核才进入密度路径。
- 退火（constant/linear/piecewise/cosine，以环境步为时钟，一次更新用同一不可变快照）。
- 控制器：向量 PID（anti-windup、角度 wrap，InvertedPendulum 符号由实测确定）、Reacher 关节 PD+解析 IK、CEM-MPC（独立 MjModel，状态只由公开观测重建，固定查询随机性）、冻结 PPO（使用 artifact 自带归一化）。
- 先验 artifact：哈希校验、环境/维度/schema 校验，拒绝有状态/循环 checkpoint。
- 质量分档：仅用独立校准 seed。

**训练与工程**
- 显式逐槽 reset 的环境池；`terminated`/`truncated` 分离，截断样本用真实终止前观测 bootstrap；控制器每步 `advance` 一次，下一条件在 reset 前构造。
- 以真实 transition 计数的分数 UTD；命名 RNG 流（评估不消耗训练随机性）；观测归一化快照。
- 原子检查点（临时文件+fsync+rename、sha256 sidecar）；三种恢复模式；日志按检查点截断并保留被丢弃记录。
- 评估：独立环境/控制器/RNG，冻结 source 与 normalizer，逐回合 CSV；分布诊断（位移、协方差、双峰系数）；时延基准；扰动环境包装。
- 实验矩阵：展开 + 校验，不支持组合记为 `skipped` 并写明原因；独立进程、并发上限、仅对基础设施类退出码重试、NaN 失败不重试；聚合为 seed 级与组合级 CSV（seed 为统计单位，bootstrap CI，未达阈值记 `censored`）。

**聚变接口（mock）**
`FusionBackend` 协议、命名字段观测装配（缺测掩码，不填"正常 0"）、参考轨迹（哈希+前瞻权限）、7→12 展开与执行器所有权（只执行自己拥有的延迟/限速阶段）、安全过滤器（如实报告干预）、可快照的 mock 后端。`env=fusion_stub` 可端到端训练。

## 输出目录

```
outputs/runs/<run>/
  config_resolved.yaml   manifest.json(来源/哈希/结果)   metrics.jsonl(原始，每条带 env step)
  events.jsonl(恢复/回退/NaN/WandB 降级…)   tensorboard/   evaluation/{episodes.csv,summary.json}
  checkpoints/{last,best,final}.pt + .json(sha256) + replay/
outputs/experiments/<matrix>/
  manifest.json  summary.csv  summary_seed_level.csv  failures.json  figures/  logs/  runs/
artifacts/priors/<env>_ppo_seed<N>/stage_<steps>/   # weights.pt manifest.json normalizer.npz ... + tiers.json
```

## 测试覆盖

`tests/unit`、`tests/integration`、`tests/numerical`：tanh Jacobian 与极值稳定性；混合源（矩、β 端点、完整密度、回退一致、生成器可复现）；逆向核重构与归一化（平移/异方差/混合源）；MeanFlow JVP 对有限差分（r 固定）、线性流恒等式、目标无梯度；密度在解析仿射场上的符号与数值、Heun 收敛、Hutchinson 无偏；温度方向；回放配对；分数 UTD；PID 符号（真实 MuJoCo）与 MPC 隔离；终止语义与逐槽控制器状态；exact resume 与不间断运行一致；检查点校验和；先验 artifact 篡改/不匹配拒绝；JSONL 与 TensorBoard 一致及恢复截断；聚变 mock 契约；精确速度场把源输运到 Q 加权目标。

`pytest -m "not slow"`（136 个，通过）；`pytest -m slow`（2 个双峰训练，目前 1 个失败，见上）。

## 未完成 / 下一步

- 查清并修复 Q 重加权幅度问题（先跑"精确速度目标回归"判断是否网络/优化问题，再扫 K 与 `K_local`）。
- 在 Reacher / HalfCheetah / Hopper 上做正式 5-seed 矩阵（当前仅 InvertedPendulum 小规模验证）。
- 真实 HFM：拿到 SDK、64 维字段顺序、7→12 矩阵、F1/F2 任务规格后接入；先用 `docs/fusion_integration.md` 清单。
- `flame_reference`（固定上游 commit 复现）、TD3 与扩散类基线、CUDA 验证、WandB 在线上传验证。
