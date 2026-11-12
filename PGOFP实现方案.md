# PGOFP 实现结构与实现细节

> 版本：v1.0；日期：2026-10-20。本文是实施规格，不包含可运行算法代码，也不表示实验结果已经得到验证。
>
> 阅读对象：后续负责算法、环境、先验控制器、实验管理与验证的实现 agent。项目名称 PGOFP = Prior-Guided One-Step Flow Policy。

## 目录

1. [依据、目标与范围](#1-依据目标与范围)
2. [必须先统一的算法约定](#2-必须先统一的算法约定)
3. [总体结构与数据流](#3-总体结构与数据流)
4. [状态、动作与数据协议](#4-状态动作与数据协议)
5. [控制器与条件源分布](#5-控制器与条件源分布)
6. [单步流策略与训练目标](#6-单步流策略与训练目标)
7. [密度、熵与温度](#7-密度熵与温度)
8. [在线采样与训练循环](#8-在线采样与训练循环)
9. [MuJoCo 环境与控制先验落地](#9-mujoco-环境与控制先验落地)
10. [PPO 先验的训练与导出](#10-ppo-先验的训练与导出)
11. [后续聚变模拟器接口](#11-后续聚变模拟器接口)
12. [完整目录结构](#12-完整目录结构)
13. [逐文件实现职责](#13-逐文件实现职责)
14. [配置与默认实验参数](#14-配置与默认实验参数)
15. [消融矩阵与公平比较](#15-消融矩阵与公平比较)
16. [训练记录、TensorBoard 与 WandB](#16-训练记录tensorboard-与-wandb)
17. [评估、统计与诊断](#17-评估统计与诊断)
18. [检查点、恢复与部署](#18-检查点恢复与部署)
19. [验证要求与验收标准](#19-验证要求与验收标准)
20. [实施顺序与工作交接](#20-实施顺序与工作交接)
21. [风险与待接入信息](#21-风险与待接入信息)
22. [参考资料](#22-参考资料)

## 1. 依据、目标与范围

### 1.1 本地文档与当前仓库状态

已阅读根目录的三个正文文档，忽略 macOS 生成的 `._*` 元数据文件：

| 文档 | 在本方案中的作用 |
|---|---|
| `Online-FLAME设计.md` | 提供在线 Actor–Critic、Q 重加权、训练期密度估计和高效部署的思路；示例存在不一致，见第 2 节。 |
| `开题报告.md` | 定义研究主线：PID/MPC/PPO 等控制先验替代高斯源分布；研究先验类型、比例、质量、退火、时间采样；后续接 HFM。 |
| `Online-OFQL设计.md` | 对照 MeanFlow 两时间变量、JVP、一致性训练，以及 OFQL 比较基线的定义。 |

当前 PGOFP 文件夹只有上述文档及系统元数据，没有可直接修改的算法实现、依赖锁文件或 HFM SDK。本方案按新建工程设计；开题报告中提到的已有代码，需要在后续获得路径后再评估复用。不得把报告中的“已实现”视为当前文件夹里已经有代码。

### 1.2 必须完成的工程目标

1. 建立以 FLAME 的 Q 重加权、最大熵学习和 MeanFlow 单步生成思路为基础的 PGOFP 框架。
2. 源分布统一支持：标准高斯、PID+扰动、MPC+扰动、冻结 PPO 分布、多控制器混合以及先验/高斯混合。
3. 环境交互、Actor 训练、Critic 目标、密度估计、评估和部署使用同一套条件源分布定义。
4. 首先在 Gymnasium MuJoCo 任务验证，具备可靠的训练、评估、恢复、消融和统计链路。
5. TensorBoard 记录训练曲线，WandB 同步并管理配置、曲线、评估表、视频和模型产物；断网时仍能训练。
6. 预留 HFM/其他聚变模拟器适配接口；后续接入时不修改核心算法。

### 1.3 本轮与第一阶段边界

- 本轮只写本文，不创建空代码框架、不安装依赖、不启动训练、不连接 WandB 账号。
- 第一阶段实现范围：向量观测、连续有界动作、单机单训练进程，可用多环境采样；优先 CPU/CUDA，Apple Silicon 先验证 CPU 路径。
- 第一阶段核心实验：MuJoCo 中的 PGOFP 与同实现标准高斯源基线；PID/MPC 只在有合理控制设计的任务启用。
- HFM 现阶段交付接口、schema、模拟替身与接入说明；真实动作展开矩阵、SDK 和参考轨迹未获得前，真实接入明确标为未完成。
- 分布式学习、视觉编码器、在线联合更新 PPO 先验、大规模 Transformer、完整约束 RL 不在第一阶段。
- 多步扩散、TD3、Online-OFQL 是扩展比较；SAC、PPO、控制器单独执行和高斯源 PGOFP 是优先比较基线。

## 2. 必须先统一的算法约定

### 2.1 本地设计不能直接逐行翻译成代码

核查 FLAME 论文后可见：论文中的单步版本依赖 MeanFlow；Q 重加权还涉及逆向采样；FLAME-R 与 FLAME-M 的命名也不同于本地示例。论文中的 R 使用连续流路线，M 使用 MeanFlow 单步生成。后续以明确公式和版本标识定义实现，避免只依据简称判断行为。[FLAME 原论文](https://arxiv.org/html/2602.01606v1)

| 本地文档中的问题 | 本方案的实施决策 |
|---|---|
| 用单个时间参数的普通 FM 回归，直接宣称得到准确单步策略。 | 主算法使用平均速度网络 `u(z, r, t, condition)`，训练包含 MeanFlow JVP 目标；普通 FM 只作消融或连续流对照。 |
| 插值从噪声到动作，但推理又采用反方向减号。 | 全工程采用 `t=0` 为目标动作、`t=1` 为源样本；第 6 节给出全部符号。 |
| 从当前策略生成终点，直接乘 `exp(Q/alpha)`，没有说明采样测度。 | 主算法明确采用条件逆向候选采样；另提供有已知 proposal 密度的端点重要性采样校验路径。不得混用两种权重。 |
| 密度估计接收指定动作，却重新采独立噪声向前积分。 | 指定动作的 log-density 必须从该动作回到源端，或者使用同轨迹联合采样结果。独立样本的密度不能配给另一动作。 |
| 把平均速度直接当作瞬时 ODE 速度。 | 连续密度路径使用 `f(z,t)=u(z,t,t)`；区间平均速度的局部离散密度估计作为另一 estimator 实现。 |
| 将 argmax 多候选选动作等同于最大熵策略采样。 | 默认环境动作是一次随机源采样+一次流前向；best-of-K 独立标记为行为/评估变体。 |
| 对动作硬 clip 后仍使用原连续策略密度。 | 主实现采用 pre-tanh 动作坐标；硬投影后的策略密度不再冒充原密度。 |
| `done` 同时表示真正终止与时间限制。 | 数据保留 `terminated` 和 `truncated`，bootstrap 使用明确掩码。 |
| 熵太低时 alpha 更新方向在公式与示例中不一致。 | 使用第 7 节的损失与方向测试，不照搬符号。 |
| 将 NFE=1 直接视为满足实时控制。 | 同时测量先验控制器、数据准备、流网络、动作变换的端到端延迟。 |

### 2.2 三条方法路线必须分别命名

| `algorithm.name` | 定义 | 用途 |
|---|---|---|
| `prior_flame_m` | 本文主算法：条件先验源、MeanFlow、逆向 Q 重加权、训练期密度代理、单步动作。 | 研究方法。 |
| `gaussian_flame_m` | 与主算法完全同代码，只有源分布改为标准高斯。 | 判断“替换源分布”的增益；主对照。 |
| `flame_reference` | 对固定版本官方实现进行适配和复现，记录所有偏差。 | 外部复现基准，不与本文适配版混为一谈。 |

`online_ofql` 是另外的 FBC+可微 Q 引导基线。不能把主算法改成 `FBC - lambda*Q` 后仍称 FLAME。`source_only` 表示取消策略改进的源/控制器评估，不作为主算法别名。

### 2.3 理论陈述的界限

- 控制先验改善位移目标幅值、训练稳定性和样本效率是待验证假设，不保证每个任务获益。
- 源分布非零方差、可查询密度与参考条件一致，是本文密度路径的重要条件。
- 控制先验不提供状态安全保证；确定性控制器叠加高斯扰动后尤其不能直接称为安全控制器。
- 任意单步神经映射可能不可逆。多步连续流密度与实际单步映射密度不必相同；主算法需把该项标记为近似，见第 7 节。
- 论文中的理想最优性论证不能自动覆盖有限候选 SNIS、截断时间、退火源分布、pre-tanh 变换和有限容量网络。
- 报告中的“先验更近时 Jacobian 必然更小”不作为定理使用；记录位移与 Jacobian 诊断，再做实证分析。

## 3. 总体结构与数据流

```text
resolved config + seed + artifact manifest
                   |
                   v
Environment adapter -> raw observation / reference / public context
                   |                         |
                   |                  Controller manager
                   |                  (PID / MPC / frozen PPO)
                   |                         |
                   +---- ConditionBuilder <--+-- controller snapshot
                                 |
                          SourceDistribution
                          sample / log_prob
                                 |
                         MeanFlow one-step map
                                 |
                    tanh -> action affine transform
                                 |
                        optional safety adapter
                                 |
                          Environment.step
                                 |
                          Replay transition
                                 |
             +-------------------+---------------------+
             |                   |                     |
       soft critic update   Q-weighted MeanFlow    density diagnostics
             |               actor update              |
             +-------------------+---------------------+
                                 |
                 metrics / checkpoint / evaluator
                                 |
               JSONL + TensorBoard -> WandB sync
```

### 3.1 层次与依赖约束

- `core`、`math`、`networks` 只依赖类型定义和数值库。
- `controllers` 不依赖 flow agent；接收公开观测、参考与自己的快照。
- `priors` 将控制器输出转换成可采样、可计算密度的源分布。
- `algorithms` 依赖 `core/math/networks/priors`，不直接导入 MuJoCo、HFM 或 WandB。
- `envs` 负责环境差异与物理动作映射，不处理 Actor/Critic 损失。
- `training` 编排采样、更新、记录和恢复；不在 runner 内重复写损失公式。
- `logging` 接收普通标量和产物描述；外部服务故障不能中断数值更新。
- `evaluation` 使用独立环境、随机数流和控制器状态，不能改变训练状态。

### 3.2 面向 agent 的最小接口

以下是接口规格，不是本轮实现：

| 对象 | 必须提供的能力 |
|---|---|
| `EnvAdapter` | `reset(seed, options)`、`step(action_command)`、`build_context()`、`spec`、`close()`；可选 `get_snapshot/set_snapshot`。 |
| `Controller` | `reset(mask)`、纯查询 `predict(input, snapshot)`、动作执行后 `advance(...)`、`state_dict/load_state_dict`、能力声明。 |
| `SourceDistribution` | `sample(condition, sample_shape, generator)`、`log_prob(z, condition)`、`representative(condition)`、版本化的 `spec`。 |
| `FlowPolicy` | `sample(condition, source_spec, generator)`、`map_source(z, condition)`、`mean_velocity(z,r,t,condition)`。 |
| `DensityEstimator` | `estimate_at(action_latent, condition, policy, source_spec)`；返回值须带近似语义和误差诊断。 |
| `Agent` | `act(...)`、`update(batch, update_context)`、`state_dict/load_state_dict`；不持有在线环境对象。 |
| `MetricLogger` | `write_scalars(record)`、`write_artifact(...)`、`flush()`、`close()`。 |
| `CheckpointManager` | 原子保存、完整恢复校验、推理包导出；不能只保存 Actor 权重。 |

## 4. 状态、动作与数据协议

### 4.1 三套动作坐标

统一使用下列命名，变量名不得只写含义不明的 `action`：

| 坐标 | 符号 | 含义 |
|---|---|---|
| flow latent | `z ∈ R^d` | 无界 pre-tanh 空间；源分布与速度网络工作空间。 |
| normalized command | `a_norm = tanh(z)` | 严格位于 `(-1,1)^d`，Critic 的动作输入。 |
| physical command | `a_phys = b + c ⊙ a_norm` | `b=(high+low)/2`，`c=(high-low)/2`；传给任务适配器。 |

对 MuJoCo，适配器再把 `a_phys` 传给原始环境。对 HFM，归一化 7 维指令经装置配置映射到 12 维线圈电压；实际执行器响应属于环境动力学。

所有主实验的熵定义在 `a_norm` 坐标，避免环境单位改变温度意义。若额外输出物理动作密度，显式减去 `sum(log(c))`。不对 7→12 维非方阵展开计算 12 维 Lebesgue 密度。

稳定计算：`log|J_tanh(z)| = sum(2*(log(2)-z-softplus(-2*z)))`。仅在将外部有界动作导入 latent 时使用 `atanh(clamp(a_norm,-1+eps,1-eps))`，并记录边界 clamp 比率。不能通过截断策略输出 latent 来隐藏饱和问题。

### 4.2 策略条件 `PolicyCondition`

策略真正依赖的条件写作 `c`，不再将所有信息都含混称为状态 `s`：

| 字段 | 建议形状 | 语义 |
|---|---|---|
| `obs_raw` | `[B,d_o]` | 环境公开原始观测，控制器物理计算与离线检查使用。 |
| `obs_features` | `[B,d_f]` | 经过训练期归一化和确定性特征构造后的网络输入。 |
| `reference` | `[B,d_r]` | 当前参考及约定的前瞻窗口；无目标轨迹任务为零宽张量。 |
| `history` | `[B,H,d_h]` 或展平 | 可选有限历史，用于 POMDP、动作延迟；首版不使用 RNN。 |
| `controller_features` | `[B,d_c]` | 积分误差、上一指令、滤波状态、必要 MPC 内部状态等可部署特征。 |
| `source_features` | `[B,d_p]` | 当前先验参数、beta/sigma、先验有效标志；由当前 `SourceSpec` 构造。 |
| `phase` | `[B,d_phase]` | 有限时域的剩余时间或参考相位；避免同观测不同阶段含义冲突。 |
| `context_id` | 元数据 | 参考文件、shot、模型参数版本等；不是自动作为网络输入的整数 ID。 |

源分布退火时 `source_features` 必须体现当前分布参数，Actor、Critic、密度估计器均接收相同条件。第一轮主实验采用固定源参数，以减少非平稳性。配置必须声明 `condition_mode`：默认 `public_state` 不额外把 PPO/MPC 的动作均值作为网络输入；有状态 PID 所需记忆仍纳入增强状态。`prior_features` 才显式加入先验中心/尺度，作为独立增强方案。

先验参数若直接进入网络，就同时改变了源分布和输入信息，不能把全部收益都归因于源替换。增加“高斯源+同样先验特征”与“先验源+同样先验特征”的配对比较；每对实验保持输入维度、公开观测和控制器记忆一致。固定参数无需输入时可用零宽 source features；退火的 beta/sigma 等调度变量仍在所有比较方法中按同一协议条件化。

控制器记忆会影响后续先验，因此必须进入增强状态或有限历史。仅保存输出均值有助于重采样，但不足以保证增强过程具有 Markov 性。对 warm-start MPC，首版采用固定种子、无历史 warm-start 的纯查询模式；启用有记忆版本时必须扩大条件协议。

### 4.3 批次形状与 dtype

- 观测、动作、速度、reward、log-density：训练默认 `float32`；数值参考测试使用 `float64`。
- 环境输入输出可为 NumPy；统一在 collector/batch collator 边界转换，算法内不反复转换。
- 标量训练向量统一 `[B]`，reward 不混用 `[B,1]`；候选维度固定在第二维 `[B,K,d_a]`。
- 时间 `t,r` 为 `[B,1]`，候选展开后 `[B,K,1]`。
- 标记为 bool，索引为 int64。每个 loss 入口断言形状，防止 `[B]` 与 `[B,1]` 广播成 `[B,B]`。
- `B=1`、`K=1`、`n_envs=1` 都不能被无条件 `squeeze()` 破坏。

### 4.4 `ReplayTransition` 的必需字段

| 字段 | 用途 |
|---|---|
| `obs_raw, next_obs_raw` | 真实当前与下一观测；下一观测必须是结束前的 final observation。 |
| `reference, next_reference` | 与对应控制时刻同源；不得从当前环境在线读取旧 transition 的参考。 |
| `history, next_history` | 可选，或可由同 episode 序列恢复。 |
| `controller_snapshot, next_controller_snapshot` | 当前/下一时刻先验控制器状态；回放中不得推进在线控制器。 |
| `prior_parameters, next_prior_parameters` | 冻结控制器的规范化分布输出缓存；必须带 controller/model/normalizer hash。 |
| `action_policy_norm` | 原始单步策略给出的归一化动作。 |
| `action_command_norm` | 外部过滤/回退后实际送给环境的归一化指令；默认 Critic 使用它。 |
| `action_applied_phys` | 可选：环境报告的实际执行器动作，主要用于聚变延迟与限幅诊断。 |
| `reward_raw`、可选 `reward_train` | 原始回报用于报告；奖励变换如启用须全方法一致并写入配置。 |
| `terminated, truncated, bootstrap_mask` | 分别保留结束语义；普通 MuJoCo 时间限制默认允许 bootstrap。 |
| `episode_id, step_in_episode, env_id` | 轨迹恢复、防跨回合历史拼接、定位控制器状态。 |
| `source_version, beta_behavior, sigma_behavior` | 行为时源分布配置，用于审计，不代表训练必须重用旧退火值。 |
| `controller_valid, fallback_reason` | 超时、求解失败、NaN、缺少参考等。 |
| `extras` | schema 明确的低频诊断；禁止把无限增长的完整 `info` 对象直接塞入 buffer。 |

存储原则：回放保存原始输入和冻结先验参数，学习时使用当前训练调度的 beta/sigma 重建源分布。这是 off-policy 学习新策略，不要求复用行为时退火值。若先验本身未来可训练，必须另设计先验版本重算/目标先验策略，不能沿用冻结缓存假装一致。

## 5. 控制器与条件源分布

### 5.1 控制器和源分布必须分层

`PIDController/MPCController/PPOPriorController` 给出控制建议或分布参数；`ControllerGaussianSource/NativePolicySource/MixtureSource` 决定如何在 flow latent 空间形成源分布。这样同一个 PID 可以分别用于控制器基线、先验源、热身策略与回退策略，不复制控制公式。

`ControllerPrediction` 至少包含：动作坐标声明、均值/尺度或粒子、valid、耗时、诊断、版本。源分布构造前必须检查维度、坐标和数值有效性。

### 5.2 首版源分布

所有下式均以同一策略条件 `c` 为条件。

**无信息高斯：**

`p_base(z|c) = N(0, sigma_base² I)`，首版 `sigma_base=1`。Gaussian 对照与主算法只在 `SourceSpec` 处不同。

**确定性控制器加扰动：**

先将控制器动作转换为 `a_prior_norm`，然后

`mu_prior = atanh(clamp(a_prior_norm, -1+eps, 1-eps))`

`q_prior(z|c) = N(mu_prior(c), diag(sigma_prior(c)²))`。

这是“pre-tanh 中心+扰动”，不等同于动作空间高斯。文档和实验标签必须写明。零扰动时经过 tanh 恢复控制器中心；非零扰动时动作均值不一定等于控制器中心。

**PPO 原生策略：**

- 首选预训练 tanh-Gaussian PPO：直接导出其 pre-tanh `mu/log_std`，不再多加一份相同用途噪声。
- 常规 Gaussian+clip 的 PPO 可直接用于 PPO 比较基线；其 clip 后分布存在边界质量，不能原样当作连续 flow 源密度。
- 如只有这种外部 checkpoint，适配器可以显式构造“PPO 中心的 latent Gaussian”，但必须命名为 `ppo_center_gaussian`，不能标为原生 PPO 分布。
- 导出物必须包含 PPO 观测归一化，且在 PGOFP 中冻结；不能改用 PGOFP 的 running statistics 输入冻结 PPO。

**先验/高斯混合：**

`p_src(z|c) = (1-beta) p_base(z) + beta q_prior(z|c)`。

- 采样用逐样本 Bernoulli 选择分量；不是线性插值两个采样值。
- `log_prob` 用 `logsumexp(log(1-beta)+log_p_base, log(beta)+log_q_prior)`。
- beta 为 0/1 时走单分量精确分支，避免 `log(0)`。
- 密度始终是完整混合密度，不能只返回被抽中分量的 log-density。

**多控制器与 MPPI 粒子：**

- 多控制器使用有限混合高斯，权重和为 1，分别保存 component hash。
- MPPI 经验粒子直接作为 Dirac 混合时没有普通连续密度。密度路径必须使用带正方差核的 Gaussian mixture，或者显式注册为仅可采样源并拒绝需要 log_prob 的算法配置。
- 正方差下限首版 `sigma_min=0.05`，上限 `2.0`，均为 latent 单位，属于待调超参。

### 5.3 有状态控制器的一致性

以 PID 为例：积分状态 `I_n` 属于当前条件；`predict(c_n,I_n)` 是纯函数，多次调用不积累积分。环境步完成后才执行一次 `advance` 更新 `I_{n+1}`，包括抗积分饱和所需的实际执行指令。下一状态先验在 reset 之前构造并缓存。

向量环境按 `env_id` 保存独立控制器状态。只有结束的槽位 reset，不能在每次 learner 更新或评估时 reset 全部 PID。

### 5.4 退火和版本

- `constant` 是默认主实验；`linear/piecewise/cosine` 是显式消融。
- 进度统一为 `env_transitions / schedule_total_steps`，不以 optimizer step 或 wall-clock 作为默认时钟。
- 一次更新使用不可变 `SourceSpec` 快照；Actor、Critic target、density、alpha 使用同一快照。
- `schedule_max_delta` 控制每次改变幅度；具体阈值先设小规模扫描，不宣称已有理论下界。
- 冻结评估时使用 checkpoint 中的 beta/sigma；不能因评估 episode 增加而继续退火。
- 源分布的参数或规范变化必须改变 `source_version`；日志记录 beta、各通道 sigma、分量使用率。

### 5.5 失败与回退

MPC 超时/无解、PPO checkpoint 不匹配、PID 数值异常要明确处理。主实验默认对配置错误立即失败；运行时短暂 controller failure 使用同一张量内显式的 fallback component，并把 valid 标志纳入条件。源采样和 log_prob 必须共同指向回退后的实际分布。

不得在 sampler 中静默换成高斯，却在 density 中继续查询旧先验。记录回退比例；高回退率的实验不能归类为正常“强 MPC 先验”结果。

## 6. 单步流策略与训练目标

### 6.1 统一方向与符号

用 `n` 表示环境时间，`t/r` 只用于流时间；用 `reward` 表示奖励，避免与流时间 `r` 冲突。

- `z_a`：目标动作的 latent，处于流时间 0。
- `epsilon ~ p_src(.|c)`：源样本，处于流时间 1。
- `z_t = (1-t) z_a + t epsilon`。
- `v = epsilon - z_a`：沿递增流时间的条件瞬时速度。
- `u_theta(z_t,r,t;c)`：从 `r` 到 `t` 的平均速度，`0≤r≤t≤1`。
- 单步生成：`z_out = epsilon - u_theta(epsilon,0,1;c)`。
- 输出：`a_norm=tanh(z_out)`，再转换物理动作。

首版网络采用 `[256,256,256]` MLP、SiLU、可配置 LayerNorm；分别编码 `r,t` 后与 `z_t`、condition 拼接。最后速度层零初始化，使初始策略恰好等于源经 tanh 后的策略。这个初始化在所有源分布实验中保持一致。

### 6.2 主算法：条件逆向采样

以下为本项目在条件源和 pre-tanh 坐标下的实现定义，必须通过数值校验，不以“替换 randn”代替。

固定一个条件 `c`、流时刻 `t∈[delta,1-delta]` 和中间点 `z_t`，有：

`p_t(z_t | z_a,c) = t^(-d) p_src((z_t-(1-t)z_a)/t | c)`。

定义可采样的逆向核：

`phi(z_a | z_t,c,t) = ((1-t)/t)^d p_src((z_t-(1-t)z_a)/t | c)`。

因此 `p_t(z_t|z_a,c)=(1-t)^(-d) phi(z_a|z_t,c,t)`。对任意具有连续密度的条件源，逆向采样可写成：

1. 采样 `epsilon_k ~ p_src(.|c)`。
2. 计算 `z_a_k=(z_t-t*epsilon_k)/(1-t)`。
3. 目标瞬时速度 `v_k=epsilon_k-z_a_k`。

当源为控制器中心高斯或混合高斯时不需要继续使用标准高斯的旧逆向核。这里控制先验通过逆向候选测度参与训练。

本项目最大熵目标定义在 `a_norm` 中，因此 latent 目标能量必须包含变换 Jacobian：

`E_log(z_a,c) = min(Q1(c,tanh(z_a)), Q2(c,tanh(z_a))) / alpha_ent + log|J_tanh(z_a)|`。

每个固定的 `(c,t,z_t)` 生成 `K_train` 个逆向候选，计算

`logw_k = stop_gradient(E_log(z_a_k,c))`

`w_k = softmax(logw, dim=candidate)`。

这一条件归一化在有限 K 时是有偏 SNIS 近似；K 增大时近似目标条件分布。时间因子 `(1-t)^(-d)` 对固定中间点的候选相同，在条件权重中抵消。实现不得在全 batch 跨状态 softmax。

中间点 proposal 首版取固定全支撑 `h(z_t|c)=N(0,I)`，保持各先验比较一致；策略 proposal、先验中心 proposal 是额外实验。h 在本条件匹配目标中决定覆盖的中间点，其 density 不进入上述候选权重。不得在此公式上随意再加 `-log p_src`；那会改变目标。

### 6.3 Q 加权 MeanFlow 损失

对每个候选，使用同一 `(z_t,r,t,c)`，计算：

`D_u = J_z u_theta(z_t,r,t;c) @ v_k + partial_t u_theta(z_t,r,t;c)`

`u_target_k = stop_gradient(v_k - (t-r)*D_u)`

`L_actor = mean_b sum_k w_bk * mean_action_dim((u_pred_b - u_target_bk)^2)`。

实施要求：

- JVP 的 primal 为 `(z_t,r,t)`，切向量为 `(v_k,0,1)`；condition、先验参数和 r 都固定。
- 网络参数在 target 分支中不得产生二阶训练图；target 完整 detach。
- `u_pred` 是正常可微前向；只对这个预测分支优化 Actor。
- 候选和权重不反传 Critic、proposal、PPO 或控制器；不额外叠加可微 `-Q`，除非选择 OFQL 基线。
- 同一中间点的 `u_pred` 可复用；JVP 中的 action Jacobian 依赖方向 v，需要候选批量化或候选分块。
- Actor 更新时冻结 Critic 参数；Critic 更新时 Actor 无参数梯度。
- `r=t` 时修正项为 0，退化为瞬时速度匹配；非零比例样本学习对角场，为密度估计提供训练信号。

### 6.4 时间对采样

主配置：`t ~ Uniform(delta,1-delta)`，`delta=0.02` 为数值起点；以概率 `flow_ratio=0.5` 令 `r=t`，否则 `r=t*q`、`q~Uniform(0,1)`。

- 逆向公式在 `t→1` 时可能放大 latent 与速度；不得采到端点。
- `delta` 取值改变了训练覆盖，必须记录并消融，不认为越小必然越好。
- `logit_normal` 作为报告中提出的采样消融；使用相同端点界，明确非对角时 r 的采样规则。
- 推理仍固定 `(r,t)=(0,1)`。内部训练区间到端点的泛化是实际误差来源，需要记录一步/多步差异。
- `flow_ratio=0`、`1` 是边界消融；前者对角场缺乏直接监督，后者不提供单步平均速度训练，结果应带清晰标记。

### 6.5 独立校验路线：已知端点 proposal

用于验证逆向采样和权重，不默认用回放动作假装有已知密度。

1. `a_k ~ Uniform((-1,1)^d)`，`z_a_k=atanh(a_k)`。
2. 独立 `epsilon_k~p_src(.|c)`，按线性插值构造 `z_t`。
3. 一般权重为 `exp(Q/alpha_ent + logJ_tanh - log q_z)`。
4. 上述 uniform-action proposal 对应 `q_z=2^(-d)|J_tanh|`，所以权重简化为 `exp(Q/alpha_ent)`，常数在同状态候选归一化中抵消。

这个路径属于端点重要性采样，和 6.2 的条件逆向权重不同。两者使用不同 loss-builder 实现并在日志中记录 `actor_objective`。高维动作下 uniform proposal 可能效率低，主要用于低维数学验证与敏感性比较。

### 6.6 Critic 与 target

两套独立 Q 网络接收 `(condition_features, a_command_norm)`，默认 `min(Q1,Q2)`。不把 `action_applied_phys` 偷换为 Critic 输入，因为执行器动态本来属于环境转移；若改变 MDP 动作定义需全链路同步。

`a_next ~ one_step_policy(theta_target, p_src(.|c_next))`

`y = reward_train + gamma * bootstrap_mask * (min(Q1_target,Q2_target)(c_next,a_next) - alpha_ent*logpi_proxy(a_next|c_next))`

`L_critic = MSE(Q1(c,a_command),y) + MSE(Q2(c,a_command),y)`。

- target 全部 stop-gradient；密度所需的动作导数可以局部启用，不能向 Actor 或温度传播 target 梯度。
- 默认使用目标 Actor，目标 Q；所有方法使用相同 target 策略约定。官方复现若不同单独记录。
- EMA 统一命名 `target_tau=0.005`：`target ← (1-tau)*target + tau*online`，禁止与本地文档的 `rho=0.995` 混用。
- `use_entropy=false` 时使用同一 Q 重加权 Actor 温度，Bellman 不加熵；这是一项明确的目标消融。

## 7. 密度、熵与温度

### 7.1 密度估计必须带语义

`DensityResult` 至少返回：`log_prob_norm`、`method`、`is_exact_for_sampled_policy`、`nfe`、`inverse_residual`、`finite_mask`。日志使用 `entropy_proxy`，除非已经证明/验证该结果对应实际采样策略。

主配置使用 `continuous_inverse_proxy`。实际部署策略是单步映射，连续流路径提供训练期 log-density 代理。增加积分步数只能减少数值积分误差，不能自动消除单步策略与连续流策略的分布差异。

### 7.2 指定动作的连续逆向密度代理

令 `f_theta(z,t;c)=u_theta(z,t,t;c)`。给定由单步 Actor 得到的 `z_a`：

1. 从目标端 `z(0)=z_a` 沿递增 t 积分 `dz/dt=f_theta(z,t;c)` 到 t=1，得到源端 `epsilon_hat`。
2. 同时积分 `D = integral_0^1 div_z f_theta(z(t),t;c) dt`。
3. 计算 `logpi_z_proxy(z_a|c)=log p_src(epsilon_hat|c)+D`。
4. `logpi_norm_proxy=logpi_z_proxy-log|J_tanh(z_a)|`。

该符号方向必须用一维仿射速度场解析密度验证。不能在 t=0 动作端启动后又采用从源到目标方向的散度符号。

首版固定网格 Heun，`N_est=5`；数值参考采用更多步数和 float64。Heun 每步不止一次场评估，日志记录实际 NFE，不能把 N_est 等同于 NFE。

### 7.3 局部平均速度估计器

可选 `local_meanflow_proxy`：按区间 `[r_i,t_i]` 使用平均速度更新并估计局部离散 Jacobian 的对数行列式；它与 `u(z,t,t)` 连续积分不是同一实现。首版先在仿射映射下精确 `slogdet` 验证，再引入 trace 近似。

具体离散映射定义为 `T_i(z_t)=z_t-h_i*u(z_t,r_i,t_i;c)`，`h_i=t_i-r_i`，从源时间 1 逐段走到 0。若要查询指定动作的密度，则从该动作起逐段解 `z_t-h_i*u(z_t,r_i,t_i;c)=z_r`，恢复源端，最后计算 `log p_src(z_1|c)-sum_i log|det(I-h_i*J_z u_i)|`。首版低维参考采用完整 Jacobian、线性解/迭代反演，配置最大迭代次数、残差容差和失败处理；不能另采一条前向路径替代指定动作的逆像。

局部反演仅在满足适当可逆性条件时有唯一解；例如小步长与局部 Lipschitz 界可支持收缩求解，但实现不默认网络全局满足。若使用 `-h_i*trace(J_u)` 近似 log-determinant，应另计 trace 截断误差。该密度首先对应局部分步映射；其输出与一次端点映射不同的误差仍需诊断，`is_exact_for_sampled_policy` 默认 false。

原论文的 decoupled estimator 应在 `flame_reference` 中按锁定代码版本复现。主算法的 estimator 选择必须写入 run 标签，不能实现一个代理后笼统称“原版准确熵”。

### 7.4 散度计算

- `exact_trace`：小动作维度的正确性参考，逐维求 action Jacobian 对角和。
- `hutchinson`：默认 Rademacher 向量，估计 `v^T J v`；同一积分轨迹复用该向量，减少额外噪声。
- `autograd.grad(f,z,grad_outputs=v)` 得到 VJP，可用于 `v^T J v`；变量不能误命名为完整 JVP。
- 可配置 `n_trace_samples`，首版 1，诊断 4/8；记录和 exact_trace 的偏差与方差。
- 不使用全局 `inference_mode` 包裹需要动作梯度的 density；内部 `enable_grad` 后只对 z 建图，参数冻结，结果 detach。
- 大批次按 chunk 做 density；NaN 应触发明确事件、保存诊断并停止该更新，不能无声设成 0 继续训练。

### 7.5 两类误差分别测量

1. **积分/trace 误差**：相同连续策略下比较 N=5 与 N=50/100、Hutchinson 与精确 trace。
2. **单步/连续策略差异**：固定条件和源样本，对比一步输出与多步输出的动作差、分布距离、Q 差；多峰玩具分布上可用 histogram/KDE 作低维参考。

额外诊断：在可逆仿射单步映射下用解析离散密度核对；一般神经映射不默认全局可逆。若使用数值反演估计离散密度，必须报告反演残差、多解处理与失败率，不能直接把单个局部 Jacobian 的 determinant 当成任意映射的全局密度。

### 7.6 熵温度

命名 `alpha_ent`，不与 OFQL 的 Q 引导系数混用。第一轮主实验固定 alpha，候选值 `0.05/0.1/0.2`，所有先验类型使用相同调参预算；小规模验证后才能开启自动温度。

自动温度定义：`alpha_ent=exp(log_alpha)`，`H_hat=mean(-logpi_norm_proxy)`，

`L_alpha = log_alpha * stop_gradient(H_hat-H_target)`。

梯度下降时，熵低于目标，log_alpha 增大；熵高于目标则减小。损失使用当前在线 Actor 的动作及对应在线 estimator，不把 target Actor 的混合输出冒充当前熵。alpha 更新频率默认与 Actor 一致。

`H_target=-d_a` 仅作 SAC 风格起点，写入动作坐标，必须消融。`alpha_min/max` 用于数值保护，触边记录比例。将学习率、目标熵与先验方差的调节分别记录；关闭自动温度时仍保留 Actor 重加权所需的固定 alpha。

## 8. 在线采样与训练循环

### 8.1 完整顺序

```text
启动：解析配置 -> 校验能力/维度 -> 加载冻结先验 -> 固定随机种子
    -> 创建训练/评估环境 -> 构造 agent/buffer/logger -> 恢复或初始化

每个环境槽位：
    取得 obs/reference/controller snapshot
    构造本时刻 condition 和 SourceSpec
    热身时按配置策略采样；否则源采样+单步 flow+tanh
    可选过滤/回退 -> 执行环境步
    更新一次控制器状态 -> 构造真实 next condition
    保存 transition -> 对结束槽位 reset

每轮采样后：
    update_credit += 新增真实 transition 数 * utd_ratio
    如样本达到 learning_starts：
        while update_credit >= 1：
            固定本次 SourceSpec、normalizer snapshot
            从 buffer 采样并构建当前/下一条件
            更新 Critic
            按频率更新 Actor -> 更新 alpha
            更新 target 网络
            update_credit -= 1
    按环境 transition 数触发日志/评估/检查点
退出：关闭 writer/环境，保存状态并标记 run outcome
```

### 8.2 计数与 UTD

`env_transitions` 是所有环境实际产生的转移总数；`vector_steps` 是 vector.step 的调用次数；`gradient_steps`、`actor_updates`、`critic_updates` 独立记录。

`utd_ratio=0.2` 用浮点积分信用额实现，不能 `range(0.2)` 或截断为零。warmup 阶段不累积一笔必须在结束时补做的历史更新债务。auto-reset 产生的无动作 reset 槽位不能计入真实 transition。

### 8.3 热身

主比较默认 `warmup_source=uniform_action`，所有 flow 源类型一致，隔离 source 的作用。另设 `warmup_source=prior` 和 `mixed` 消融。

零初始化速度时策略已经接近源控制器，但 `learning_starts` 前仍不更新网络。若用先验热身，热身步数也计入 PGOFP 的交互预算；先验预训练预算单独报告。

### 8.4 终止与向量环境

MuJoCo 的外部 TimeLimit 默认 `bootstrap_mask=1-terminated`；真正失败/成功终止不 bootstrap。Gymnasium 明确区分 termination 与 truncation，不能只用 done。[时间限制处理](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

首版优先普通环境或显式管理 reset 的同步环境池，先消除自动 reset 歧义。启用 Gymnasium vector 环境后固定 autoreset 模式；adapter 根据锁定版本解释 `final_obs/final_observation` 等字段和有效掩码，最终向 learner 提供统一 `next_obs_raw`。不同版本 key 的处理集中在 adapter，算法不得判断字符串。

有限时域 HFM 若任务在第 300/500 步自然结束，应根据任务语义设真正终止，并把剩余时间纳入条件；不能把所有时长到期统一当成可 bootstrap 的外部限制。

### 8.5 归一化和随机数

- 观测 running statistics 仅由训练环境更新；评估冻结，控制器使用原始物理观测或自己的冻结 normalizer。
- learner 单次更新使用同一个 normalizer snapshot，避免 next/source/density 条件不一致。
- 主实验不缩放 reward；必要时另开明确定义的实验。
- 随机数流至少分为训练环境、行为源采样、replay 抽样、时间采样、逆向候选、trace、评估环境、评估策略和先验训练。
- 评估不能消耗训练 generator；每条 generator 状态进入 checkpoint。
- 首版 flow/JVP/density 使用 float32，不默认 AMP；优化混合精度前必须通过数值检查。

## 9. MuJoCo 环境与控制先验落地

使用 Gymnasium 的 MuJoCo v5 环境，锁定 Gymnasium、MuJoCo、环境参数和 XML 版本。各版本回报不能无条件横比，精确复现也需要一致的模拟器版本。[MuJoCo 环境与版本](https://gymnasium.farama.org/environments/mujoco/)

### 9.1 首批任务

| 环境 | 目的 | 初始先验配置 | 实施注意 |
|---|---|---|---|
| `InvertedPendulum-v5` | 最低维闭环、density 符号、控制器状态与失败终止。 | PID/PD、小模型 MPC、PPO。 | 奖励较易饱和，不能只靠最终 return 证明改进；加扰动诊断另命名。 |
| `Reacher-v5` | 多动作、目标条件与参考一致性。 | 关节 PD/PID、采样 MPC、PPO。 | 由公开目标坐标求二维 IK 参考；固定肘部支路规则，参数独立保存。 |
| `HalfCheetah-v5` | 中等维动作与样本效率。 | PPO；MPC 后续。 | 不把“每关节 PID 保持零角度”假装成有用运动先验。 |
| `Hopper-v5` | 易失败动力学与训练波动。 | PPO；有经过独立验证的步态控制器才加 PID。 | 记录摔倒与 time-limit；不能用训练期 privileged 状态偷偷帮助先验。 |

后续增加 `Walker2d-v5`、`Ant-v5`，规模足够后再考虑 Humanoid。首轮无需对每个任务凑齐所有先验类型；不支持的组合在 manifest 中明确为 skipped。

### 9.2 PID 具体实现

- 控制律包括 P/I/D、dt、导数滤波、积分限幅、输出限幅与 anti-windup。
- 首版 InvertedPendulum 可用 `u=Ktheta*theta + Komega*theta_dot + Kx*x + Kv*x_dot + Ki*I`；正负号依据实际动作方向的小扰动响应确定，禁止凭符号猜测。
- Reacher 使用公开的末端目标构造关节参考，再对关节误差闭环；关节角误差 wrap 到正确区间。
- 增益通过单独 calibration seeds 搜索，不在最后测试 seeds 上挑选。
- good/medium/poor 按独立校准回报和失败率分档，而不是仅用“某个 gain 更大”推断质量。
- controller 单独执行、controller+同等扰动、PGOFP 三者都要测，以区分随机化与 flow 改进。

### 9.3 MPC 具体实现

首版选采样 MPC（CEM），在 InvertedPendulum/Reacher 上运行：

1. 构造独立 planning simulator 或显式动力学模型，不在真实环境对象上试走候选。
2. 优化短时域归一化动作序列；代价由公开观测可计算的角度/位置误差、速度、动作范数组成。
3. CEM 采样序列、筛 elite、更新均值方差，输出首个动作；保持动作约束。
4. 固定迭代数、population、elite fraction、horizon、seed 与 timeout；首版关闭跨时刻 warm start。
5. 支持 `model_access=observation_model` 与 `privileged_sim_state` 两种能力声明。后者若读取隐藏状态只可作为标记清楚的 oracle 上界，不能与普通观测方法当成公平同预算比较。
6. observation-only 模式无法重建完整 MuJoCo 状态时应报不支持；使用公开观测辨识的小模型或单独标记 oracle，不能静默访问 `env.unwrapped.data`。
7. 对 MPC 输出构造 source 时，默认使用固定 latent 方差；如使用 elite 方差，应明确那是构造出的第一步动作分布而非完整最优控制后验。

MPC planning 消耗的模型 rollout、模型训练数据、求解时间另计。用缓存加速 learner，但真实部署路径的先验时延必须实测。

### 9.4 不修改标准任务来制造结果

主结果保留原始 observation、reward、termination 与目标随机化分布。目标跟踪误差、动作平滑度等作为诊断记录，不暗中加入训练 reward。若增加扰动、目标切换、参数偏移或跟踪任务，必须另设任务配置和名称，原标准任务结果同时保留。

## 10. PPO 先验的训练与导出

### 10.1 两种明确入口

- `native`：工程内最小 tanh-Gaussian PPO，实现 GAE、clipped surrogate、value loss、Gaussian entropy proxy/采样熵选项、minibatch epochs、梯度裁剪和优势归一化；支持直接导出 pre-tanh 分布。
- `external`：加载用户已有或第三方训练 checkpoint，必须通过适配器声明观测、动作、归一化和分布语义。不能仅凭层大小匹配就加载。

优先 native 路线保证分布语义清楚；PPO 是先验制备和经典比较方法，不能在 PGOFP learner 内继续 PPO 更新。

### 10.2 质量与预算

- 每任务保存多个预定义训练阶段的 checkpoint，再在独立 calibration seeds 上分档。
- 不使用 PGOFP 最终测试回报来挑选“最合适先验”。
- 固定并记录 `prior_seed`；flow 的多个 seed 可以共享冻结先验进行主比较，再用多个 prior seeds 做稳健性补充实验。
- 分开报告 `PGOFP_online_steps`、`prior_pretrain_steps`、`total_steps`、`prior_gpu_hours`。
- `PPO_prior_only` 与 `PPO+继续训练` 都是有价值的比较，后者按相同新增交互预算。

### 10.3 `PriorArtifact` 内容

```text
weights.pt                # PPO actor，或传统控制器的参数文件
manifest.json             # schema/version/checksum/env/task/action chart
normalizer.npz            # 冻结 PPO 的原观测统计量
controller_config.yaml    # 网络结构、增益、MPC model hash、输入特征
calibration_metrics.json  # 独立校准表现与质量档位依据
budget.json               # 训练/辨识/校准预算，来源 seed
```

每个字段有 schema 校验；有状态 PPO/RNN 的 checkpoint 第一版拒绝，需要扩展隐状态历史协议后才能支持。先验 artifact 可放本地或 WandB Artifact，运行 manifest 必须记录不可变版本和内容 hash。

## 11. 后续聚变模拟器接口

### 11.1 可提前定义的接口

`FusionBackend` 协议包含：

- `describe()`：观测字段、单位、动作通道、范围、控制周期、能力声明。
- `reset(shot_id, reset_params, seed)`：返回公开观测及参考定位信息。
- `advance(voltage_command)`：返回观测、奖励分项、结束原因、实际执行器量和诊断。
- `get_reference(time_index, horizon)`：只返回部署时允许取得的参考。
- `get_snapshot/set_snapshot`：可选，若不支持则 checkpoint 明确只能从新回合恢复。

环境、控制器、算法各自只接触其协议，不直接调用私有 HFM API。真实 SDK 导入延迟到创建 backend 时；没有 SDK 也必须能运行全部 MuJoCo 和 mock 测试。

### 11.2 依据报告的暂定规格

| 项目 | 暂定值 | 接入动作 |
|---|---|---|
| 观测 | 64 维 | 用字段名、单位、缺失掩码、顺序验证；参考和历史加入后网络维度通常大于 64。 |
| 策略动作 | 7 维归一化指令 | 保留为算法动作空间。 |
| 装置动作 | 12 维线圈电压 | 使用用户确认的 `M ∈ R^(12×7)`、通道顺序与量程。 |
| 周期 | 10 ms / 100 Hz | 测端到端 p50/p95/p99 与 deadline miss rate。 |
| 执行器 | 延迟、变化率限制、仿射标定 | SDK 已提供时不得 wrapper 再执行一次；通过 capability 声明决定谁负责。 |
| F1/F2 | 报告暂记 300/500 步 | 由 backend/任务 spec 最终确认，明确 terminal 语义。 |
| shot | 暂记 4 组 | 配置化 train/calibration/test 划分和 reset 扰动。 |

报告中“7 个分量的线圈组解释”与 12 通道展开需要实际矩阵核对。不得凭文字自行编造矩阵；`coil_map.yaml` 在真实值未提供时保持 `verified:false`，真实运行必须拒绝未确认配置。

### 11.3 观测与参考

- `ObservationSchema` 声明 Ip、位置、轮廓、X 点、磁通等字段，避免靠硬编码索引猜物理含义。
- 参考轨迹需包含 `shot_id/task_id/reference_hash`、时间轴、可用前瞻和单位。
- 对接缺测值提供 mask；不要把物理缺测直接填成“正常的零”而不标识。
- POMDP 首版使用有限历史、上一指令和执行器公开状态；有实时状态估计器时作为独立可替换模块。
- 缓冲存当前与下一参考；禁止对旧样本读“现在的参考时刻”。

### 11.4 动作与安全适配

明确区分：7 维策略动作、12 维电压设定、延迟后电压、限速电压、实际施加电压。奖励与转移按真实 backend 结果，Critic 默认以发出的 7 维指令为动作。

可选安全层返回 `raw_command/filtered_command/intervention/reason`。主算法密度描述过滤前策略，过滤导致的 pushforward 密度一般未知；启用过滤后日志写 `density_before_filter`，并另做过滤开关比较。安全过滤不是已证明的受约束策略最优解。

后续如需严格 CMDP，可引入 cost、cost critic 和约束优化模块；本文首阶段只预留 cost 字段，不把奖励惩罚包装成约束保证。

## 12. 完整目录结构

下列为未来实现目录；本轮只增加本文件。所有列出的文件均在第 13 节分配职责。`__init__.py` 统一只导出公共对象，不执行环境创建、联网、WandB 初始化或注册外部 SDK。

```text
PGOFP/
├── Online-FLAME设计.md                       # 已有，保留
├── Online-OFQL设计.md                        # 已有，保留
├── 开题报告.md                              # 已有，保留
├── PGOFP实现方案.md                         # 本文
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .env.example
├── docs/
│   ├── algorithm_contract.md
│   ├── source_prior_contract.md
│   ├── environment_contract.md
│   ├── experiment_protocol.md
│   ├── fusion_integration.md
│   └── reproduction_notes.md
├── configs/
│   ├── train.yaml
│   ├── prior_train.yaml
│   ├── evaluate.yaml
│   ├── algorithm/
│   │   ├── prior_flame_m.yaml
│   │   ├── gaussian_flame_m.yaml
│   │   ├── flame_reference.yaml
│   │   ├── online_ofql.yaml
│   │   ├── sac.yaml
│   │   ├── td3.yaml
│   │   ├── ppo.yaml
│   │   └── controller_only.yaml
│   ├── env/
│   │   ├── inverted_pendulum.yaml
│   │   ├── reacher.yaml
│   │   ├── half_cheetah.yaml
│   │   ├── hopper.yaml
│   │   ├── walker2d.yaml
│   │   └── fusion_stub.yaml
│   ├── prior/
│   │   ├── gaussian.yaml
│   │   ├── pid.yaml
│   │   ├── mpc.yaml
│   │   ├── ppo.yaml
│   │   └── multi_controller.yaml
│   ├── source/
│   │   ├── constant.yaml
│   │   └── annealed.yaml
│   ├── logging/
│   │   ├── local.yaml
│   │   ├── wandb_online.yaml
│   │   └── wandb_offline.yaml
│   ├── experiment/
│   │   ├── smoke.yaml
│   │   ├── mujoco_main.yaml
│   │   ├── source_type.yaml
│   │   ├── prior_ratio.yaml
│   │   ├── prior_quality.yaml
│   │   ├── schedule_time.yaml
│   │   ├── flame_components.yaml
│   │   └── robustness.yaml
│   ├── controller/
│   │   ├── inverted_pendulum_pid.yaml
│   │   ├── reacher_pid.yaml
│   │   └── cem_mpc.yaml
│   └── fusion/
│       ├── observation_schema.yaml
│       ├── coil_map.yaml
│       └── tasks.yaml
├── src/pgofp/
│   ├── __init__.py
│   ├── cli.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   ├── protocols.py
│   │   ├── specs.py
│   │   ├── registry.py
│   │   └── exceptions.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── schema.py
│   │   ├── compose.py
│   │   └── validate.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── factory.py
│   │   ├── adapters.py
│   │   ├── wrappers.py
│   │   ├── vector.py
│   │   ├── mujoco/
│   │   │   ├── __init__.py
│   │   │   ├── tasks.py
│   │   │   ├── features.py
│   │   │   └── planning.py
│   │   └── fusion/
│   │       ├── __init__.py
│   │       ├── protocol.py
│   │       ├── adapter.py
│   │       ├── observation.py
│   │       ├── reference.py
│   │       ├── actuator.py
│   │       ├── safety.py
│   │       └── mock_backend.py
│   ├── controllers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── manager.py
│   │   ├── pid.py
│   │   ├── mpc.py
│   │   ├── ppo.py
│   │   └── artifacts.py
│   ├── priors/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── gaussian.py
│   │   ├── controller_source.py
│   │   ├── native_policy_source.py
│   │   ├── mixture.py
│   │   ├── particle_source.py
│   │   ├── schedules.py
│   │   └── cache.py
│   ├── networks/
│   │   ├── __init__.py
│   │   ├── mlp.py
│   │   ├── embeddings.py
│   │   ├── conditioning.py
│   │   ├── mean_flow.py
│   │   ├── critics.py
│   │   └── gaussian_policy.py
│   ├── math/
│   │   ├── __init__.py
│   │   ├── action_transform.py
│   │   ├── time_sampling.py
│   │   ├── reverse_sampling.py
│   │   ├── importance.py
│   │   ├── meanflow_target.py
│   │   ├── divergence.py
│   │   └── integration.py
│   ├── density/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── continuous_proxy.py
│   │   ├── local_meanflow_proxy.py
│   │   └── diagnostics.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── prior_flame.py
│   │   ├── objectives.py
│   │   ├── temperature.py
│   │   ├── flame_reference.py
│   │   ├── online_ofql.py
│   │   └── baselines.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── replay_buffer.py
│   │   ├── rollout_buffer.py
│   │   ├── batch.py
│   │   └── trajectory.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── runner.py
│   │   ├── collector.py
│   │   ├── update_scheduler.py
│   │   ├── prior_trainer.py
│   │   ├── checkpoint.py
│   │   └── hooks.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── matrix.py
│   │   ├── launcher.py
│   │   ├── manifest.py
│   │   └── aggregate.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluator.py
│   │   ├── metrics.py
│   │   ├── distributions.py
│   │   ├── latency.py
│   │   └── robustness.py
│   ├── logging/
│   │   ├── __init__.py
│   │   ├── logger.py
│   │   ├── jsonl.py
│   │   ├── tensorboard.py
│   │   ├── wandb.py
│   │   └── artifacts.py
│   └── utils/
│       ├── __init__.py
│       ├── seed.py
│       ├── normalization.py
│       ├── torch_utils.py
│       ├── timing.py
│       └── provenance.py
├── scripts/
│   ├── train.py
│   ├── train_prior.py
│   ├── evaluate.py
│   ├── launch_experiments.py
│   ├── calibrate_controller.py
│   ├── analyze_results.py
│   ├── benchmark_latency.py
│   ├── inspect_policy.py
│   ├── export_policy.py
│   └── validate_install.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_action_transform.py
│   │   ├── test_sources.py
│   │   ├── test_controller_state.py
│   │   ├── test_reverse_sampling.py
│   │   ├── test_meanflow_target.py
│   │   ├── test_density.py
│   │   ├── test_temperature.py
│   │   ├── test_replay.py
│   │   └── test_update_scheduler.py
│   ├── integration/
│   │   ├── test_training_smoke.py
│   │   ├── test_vector_terminal.py
│   │   ├── test_resume.py
│   │   ├── test_prior_artifact.py
│   │   ├── test_logging.py
│   │   └── test_fusion_contract.py
│   └── numerical/
│       ├── test_weighted_transport.py
│       ├── test_density_convergence.py
│       └── test_one_step_gap.py
├── assets/
│   └── README.md
├── artifacts/                                # 运行生成，不提交大权重
│   └── priors/<prior_id>/                     # 第 10.3 节定义文件
└── outputs/                                  # 运行生成，不提交
    ├── experiments/<matrix_id>/               # 第 16 节定义文件
    └── runs/<run_id>/                         # 第 16 节定义文件
```

## 13. 逐文件实现职责

下表路径相对于项目根目录。包内 `__init__.py` 的职责统一见第 12 节；根目录三个既有文档保持原文。未来目录不得只创建空文件宣称完成：每项交付必须有对应行为、调用方和验收证据。

### 13.1 根目录与说明文档

| 文件 | 职责、内容与验收 |
|---|---|
| `README.md` | 最短安装与训练路线、先验制备顺序、主算法定义、实际支持的任务、结果目录；标出尚未实现的功能，不贴虚构曲线。 |
| `pyproject.toml` | 包名、Python 支持范围、console entrypoints、依赖 extra（mujoco、wandb、dev、reference）、pytest/ruff 配置；基础安装不强制导入 HFM。 |
| `uv.lock` | 在选定机器上实际解析产生的依赖锁；不可手写不存在的包版本；包含 PyTorch 平台差异说明。 |
| `.gitignore` | outputs、wandb 缓存、权重、大 buffer、密钥、虚拟环境、`._*`；不忽略小型实验 manifest 和 schema。 |
| `.env.example` | WandB entity/project/mode 和本地 artifact 路径示例；API key 只留空占位，不放真实凭据。 |
| `docs/algorithm_contract.md` | 从本文拆分的最终公式、坐标、梯度边界、估计器语义；随实现变化维护，不能只写概述。 |
| `docs/source_prior_contract.md` | ControllerPrediction、SourceSpec、源分布坐标、状态快照与混合密度规则。 |
| `docs/environment_contract.md` | EnvSpec、终止语义、vector autoreset 兼容、公开/特权观测与动作命名。 |
| `docs/experiment_protocol.md` | 数据划分、预算、seed、消融矩阵、统计和选 checkpoint 规则。 |
| `docs/fusion_integration.md` | 真实 SDK 所需方法映射、单位检查、接入步骤与待确认列表；不得杜撰私有接口。 |
| `docs/reproduction_notes.md` | 固定 FLAME upstream commit、许可、算法/依赖差异和每项复现偏差；没有复现时明确标未完成。 |
| `assets/README.md` | 说明参考轨迹/校准小数据的下载或放置方式、许可、hash、字段；不提交 HFM 私有大文件。 |

### 13.2 配置文件

| 文件 | 每个文件的具体职责 |
|---|---|
| `configs/train.yaml` | 训练总入口、defaults 组合、seed、device、budget、checkpoint；所有组有唯一来源。 |
| `configs/prior_train.yaml` | PPO 先验训练入口与校准/导出路径，区别 prior steps 和 flow steps。 |
| `configs/evaluate.yaml` | checkpoint 路径、冻结 source 参数、eval seeds、episodes、视频和分布检查开关。 |
| `configs/algorithm/prior_flame_m.yaml` | MeanFlow、reverse objective、K_train、target_tau、熵代理、温度和学习率。 |
| `configs/algorithm/gaussian_flame_m.yaml` | 继承主算法并强制 gaussian source；不额外改网络、训练预算或候选数。 |
| `configs/algorithm/flame_reference.yaml` | upstream 版本和 reference adapter 参数；未配置 commit/依赖时拒绝启动。 |
| `configs/algorithm/online_ofql.yaml` | FBC 与可微 Q 权重、flow_ratio、额外探索噪声；不复用 alpha_ent 字段。 |
| `configs/algorithm/sac.yaml` | 双 Q、tanh Gaussian、自动温度配置、与主算法一致的环境预算。 |
| `configs/algorithm/td3.yaml` | 确定性 Actor、target smoothing、policy delay；第二阶段比较。 |
| `configs/algorithm/ppo.yaml` | rollout 长度、GAE、clip ratio、epochs、value/entropy loss；同时支持 PPO 比较运行。 |
| `configs/algorithm/controller_only.yaml` | 直接 controller/source 策略，无 learner 更新；区分 deterministic controller 与 noisy source-only。 |
| `configs/env/inverted_pendulum.yaml` | 精确环境 ID、make kwargs、观测特征与可用 PID/MPC 控制参考。 |
| `configs/env/reacher.yaml` | 原任务目标字段与 IK/角度 feature 配置；不改变标准 reward。 |
| `configs/env/half_cheetah.yaml` | 原任务 kwargs、仅启用已支持先验，维度从运行时 EnvSpec 检查。 |
| `configs/env/hopper.yaml` | 原任务健康终止、time-limit、公开观测规则。 |
| `configs/env/walker2d.yaml` | 后续扩展预设；未通过 smoke 前不纳入默认 main matrix。 |
| `configs/env/fusion_stub.yaml` | 选择 mock backend、公开 observation schema 和虚拟参考；不能冒充真实 HFM。 |
| `configs/prior/gaussian.yaml` | base sigma、坐标与可查询密度能力，无 controller artifact。 |
| `configs/prior/pid.yaml` | controller config 引用、latent perturbation sigma、fallback 和版本。 |
| `configs/prior/mpc.yaml` | MPC controller、model access、缓存、方差来源、超时和 fallback。 |
| `configs/prior/ppo.yaml` | artifact hash、native/center 分布语义、冻结 normalizer。 |
| `configs/prior/multi_controller.yaml` | component 列表、权重、质量档位、所有 component 的坐标一致性。 |
| `configs/source/constant.yaml` | 固定 beta/sigma、base sigma、sigma_min/max；主实验默认。 |
| `configs/source/annealed.yaml` | schedule 类型、起止值、总步数、最大变化；不改 controller checkpoint。 |
| `configs/logging/local.yaml` | JSONL/TB 开关、flush 频率、磁盘目录、禁用外部服务。 |
| `configs/logging/wandb_online.yaml` | online 模式、project/entity、TB 同步、artifact 策略；凭据来自环境。 |
| `configs/logging/wandb_offline.yaml` | offline 模式、后续同步说明，不要求网络可用。 |
| `configs/experiment/smoke.yaml` | 极短训练预算、小网络/批次；只验证链路，标签禁止进入正式统计。 |
| `configs/experiment/mujoco_main.yaml` | 首批任务×主算法×先验×5 seeds 的合法组合，写明不支持组合。 |
| `configs/experiment/source_type.yaml` | 同预算替换 Gaussian/PID/MPC/PPO，其他参数固定。 |
| `configs/experiment/prior_ratio.yaml` | beta=0/0.25/0.5/0.75/1 扫描，固定先验质量与 sigma。 |
| `configs/experiment/prior_quality.yaml` | 校准档位、多个 artifact 或 gains；先验独立校准指标随结果导出。 |
| `configs/experiment/schedule_time.yaml` | 固定/退火、sigma、时间采样与 flow_ratio 的分阶段扫描，避免无预算全笛卡尔积。 |
| `configs/experiment/flame_components.yaml` | 熵、Q 重加权、density steps、K_train、K_exec、源替换位置等诊断。 |
| `configs/experiment/robustness.yaml` | 扰动、模型偏差、目标变化、多个 prior seed；不覆盖标准任务名。 |
| `configs/controller/inverted_pendulum_pid.yaml` | 公开观测字段、带单位增益、dt、限幅、积分和滤波。 |
| `configs/controller/reacher_pid.yaml` | 目标到关节参考规则、IK 支路、各关节增益和力矩范围。 |
| `configs/controller/cem_mpc.yaml` | horizon、population、elite、iterations、cost、planning dt、model access。 |
| `configs/fusion/observation_schema.yaml` | 暂定字段规范与 verified 标志；真实 64 维字段顺序未获得时不填虚构索引。 |
| `configs/fusion/coil_map.yaml` | 7→12 映射、通道顺序、单位和限值；真实矩阵由设备资料填写且 hash 固定。 |
| `configs/fusion/tasks.yaml` | F1/F2 时域、参考轨迹和 shot 划分；待 SDK 核实字段标记 pending。 |

### 13.3 核心类型与配置解析

| 文件 | 职责与关键对象 |
|---|---|
| `src/pgofp/cli.py` | 统一 CLI 子命令路由，返回明确退出码；不包含训练逻辑。 |
| `src/pgofp/core/types.py` | PolicyCondition、ControllerPrediction、SourceSample、ReplayTransition、ReplayBatch、ActionResult、DensityResult、MetricRecord。 |
| `src/pgofp/core/protocols.py` | EnvAdapter、Controller、SourceDistribution、Agent、DensityEstimator、Logger 的 Protocol；实现之间以此解耦。 |
| `src/pgofp/core/specs.py` | EnvSpec、ActionSpec、SourceSpec、CapabilitySpec、ArtifactSpec；有 schema_version 与不可变 hash。 |
| `src/pgofp/core/registry.py` | 显式注册 task/controller/source/agent 构造器；未知名称报错，不以字符串 eval 动态加载。 |
| `src/pgofp/core/exceptions.py` | 配置不兼容、先验失效、数值失效、恢复不兼容、backend 不可用等明确异常。 |
| `src/pgofp/config/schema.py` | 结构化配置 dataclass，限制字段类型/可选项，保存默认值与解释。 |
| `src/pgofp/config/compose.py` | Hydra/OmegaConf 组合、命令行覆盖、保存 fully resolved yaml；不隐式修改算法参数。 |
| `src/pgofp/config/validate.py` | 在创建训练环境前检查组合、artifact、source 密度能力、动作坐标、退火条件输入、HFM verified 状态。 |

### 13.4 环境模块

| 文件 | 职责与禁止事项 |
|---|---|
| `src/pgofp/envs/factory.py` | 根据 config 创建独立 train/eval/planning 环境及 wrappers；显式传 seed/render_mode。 |
| `src/pgofp/envs/adapters.py` | Gymnasium reset/step 转标准结果、terminated/truncated/final observation 兼容；不改变 reward。 |
| `src/pgofp/envs/wrappers.py` | episode 统计、动作 affine 变换、可选 history/reference wrapper；每个 wrapper 声明次序。 |
| `src/pgofp/envs/vector.py` | 同步环境池和向量 autoreset 适配、槽位 reset mask、真实转移计数；CPU 环境不直接接 GPU 张量。 |
| `src/pgofp/envs/mujoco/tasks.py` | 支持任务表、make kwargs、控制周期、可用 controller 与 observation schema。 |
| `src/pgofp/envs/mujoco/features.py` | 从公开 obs 获取角度/速度/目标，Reacher IK 特征、PID error；无隐藏状态默认访问。 |
| `src/pgofp/envs/mujoco/planning.py` | 独立 planning model/环境快照 adapter，区分 observation model 与 oracle；保证 planning 不修改在线环境。 |
| `src/pgofp/envs/fusion/protocol.py` | FusionBackend 与 Observation/Reference/Actuator schema，SDK 能力声明。 |
| `src/pgofp/envs/fusion/adapter.py` | Gymnasium 风格聚变环境、SDK 延迟导入、奖励分项和 terminal 映射；真实类名待 SDK 提供。 |
| `src/pgofp/envs/fusion/observation.py` | 按字段解析和拼接、单位检查、缺失 mask、公开状态与参考隔离。 |
| `src/pgofp/envs/fusion/reference.py` | 参考文件加载、hash、时间索引、前瞻权限、当前/下一参考查询。 |
| `src/pgofp/envs/fusion/actuator.py` | 7→12 展开、单位/范围、执行器 ownership；禁止重复 delay/rate limit。 |
| `src/pgofp/envs/fusion/safety.py` | 可选过滤器与回退，返回 intervention 数据；不隐藏策略实际动作被修改。 |
| `src/pgofp/envs/fusion/mock_backend.py` | 无 SDK 的确定性小状态空间替身，模拟参考、延迟、超限与结束；只用于接口测试。 |

### 13.5 控制器与源分布模块

| 文件 | 职责与关键接口 |
|---|---|
| `src/pgofp/controllers/base.py` | 控制器基础类、纯 predict 与 state advance 约定、reset mask、耗时/能力声明。 |
| `src/pgofp/controllers/manager.py` | 每环境控制器状态、当前/下一预测顺序、纯批量查询、失败回退；训练/评估实例完全分离。 |
| `src/pgofp/controllers/pid.py` | 向量 PID、角度 wrap、导数滤波、积分饱和与 anti-windup；参数来自配置，不硬编码 task index。 |
| `src/pgofp/controllers/mpc.py` | CEM planner、cost adapter、固定查询随机性、超时/无解状态；输出动作或第一步分布参数。 |
| `src/pgofp/controllers/ppo.py` | 只加载/查询冻结 PPO，原 normalizer、pre-tanh 参数与外部 adapter；不调用 optimizer。 |
| `src/pgofp/controllers/artifacts.py` | PriorArtifact schema、保存/加载/checksum、环境和动作兼容检查；支持本地不可变路径。 |
| `src/pgofp/priors/base.py` | SourceDistribution 接口、条件广播、sample metadata、连续密度能力；sample 和 rsample 能力分开。 |
| `src/pgofp/priors/gaussian.py` | 无信息 Gaussian、对角 Gaussian，显式 generator 与 sigma 检查。 |
| `src/pgofp/priors/controller_source.py` | 确定性动作到 latent Gaussian 的构造、坐标变换、边界诊断。 |
| `src/pgofp/priors/native_policy_source.py` | 冻结 tanh-Gaussian PPO 的 pre-tanh 分布，保持原 log_std 和归一化语义。 |
| `src/pgofp/priors/mixture.py` | Bernoulli/多分量采样、完整 logsumexp 密度、beta 端点、逐样本 component id。 |
| `src/pgofp/priors/particle_source.py` | MPPI/CEM 粒子转正方差核混合，拒绝 Dirac 源进入连续 log-density 路径。 |
| `src/pgofp/priors/schedules.py` | beta/sigma 调度和冻结 SourceSpec、时钟与最大变化约束；可保存/恢复状态。 |
| `src/pgofp/priors/cache.py` | 只缓存冻结 controller 的分布参数，key 含原输入/参考/状态/version hash；不长期缓存随机 epsilon。 |

### 13.6 网络、数学工具与密度模块

| 文件 | 职责与验收重点 |
|---|---|
| `src/pgofp/networks/mlp.py` | 共享 MLP block、初始化、激活、LayerNorm；不混入算法损失。 |
| `src/pgofp/networks/embeddings.py` | r/t 的正弦编码及可选环境相位编码；区分流时间与控制时间。 |
| `src/pgofp/networks/conditioning.py` | 按 schema 组装 condition tensor，当前/下一状态同序，处理可选 history/先验特征。 |
| `src/pgofp/networks/mean_flow.py` | 平均速度网络和 one-step policy wrapper；速度输出零初始化，支持对角场调用。 |
| `src/pgofp/networks/critics.py` | 独立双 Q，可批量评价 `[B,K]` 候选；动作输入固定 normalized command。 |
| `src/pgofp/networks/gaussian_policy.py` | PPO/SAC 的 tanh-Gaussian、value head 和 TD3 可选确定性 head，稳定 log_prob。 |
| `src/pgofp/math/action_transform.py` | tanh/inverse/affine/Jacobian，统一 dtype、单位、边界和形状。 |
| `src/pgofp/math/time_sampling.py` | uniform/logit-normal 时间对、flow_ratio、端点界；每次返回 sampler metadata。 |
| `src/pgofp/math/reverse_sampling.py` | 6.2 条件逆向核，返回 z_t/epsilon/z_a/v；源类型无关，不内部写 randn。 |
| `src/pgofp/math/importance.py` | logsumexp、候选 softmax、ESS、max_weight、finite 检查；禁止跨 B 归一化。 |
| `src/pgofp/math/meanflow_target.py` | 单一 MeanFlow JVP 公式、target detach；可用有限差分检查，不能复制到多处。 |
| `src/pgofp/math/divergence.py` | exact trace、Hutchinson VJP/JVP、显式 trace generator、动作梯度隔离。 |
| `src/pgofp/math/integration.py` | 固定步长 Euler/Heun、增广状态、NFE 统计和积分方向；自适应 solver 后续可插拔。 |
| `src/pgofp/density/base.py` | DensityEstimator Protocol、结果语义、exact/proxy 能力标签。 |
| `src/pgofp/density/continuous_proxy.py` | 7.2 从指定动作到源端的积分和完整条件源密度，计算 tanh 修正。 |
| `src/pgofp/density/local_meanflow_proxy.py` | 局部平均速度与离散 Jacobian proxy；先实现仿射精确参考，再加 trace 近似。 |
| `src/pgofp/density/diagnostics.py` | 一步/多步 gap、inverse residual、解析密度对照；不参与 Actor 梯度。 |

### 13.7 算法与数据模块

| 文件 | 职责与调用关系 |
|---|---|
| `src/pgofp/algorithms/base.py` | Agent 与 UpdateResult 基类、train/eval mode 和 state dict 规范。 |
| `src/pgofp/algorithms/factory.py` | 构造算法、source、estimator、optimizers；唯一依赖组合入口。 |
| `src/pgofp/algorithms/prior_flame.py` | 主 agent 的 act/update 顺序、参数冻结、EMA；Gaussian 版复用同一类。 |
| `src/pgofp/algorithms/objectives.py` | soft critic loss、逆向 QRMF、端点校验 objective 和无 Q 消融；返回细分标量。 |
| `src/pgofp/algorithms/temperature.py` | 固定/自动 alpha、target entropy、上下限、正确方向与 optimizer 状态。 |
| `src/pgofp/algorithms/flame_reference.py` | 固定 upstream 版本 wrapper，协议与日志适配；不宣称当前 main 自动等于论文版本。 |
| `src/pgofp/algorithms/online_ofql.py` | 按本地 OFQL 文档实现对照，FBC 与可微 Q 指导、单独 eta/q_scale。 |
| `src/pgofp/algorithms/baselines.py` | SAC/TD3/PPO/controller-only 的统一 adapter；PPO 训练核心复用 prior_trainer，避免两套实现。 |
| `src/pgofp/data/replay_buffer.py` | 环形 CPU 存储、均匀采样、schema 校验、容量统计、可选分片保存；拒绝存 GPU graph。 |
| `src/pgofp/data/rollout_buffer.py` | PPO on-policy rollout、GAE、截断 bootstrap；不与 off-policy replay 混用。 |
| `src/pgofp/data/batch.py` | transition collate、GPU 转移、condition 构造、当前/下一先验重建。 |
| `src/pgofp/data/trajectory.py` | 固定评估轨迹和分布快照 NPZ、字段 schema、episode 切分；不把视频帧写进 replay。 |

### 13.8 训练、实验、评估与日志

| 文件 | 职责与交付 |
|---|---|
| `src/pgofp/training/runner.py` | 8.1 的训练生命周期、预算、信号退出、hook 调度；不实现数值损失。 |
| `src/pgofp/training/collector.py` | 行为动作、controller advance、env step、真实 next condition、终止前样本存储。 |
| `src/pgofp/training/update_scheduler.py` | 实际 transitions 驱动的 fractional UTD、频率触发和恢复；避免 vector env 预算放大。 |
| `src/pgofp/training/prior_trainer.py` | native PPO、GAE/mini-batch updates、prior 校准/导出和预算；适配 classic PPO 比较。 |
| `src/pgofp/training/checkpoint.py` | 原子保存、resume 配置校验、完整状态、artifact hash 与部署导出。 |
| `src/pgofp/training/hooks.py` | 日志/评估/视频/保存的触发器；异常处理不掩盖训练数值失败。 |
| `src/pgofp/experiments/matrix.py` | 展开 task×method×seed×ablation，过滤不支持组合并显式写 skipped 原因。 |
| `src/pgofp/experiments/launcher.py` | 独立进程启动、并发/GPU 分配、状态机、重试和退出码；一个 run 不自动创建 Codex chat。 |
| `src/pgofp/experiments/manifest.py` | 稳定 config hash、run id、prior lineage、预算、开始/结束/失败状态。 |
| `src/pgofp/experiments/aggregate.py` | 本地结果汇总、seed-level CI、AUC、阈值样本数、缺失/失败 run 清单。 |
| `src/pgofp/evaluation/evaluator.py` | 独立 eval env/controller/RNG、冻结归一化与 source spec、checkpoint 评估。 |
| `src/pgofp/evaluation/metrics.py` | 回报、轨迹长度、失败率、跟踪误差、平滑度和 action saturation；无定义指标写 N/A。 |
| `src/pgofp/evaluation/distributions.py` | 固定 condition bank、多样本采集、PCA/投影/KDE、与 source 的位移和多模态诊断。 |
| `src/pgofp/evaluation/latency.py` | 先验/flow/后处理/端到端计时、warmup、CPU/CUDA 同步、p99 与 deadline miss。 |
| `src/pgofp/evaluation/robustness.py` | 预定义扰动和 held-out task/shot 参数；不在测试集上重调 controller。 |
| `src/pgofp/logging/logger.py` | MetricRecord 多后端扇出、主轴、去重、flush；训练代码只依赖它。 |
| `src/pgofp/logging/jsonl.py` | append-only metrics/events、每条完整带 step；本地原始审计数据。 |
| `src/pgofp/logging/tensorboard.py` | 唯一标量写入 SummaryWriter、直方图/图像频率、resume step 处理。 |
| `src/pgofp/logging/wandb.py` | online/offline/disabled 初始化、TB sync、run id/group/tags、结果 tables；不双写同一 scalar。 |
| `src/pgofp/logging/artifacts.py` | 产物引用、版本/checksum、上传 best/final/报告；大 replay 默认仅本地。 |
| `src/pgofp/utils/seed.py` | seed 派生和 RNG generator/state 序列化；不依赖调用顺序生成 eval seed。 |
| `src/pgofp/utils/normalization.py` | running mean/var、冻结快照、shape 及 version，PPO 先验统计单独保存。 |
| `src/pgofp/utils/torch_utils.py` | 参数冻结、EMA、梯度 norm、device 选择；不隐式 `.data` 修改计算图。 |
| `src/pgofp/utils/timing.py` | 性能计时与阶段耗时，不用异步 GPU launch 时间冒充真实延迟。 |
| `src/pgofp/utils/provenance.py` | code/config/dependency/XML/artifact hash、硬件、启动参数与 UTC 时间；用户展示可另转本地时间。 |

### 13.9 脚本入口

所有脚本只解析参数并调用包内函数，不复制算法。运行例子见第 14 节。

| 文件 | 职责 |
|---|---|
| `scripts/train.py` | 单个 flow/baseline run 的启动与 resume。 |
| `scripts/train_prior.py` | 训练 PPO prior，校准分档并导出 PriorArtifact。 |
| `scripts/evaluate.py` | 对给定 checkpoint 在固定 seeds 上评估并导出逐回合数据。 |
| `scripts/launch_experiments.py` | dry-run manifest、执行/继续矩阵、限制并发、列出失败。 |
| `scripts/calibrate_controller.py` | PID/MPC 增益/时域在 calibration split 上搜索，记录每次预算和结果。 |
| `scripts/analyze_results.py` | 聚合矩阵，输出 CSV、PDF/PNG 曲线和失败清单。 |
| `scripts/benchmark_latency.py` | 指定硬件与 batch=1 测全链路延迟；区分 cached/真实先验路径。 |
| `scripts/inspect_policy.py` | 固定状态动作采样、source/target 投影、Q/位移/Jacobian/density gap 诊断。 |
| `scripts/export_policy.py` | 推理包、source/normalizer/controller 元数据与最小 smoke；不能只导出 flow 权重。 |
| `scripts/validate_install.py` | 检查 Python/PyTorch/device/JVP、MuJoCo reset/step、可选渲染与本地 writer；WandB 网络非必需。 |

### 13.10 测试文件

| 文件 | 必须验证的行为 |
|---|---|
| `tests/conftest.py` | 确定性 toy env、仿射/零速度网络、小型 prior artifact、临时目录和固定 RNG fixtures。 |
| `tests/unit/test_config.py` | 错误组合拒绝、beta/sigma 范围、坐标/维度/版本不兼容、禁用未知参数。 |
| `tests/unit/test_action_transform.py` | tanh/affine 往返、稳定 Jacobian、极端值、单位和批次形状。 |
| `tests/unit/test_sources.py` | Gaussian moments、混合比例、beta 端点、完整混合密度、positive sigma、generator 可复现。 |
| `tests/unit/test_controller_state.py` | predict 不改状态、advance 一次、mask reset、回放重算不污染在线状态、失败分布一致。 |
| `tests/unit/test_reverse_sampling.py` | 第 6.2 核关系、非零均值/异方差/混合源、反推重建路径和向量形状。 |
| `tests/unit/test_meanflow_target.py` | JVP 对有限差分、r 固定的导数、r=t 退化、detach 后无不应有梯度。 |
| `tests/unit/test_density.py` | 恒等和仿射场解析密度、散度方向、指定动作依赖、混合源、tanh 修正、VJP trace。 |
| `tests/unit/test_temperature.py` | 熵不足 alpha 上升、熵过高下降、detach 和上/下限事件。 |
| `tests/unit/test_replay.py` | 环形覆盖、真实 next obs、controller/reference 配对、抽样/device/shape。 |
| `tests/unit/test_update_scheduler.py` | UTD=0.2/1/2、多环境真实步数、warmup 无历史更新债务、恢复不重触发评估。 |
| `tests/integration/test_training_smoke.py` | 各主 source 的短训练有限 loss、参数更新、prior 冻结，Gaussian 版共用主算法。 |
| `tests/integration/test_vector_terminal.py` | 同步/auto-reset 两条路径；terminated 不 bootstrap，truncated 使用 terminal next obs；PID 槽位独立。 |
| `tests/integration/test_resume.py` | toy env 精确 continuation、RNG/UTD/source/normalizer 恢复；不支持环境快照时明确 episode restart。 |
| `tests/integration/test_prior_artifact.py` | 导出/载入 PPO 参数与分布一致，normalizer/hash/schema 错误被拒绝。 |
| `tests/integration/test_logging.py` | 本地/TB 同步标量与轴、offline 无网、resume 单调性、writer 正常关闭；真实在线检查单独可选。 |
| `tests/integration/test_fusion_contract.py` | mock 参考时刻、7→12 测试矩阵、单位/限幅 ownership、terminal/fallback 和 SDK 缺失行为。 |
| `tests/numerical/test_weighted_transport.py` | 低维双峰/能量分布验证 Q 重加权方向，条件逆向与已知端点 proposal 近似结果一致。 |
| `tests/numerical/test_density_convergence.py` | 数值积分步数和 trace 样本数对解析/高精度参考的收敛；不把代理误差自动视为消失。 |
| `tests/numerical/test_one_step_gap.py` | 仿射一致场的一步/多步一致，以及刻意构造不一致网络时 gap 被正确检出。 |

## 14. 配置与默认实验参数

### 14.1 依赖与平台

计划采用 Python 3.11、PyTorch、Gymnasium MuJoCo、NumPy、Hydra/OmegaConf、TensorBoard、WandB、pytest、ruff，以及结果分析所需 pandas/Matplotlib/SciPy。具体兼容版本在实现时解析并实际验证后锁定，不在本文虚构锁文件内容。

JVP 首选 `torch.func.jvp`；首次安装验收需验证所选 PyTorch/device 的 AD 支持。MPC CPU 计算与训练 GPU 计算解耦。macOS 本地优先 CPU smoke，服务器 CUDA 正式训练；MPS 未通过 JVP/density 数值验证前不列入正式支持。

### 14.2 启动超参

以下是实施起点，不是已验证最优值：

| 配置 | 初始值 | 说明 |
|---|---|---|
| `seed` | 0，正式 0–4 | calibration/eval 使用单独 seed 列表。 |
| `train.num_envs` | 1 | 先验证一致性，再试 4/8。 |
| `train.total_env_steps` | 1,000,000 | 简单任务可预定义较短预算，不能看结果后裁切。 |
| `train.learning_starts` | 5,000 | 与 warmup 一致。 |
| `train.utd_ratio` | 1.0 | 按真实 transitions。 |
| `replay.capacity` | 1,000,000 | controller/history 字段需提前估算内存。 |
| `train.batch_size` | 256 | 所有方法记录实际 update batch。 |
| `actor/critic.hidden_dims` | [256,256,256] | 同任务各 source 保持相同网络。 |
| `optimizer.actor_lr/critic_lr` | 3e-4/3e-4 | 首版固定学习率，后续单独调度。 |
| `algorithm.gamma` | 0.99 | reward 不默认缩放。 |
| `algorithm.target_tau` | 0.005 | 新参数权重。 |
| `algorithm.actor_update_interval` | 1 | 与 critic 更新频率独立记录。 |
| `algorithm.K_train` | 32 | 同中间点逆向候选数。 |
| `policy.K_exec/K_eval` | 1/1 | 主要结果严格单样本。 |
| `flow.time_delta` | 0.02 | 逆向核端点保护。 |
| `flow.flow_ratio` | 0.5 | 对角场样本比例。 |
| `source.beta` | 0.75 | 与 beta=0 同实现 baseline 比较。 |
| `source.sigma_prior` | 0.3 | latent 单位；PPO native 使用导出尺度。 |
| `source.schedule` | constant | 退火另做消融。 |
| `temperature.mode/alpha` | fixed/0.1 | 代理校验后再启用 auto。 |
| `density.method/steps` | continuous_inverse_proxy/5 | Heun 的真实场 NFE 单独统计。 |
| `density.trace/samples` | Hutchinson/1 | 小维参考用 exact_trace。 |
| `optimizer.max_grad_norm` | 10 | 记录裁剪前 norm 与 clipping rate。 |
| `eval.interval/episodes` | 10,000/10 | 按环境 transition 触发。 |
| `checkpoint.interval` | 50,000 | best、last、final 另存。 |
| `logging.interval` | 1,000 | 每次梯度更新内聚合，不逐次联网。 |

### 14.3 配置校验与运行命令规格

实现后的目标命令如下，当前不可直接运行：

```bash
python scripts/validate_install.py
python scripts/train_prior.py env=half_cheetah algorithm=ppo seed=100
python scripts/calibrate_controller.py env=inverted_pendulum prior=pid
python scripts/train.py experiment=smoke logging=local
python scripts/train.py env=reacher algorithm=prior_flame_m prior=pid source=constant seed=0
python scripts/train.py env=hopper algorithm=prior_flame_m prior=ppo prior.artifact_path=/absolute/path/to/prior logging=wandb_online
python scripts/launch_experiments.py experiment=prior_ratio --dry-run
python scripts/launch_experiments.py experiment=prior_ratio --execute
python scripts/evaluate.py checkpoint=/absolute/path/to/checkpoint.pt
python scripts/analyze_results.py matrix=/absolute/path/to/manifest.json
python scripts/benchmark_latency.py bundle=/absolute/path/to/policy_bundle
```

`--dry-run` 只输出 resolved configs 和 manifest，不创建训练进程；`--execute` 才运行，避免误启动大矩阵。未知 override、缺少 PPO artifact、不支持的 prior/task、缺失 log_prob、未验证 coil map、错误 action chart 必须在训练前报错。

## 15. 消融矩阵与公平比较

### 15.1 分批推进，不一次运行全部组合

| 编号 | 问题 | 组合 | 固定项 |
|---|---|---|---|
| A0 | 标准高斯基础链路是否正常？ | Gaussian PGOFP、SAC、PPO。 | 相同任务版本、预算、seed、评估协议。 |
| A1 | 控制先验源是否有效？ | Gaussian、PID、MPC、PPO native/center 分别命名。 | 相同算法网络、K_train、温度、固定 beta/sigma。 |
| A2 | 改进是否来自先验本身？ | controller-only、noisy-source-only、PGOFP、PPO 继续训练。 | 相同先验 artifact，增加预算清楚列出。 |
| A3 | 先验占比影响？ | beta=0/0.25/0.5/0.75/1。 | 同 prior、sigma、时间采样、初始权重。 |
| A4 | 先验扰动尺度影响？ | sigma=0.1/0.3/0.6/1.0。 | 同 prior 中心和 beta；PPO native 做 std multiplier 单独命名。 |
| A5 | 先验质量影响？ | 独立校准的 poor/medium/good、可选错误方向 prior。 | 冻结质量档位，记录 prior-only 表现。 |
| A6 | 退火有无收益？ | constant 与慢速 beta↑/sigma↓、较快 schedule 对照。 | 同起止范围，source 参数显式条件化。 |
| A7 | 时间采样耦合？ | uniform/logit-normal，flow_ratio=0/0.5/1。 | 在精选任务和固定 prior 比较；包含 delta 敏感性。 |
| A8 | FLAME 组件贡献？ | 无 Q、无 Bellman 熵、N_est=1/5/10、K_train=1/8/32。 | 一次改变一个因素；K=1 不具备候选内 Q 偏好。 |
| A9 | 替换位置是否重要？ | all、仅执行、仅 Actor、仅 target。 | 后三者标为故意不一致的诊断，不当成合法最终算法。 |
| A10 | 延迟与选择策略影响？ | K_exec=1 与 best-of-4/16。 | 报行为策略变更和总耗时；不纳入单样本主结果。 |
| A11 | 增益是否依赖更强信息？ | observation-only 与 MPC oracle。 | oracle 单列，记录 model rollouts。 |
| A12 | 跨先验 seed/扰动鲁棒性？ | 多个 prior seed、held-out 环境/目标参数。 | 固定测试集，禁止反馈调参。 |
| A13 | 是否只是增加了先验输入信息？ | 高斯源/先验源分别配 public_state/prior_features。 | 同一配对使用相同网络输入维度与 controller 记忆，分开识别输入信息和源分布效应。 |

无 Q 消融在 action-coordinate MaxEnt 定义下将 Q 项置 0，保留 `logJ_tanh`；另可提供“全 uniform latent weights”作为不同目标的对照。二者不能同名为 no-Q。

### 15.2 预算与 seed

- 每组合先做 1 seed 链路检查，再 3 seeds 小规模筛选，正式核心表至少 5 seeds。
- 相同 flow seed 使用相同初始化种子派生规则和环境 seed；先验分布改变后轨迹不同，不要求逐步动作一致。
- 统一记录交互预算、gradient steps、K_train、density NFE、wall-clock、GPU 小时和先验成本。
- 先验校准与 PPO 训练消耗单列；同时给“已有先验时新增样本效率”和“含先验制备总成本”两种视角。
- 所有方法同等超参搜索次数；不能主方法反复调参而基线只运行一次默认值。
- 主结果采用 final checkpoint 或预先规定的尾段评估均值；best checkpoint 单独给，不能用测试最佳点替代主结果。
- 训练失败 run 保留并汇报比例；基础设施失败可按规则重试，不能无标记删除不利 seed。

## 16. 训练记录、TensorBoard 与 WandB

### 16.1 一条标量数据通路

```text
MetricRecord -> 本地 JSONL
             -> TensorBoard SummaryWriter -> WandB TensorBoard sync
非标量产物 --------------------------------> WandB Table/Artifact/Video
```

同名 scalar 不同时调用 `wandb.log` 再同步 TB，避免重复曲线和不同 step。WandB 独立记录配置、summary 和产物；标量只由 SummaryWriter 产生。WandB 官方 SDK 提供 TB patch 和同步支持，实施时按锁定 SDK 版本做最小同步验证。[WandB TB 集成代码](https://github.com/wandb/wandb/blob/main/wandb/integration/tensorboard/monkeypatch.py)

### 16.2 初始化与失败处理

1. 创建唯一 run 目录，保存 resolved config/manifest，确认恢复时的 run id。
2. online/offline 模式下，在构造 SummaryWriter 之前启用 SDK TB patch/sync，root 指向本 run 的 tensorboard 目录。
3. 初始化 WandB 的 project/entity/name/group/tags/config，支持 `sync_tensorboard=True`；具体 patch 参数以安装版本测试为准。
4. 创建唯一 writer；所有曲线以 `global_step=env_transitions` 写入。
5. 梯度步等其他时钟作为数值字段记录，分析更新效率时从本地数据重建，不让 WandB 自动内部 step 取代环境步。
6. offline 保存本地 W&B run，后续使用官方同步命令；disabled 不导入/连接网络也能训练。
7. 退出时 writer.flush/close、WandB.finish，产物 flush；外部连接失败降级并写事件。

恢复时加载上次持久化 `last_logged_step`，防重复事件；TB 使用经过测试的 purge/resume 机制。WandB 若版本无法可靠追加原 run，应创建带 `parent_run_id` 的 continuation run 并在汇总层拼接，不能默默覆盖旧曲线。

### 16.3 指标命名

| 类别 | 必需指标 |
|---|---|
| `charts/*` | `env_steps`、`gradient_steps`、`episode_return`、`episode_length`、`steps_per_second`、`wall_time_sec`。 |
| `loss/*` | `critic`、`actor_qrmf`、`meanflow_diag`、`meanflow_offdiag`、`alpha`。 |
| `critic/*` | Q1/Q2 mean/std、target mean/std、TD error、Q gap。 |
| `weights/*` | ESS mean/p10、max_weight、candidate_Q_std、finite_ratio。 |
| `flow/*` | source-target displacement norm、velocity norm、target norm、JVP norm、time distribution、one_step_gap。 |
| `prior/*` | beta、sigma mean/min/max、component 使用率、valid/fallback rate、center norm、controller latency。 |
| `entropy/*` | `proxy_mean`、`alpha_value`、target entropy、estimator NFE、integration error、trace variance；不确定时不叫 exact entropy。 |
| `actions/*` | norm、delta norm、saturation fraction、raw-vs-filtered difference。 |
| `eval/*` | return mean/std、length、task-defined success/failure、tracking error、各 eval seed 数据。 |
| `timing/*` | env、source/controller、critic、actor/JVP、density、logging、checkpoint 时间。 |
| `system/*` | CPU/GPU 内存与资源信息，writer 不可用时也记录核心 local 指标。 |

每次记录窗口按样本数/更新数做正确加权平均。不同 episode 长度不能通过先平均各 collector 再平均造成隐含不等权；评估原始每回合数据永远保留。

### 16.4 run 身份与产物目录

run 名建议：`task__algorithm__prior-type-quality__beta__seed__config-hash`。随机 UUID 和 config hash 分开；重跑相同配置不覆盖。

```text
outputs/experiments/<matrix_id>/
├── manifest.json              # 所有组合、config hash、planned/running/completed/failed/skipped
├── summary.csv                # seed 级与组合级汇总，标明统计层级
├── failures.json              # 缺失/失败/重试原因
└── figures/                   # 文件名由指标、任务和比较组确定，索引写 summary

outputs/runs/<run_id>/
├── config_resolved.yaml
├── manifest.json              # code/deps/env/prior/schema/hardware/checksum
├── metrics.jsonl              # scalar 原始数据
├── events.jsonl               # fallback、NaN、resume、exception 等事件
├── train.log                  # 人可读进程日志
├── tensorboard/               # SDK SummaryWriter 事件文件
├── wandb/                     # W&B 本地缓存/离线 run
├── checkpoints/
│   ├── last.pt                # 最新完整训练状态
│   ├── best.pt                # 预定义校准指标最佳状态
│   ├── final.pt               # 预算结束状态
│   └── replay/                # 完整续训所需 replay 分片+索引
├── evaluation/
│   ├── episodes.csv           # checkpoint step、eval seed、回报、终止原因
│   ├── summary.json
│   └── condition_bank.npz     # 固定 obs/ref/controller context
├── trajectories/              # 低频选定轨迹 npz，schema 入 manifest
├── videos/                    # 低频 rgb 视频，按 env_step/eval_seed 命名
├── figures/                   # 指标和固定状态动作分布图
└── exports/
    └── policy_bundle/         # 第 18 节定义完整推理包
```

运行生成的动态文件名由 manifest 索引，不要求穷举每段 event 文件或 replay 分片名称。产物上传默认 best/final、config、summary、prior manifest 和少量视频；完整 replay 不默认上传。密钥从环境或已有登录读取，不写入 config、日志或 checkpoint。

## 17. 评估、统计与诊断

### 17.1 性能与稳定性

- 每个固定评估时间点运行相同数量 episode，冻结 normalizer、source schedule 和 controller 参数。
- 默认采样策略评估 `epsilon~p_src`、K_eval=1；固定评估 RNG 保证可复验。
- “用源均值生成动作”只能作为 `representative_action` 附加指标，不等于随机非线性策略的均值。
- return mean/std、episode length、环境定义的 failure；无 success 标准的 locomotion 任务不凭空发明 success rate。
- 样本效率：相同训练区间的 return AUC、达到预注册阈值所需步数；未达到标为 censored，不能填总预算当作真实到达值。
- 稳定性：跨 seed 方差、训练尾段评估方差、最大回撤、失败 run 比例。不得从 loss 波动直接推断控制稳定。
- raw 曲线和展示用平滑曲线同时保留，平滑窗口预定义；显著结论基于 raw 数据。

### 17.2 统计

以训练 seed 为统计独立单位，先在每个 seed 内聚合 eval episodes，再跨 seed 计算均值、标准差和 bootstrap 95% CI。不能把 5 seeds×10 episodes 当成 50 次独立训练。

跨任务回报尺度不同，主表逐任务报告；需要综合指标时预定义归一化参考并保留原值，不直接平均 Reacher 负回报和 HalfCheetah 正回报。seed 少时说明 CI 不稳定，避免把微小均值差写成确定优势。

### 17.3 表达能力与机理

- condition bank 包含固定原始观测、参考、控制器快照和 source 参数，比较所有方法时使用同一可兼容条件。
- 每个 condition 采样如 2,048 个动作，保存 source/one-step output 的完整样本，而非仅绘制两维散点。
- 固定投影、PCA、每维分布、动作协方差和多样性；聚类数依赖参数，不能单靠 KMeans 分成两簇声称多峰。
- 在预定义低维双峰控制/能量玩具任务上检验模式覆盖，MuJoCo 投影作为补充，避免把所有任务都假定存在多个最优模式。
- 测量位移范数、JVP 范数、目标方差、Q 加权 ESS 与 return 的关系，检验先验是否降低学习难度。
- 与高 Q 区域重合只能视为 Critic 诊断；要用环境 rollout/参考模型验证，不能把当前 Q 自评当成真实性能。

### 17.4 推理时延

测量路径包括：原始观测/参考准备、PID/MPC/PPO 先验查询、源采样、单步网络、tanh/物理变换和必要过滤。分别给 cached 与 uncached，最终实时结论以真实部署所需路径为准。

硬件固定、batch=1、提前 warmup、至少数千次计时；CUDA 同步，CPU 设置线程数；报告 p50/p95/p99、最大值与超 10 ms 比例。训练时使用 cached MPC 参数不意味着在线 MPC 没有成本。必要时后续蒸馏 MPC 先验并作为单独方法比较。

NFE=1 仅指 flow 网络的一次动作生成；冻结 PPO 和 MPC 求解成本另算。K_exec>1 即使向量化成一次 batched call，也需报告候选数量、处理样本数和吞吐，不把它等同于单动作工作量。

## 18. 检查点、恢复与部署

### 18.1 完整训练 checkpoint

必须包含：

- Actor、target Actor、双 Critic、target Critic、所有 optimizer、可学习 log_alpha。
- resolved config、schema_version、code/dependency/env/XML hash。
- replay buffer 分片与指针，PPO rollout 状态若适用。
- env_transitions、vector_steps、更新计数、UTD 信用额、下次评估/保存阈值。
- 训练 normalizer、先验 normalizer 引用、SourceSpec、schedule 状态。
- 每环境 controller 状态、参考进度、history、可用环境 snapshot。
- Python/NumPy/Torch CPU/CUDA 和各显式 generator 状态。
- run id、parent run id、WandB id、last_logged_step、先验 artifact hash。

先写临时文件，再原子 rename；维护 checksum，保存失败不得损坏 last。大 replay 分片用独立 manifest 标明完整性。只保存网络权重的文件应叫 `policy_weights`，不能称完整可恢复 checkpoint。

### 18.2 恢复语义

- `exact_resume`：环境支持完整 snapshot、版本一致，可继续当前 episode；toy env 用于严格复现。
- `episode_restart_resume`：恢复模型/replay/时钟/RNG，但重新 reset 环境和 controller；写事件并明确不保证逐步轨迹一致。
- `weights_only`：仅初始化权重，新 run、新预算，不能沿用旧 run 宣称无缝续训。

修改任务/动作坐标/先验 artifact/网络/参考 schema 必须拒绝 exact resume。若只延长总预算，退火的原 schedule_total_steps 默认不自动变化，防止继续训练时源参数突然回退。

### 18.3 推理包

```text
policy_bundle/
├── actor.pt                  # flow 权重/结构，经过数值对照
├── prior/                    # 第 10.3 节 PriorArtifact 或外部明确引用
├── normalizer.npz
├── source_spec.json          # 最终 beta/sigma/chart/version
├── policy_spec.json          # condition schema、feature 顺序、action transform
├── deployment_config.yaml    # dt、device、线程、reference 接口、可选过滤
└── manifest.json             # checksum、依赖和 smoke/latency 结果
```

真实部署若需要在线 PID/MPC/PPO，推理包必须带其实现配置和依赖；不能导出单个 flow Actor 就宣称控制先验方法可以独立运行。无 Critic 的 K=1 默认部署不导出训练期 density estimator。

## 19. 验证要求与验收标准

### 19.1 数学与协议验收

| 验收项 | 通过条件 |
|---|---|
| 源替换 | 在行为、target、Actor 候选、density、eval 中注入同一带版本的测试 source，确认无隐藏标准高斯采样。 |
| 零速度策略 | 输出恰为 tanh(source)，beta=0/1 的采样与密度符合解析值。 |
| 逆向核 | 非零均值/混合源的路径重构残差接近浮点精度；低维数值积分验证核归一与比例。 |
| MeanFlow | 仿射/多项式网络的 JVP 对有限差分；r=t 修正为零；权重/target/先验无梯度。 |
| density | 零场和可解析仿射场的 log-density 与解析值一致，方向正确；代理误差字段真实存在。 |
| alpha | 熵不足 alpha 增大、熵过高减小，且梯度不会进入 Actor。 |
| 终止语义 | truncated 使用终止前观测 bootstrap，terminated 不 bootstrap；reset observation 不混入 target。 |
| 有状态 prior | 多次 replay 查询不改变在线 PID/MPC 状态，评估不会污染训练。 |

容差按 float64 参考设定并在测试里说明：代数恒等式通常 `1e-8` 量级，float32 `1e-5` 起；有限差分/积分误差使用与步长匹配的阈值，不设掩盖错误的宽泛容差。

### 19.2 工程验收

1. 无 WandB 登录、无 HFM SDK、无显示服务时，MuJoCo 非渲染训练和 local/offline logging 可运行。
2. Gaussian/PID/PPO 的短训练均无 NaN，Actor/Critic 更新，冻结先验 hash 保持不变；MPC 在支持任务完成相同链路。
3. 单环境和多环境的预算/UTD 等价，结束槽位的 controller/reference 状态正确。
4. Ctrl+C/正常退出后有有效 last checkpoint，恢复后曲线环境步单调，产物不覆盖。
5. 本地 JSONL、TensorBoard 曲线、WandB 同步值和环境步一致；真实在线上传用一个单独小 run 验证。
6. mock fusion 满足 schema、动作映射、执行器 ownership、参考和异常协议；这只证明接口，不证明 HFM 可用。

### 19.3 研究验收

完成核心 5-seed 矩阵、原始数据、统计表、失败清单、source-only 比较、成本与完整延迟报告，即可认为实验框架交付。PGOFP 是否优于 Gaussian 是实验结论，不是允许删失败结果的验收条件。

数学和协议测试先执行；通过后做一次 10k–20k transitions 的小任务 smoke，再进入正式预算。只有改动、失败或新疑点时增加检查，不在每次纯文档/低影响改动后重复昂贵训练。

## 20. 实施顺序与工作交接

### 20.1 推荐里程碑

| 阶段 | 交付 | 进入下一阶段的门槛 |
|---|---|---|
| M0：统一约定 | schema、动作坐标、方向、SourceSpec、Gradient contract、依赖锁。 | 设计无未定义变量和明显坐标冲突。 |
| M1：数学基础 | distributions、action transforms、reverse kernel、MeanFlow JVP、解析 density 测试。 | 数值与梯度边界检查通过。 |
| M2：高斯源基线 | MuJoCo adapter/replay/runner、gaussian_flame_m、local logging/checkpoint。 | 小任务学习链路工作，density proxy 限制有诊断。 |
| M3：控制先验 | PID/MPC 支持任务、PPO prior 训练/导出、source mixture、状态与缓存。 | controller-only 校准、全采样位置一致性通过。 |
| M4：实验框架 | 矩阵 launcher、5-seed 汇总、TensorBoard/WandB、完整恢复与延迟测试。 | smoke 与数据审计通过。 |
| M5：研究实验 | 主比较、比例/质量消融、关键组件诊断、图表与结论。 | 原始数据、预算、统计和失败结果齐全。 |
| M6：真实聚变接入 | SDK adapter、真实 schema/coil map/reference、实测 HFM smoke。 | 获得实际资料，mock 不能替代验收。 |

官方 FLAME 复现应在 M2–M4 间单独完成并记录偏差；若复现遇到依赖或公开代码版本问题，保留 Gaussian 同实现对照，不能将未经验证的结果标成“复现成功”。

### 20.2 未来 agent 工作包

本节仅定义交接边界，不要求本轮创建或启动 agent。

| 工作包 | 负责文件范围 | 依赖和交付说明 |
|---|---|---|
| W1 数学/算法 | math、networks、density、algorithms | 先依赖 core 类型；提交公式到函数映射、梯度说明、数值验证、代理限制。 |
| W2 环境/先验 | envs、controllers、priors | 与 W1 先确认坐标和 SourceSpec；提交 controller 状态时序与独立校准结果。 |
| W3 训练/实验 | data、training、experiments、logging、scripts | 依赖接口稳定；提交一条可恢复 smoke、目录产物和矩阵 dry-run。 |
| W4 评估/接入 | evaluation、tests、fusion 文档 | 按协议验收而非复制实现；真实 HFM 信息缺失时保留明确 blocker。 |

每个工作包完成后更新：已实现文件、接口变化、验证命令与结果、未支持组合、数值近似/剩余风险。公共 schema 或动作方向变更必须同步所有调用方和本文衍生 contract，不能私自新增另一套同义字段。

### 20.3 agent 开始实施时的检查顺序

1. 阅读本文第 2、4、5、6、7 节，确认方法和坐标；再看各自文件清单。
2. 查看当前实际文件、用户后续指令及已有实现，避免覆盖其他工作。
3. 先实现最小端到端路径及关键数学验证，再补任务和扩展基线，不先堆空壳。
4. 任何理论或工程捷径必须写进 config、method tag 和 reproduction notes。
5. 不在报告中使用“保证安全”“准确熵”“满足 10 ms”等尚未验证的断言。

## 21. 风险与待接入信息

### 21.1 已明确的研究风险

- 先验太差/太强会限制有效探索；依靠 beta、sigma、质量档位和 source-only 比较分析。
- 有限候选 SNIS 在高维可能 ESS 退化；增加 K 的计算成本必须与收益同时报告。
- 逆向时间靠近源端时 latent/velocity 放大；用时间界和稳定 log-Jacobian，保留诊断。
- 一步策略与连续 density proxy 可能不一致；即使 loss 下降，也要检查 gap 与 Critic 稳定性。
- cached MPC 只能减训练开销，不能消除真实部署的 MPC 求解时延。
- 有状态先验、参考轨迹和归一化不一致会制造隐藏分布偏移；schema 和 replay 快照是必要约束。
- 有用先验、短位移和高回报之间没有必然关系；需允许得到“收益只在某些先验/任务出现”的结论。

### 21.2 不影响 MuJoCo 首阶段、但真实接入前必须获得的信息

| 信息 | 未获得时的处理 |
|---|---|
| 开题报告提到的已有 OFQL/HFM 代码路径和许可证 | 按本文新建工程；之后评估迁移，不假定现有 API。 |
| HFM SDK、版本、启动方式、依赖和示例 | 只交付协议和 mock，不宣称已接入。 |
| 真实 64 维 observation 字段顺序/单位 | schema 保持待确认，不编造索引。 |
| 7→12 维矩阵、线圈通道顺序和量程 | `verified:false`；真实 backend 拒绝运行。 |
| 电源 delay/rate limit/PSM 已由哪层实现 | 接口能力协商，避免重复变换。 |
| F1/F2 的正式奖励、终止与参考文件 | 使用报告暂定值作说明，不作为真实验收依据。 |
| 目标服务器硬件、WandB project/entity、网络模式 | 支持可配置和 offline；先不影响本地数学/环境实现。 |
| 已有 PPO/PID/MPC 先验及其输入、checkpoint 格式 | 可先训练/校准 MuJoCo native prior；外部 prior 按 artifact 协议接入。 |

## 22. 参考资料

本文主要依据三个本地文档。下列外部资料用于核对原始方法及工程 API；第 4–21 节中具体接口、目录与数值方案是本项目拟议实现，不能当成原论文已经验证的扩展。

- [Boosting Maximum Entropy Reinforcement Learning via One-Step Flow Matching：FLAME 原论文](https://arxiv.org/html/2602.01606v1)：核对 Q 重加权、逆向采样、MeanFlow 与训练/推理分离；实现时固定版本。
- [FLAME 官方仓库](https://github.com/lzqw/FLAME)：后续选择可复现 commit，记录论文与代码差异，不直接依赖浮动 main。
- [Gymnasium MuJoCo 文档](https://gymnasium.farama.org/environments/mujoco/)：环境版本和模拟器依赖。
- [Gymnasium 时间限制处理](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)：terminated/truncated 的 bootstrap 语义。
- [Gymnasium SyncVectorEnv](https://gymnasium.farama.org/api/vector/sync_vector_env/)：向量环境和 autoreset 配置；具体 final observation 字段按锁定版本测试。
- [WandB TensorBoard 集成源码](https://github.com/wandb/wandb/blob/main/wandb/integration/tensorboard/monkeypatch.py)：核对同步初始化和 root 目录行为。

