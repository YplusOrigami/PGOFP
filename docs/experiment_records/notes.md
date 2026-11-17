# 人工研究备注

- β=1 原始运行及同 seed 重跑均出现 critic loss 非有限值，停止步数分别为 13632、10490、7345；不构成 300k 性能比较。
- β=0.25/0.5 六组由 250k 检查点以 episode_restart 恢复，原始日志中途进度与恢复后曲线需要区分。

## InvertedPendulum PID β 扫描（2026-11-04）

- 新实验：β=0/0.5/1，seed=0/1/2，每 run 300k，标准 Critic 熵项开启；sigma_prior=0.3，sigma_base=1。
- β=0 保留相同 PID context，但源采样和密度严格退化为基础高斯；各组 condition=public_state，不额外加入控制器特征。
- 使用原有密钥进行在线预检返回 AuthenticationError，未修改或复制密钥。实验实际采用 WandB offline＋本地 JSONL/TensorBoard，未声称已在线同步。
- 并发上限 3，先覆盖三个 β 的 seed0，其余 seed 排队；数值失败保留记录，不自动重跑。结束后自动聚合并更新总记录。
- 倒立摆回报上限为 1000，除最终回报外，应比较学习曲线、达到稳定回报所需步数、失败率和动作饱和率。

## 倒立摆 PID 扫描重跑（retry1）

上一轮 seed0 三组在 10k/20k/30k 保存检查点时出现 ENOSPC；旧进程已退出，未完成其余六组。本轮从头运行、保留所有旧记录，超参数与 seed/预算不变，WandB 仍为 offline。新输出目录通过 8 MiB 写入、fsync、原子重命名、读回校验；该检查仅说明当前可写，未确定之前写入失败根因。

## 倒立摆 PID 扫描最终分析（2026-11-10）

retry1 九组结束，7 完成、2 数值失败。β=0 最终随机评估 814.2±16.4，β=0.5 为 968.6±54.4（seed 间样本标准差）。末段250k–300k均值808.8 vs905.4，全程10k–300k归一化AUC830.1 vs870.1。β=1 seed0/2在165348/50968步Critic loss非有限值退出，seed1完成但最终随机回报3.3、代表动作3.8。β=0/0.5各seed代表动作评估均1000，提示随机执行鲁棒性差异，尚未证明机制。失败seed不填零、不作为300k结果，未做显著性检验。图和完整分析见项目根目录实验总结.md第三节。

## Reacher 控制器候选任务筛查（2026-11-11）

无RL训练、无调参，校准seed50000–50019，共20回合。PID return−5.73±2.15、MPC−6.26±2.20；随机−45.04、零动作−8.88。MPC回退率0，模型rollout步数2560000；环境交互四种各1000步。PID末端已很接近目标，收益需看轨迹和控制成本；MPC现有目标与环境奖励权重不完全一致。详情见项目根目录Reacher先验评估.md。

## 倒立双摆 MPC 独立评估（2026-11-11）

固定 CEM H10/P64/I4，20个校准回合。return 779.24±406.60，平均存活84.65步，跌倒率100%，规划回退率0.00%。未训练Flow、未实现PID；源码新增task/公开状态重建/规划代价且通过3个集成测试。详细协议和源码哈希保存在outputs/controller_screening/inverted_double_pendulum_20261111，报告为根目录倒立双摆先验评估.md。

## EXP-010 · 服务器 MuJoCo 30k 扫描（2026-11-18 补记）

- 原始批次 `outputs/experiments/mujoco_30k_a100_v1`，Flow 矩阵从北京时间 2026-11-13 12:14:13 到 2026-11-17 11:00:55。结果数据提交为 `e8f7036`；本次仅离线整理，不启动、续训或重试实验。
- PPO 六任务、PID 倒立摆/Reacher、CEM-MPC Reacher/双摆，β=0/0.25/0.5/0.75/1，seeds=0/1/2，各 30k，共 150 run。129 完成；21 个均在 β=1 时 Critic loss 非有限值退出。未完成分数不能当成 30k 终点；数值失败不是环境 terminated 指标。
- PPO 先验每任务 seed100、实际 1,001,472 步；HalfCheetah 复用历史，其余五个新制备，三训练 seed 共享。六套内容 hash 已复算核对。直接先验每 run 评估十回合，但同组合的十五次使用固定相同种子，不是十五个独立样本。
- Reacher/PPO、HalfCheetah/PPO、Hopper/PPO、Reacher/MPC 的 β=.75 在三个配对 seed 上均优于各自 β0；Reacher/MPC 的最终 −7.31±0.67 也在三个 seed 上超过直接 MPC（−8.29）。六个 PPO 任务的最优组均值全部低于直接 PPO；两个 PID 任务非零 β 没有平均优势。
- 双摆 MPC β1 的三个终点为 9352.69/4855.40/297.06，均值 4835.05±4527.85；只能作为高方差探索结果。倒立摆 PPO β1 虽完成但均值仅 77.5；“没数值退出”不等于控制表现稳定。
- 本轮最优 β 为事后选择，仅三个训练 seed；不做显著性、跨任务通用最优或总样本效率优势结论。AUC 只覆盖 10k–30k；30k 周期/最终重复取最后一条。主分数随机评估十回合，代表动作五回合单独解释。
- Reacher/PID 与 MPC 的 β0 曲线相同，不能当六个独立 seed。PPO 包装器构造消耗全局 Torch RNG，跨先验组同 seed 不保证初始网络相同；先按组内 β 配对比较。147 个 run 来源 `4e154ad` 且 dirty，3 个无 Git 信息，没有完整运行时源码 diff。
- WandB disabled，150 个 run 均有本地 JSONL、TensorBoard、先验基线；129 个成功终点与原始 JSONL/逐回合 CSV/服务器汇总核对。Flow checkpoint/replay 未同步，策略录像或重评需另取服务器文件。
- 接续记录见 [EXP-010/实验记录.md](EXP-010/实验记录.md)，含 150 份完整配置、run_index.json 和 10 张逐 seed 曲线；人读报告及五图集中在 [实验总结](../experiment_reports/mujoco_30k_a100_v1/实验总结.md)。后续刷新用 `scripts/update_experiment_records.py`，本批专项生成逻辑会自动保留索引；人读报告用 `scripts/summarize_mujoco_30k.py` 复算。

### EXP-010 运行效率排查（2026-11-18）

只读复算确认 150 个 Flow run 最大实际并发=1，子进程总时间 94.7547h、整体跨度 94.7781h；每 run num_envs=1、torch_threads=1。PPO prepare_priors 也串行，不受 Flow 的 --max-parallel 控制。进度阶段时间：热身+训练 84.80h、Flow 评估 8.51h、先验评估 0.57h、保存约 2 分钟。不要把 sync_device_timing=false 的未完整 flush 阶段计时当成纯 GPU 时间，或把 Runner.summary 计时当完整子进程计时。两组 MPC 共 36.16h，β0 仍查询控制器；MPC 0.5s 墙钟超时使过度并发可能改变回退率。下一步先测独立 run 1/2/3 并发与 CPU/CUDA profile，再优化标量同步、规划线程及重复评估；不能由平台 20% 利用率承诺 5 倍加速。本次未改算法、并发默认值或启动训练。详见[性能排查](../experiment_reports/mujoco_30k_a100_v1/性能排查.md)及 outputs/analysis/mujoco_30k_a100_v1/performance_audit.json。

## 倒立双摆 300k 新批次计划（2026-11-18；正式训练未启动）

用户最终指定仅 MPC/PPO，不包含 PID。新矩阵 `double_pendulum_prior_beta_300k_v1`：InvertedDoublePendulum-v5，β0/.5/1，seeds0/1/2，共18 run，每run300k、默认CUDA、并发3；其余学习/评估/保存协议沿用30k批次，WandB disabled。优先复用上一轮seed100冻结PPO，缺失时在新目录独立制备。入口为根目录 launch_double_pendulum_300k.sh（**无参数启动正式训练**，--plan仅计划，--validate是隔离短验证）。正式计划存于 outputs/experiments/double_pendulum_prior_beta_300k_v1，仅有计划和解析配置，不应记为已启动或已完成实验。详情见[运行说明](double_pendulum_300k/运行说明.md)。

本地隔离 CPU 短验证6/6完成、数值失败0、实际最大并发3，记录核验通过；目录 outputs/validation/double_pendulum_prior_beta_300k_v1_20261118_012350_dbb4ae。正式目录 validation_receipt.json 记录验证结果，不是训练 manifest。正式 PPO hash 仍为5e9e89068316fda9691be6aecb056bc006055d01dfba76af5c184016b227014b；300k正式训练未启动。
