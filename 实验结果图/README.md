# MuJoCo 30k 实验训练曲线

本轮批次：`mujoco_30k_a100_v1`。每个环境/先验一张图，共 10 张；每个 β 一条曲线。

## 读图口径

- 横轴为 Flow 训练的环境步数，单 run 预算 30k；纵轴为随机策略评估平均 return，越高越好。
- 正常完成的组：每条实线为训练 seeds=0/1/2 的评估均值，每个 seed 每次评估 10 回合。只画均值线，不加误差带。
- 原始评估在 10k/20k/30k；30k 重复评估取最后一次。没有初始 Flow 回报测量，不补 0 步分数，不平滑、不外推。
- 失败组：虚线与 × 仅画当时有评估记录的 seed 的均值，并逐点标出 n；参与 seed 可变化，不能视为固定三 seed 曲线或完整 30k 成绩。
- 失败且没有任何评估记录的 β 只显示图例，不造曲线；图下注明各 seed 停止步数。尤其 Hopper/PPO 的 β=1 三个 seed 均未到首次评估。
- 每张图采用独立纵轴范围。直接 PPO/控制器基线请对照人读报告，本组图片只比较 β 曲线。

## 图片

| 环境 | 先验 | 图片 |
|---|---|---|
| InvertedPendulum-v5 | PPO | [InvertedPendulum-v5_PPO.png](InvertedPendulum-v5_PPO.png) |
| Reacher-v5 | PPO | [Reacher-v5_PPO.png](Reacher-v5_PPO.png) |
| HalfCheetah-v5 | PPO | [HalfCheetah-v5_PPO.png](HalfCheetah-v5_PPO.png) |
| Hopper-v5 | PPO | [Hopper-v5_PPO.png](Hopper-v5_PPO.png) |
| Walker2d-v5 | PPO | [Walker2d-v5_PPO.png](Walker2d-v5_PPO.png) |
| InvertedDoublePendulum-v5 | PPO | [InvertedDoublePendulum-v5_PPO.png](InvertedDoublePendulum-v5_PPO.png) |
| InvertedPendulum-v5 | PID | [InvertedPendulum-v5_PID.png](InvertedPendulum-v5_PID.png) |
| Reacher-v5 | PID | [Reacher-v5_PID.png](Reacher-v5_PID.png) |
| Reacher-v5 | MPC | [Reacher-v5_MPC.png](Reacher-v5_MPC.png) |
| InvertedDoublePendulum-v5 | MPC | [InvertedDoublePendulum-v5_MPC.png](InvertedDoublePendulum-v5_MPC.png) |

数据与统计索引：[curve_data.json](curve_data.json)。含每个点的参与 seed、原始分数、均值与失败状态。

完整报告：[实验总结](../docs/experiment_reports/mujoco_30k_a100_v1/实验总结.md)。

## 离线重画

在项目根目录运行，不训练、不联网：

```bash
.venv/bin/python scripts/plot_mujoco_30k_curves.py
```
