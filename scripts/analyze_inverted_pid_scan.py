"""Analyze completed PID scan without imputing scores after numerical failures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pgofp-matplotlib')
from pathlib import Path
import json,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'outputs/experiments/inverted_pendulum_pid_beta_sweep_300k_retry1';OUT=ROOT/'experiment_summary_figures';OUT.mkdir(exist_ok=True)
G={0:[],.5:[],1:[]}
for d in sorted((P/'runs').glob('inverted*')):
 m=json.loads((d/'manifest.json').read_text());c=__import__('yaml').safe_load((d/'config_resolved.yaml').read_text());rows=[json.loads(l) for l in (d/'metrics.jsonl').read_text().splitlines()];ev={v['step']:v['eval/return_mean'] for v in rows if 'eval/return_mean'in v};xs=sorted(ev)
 r=dict(seed=c['seed'],outcome=m['outcome'],stop=m['summary']['env_transitions'],final=m['summary'].get('final_eval',{}).get('eval/return_mean'),failure=m['summary'].get('final_eval',{}).get('eval/failure_rate'),representative=m['summary'].get('final_eval',{}).get('eval_representative/return_mean'),first900=next((x for x in xs if ev[x]>=900),None),first1000=next((x for x in xs if ev[x]>=1000),None),tail=statistics.mean(ev[x] for x in xs if 250000<=x<=300000) if 300000 in ev else None,auc=float(np.trapezoid([ev[x] for x in xs],xs)/290000) if 300000 in ev else None)
 early=[x for x in xs if x<=100000];r['early_auc']=float(np.trapezoid([ev[x] for x in early],early)/90000) if 100000 in ev else None
 G[c['source']['beta']].append((r,rows,ev))
fig,axs=plt.subplots(2,2,figsize=(12,8));colors=['#2563eb','#e87918','#16834a']
for ax,(b,items) in zip(axs.flat,list(G.items())):
 for r,rows,ev in items:
  xs=sorted(ev);ax.plot(xs,[ev[x] for x in xs],label=f'seed {r["seed"]}',color=colors[r['seed']])
  if r['outcome']!='completed':ax.axvline(r['stop'],color=colors[r['seed']],ls=':',alpha=.7);ax.scatter([xs[-1]],[ev[xs[-1]]],marker='x',color=colors[r['seed']],s=60)
 ax.set(title=f'beta={b:g}: raw evaluation (dotted = numerical failure)',xlabel='Environment steps',ylabel='Evaluation return',ylim=(-20,1050),xlim=(0,300000));ax.grid(alpha=.2);ax.legend()
ax=axs[1,1]
for r,rows,ev in G[1]:
 vs=[v for v in rows if 'critic/loss'in v and np.isfinite(v['critic/loss'])];ax.plot([v['step'] for v in vs],[v['critic/loss'] for v in vs],color=colors[r['seed']],label=f'seed {r["seed"]}')
ax.set(title='beta=1: Critic loss (log scale)',xlabel='Environment steps',ylabel='Window mean loss',yscale='log');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'inverted_pendulum_pid_scan.png',dpi=160);plt.close(fig)
results={str(b):[r for r,_,_ in items] for b,items in G.items()};(P/'analysis_metrics.json').write_text(json.dumps(results,indent=2))
def avg(b,k):return statistics.mean(r[k] for r,_,_ in G[b])
section='''## 3. InvertedPendulum-v5：PID 先验扫描（2026-11-10 汇总）

β=0/0.5/1，各 seed=0/1/2，每组预算 300k；sigma_prior=0.3、sigma_base=1.0、固定 α=0.1，Critic 熵项开启。PID 使用现有固定参数。β=0 保留相同控制器上下文，但源采样和密度退化为基础高斯，网络不额外读取控制器特征。

这轮 retry1 已全部结束：7 组完成、2 组数值失败。此前一轮因检查点写入 ENOSPC 中断，单独保留，不与本轮合并统计。WandB 离线保存，本次分析采用本地日志。

| β | seed 0 最终回报 | seed 1 | seed 2 | 完成数 | 最终均值 ± seed 间样本标准差 |
|---:|---:|---:|---:|---:|---:|
| 0 | 825.8 | 821.4 | 795.5 | 3/3 | **814.2 ± 16.4** |
| 0.5 | 1000.0 | 1000.0 | 905.7 | 3/3 | **968.6 ± 54.4** |
| 1 | 165,348 步数值失败 | 3.3 | 50,968 步数值失败 | 1/3 | 不汇总为三 seed 最终成绩 |

![倒立摆 PID 扫描与 β=1 Critic loss](experiment_summary_figures/inverted_pendulum_pid_scan.png)

曲线为每个 seed 的原始随机策略评估，每次 10 回合；同一步多次评估取最后一条，不做平滑。β=1 的叉号是最后有效评估，竖虚线为实际数值失败步数；失败后不补零、不延长曲线。Critic loss 图只绘有限日志值。

| 指标（先按 seed 计算，再取三 seed 均值） | β=0 | β=0.5 |
|---|---:|---:|
| 最终回报 | 814.2 | 968.6 |
| 末段平均回报（250k–300k，含端点共 6 次） | 808.8 | 905.4 |
| 10k–300k 归一化评估 AUC | 830.1 | 870.1 |
| 10k–100k 归一化评估 AUC | 697.5 | 838.9 |
| 最终评估回合失败率 | 33.3% | 6.7% |

AUC 用梯形积分除以所注明的步数跨度，单位仍为回报，不包含 0–10k 的未测区间。失败率指环境回合提前终止，不是训练数值失败率；最终各组共 30 个评估回合，分别 10/30 与 2/30 回合失败。β=1 未完成 run 的 AUC 不与完整 300k AUC 混合。

**结论一：本轮中等比例 PID 先验有收益，但收益量级和稳定性要分别看。** β=0.5 的最终均值比 β=0 高约 19.0%，末段均值高约 11.9%，全程 AUC 高约 4.8%；三个配对 seed 的最终回报都更高。早期 AUC 均值也更高，但首次达到回报 900 的步数分别为 β=0 的 [70k,20k,20k]、β=0.5 的 [40k,30k,20k]，不能声称每个 seed 都收敛更快。首次满分之后仍有大幅回落，首次达到阈值不等于持续稳定收敛。

**结论二：β=1 的问题是长期训练失稳，而非单纯起点不好。** 三个 seed 都曾评估满分，但 seed0/seed2 最终因 Critic loss 非有限值退出；seed1 虽完成 300k，最终随机/代表动作评估回报仅 3.3/3.8，末段平均回报 2.45。两种现象需分开：数值失败 2/3，性能崩溃也出现在未触发数值异常的 run 中。不能用 best checkpoint 满分掩盖退化，也不能将失败样本从统计中删除。

**结论三：此次混合先验的优势主要体现在随机执行的鲁棒性。** β=0 和 0.5 的最终代表动作评估（固定源代表点，每 seed 5 回合）全部为 1000，随机执行评估则存在差异。这提示最终差距与动作分布、尾部行为或噪声鲁棒性有关，但不是因果证明；代表动作不是策略分布的动作均值，两类评估的回合数也不同。

**局限与下一步：** 每组仅三个训练 seed、评估环境种子固定，未作显著性检验，且任务回报上限为 1000。此结果不证明先验越强越好，也不能直接推广到 HalfCheetah。现有短验证中的 PID-only 满分不能替代本轮同协议基线；还需 PID-only、加噪源不训练 Flow、更多独立评估种子的对照，才能判断学习改进是否超越原控制器。β=1 在 PID 与 PPO 两类源上均失稳，说明问题不限于 PPO；候选估计、反向积分和熵反馈的具体根因仍未确定。

配置：[PID 扫描配置](configs/experiment/inverted_pendulum_pid_beta_sweep_300k_retry1.yaml)；逐 seed 指标：[分析数据](outputs/experiments/inverted_pendulum_pid_beta_sweep_300k_retry1/analysis_metrics.json)；完整证据见 [实验记录](docs/experiment_records/实验记录.md)。

'''
report=ROOT/'实验总结.md';s=report.read_text();marker='## 当前判断与下一步';assert marker in s
if '## 3. InvertedPendulum-v5：PID 先验扫描' in s:
 start=s.index('## 3. InvertedPendulum-v5：PID 先验扫描');end=s.index(marker,start);s=s[:start]+section+s[end:]
else:s=s.replace(marker,section+marker,1)
s=s.replace('环境为 HalfCheetah-v5，使用单步 Flow 策略','前两节环境为 HalfCheetah-v5；第三节为 InvertedPendulum-v5。使用单步 Flow 策略',1)
s=s.replace('**性能方面：** 已完成的 300k 比较中，Gaussian 均值最高，β=0.25 接近，高占比先验未显示优势；β=1 没有可比较的最终成绩。','**性能方面：** HalfCheetah 上 Gaussian 最终均值最高，β=0.25 接近；倒立摆上 β=0.5 的随机执行回报优于 Gaussian。收益依赖任务与先验，不能归纳为先验普遍优于高斯。')
report.write_text(s)
print('Analysis saved. Final means:',avg(0,'final'),avg(.5,'final'))
