"""Rebuild the local experiment ledger from immutable local evidence; no network access."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/pgofp-matplotlib')
from pathlib import Path
import json, shutil, statistics
from datetime import datetime, timezone, timedelta
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/experiment_records'; OUT.mkdir(exist_ok=True)
TZ=timezone(timedelta(hours=8))
def read(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def rows(p):
 result=[]
 if p.exists():
  for line in p.read_text().splitlines():
   try:result.append(json.loads(line))
   except ValueError:pass
 return result
def date(s):
 try:return datetime.fromisoformat(s.replace('Z','+00:00')).astimezone(TZ).strftime('%Y-%m-%d %H:%M:%S')
 except (ValueError,AttributeError):return '未知'
def link(p,label=None):return f'[{label or p.name}]({os.path.relpath(p,OUT)})'
experiments=[]
for p in (ROOT/'outputs/experiments').iterdir():
 if not p.is_dir():continue
 runs=[]
 for r in (p/'runs').glob('*'):
  if not r.is_dir() or not (r/'config_resolved.yaml').exists():continue
  m=read(r/'manifest.json'); c=yaml.safe_load((r/'config_resolved.yaml').read_text());met=rows(r/'metrics.jsonl');events=rows(r/'events.jsonl')
  runs.append(dict(path=r,m=m,c=c,met=met,events=events))
 if runs:
  matrix=read(p/'manifest.json')
  for run in runs:
   candidates=[rec.get('started_utc') for rec in matrix.get('runs',[]) if run['path'].name.startswith(rec.get('run_key','!')+'__') and rec.get('started_utc')]
   run['started']=min(candidates) if candidates else run['m'].get('provenance',{}).get('utc_time','9999')
  experiments.append((min(x['started'] for x in runs),p,runs))
experiments.sort(key=lambda x:(x[0],x[1].name))
text=['# PGOFP 本地实验记录','', '## Agent 记录写入规范','',
'1. 本文件是实验索引；原始证据为各 run 的 config_resolved.yaml、manifest.json、metrics.jsonl、events.jsonl 和检查点。不得依赖 WandB 作为唯一记录。',
'2. 每次新实验或重跑分配独立序号、目录和 run；保留失败、恢复及历史记录，禁止用新结果覆盖旧实验。相同 seed 重跑属于复现，不得计作新增独立 seed。',
'3. 日期统一北京时间（UTC+8），优先采用日志中的 UTC 时间换算；缺失时间写“未知”，不得把文件修改时间当作实验开始时间。恢复后的 manifest 可能覆盖首次启动时间，须结合事件和矩阵记录判断。',
'4. 启动前记录目的、配置路径、全部解析配置、seed、预算、先验版本和实验差异；启动后记录状态；完成或失败后补上实际步数、评估、异常与图像。算法改动必须记录改动文件及版本/哈希。',
'5. 最终回报仅取 summary.final_eval；未完成实验不得把中途评估写作最终结果。均值和样本标准差以 seed 为单位；评估回合数与训练 seed 数分别注明。失败率须保留失败 seed，不得只报告成功样本。',
'6. 曲线横轴为环境步数，使用本地原始评估，不平滑；同一步重复评估取最后一条。当前图像分别绘制每个 seed，不使用阴影。缺失评估不补造，不外推。',
'7. 恢复必须记录恢复模式与检查点步数；本地 metrics_discarded.jsonl 保留回退数据，恢复曲线以当前 metrics.jsonl 为准。MuJoCo episode_restart 不保证逐步一致。',
'8. 每次更新运行 `.venv/bin/python scripts/update_experiment_records.py` 重建本报告与图片。自动生成区请勿手工维护；研究解释写入同目录 notes.md，避免刷新时丢失。不得在记录中写入账号密钥或 .env 内容。',
'9. 新配置后续默认启用 JSONL/TensorBoard 本地记录；WandB 账号已不可用，不应把在线连接作为实验启动前提。更改日志后端须显式记录，已有历史配置快照保持原样。','',
'## 数据范围与统计口径','', '本报告扫描 outputs/experiments 下具有解析配置的所有 run；包括链路验证、正式预算的探索实验和失败重跑。先验训练 artifact 单列。日期为本地证据，状态来自每个 run，矩阵 launcher 状态可能滞后。','',
'## 实验索引','', '| 序号 | 日期（北京时间，见时间说明） | 实验 | 完成/总 run |','|---|---|---|---|']
for i,(start,p,rs) in enumerate(experiments,1):
 text.append(f'| EXP-{i:03d} | {date(start)} | {p.name} | {sum(r["m"].get("outcome")=="completed" for r in rs)}/{len(rs)} |')
for i,(start,p,rs) in enumerate(experiments,1):
 ident=f'EXP-{i:03d}';asset=OUT/ident;asset.mkdir(exist_ok=True)
 text+=['',f'## {ident} · {p.name}','',f'配置入口：{link(ROOT/"configs/experiment"/(p.name+".yaml"))}；原始目录：{link(p)}。','',
'| run / seed | 状态 | 开始/最近清单时间（北京） | 实际步数 | 最终回报 | 最后中途评估（步数:回报） |','|---|---|---|---:|---:|---|']
 groups={};fig,ax=plt.subplots(figsize=(9,4.6));curves={}
 for j,r in enumerate(sorted(rs,key=lambda r:r['path'].name),1):
  d,m,c=r['path'],r['m'],r['c']; ev={v['step']:v['eval/return_mean'] for v in r['met'] if 'eval/return_mean' in v};final=m.get('summary',{}).get('final_eval',{}).get('eval/return_mean');errors=[e for e in r['events'] if e.get('kind')=='numerical_error'];step=errors[-1]['env_transitions'] if errors else m.get('summary',{}).get('env_transitions',max((v['step'] for v in r['met']),default=0));label=d.name.split('__')[1] if '__' in d.name else d.name
  snapshot=asset/f'run{j:02d}_config.yaml';shutil.copyfile(d/'config_resolved.yaml',snapshot)
  text.append(f'| {link(d,label)} / {c["seed"]} | {m.get("outcome","未知")} | {date(r.get("started"))} | {step} | {f"{final:.2f}" if final is not None else "—"} | {str(max(ev))+":"+format(ev[max(ev)],".2f") if ev else "—"} |')
  groups.setdefault(label,[]).append(final)
  if ev:
   xs=sorted(ev);ys=[ev[x] for x in xs];ax.plot(xs,ys,label=f'{label} / seed {c["seed"]}',marker='.' if len(xs)<3 else None);curves.setdefault(label,[]).append(ev)
  r['snapshot']=snapshot
 ax.set(xlabel='Environment steps',ylabel='Evaluation return (raw)',title=p.name);ax.grid(alpha=.2)
 if curves:ax.legend(fontsize=7)
 else:ax.text(.5,.5,'No evaluation reached',transform=ax.transAxes,ha='center')
 fig.tight_layout();fig.savefig(asset/'eval_return.png',dpi=150);plt.close(fig)
 text+=['',f'![{ident} 评估曲线]({ident}/eval_return.png)','']
 for label,values in groups.items():
  vs=[v for v in values if v is not None]
  text.append(f'- {label}：完成最终评估 {len(vs)}/{len(values)}；'+(f'最终均值 {statistics.mean(vs):.2f}，seed 间样本标准差 {statistics.stdev(vs):.2f}。' if len(vs)>1 else '无可汇总的多 seed 最终结果。'))
 for r in sorted(rs,key=lambda r:r['path'].name):
  c=r['c'];text+=['',f'### {r["path"].name}','',f'完整配置快照：{link(r["snapshot"])}；日志：{link(r["path"]/"metrics.jsonl")}；事件：{link(r["path"]/"events.jsonl")}。','',f'配置哈希：`{r["m"].get("config_hash","未知")}`；来源版本：`{json.dumps(r["m"].get("controller_versions",[]))}`。','']
  text+=['```yaml',yaml.safe_dump({k:c[k] for k in ['seed','env','algorithm','source','prior','network','optimizer','train','eval','checkpoint','condition','device'] if k in c},allow_unicode=True,sort_keys=False).rstrip(),'```','']
  for e in r['events']:
   if e.get('kind') in ['resume','numerical_error','log_truncated_on_resume']:
    text.append(f'- {date(e.get("utc"))}：`{json.dumps(e,ensure_ascii=False)}`')
text+=['','## 先验训练与 artifact 证据','', '下列文件保留先验身份、校准与来源信息；artifact 不等同于本报告中的下游 Flow 训练。']
for f in sorted((ROOT/'artifacts/priors').rglob('*.json')):
 if not f.name.startswith('._'):text.append(f'- {link(f)}')
(OUT/'实验记录.md').write_text('\n'.join(text)+'\n')
notes=OUT/'notes.md'
if not notes.exists():notes.write_text('# 人工研究备注\n\n- β=1 原始运行及同 seed 重跑均出现 critic loss 非有限值，停止步数分别为 13632、10490、7345；不构成 300k 性能比较。\n- β=0.25/0.5 六组由 250k 检查点以 episode_restart 恢复，原始日志中途进度与恢复后曲线需要区分。\n')
print('Written',OUT/'实验记录.md','experiments',len(experiments))
