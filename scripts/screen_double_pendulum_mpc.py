"""Fixed-budget, no-learning CEM-MPC assessment on the standard double pendulum."""
from pathlib import Path
import json,time,hashlib,platform
from datetime import datetime,timezone
import numpy as np
import gymnasium as gym
import mujoco
from omegaconf import OmegaConf
from pgofp.config.compose import compose
from pgofp.envs.factory import make_pool
from pgofp.controllers.factory import build_controllers
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/controller_screening/inverted_double_pendulum_20261111'
OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'protocol.json').exists(),'Existing assessment must not be overwritten'
N=20;BASE=50000
cfg=compose(['env=inverted_double_pendulum','prior=mpc','prior.controller_config=inverted_double_pendulum_mpc','logging=local','seed=0'])
pool=make_pool(cfg,1,BASE,'screening');controllers,_=build_controllers(cfg,pool.spec,seed=0);pool.close();ctrl=controllers[0]
protocol={'created_utc':datetime.now(timezone.utc).isoformat(),'env':cfg.env.id,'episodes_per_method':N,'environment_seed_range':[BASE,BASE+N-1],'controller_seed':0,'methods':['random','zero','mpc'],'no_learning':True,'max_steps':1000,'deployment_action_limit':.999,'gymnasium_version':gym.__version__,'mujoco_version':mujoco.__version__,'python':platform.python_version(),'controller':OmegaConf.to_container(cfg.prior.controller,resolve=True),'model_access':'public observation only; clipped velocities; independent planning model','cost':'environment distance+velocity penalty plus 10 per absorbing planned-fall step; no extra action penalty','source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ['src/pgofp/envs/mujoco/planning.py','src/pgofp/controllers/mpc.py','scripts/screen_double_pendulum_mpc.py']}}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));(OUT/'config_resolved.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
results={};rng=np.random.default_rng(BASE)
with (OUT/'episodes.jsonl').open('x',buffering=1) as log:
 for method in protocol['methods']:
  env=gym.make(cfg.env.id);episodes=[];started=time.time()
  try:
   for seed in range(BASE,BASE+N):
    obs,_=env.reset(seed=seed);ret=0.;n=0;fallback=0;clip=0;latencies=[];qstats=[];us=[];distance_penalty=velocity_penalty=0.;before=ctrl.model_rollouts
    while True:
     if method=='random':action=rng.uniform(-1,1,(1,)).astype(np.float32)
     elif method=='zero':action=np.zeros(1,np.float32)
     else:
      t=time.perf_counter();pred=ctrl.predict(obs[None],np.zeros((1,0)),{});latencies.append(time.perf_counter()-t)
      if bool(pred.valid[0,0]):action=np.clip(pred.center_norm[0,0],-.999,.999).astype(np.float32)
      else:fallback+=1;action=np.zeros(1,np.float32)
     clip+=int(np.any(np.abs(obs[5:8])>=10))
     obs,reward,terminated,truncated,info=env.step(action);ret+=float(reward);n+=1;us.append(float(action[0]))
     qstats.append([np.arctan2(obs[1],obs[3]),np.arctan2(obs[2],obs[4])]);distance_penalty-=float(info.get('distance_penalty',0));velocity_penalty-=float(info.get('velocity_penalty',0))
     if terminated or truncated:break
    angles=np.array(qstats)
    rec={'method':method,'seed':seed,'return':ret,'length':n,'terminated':bool(terminated),'truncated':bool(truncated),'full_horizon':bool(truncated and n==1000 and not terminated),'fallback_count':fallback,'fallback_rate':fallback/n,'observation_velocity_clipped_steps':clip,'q1_rms_rad':float(np.sqrt(np.mean(angles[:,0]**2))),'q2_relative_rms_rad':float(np.sqrt(np.mean(angles[:,1]**2))),'mean_action_squared':float(np.mean(np.square(us))),'distance_penalty_sum':distance_penalty,'velocity_penalty_sum':velocity_penalty,'planning_latency_mean_ms':1000*float(np.mean(latencies)) if latencies else 0.,'planning_latency_max_ms':1000*max(latencies,default=0.),'model_control_rollout_steps':ctrl.model_rollouts-before}
    episodes.append(rec);log.write(json.dumps(rec)+'\n');print(f'{method} seed={seed}: return={ret:.2f}, steps={n}, fallback={fallback}',flush=True)
   vals=np.array([r['return'] for r in episodes]);summary={'return_mean':float(vals.mean()),'return_std':float(vals.std()),'length_mean':float(np.mean([r['length'] for r in episodes])),'failure_rate':float(np.mean([r['terminated'] for r in episodes])),'full_horizon_rate':float(np.mean([r['full_horizon'] for r in episodes])),'fallback_rate':sum(r['fallback_count'] for r in episodes)/sum(r['length'] for r in episodes),'total_env_steps':sum(r['length'] for r in episodes),'model_control_rollout_steps':sum(r['model_control_rollout_steps'] for r in episodes),'wall_time_sec':time.time()-started}
   results[method]={'summary':summary,'episodes':episodes};(OUT/'progress.json').write_text(json.dumps(results,indent=2));print('SUMMARY',method,json.dumps(summary),flush=True)
  finally:env.close()
(OUT/'results.json').write_text(json.dumps({'protocol':protocol,'results':results},indent=2));print('Completed',OUT,flush=True)
