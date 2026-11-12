"""Evaluate existing Reacher controllers without learning or gain search, on calibration seeds only."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/pgofp-matplotlib')
from pathlib import Path
from dataclasses import asdict
import json,time
import numpy as np
from omegaconf import OmegaConf
from pgofp.config.compose import compose
from pgofp.envs.factory import make_pool
from pgofp.controllers.factory import build_controllers
from pgofp.controllers.manager import ControllerManager
from pgofp.evaluation.evaluator import Evaluator,summarize
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/controller_screening/reacher_20261111';OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'results.json').exists(), 'Results exist; use a new output directory for another evaluation'
N=20;BASE=50000;results={}
for method in ['random','zero','pid','mpc']:
 cfg=compose(['env=reacher','prior=mpc' if method=='mpc' else 'prior=pid','logging=local','seed=0'])
 pool=make_pool(cfg,1,BASE,'controller_screening');ctrls=[]
 if method in ('pid','mpc'):ctrls,_=build_controllers(cfg,pool.spec,seed=0)
 manager=ControllerManager(ctrls,1,pool.spec.action_dim);rng=np.random.default_rng(BASE)
 if method=='random':policy=lambda ctx:rng.uniform(-1,1,(1,pool.spec.action_dim)).astype(np.float32)
 elif method=='zero':policy=lambda ctx:np.zeros((1,pool.spec.action_dim),np.float32)
 else:policy=lambda ctx:np.tanh(ctx['prior_mu'][:,0]).astype(np.float32)
 print('Starting',method,flush=True);start=time.time()
 try:
  episodes=Evaluator(pool,manager,pool.spec).run(policy,N,BASE)
  summary=summarize(episodes)
  summary['prior_fallback_rate_mean']=float(np.mean([e.extras.get('prior/fallback_rate',0) for e in episodes]))
  summary['wall_time_sec']=time.time()-start
  summary['model_rollout_steps']=sum(getattr(c,'model_rollouts',0) for c in ctrls)
  results[method]={'summary':summary,'episodes':[asdict(e) for e in episodes],'controller':OmegaConf.to_container(cfg.prior.controller,resolve=True) if ctrls else None,'controller_seed':0 if ctrls else None}
  (OUT/f'{method}_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
  (OUT/'progress.json').write_text(json.dumps(results,indent=2));print(method,json.dumps(summary),flush=True)
 finally:pool.close()
(OUT/'results.json').write_text(json.dumps({'env':'Reacher-v5','episodes':N,'environment_seeds':[BASE,BASE+N-1],'purpose':'Independent calibration screening, no learning or tuning; not 20 training seeds','results':results},indent=2))
print('Saved',OUT/'results.json',flush=True)
