"""Run the bounded local-only diagnostic, then generate report even for failed runs."""
from pathlib import Path
import subprocess,sys
from pgofp.experiments.manifest import MatrixManifest
from pgofp.experiments.launcher import Launcher
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'outputs/experiments/halfcheetah_ppo_beta1_noentropy_20k'
try:
    Launcher(MatrixManifest.load(p/'manifest.json'),ROOT,max_parallel=3,max_retries=0,
             script='scripts/train_entropy_diagnostic.py').run()
finally:
    subprocess.run([sys.executable,str(ROOT/'scripts/report_entropy_diagnostic.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/update_experiment_records.py')],cwd=ROOT,check=True)
