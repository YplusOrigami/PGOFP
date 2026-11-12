"""Background PID beta sweep; independent runs, bounded concurrency, local ledger refreshed at the end."""
from pathlib import Path
import subprocess, sys
from pgofp.experiments.launcher import Launcher
from pgofp.experiments.manifest import MatrixManifest
ROOT = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else 'inverted_pendulum_pid_beta_sweep_300k'
p = ROOT / 'outputs/experiments' / name
man = MatrixManifest.load(p / 'manifest.json')
# Start one run for each beta before subsequent seeds.
man.runs.sort(key=lambda r: (r['seed'], r['variant']))
man.save()
try:
    Launcher(man, ROOT, max_parallel=3, max_retries=0).run()
finally:
    subprocess.run([sys.executable, str(ROOT / 'scripts/analyze_results.py'), f'matrix={p / "manifest.json"}'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/update_experiment_records.py')], cwd=ROOT, check=True)
