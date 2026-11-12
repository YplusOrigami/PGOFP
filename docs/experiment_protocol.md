# Experiment protocol

## Seeds and splits
- flow-training seeds `0–4` (core table), controller calibration seeds `50000+`, in-training evaluation seeds `10000+`, final evaluation seeds
  `20000+`. Prior seeds are `100+`. Splits are disjoint; nothing is tuned on the final-evaluation seeds.
- Stage order per combination: 1 seed link check → 3 seeds screening → ≥ 5 seeds for the core tables.

## Budgets
Report separately: `PGOFP_online_steps`, `prior_pretrain_steps`, `total_steps`, `prior_gpu_hours`, model-rollout steps (MPC), density NFE, wall-clock.
Both views are produced: "new samples given an existing prior" and "including prior preparation". Every method gets the same number of
hyper-parameter trials; failed runs are kept and counted.

## Statistics
Training seed = independent unit. Eval episodes are first averaged within a seed; the CI is a percentile bootstrap over seeds
(`evaluation/metrics.py:bootstrap_ci`). Main numbers: tail mean of the last 20 % of evaluations, return AUC, max drawdown, failure rate;
steps-to-threshold uses `censored` for runs that never reach the (pre-registered) threshold — never the total budget. `final` checkpoint is the
headline; `best` is reported separately. Raw curves are always kept; smoothing windows are declared in advance and used for display only.
Cross-task comparison is per task (no averaging of Reacher and HalfCheetah returns).

## Fairness rules
- Same task version, evaluation protocol, network sizes, `K_train`, temperature, β/σ for all source types in a comparison.
- `gaussian_flame_m` shares the code path with `prior_flame_m`; `source.kind=gaussian` is forced by composition (conflicts raise).
- A1 uses PPO `native` and `center_gaussian` as separately named sources.
- Prior quality tiers (good/medium/poor) come only from **independent calibration seeds** (`scripts/calibrate_controller.py`, `assign_tiers`).
- controller-only, noisy-source-only (zero-velocity flow) and PGOFP are all run with the same prior artifact.

## Matrices (`configs/experiment/*.yaml`, expanded by `scripts/launch_experiments.py`)
`--dry-run` writes `outputs/experiments/<name>/manifest.json` with every planned/skipped combination and config hash and starts nothing;
`--execute` runs the pending ones (bounded parallelism, separate processes, one retry only for infrastructure exits, never for NaN failures).
`analyze_results.py` writes `summary_seed_level.csv`, `summary.csv`, `failures.json` and figures.

| Id | Question | Config |
|---|---|---|
| A0/A1 | Gaussian base chain; source type | `mujoco_main`, `source_type` |
| A2 | prior itself vs flow improvement | `validation_small` pattern: controller_only / noisy_source_only / PGOFP |
| A3/A4 | β, σ | `prior_ratio`, `schedule_time` |
| A5 | prior quality | `prior_quality` |
| A6/A7 | annealing, time sampling, flow_ratio | `schedule_time` |
| A8–A10 | no-Q, no-entropy, N_est, K, `K_local`, source placement, best-of-K | `flame_components` |
| A12 | prior seeds / held-out | `robustness` |
| A13 | input information vs source replacement | `flame_components` (paired `prior_features`) |
| A11 | MPC oracle vs observation-only | not separate: identical for the supported tasks (see environment contract) |
