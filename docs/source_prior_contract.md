# Source / prior contract

## Layering

`Controller` (PID, CEM-MPC, frozen PPO) → `ControllerPrediction` → `ControllerManager` (per-env state, cache of **frozen** parameters)
→ replay stores `prior_mu / prior_logstd / prior_valid` → `SourceSpec` (immutable snapshot of β, σ, …) → `build_source()` →
`GaussianMixtureSource` (`sample`, `log_prob`). The same controller can serve as baseline, source, warm-up and fallback.

## Distribution (all in the pre-tanh latent z)

```
p_src(z|c) = (1-β) N(0, σ_base² I) + β Σ_m w_m N(μ_m(c), diag σ_m²)
```

- Deterministic controller: `μ = atanh(clamp(a_prior_norm, ±(1-ε)))`, `σ = σ_prior` (default 0.3 latent units). This is a **pre-tanh center
  plus latent perturbation**, not an action-space Gaussian; with σ>0 the action mean differs from the controller output.
- Frozen tanh-Gaussian PPO (`prior.distribution=native`): the exported pre-tanh `(μ, log_std)` are used as-is
  (`prior_std_mode=native`, optional `std_multiplier`) — no extra noise. `center_gaussian`: only the center, explicit `σ_prior`;
  must be labelled `ppo_center_gaussian`. A Gaussian+clip PPO is never a native source (boundary mass).
- Sampling selects a component per sample (Bernoulli/categorical); `log_prob` is the full `logsumexp` mixture density. `β∈{0,1}` take exact
  single-component branches. `σ` is clamped to `[sigma_min, sigma_max] = [0.05, 2.0]`.
- Raw particle (MPPI/CEM) sets are Dirac mixtures without a density: only `priors/particle_source.py:particles_to_components` (positive
  kernel bandwidth) may feed the density path.
- Controller failure (exception, timeout, NaN): the component gets `valid=0`, its weight moves to the base Gaussian, **sampling and
  density see the same fallback distribution**; the rate is logged (`prior/fallback_rate`). Configuration errors raise.

## Where the source is used (must be the same everywhere)

behavior sampling · actor reverse candidates · target-action sampling in the Critic · entropy density · evaluation.
`algorithm.source_scope` (default all three of `behavior/actor/target`) exists only to run the deliberately inconsistent diagnostics (A9);
non-listed places fall back to the Gaussian.

## Stateful controllers

`predict(obs, ref, state)` is pure; `advance(state, obs, ref, applied_command)` runs once per executed step; `reset_slots(mask)` resets only
finished slots. Replay re-queries never touch online states (frozen parameters are stored). Integral states and the previous command that a
controller needs are exposed through `ctrl_feat` (`condition.use_controller_features`). CEM-MPC runs without warm start (fixed query
randomness from `(obs, seed)`); a warm-started MPC would need an extended condition protocol.

## Conditions and what the network sees

`condition.mode=public_state` (default): normalized public observation (+ reference, phase, controller memory if enabled). The prior center/scale
are **not** inputs. `prior_features` additionally feeds `tanh(μ)` and `valid`; pair it with a Gaussian-source twin
(`gaussian_flame_m` + `condition.mode=prior_features`) to separate input information from source replacement (ablation A13).
When the source schedule is not constant, `(β, σ_prior)` are appended to the condition for **all** compared methods (`source_features=auto`).

## Schedules (`priors/schedules.py`)

Clock = `env_transitions / total_steps`. One update uses one immutable `SourceSpec` for actor, critic target, density and alpha. Evaluation
freezes the last emitted spec (`spec_at(None)`). `version` increments when parameters change.

## Prior artifacts (`controllers/artifacts.py`)

`weights.pt, manifest.json (schema, env_id, dims, action chart, content hash), normalizer.npz, controller_config.yaml,
calibration_metrics.json, budget.json`. Loading rejects: missing files, hash mismatch, wrong env/dims, unsupported schema, stateful/recurrent
checkpoints. The frozen PPO prior always uses its **own** stored normalizer.
