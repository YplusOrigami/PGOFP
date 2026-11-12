# Algorithm contract (as implemented)

Source of truth for symbols, coordinates, gradient boundaries and estimator semantics. Code locations are given so each
formula maps to exactly one function.

## 1. Coordinates and flow direction

| Symbol | Meaning | Code |
|---|---|---|
| `z ∈ R^d` | pre-tanh flow latent (source samples and flow work here) | – |
| `a_norm = tanh(z)` | normalized command, **Critic input, entropy coordinate** | `math/action_transform.py` |
| `a_phys = b + c ⊙ a_norm` | what the task adapter receives (`b=(hi+lo)/2`, `c=(hi-lo)/2`) | `core/specs.py:ActionSpec` |
| flow time `t=0` | target action | – |
| flow time `t=1` | source sample `ε ~ p_src(·|c)` | – |
| `z_t = (1-t) z_a + t ε` | linear interpolation | `math/reverse_sampling.py` |
| `v = ε − z_a` | instantaneous velocity along increasing `t` | – |
| `u_θ(z,r,t;c)` | average velocity over `[r,t]`, `0≤r≤t≤1` | `networks/mean_flow.py` |
| one-step policy | `z_out = ε − u_θ(ε,0,1;c)`, `a = tanh(z_out)` | `MeanFlowNet.one_step` |

The last layer of `u_θ` is zero-initialized, so at initialization the policy is exactly `tanh(ε)`
(`tests/integration/test_training_smoke.py::test_zero_velocity_init_policy_is_tanh_of_source`).

Log-Jacobian: `log|J_tanh(z)| = Σ 2(log2 − z − softplus(−2z))` (stable; checked at |z|=1e3).
The entropy is defined in `a_norm` coordinates. A physical-coordinate density subtracts `Σ log c`; no 12-D density is ever
computed for the 7→12 fusion expansion.

## 2. Actor objective (`algorithms/objectives.py:actor_loss_reverse`)

For a condition `c`, flow time `t∈[δ,1−δ]` (δ=0.02) and `z_t ~ h = N(0,I)`:

1. candidates `ε_k ~ p_src(·|c)`, `z_a,k = (z_t − tε_k)/(1−t)`, `v_k = ε_k − z_a,k`  (reverse kernel
   `φ(z_a|z_t,c,t) = ((1−t)/t)^d p_src((z_t−(1−t)z_a)/t | c)`; works for any sampleable source, no standard-normal assumption);
2. `logw_k = sg[ min(Q1,Q2)(c, tanh z_a,k)/α_ent + log|J_tanh(z_a,k)| ]`, `w = softmax_k(logw)`  — **per condition only, never over the batch**;
3. MeanFlow target `u*_k = sg[ v_k − (t−r)·(J_z u · v_k + ∂_t u) ]` with the JVP tangent `(v_k, 0, 1)` on `(z, r, t)`
   (`math/meanflow_target.py`, the only implementation);
4. `L_actor = mean_b Σ_k w_bk · mean_dims (u_θ(z_t,r,t;c) − u*_k)²`.

`(r,t)`: `t~U(δ,1−δ)`; with probability `flow_ratio` (0.5) `r=t`, else `r = t·q`, `q~U(0,1)` (`math/time_sampling.py`).

Gradient boundaries: candidates, weights and targets are computed under `no_grad`; only `u_pred` carries gradient; the Critic's
parameters are frozen during the Actor step; the controller/prior is never differentiated.

### 2.1 Variance-reduced default: `K_local` (deviation from the literal spec — read this)

With the pure reverse kernel (`K_local=0`) the SNIS estimate of the posterior-mean velocity has a variance that blows up as
`t→1` (1-D check: at t=0.98, K=16 gives −5.5±5.9 against the exact −0.18; `tests/numerical/test_weighted_transport.py`).
In isolated actor training this shows as loss spikes (up to 1e4) and a policy that stays too wide.

`K_local>0` (default 16) adds candidates `z_a = z_c + σ_local ξ` around `K_local` samples `z_c` of the **current policy** and applies the
balance-heuristic multiple-importance-sampling correction `log φ − log g_mix` (`math/reverse_sampling.py:mixture_reverse_candidates`).
The target posterior is unchanged (unbiased in the limit; verified against the exact 1-D posterior), only the estimator variance changes
(measured: std ≤ 20 % of the pure kernel at t=0.95, and ESS 6.6→21 in the bandit problem). `K_local=0` reproduces the spec-literal
estimator and is kept as the ablation `pure_reverse_kernel_K_local0`. Both settings are written to every run's resolved config.

## 3. Critic (`soft_critic_loss`)

`y = r + γ·bootstrap_mask·( min Q'(c', a') − α_ent·logπ_proxy(a'|c') )`, `a' = tanh(ε' − u_θ'(ε',0,1;c'))`, `ε' ~ p_src(·|c')` with the
**same conditional source** used for acting and the Actor. Everything on the target side is stop-gradient. Target EMA:
`target ← (1−τ)·target + τ·online`, `τ = target_tau = 0.005`. Two independent Q networks, `min` for targets and weights.
`bootstrap_mask = 1 − terminated` (time-limit truncation bootstraps from the true final observation).

## 4. Density / entropy proxy (`density/`)

`continuous_inverse_proxy` (default): integrate `dz/dt = f(z,t) = u(z,t,t;c)` from the **action end `t=0` to the source end `t=1`**
for the queried action, together with `D = ∫ div f dt`; then `log π_z(z_a|c) = log p_src(ε̂|c) + D`, and
`log π_norm = log π_z − log|J_tanh(z_a)|`.
Direction/sign are pinned by analytic affine fields (`tests/unit/test_density.py`; `f=(a+bt)z` ⇒ `z(1)=z₀e^{a+b/2}`, `D=d(a+b/2)`).
Heun with `N_est=5` costs 10 field evaluations (`DensityResult.nfe`); Hutchinson uses Rademacher probes through a VJP
(`v^T J v`), one probe by default, reused along the trajectory.

`DensityResult.is_exact_for_sampled_policy` is **False**: the deployed policy is the one-step map, the density is a continuous-flow
proxy. More steps reduce integration error only; the one-step/continuous gap is measured separately
(`density/diagnostics.py:one_step_gap`, logged as `flow/one_step_gap_*`). Logs call the quantity `entropy/proxy_mean`, never "exact entropy".

`local_meanflow_proxy` (optional): invert the discrete map `T_i(z)=z−h_i u(z,r_i,t_i)` from the queried action, with full-Jacobian
`slogdet`; residuals and failures are reported, global invertibility is **not** assumed.

## 5. Temperature (`algorithms/temperature.py`)

`α_ent = exp(log α)`, `H_hat = mean(−logπ_proxy)` of the **online** actor, `L_α = log α · sg(H_hat − H_target)`, `H_target = −d_a`
(SAC-style start, to be ablated). Entropy below target ⇒ `log α` increases. First experiments use a fixed `α_ent=0.1`.

## 6. Objective variants (`algorithm.actor_objective`)

| value | meaning |
|---|---|
| `reverse_qrmf` | main objective above |
| `reverse_no_q` | Q term set to 0, **keeps** `log|J_tanh|` |
| `reverse_uniform` | all candidate weights equal (different objective; do not call it no-Q) |
| `endpoint_is` | validation path: `a_k~U((−1,1)^d)`, independent `ε_k~p_src`, weights `exp(Q/α)` (known-proposal importance sampling) |

`online_ofql` is a separate class (FBC regression to replay actions + differentiable-Q guidance with `α_q = η/EMA|Q|`) and never reuses `α_ent`.
`gaussian_flame_m` is the same `PriorFlameAgent` class with `source.kind=gaussian` forced by the config composer.

## 7. Known limits

- The prior gives no safety guarantee; controller + Gaussian perturbation is not a safe controller.
- Finite-K SNIS is a biased estimator of the conditional target; the single-step map is not guaranteed invertible.
- No theorem is used for "closer prior ⇒ smaller Jacobian"; displacement and JVP norms are logged and analysed empirically.
