# Reproduction notes

## FLAME upstream
`flame_reference` is **not implemented**: it needs a pinned checkout of the official repository (github.com/lzqw/FLAME) and a recorded
commit. The adapter (`algorithms/flame_reference.py`) refuses to start without `algorithm.params.upstream_commit` and
`upstream_path`, and then raises `BackendUnavailableError` — it never silently falls back to a floating `main` or to this repository's own
implementation. The same-implementation control is `gaussian_flame_m`.

No upstream results are claimed. License and commit of the upstream code are to be recorded here when the reference run is set up.

## Deviations of this repository's FLAME-M-style algorithm from the local design notes (`Online-FLAME设计.md`)
| Local note | Implemented |
|---|---|
| single time variable, plain FM regression | MeanFlow average velocity `u(z,r,t;c)`, JVP target (`flow_ratio` of diagonal samples) |
| interpolation noise→action but subtraction at inference | one convention: `t=0` action, `t=1` source, `z_out = ε − u(ε,0,1)` |
| weight `exp(Q/α)` on a generated endpoint | conditional reverse-kernel candidates + per-condition SNIS weights (`objectives.py`); endpoint-IS only as validation objective |
| density: independent noise integrated forward, then attached to another action | density of the **queried action** via inverse-direction augmented ODE (`density/continuous_proxy.py`) |
| `argmax Q` over `N_act` candidates as the max-ent behavior | single source sample + one flow forward; best-of-K is a separate tagged variant (`K_exec`) |
| hard clip after the policy | tanh chart; entropy in `a_norm` coordinates with the tanh log-Jacobian |
| `done` for both termination and time limit | `terminated`/`truncated` kept; `bootstrap_mask = 1 − terminated` |
| EMA `ρ = 0.995` | `target_tau = 0.005` weight on the new parameters |
| alpha update sign in the example | `L_α = log α · sg(H − H_target)` (tested for direction) |
| (addition, not in the notes) `K_local` MIS candidates | variance reduction of the same conditional target, default 16, `K_local=0` is spec-literal (see `algorithm_contract.md` §2.1) |
| (addition) sinusoidal time embedding with `max_freq=4` | the first attempt (`max_freq=64`) made the JVP term explode in isolated actor training; lower frequency was stable |

## Not done / not claimed
- No result in this repository is a formal experiment result. Run budgets used while developing are validation-sized.
- MPS is not a supported device (JVP / density numerics not validated there).
- AMP is not used.
