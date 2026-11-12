# Environment contract

- **Standard MuJoCo tasks are unmodified**: observation, reward and termination come from Gymnasium `*-v5`. Perturbed / held-out variants
  (`evaluation/robustness.py`) get their own names; standard results stay separate.
- `terminated` and `truncated` are stored separately. `bootstrap_mask = 1 − terminated`; a truncated step's `next_obs` is the real final
  observation (pre-reset). Tested in `tests/integration/test_vector_terminal.py`.
- Environment pool (`envs/vector.py:SyncEnvPool`) uses **explicit per-slot resets** (no autoreset ambiguity). Only the finished slot is reset;
  its controller state and reference progress are reset with it, other slots are untouched.
- Counters: `env_transitions` (real transitions, the single x-axis), `vector_steps`, `gradient_steps`, `actor_updates`, `critic_updates`.
  UTD is credit-based (`training/update_scheduler.py`), driven by real transitions, never truncated for `utd<1`.
- Normalization: running statistics are updated by the training environment only; every update and every evaluation uses one immutable snapshot.
- RNG: named streams (`utils/seed.py`): training env, behavior source, replay, time sampling, candidates, trace, target source, alpha source,
  eval env, eval policy, prior training. Evaluation never consumes training streams; all stream states are checkpointed.
- Warm-up (`train.warmup_source`): `uniform_action` (default, identical for all sources), `prior`, `mixed`. Warm-up transitions count toward the
  budget; prior pre-training steps are accounted separately.

## Task × prior support (`envs/mujoco/tasks.py`)

| Task | PID | CEM-MPC | PPO | Note |
|---|---|---|---|---|
| InvertedPendulum-v5 | yes | yes | yes | reward saturates (1/step): judge by episode length / failure |
| Reacher-v5 | yes (joint PD + fixed-elbow IK) | yes | yes | target is in the public observation |
| HalfCheetah-v5 | – | – | yes | no useful per-joint PID prior is assumed |
| Hopper-v5 | – | – | yes | healthy termination + time limit kept separate |
| Walker2d-v5 | – | – | yes | extension, not in the default matrix |

Unsupported pairs raise `UnsupportedCombinationError` at config validation and are written to the matrix manifest as `skipped` with the reason.

## MPC model access

`observation_model` only: an independent `MjModel` copy; the planning state is rebuilt from the **public observation** (never from
`env.unwrapped.data`). For these two tasks the public observation fully determines `(qpos, qvel)`, so this equals the oracle; `privileged_sim_state`
is therefore rejected explicitly. Model rollouts are counted (`model_rollouts`) and reported separately from environment interaction.
