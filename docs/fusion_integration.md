# Fusion (HFM) integration

**Status: interface + mock only. Nothing here has been run against the real HFM simulator.**

## What exists
`envs/fusion/`: `FusionBackend` protocol (`describe/reset/advance/get_reference/get_snapshot/set_snapshot`), `ObservationSchema` +
`ObservationAssembler` (named fields, units, explicit missing masks), `ReferenceTrack` (hash, look-ahead permission), `CoilMap` + `ActuatorWrapper`
(explicit delay/rate-limit/calibration ownership), `SafetyFilter` (reports every intervention), `MockFusionBackend` (6-D, 3 tracked
quantities, delay, rate limit, divergence termination, snapshots) and `FusionEnv` (7-D command → 12-D voltages → backend).
`env=fusion_stub` trains end-to-end on the mock (`tests/integration/test_fusion_contract.py`).

## What is deliberately NOT filled in (all `verified: false`, real backend refuses to run)
| Item | File | Needed |
|---|---|---|
| 64-D observation field order/units | `configs/fusion/observation_schema.yaml` | SDK documentation |
| 7→12 matrix, channel order, per-channel limits | `configs/fusion/coil_map.yaml` | device documents (the proposal's "7 components ↔ main coil + 4 up/down pairs" must be matched against the real matrix) |
| F1/F2 horizons, terminal semantics, shots split | `configs/fusion/tasks.yaml` | task spec (300/500 steps are provisional) |
| Who applies delay/rate limit/PSM | `ActuatorOwnership` | SDK capability; exactly one layer may own each stage |
| SDK factory | `env.kwargs.sdk_factory='module:function'` | must return a `FusionBackend` |

`env.backend=fusion` fails config validation while any of the three YAMLs is unverified. No matrix, index or unit has been invented.

## Integration steps
1. Implement a `FusionBackend` adapter around the SDK (named observation fields, masks, reward terms, termination reasons, optional snapshots).
2. Fill and verify the three YAMLs; the coil map hash goes into the run manifest.
3. Decide terminal semantics: a task that naturally ends at step 300/500 should usually be `terminated` with the remaining time in the
   condition (`phase`), not a bootstrapping time limit.
4. Run `tests/integration/test_fusion_contract.py` with the real adapter swapped in (mock-only assertions marked there), then a short real smoke.
5. Measure end-to-end latency (`scripts/benchmark_latency.py`) including the prior queries; target: the 10 ms control period, reported as
   p50/p95/p99 and deadline-miss rate. No "meets 10 ms" statement is made before this measurement.
6. If a safety filter is used, density/entropy describe the pre-filter policy (`density_before_filter`); compare filter on/off as a separate experiment.
