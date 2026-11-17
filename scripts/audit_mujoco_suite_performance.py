"""Read the finished suite's timestamps and timers; never train or contact a server."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/experiments/mujoco_30k_a100_v1"
OUTPUT = ROOT / "outputs/analysis/mujoco_30k_a100_v1/performance_audit.json"


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def timestamp(utc):
    return datetime.fromisoformat(utc.replace("Z", "+00:00")).timestamp()


def main():
    matrix = read(SOURCE / "manifest.json")
    intervals, records = [], []
    stage_totals, metric_totals = Counter(), Counter()
    grouped = defaultdict(list)
    for rec in matrix["runs"]:
        start, finish = timestamp(rec["started_utc"]), timestamp(rec["finished_utc"])
        intervals.extend([(start, 1), (finish, -1)])
        run = next((SOURCE / "runs").glob(rec["run_key"] + "__*"))
        meta = read(run / "manifest.json")
        journal = jsonl(run / "progress_events.jsonl")
        metrics = jsonl(run / "metrics.jsonl")
        baseline = read(run / "evaluation/prior_baseline.json")
        stages = Counter(process_setup=journal[0]["updated_at"] - start,
                         process_teardown=finish - journal[-1]["updated_at"])
        for previous, current in pairwise(journal):
            delta = current["updated_at"] - previous["updated_at"]
            assert delta >= 0
            stages[previous["stage"]] += delta
        assert abs(sum(stages.values()) - (finish - start)) < 1e-6
        # Manifest timestamps have 1-second precision; sub-second boundary residuals may be negative.
        counters = Counter(child_sec=finish - start, env_steps=meta["summary"]["env_transitions"],
                           gradient_steps=meta["summary"]["gradient_steps"],
                           runner_summary_sec=meta["summary"]["wall_time_sec"],
                           prior_baseline_sec=baseline["wall_time_sec"])
        for row in metrics:
            for phase in ("collect", "update", "eval", "checkpoint"):
                counters["logged_" + phase + "_sec"] += row.get("timing/" + phase + "_sec", 0)
            for prefix in ("eval", "eval_representative"):
                if prefix + "/return_mean" in row:
                    counters["flow_eval_env_steps"] += row[prefix + "/length_mean"] * row[prefix + "/episodes"]
        for prefix in ("prior_eval", "prior_eval_deterministic"):
            if prefix in baseline:
                counters["prior_eval_env_steps"] += sum(e["length"] for e in baseline[prefix]["episodes"])
        fallback = [r["prior/fallback_rate"] for r in metrics if "prior/fallback_rate" in r]
        record = {"run_key": rec["run_key"], "env": rec["axes"]["env"], "prior": rec["axes"]["prior"],
                  "beta": rec["axes"]["source.beta"], "seed": rec["seed"], "status": rec["status"],
                  "run_dir": str(run.relative_to(ROOT)), "started_utc": rec["started_utc"],
                  "finished_utc": rec["finished_utc"], "stage_seconds": dict(stages),
                  "counters": dict(counters), "torch_threads": meta["provenance"]["torch_threads"],
                  "max_logged_window_fallback_rate": max(fallback, default=None)}
        records.append(record)
        grouped[record["env"], record["prior"]].append(record)
        stage_totals.update(stages)
        metric_totals.update(counters)
    concurrency = peak = 0
    for _, delta in sorted(intervals, key=lambda x: (x[0], x[1])):
        concurrency += delta
        peak = max(peak, concurrency)
    span = max(t for t, _ in intervals) - min(t for t, _ in intervals)
    pairs = []
    for (env, prior), selected in grouped.items():
        stages, counters = Counter(), Counter()
        for rec in selected:
            stages.update(rec["stage_seconds"])
            counters.update(rec["counters"])
        pairs.append({"env": env, "prior": prior, "stage_seconds": dict(stages), "counters": dict(counters),
                      "median_completed_child_minutes": median(r["counters"]["child_sec"] / 60
                                                                for r in selected if r["status"] == "completed")})
    registry = read(SOURCE / "prior_registry.json")
    new_prior_seconds = 0
    for env, prior in registry.items():
        if env != "half_cheetah":
            path = ROOT / prior["path"].split("/PGOFP/", 1)[1]
            new_prior_seconds += read(path / "budget.json")["wall_time_sec"]
    result = {"source": str(SOURCE.relative_to(ROOT)), "audit_date_shanghai": "2026-11-18",
              "maximum_concurrent_flow_runs": peak, "flow_span_seconds": span,
              "serial_between_run_gap_seconds": span - metric_totals["child_sec"] if peak == 1 else None,
              "stage_seconds": dict(stage_totals), "counters": dict(metric_totals),
              "new_five_prior_budget_seconds": new_prior_seconds,
              "notes": ["Progress journals label host wall-clock phases, not exclusive CPU/GPU compute.",
                        "Logged timing/* are host timers with sync_device_timing=false and omit unflushed tails.",
                        "No GPU utilization series, per-process memory peaks or vGPU compute quotas were recorded."],
              "pairs": pairs, "runs": records}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("Maximum concurrency:", peak)
    print("Flow hours:", round(span / 3600, 3))
    print("Stage hours:", {k: round(v / 3600, 3) for k, v in stage_totals.items()})
    print("New prior budget hours:", round(new_prior_seconds / 3600, 3))
    print("Evidence:", OUTPUT)


if __name__ == "__main__":
    main()
