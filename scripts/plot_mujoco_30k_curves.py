"""Plot one beta-comparison learning curve per environment/prior, without training."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
from summarize_mujoco_30k import BETAS, COLORS, DEFAULT_INPUT, NAMES, PAIRS, ROOT, collect


def plot_curves(source, output):
    rows, audit = collect(source)
    output.mkdir(parents=True, exist_ok=True)
    curve_data = []
    files = []
    plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2})
    for env, prior in PAIRS:
        selected = [r for r in rows if (r["env"], r["prior"]) == (env, prior)]
        fig, ax = plt.subplots(figsize=(11, 7))
        failed_groups = []
        for beta, color in zip(BETAS, COLORS):
            group = sorted([r for r in selected if r["beta"] == beta], key=lambda r: r["seed"])
            completed = sum(r["status"] == "completed" for r in group)
            failed = completed != 3
            measured = defaultdict(list)
            for run in group:
                for point in run["curve"]:
                    measured[point["step"]].append({"seed": run["seed"], "return": point["return"]})
            steps = sorted(measured)
            values = [mean(p["return"] for p in measured[step]) for step in steps]
            label = f"beta = {beta:g}"
            if beta == 0:
                label += " (Gaussian source)"
            if failed:
                label += " (failed; partial data)" if steps else " (failed; no evaluation)"
                failed_groups.append(f"beta={beta:g}: {completed}/3 completed; stop steps = "
                                     + ", ".join(f"s{r['seed']}:{r['actual_train_steps']:,}" for r in group))
            ax.plot(steps, values, color=color, lw=2.3, marker="x" if failed else "o",
                    ms=8 if failed else 6, ls="--" if failed else "-", label=label)
            if failed:
                for step, value in zip(steps, values):
                    ax.annotate(f"n={len(measured[step])}/3", (step, value), xytext=(8, 10),
                                textcoords="offset points", color=color, fontsize=10, va="bottom")
            if not failed:
                assert steps == [10000, 20000, 30000]
                assert abs(values[-1] - mean(r["final_return_30k"] for r in group)) < 1e-8
            curve_data.append({"env": env, "prior": prior, "beta": beta, "completed": completed,
                               "failed": 3 - completed,
                               "stop_steps": {str(r["seed"]): r["actual_train_steps"] for r in group},
                               "points": [{"step": step, "return_mean": value, "n": len(measured[step]),
                                           "seed_returns": measured[step],
                                           "return_seed_sd": stdev(p["return"] for p in measured[step])
                                           if len(measured[step]) > 1 else None}
                                          for step, value in zip(steps, values)]})
        title = f"{NAMES[env]}-v5 / {prior.upper()}"
        ax.set(title=title, xlabel="Training environment steps", ylabel="Evaluation return (higher is better)",
               xlim=(0, 31500), xticks=[0, 5000, 10000, 15000, 20000, 25000, 30000])
        ax.set_xticklabels(["0", "5k", "10k", "15k", "20k", "25k", "30k"])
        ax.set_title(title, fontsize=17, fontweight="bold", pad=15)
        ax.margins(y=.12)
        ax.legend(loc="best", fontsize=10, framealpha=.95)
        # Reacher's failed beta=1 scores can compress all completed curves into a narrow strip.
        # Keep the complete range in the main axes and add a clearly labelled zoom of solid lines.
        completed_series = [c for c in curve_data if c["env"] == env and c["prior"] == prior and c["completed"] == 3]
        failed_values = [p["return_mean"] for c in curve_data if c["env"] == env and c["prior"] == prior and c["failed"] for p in c["points"]]
        completed_values = [p["return_mean"] for c in completed_series for p in c["points"]]
        if env == "reacher" and failed_values and (
                max(completed_values + failed_values) - min(completed_values + failed_values)
                > 4 * (max(completed_values) - min(completed_values))):
            inset = ax.inset_axes([.49, .34, .46, .36])
            for c in completed_series:
                inset.plot([p["step"] for p in c["points"]], [p["return_mean"] for p in c["points"]],
                           color=COLORS[BETAS.index(c["beta"])], marker="o", ms=3, lw=1.5)
            inset.set(title="Zoom: completed groups", xticks=[10000, 20000, 30000], xlim=(9500, 30500))
            inset.set_xticklabels(["10k", "20k", "30k"])
            inset.tick_params(labelsize=8)
            inset.set_title("Zoom: completed groups", fontsize=10)
            inset.grid(alpha=.2)
        footer = "Each solid line = mean of 3 training seeds (0/1/2); each seed evaluated over 10 episodes.\n"
        if failed_groups:
            footer += "Dashed line = available-seed mean before failure; n may change. No extension or zero filling.\n"
            footer += "\n".join(failed_groups)
        else:
            footer += "All 15 runs completed. Raw evaluations at 10k/20k/30k; no smoothing or extrapolation."
        fig.text(.09, .025, footer, ha="left", va="bottom", fontsize=9, color="#475569", linespacing=1.5)
        fig.tight_layout(rect=(0, .16 if failed_groups else .12, 1, 1))
        name = f"{NAMES[env]}-v5_{prior.upper()}.png"
        fig.savefig(output / name, dpi=240)
        plt.close(fig)
        files.append((env, prior, name))
    (output / "curve_data.json").write_text(json.dumps({"audit": audit, "curves": curve_data},
                                                       ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# MuJoCo 30k 实验训练曲线", "",
             "本轮批次：`mujoco_30k_a100_v1`。每个环境/先验一张图，共 10 张；每个 β 一条曲线。", "",
             "## 读图口径", "",
             "- 横轴为 Flow 训练的环境步数，单 run 预算 30k；纵轴为随机策略评估平均 return，越高越好。",
             "- 正常完成的组：每条实线为训练 seeds=0/1/2 的评估均值，每个 seed 每次评估 10 回合。只画均值线，不加误差带。",
             "- 原始评估在 10k/20k/30k；30k 重复评估取最后一次。没有初始 Flow 回报测量，不补 0 步分数，不平滑、不外推。",
             "- 失败组：虚线与 × 仅画当时有评估记录的 seed 的均值，并逐点标出 n；参与 seed 可变化，不能视为固定三 seed 曲线或完整 30k 成绩。",
             "- 失败且没有任何评估记录的 β 只显示图例，不造曲线；图下注明各 seed 停止步数。尤其 Hopper/PPO 的 β=1 三个 seed 均未到首次评估。",
             "- 每张图采用独立纵轴范围。直接 PPO/控制器基线请对照人读报告，本组图片只比较 β 曲线。", "",
             "## 图片", "", "| 环境 | 先验 | 图片 |", "|---|---|---|"]
    for env, prior, name in files:
        lines.append(f"| {NAMES[env]}-v5 | {prior.upper()} | [{name}]({name}) |")
    lines += ["", "数据与统计索引：[curve_data.json](curve_data.json)。含每个点的参与 seed、原始分数、均值与失败状态。", "",
              "完整报告：[实验总结](../docs/experiment_reports/mujoco_30k_a100_v1/实验总结.md)。", "",
              "## 离线重画", "", "在项目根目录运行，不训练、不联网：", "", "```bash",
              ".venv/bin/python scripts/plot_mujoco_30k_curves.py", "```", ""]
    (output / "README.md").write_text("\n".join(lines))
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "实验结果图")
    args = parser.parse_args()
    files = plot_curves(args.input.resolve(), args.output.resolve())
    print(f"Wrote {len(files)} learning-curve figures to {args.output}")


if __name__ == "__main__":
    main()
