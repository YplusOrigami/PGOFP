"""Generate the agent handoff record for the completed MuJoCo matrix, offline only."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from summarize_mujoco_30k import (
    BETAS,
    NAMES,
    PAIRS,
    REPORT_DIR,
    ROOT,
    collect,
    read_json,
    shanghai_time,
    summarize,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers),
            *["| " + " | ".join(map(str, row)) + " |" for row in rows]]


def number(value):
    return "—" if value is None else f"{value:,.2f}"


def write_record(source, asset, ident):
    """Return the compact ledger section; put full evidence in EXP-NNN/实验记录.md."""
    rows, audit = collect(source)
    combos, _ = summarize(rows)
    matrix = read_json(source / "manifest.json")
    matrix_runs = {r["run_key"]: r for r in matrix["runs"]}
    asset.mkdir(parents=True, exist_ok=True)
    config_dir = asset / "configs"
    config_dir.mkdir(exist_ok=True)

    def link(path, label=None):
        return f"[{label or path.name}]({Path(os.path.relpath(path, asset)).as_posix()})"

    human = REPORT_DIR / "实验总结.md"
    analysis = ROOT / "outputs/analysis/mujoco_30k_a100_v1"
    text = [f"# {ident} · mujoco_30k_a100_v1（Agent 接续记录）", "",
            "实验日期以原始日志为准；本记录于 2026-11-18 补齐。自动生成统计与证据索引，研究解释保留在 ../notes.md。", "",
            "## 状态与入口", "",
            f"- 批次已结束：150 个 run，129 completed，21 failed_numerical；没有运行中或待启动项。运行编号 `{matrix['matrix_id']}`。",
            f"- Flow 矩阵：{audit['matrix_created_shanghai']} → {audit['matrix_finished_shanghai']}，均为北京时间。先验制备在矩阵之前。",
            f"- 入口配置：{link(ROOT / 'configs/experiment/mujoco_prior_beta_30k.yaml')}；启动器：{link(ROOT / 'scripts/run_mujoco_suite.py')}。本记录不启动训练。",
            f"- 运行清单：{link(source / 'manifest.json')}；服务器验收：{link(source / 'verification.json')}；先验 registry：{link(source / 'prior_registry.json')}。",
            f"- 人读报告与 5 张综合图：{link(human, '实验总结')}；机器可读统计：{link(analysis / 'analysis.json')}。",
            f"- 本记录附带 150 份完整解析配置、{link(asset / 'run_index.json', 'run_index.json')} 和 10 张逐 seed 图。编号只表示本地索引，不是新增实验。", "",
            "## 配置、预算与统计口径", "",
            "PPO 六任务；PID 为 InvertedPendulum/Reacher；MPC 为 Reacher/InvertedDoublePendulum（cem_mpc）。每组合 β=0/0.25/0.5/0.75/1，训练 seed=0/1/2。",
            "单环境、Flow 预算 30k、前 5k uniform_action 热身、UTD=1、batch=256；Actor/双 Critic [256,256,256]；K_train/local/exec/eval=32/16/1/1；α=0.1，Critic 熵项开启，Heun 密度估计 5 步。",
            "`prior_flame_m`，public_state，不加 controller features；源为概率混合而非动作插值。β=0 是本实现高斯源对照，不是已完成官方 FLAME 复现。奖励未改写。",
            "PPO 使用 native pre-tanh 分布，PID/MPC latent σ=0.3，base σ=1。每个 run 第 0 步前评估直接先验 10 回合；PPO 另评估均值动作。所有 run 均有本地 JSONL/TensorBoard，WandB disabled。",
            "主指标为完整 30k run 的最后一次随机策略评估（10 回合，环境 seeds=10000–10009），不是 best_eval_return。10k/20k/30k 加一次最终 30k 评估；同一步取最后一条。",
            "组统计为 3 个训练 seed 的均值±样本 SD（ddof=1）；先验 SD 为回合总体 SD。AUC 仅覆盖已测 10k–30k。失败样本终点为空、不补零，最后有效评估单列。",
            f"实际 Flow 交互 {audit['actual_train_steps']:,}/4,500,000 步，梯度更新 {audit['actual_gradient_steps']:,} 次。六个 PPO 先验各 1,001,472 步（seed100，共享冻结）；本轮新训五个，共 5,007,360 步，HalfCheetah 复用历史先验。", "",
            "## 已核对结果（50 组）", "",
            "每行 n=3 计划 run；失败组没有 30k 终点均值。详细差值与配对改善数见人读报告及审计 CSV。", ""]
    text += table(["环境 / 先验", "β", "完成/计划", "30k return ± seed SD", "直接先验均值", "10k–30k AUC"],
                  [[f"{NAMES[c['env']]} / {c['prior']}", f"{c['beta']:g}", f"{c['completed']}/3",
                    f"{number(c['return_mean'])} ± {number(c['return_seed_sd'])}" if c['completed'] else "数值失败",
                    number(c['prior_return_mean']), number(c['auc_mean'])] for c in combos])

    text += ["", "## 先验身份与本地路径", "",
             "服务器绝对路径保存在原始配置中，不改写；在本地读取时将 `/inspire/hdd/project/fdu-weilong/public/PGOFP/` 前缀替换为当前 checkout 根目录。", "",
             "PPO content hash 按现有 artifact 协议对五个内容文件复算，与 artifact manifest 和 registry 三方一致；只读文件，不加载策略权重执行。", ""]
    registry = read_json(source / "prior_registry.json")
    prior_table = []
    priors = {}
    required = ("weights.pt", "normalizer.npz", "controller_config.yaml", "calibration_metrics.json", "budget.json")
    for env, rec in registry.items():
        local = ROOT / rec["path"].split("/PGOFP/", 1)[1]
        digest = hashlib.sha256()
        for name in sorted(required):
            assert (local / name).is_file(), local / name
            digest.update(name.encode())
            digest.update(sha256(local / name).encode())
        actual = digest.hexdigest()
        assert actual == rec["content_hash"] == read_json(local / "manifest.json")["content_hash"]
        priors[env] = {**rec, "local_path": str(local.relative_to(ROOT)), "verified_content_hash": actual}
        prior_table.append([NAMES[env], link(local, str(local.relative_to(ROOT))), str(rec["prior_seed"]),
                            f"{rec['actual_env_steps']:,}", f"`{actual}`"])
    text += table(["PPO 环境", "本地 artifact", "seed", "实际制备步数", "完整 content hash"], prior_table)
    text += ["", "PID 与 CEM-MPC 的参数完整保存在每份解析 YAML 的 `prior.controller` 中，版本见下述 run_index 的 controller_versions；不是 PPO 权重 artifact。", "",
             "## 全部 150 个 run 的接续索引", "",
             "编号对应 configs/runNNN.yaml。每行链接原始 run 目录，JSON 索引另列原始配置、日志、事件、逐回合评估、先验基线、数值失败文件和配置 SHA-256。", ""]
    run_table = []
    index = []
    for i, row in enumerate(rows, 1):
        run = ROOT / row["run_dir"]
        rec = matrix_runs[row["run_key"]]
        meta = read_json(run / "manifest.json")
        snapshot = config_dir / f"run{i:03d}.yaml"
        shutil.copyfile(run / "config_resolved.yaml", snapshot)
        config_sha = sha256(snapshot)
        assert config_sha == sha256(run / "config_resolved.yaml")
        files = {name: str((run / name).relative_to(ROOT)) for name in (
            "manifest.json", "config_resolved.yaml", "metrics.jsonl", "events.jsonl",
            "evaluation/episodes.csv", "evaluation/prior_baseline.json", "numerical_failure.json") if (run / name).is_file()}
        assert all(name in files for name in ("manifest.json", "metrics.jsonl", "events.jsonl", "evaluation/prior_baseline.json"))
        tensorboard = [str(f.relative_to(ROOT)) for f in run.rglob("events.out.tfevents.*") if not f.name.startswith("._")]
        assert tensorboard, run
        index.append({"record_number": i, **{k: v for k, v in row.items() if k != "curve"},
                      "config_snapshot": str(snapshot.relative_to(ROOT)), "config_file_sha256": config_sha,
                      "recorded_config_hash": meta["config_hash"], "controller_versions": meta.get("controller_versions", []),
                      "provenance": meta["provenance"], "files": files, "tensorboard_files": tensorboard,
                      "started_shanghai": shanghai_time(rec["started_utc"]),
                      "finished_shanghai": shanghai_time(rec["finished_utc"]),
                      "exit_code": rec["exit_code"], "retries": rec["retries"]})
        last = f"{row['last_eval_step']}:{number(row['last_eval_return'])}" if row["last_eval_step"] is not None else "未到首次评估"
        run_table.append([link(snapshot, f"{i:03d}"), link(run, f"{row['env']} / {row['prior']}"), f"{row['beta']:g}",
                          row["seed"], meta["outcome"], f"{row['actual_train_steps']:,}", number(row["final_return_30k"]), last])
    (asset / "run_index.json").write_text(json.dumps({"experiment_id": ident, "audit": audit, "priors": priors,
                                                   "runs": index}, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    text += table(["配置编号", "原始 run（环境/先验）", "β", "seed", "状态", "实际步数", "30k 最终回报", "最后已测步数:回报"], run_table)

    text += ["", "## 逐 seed 原始曲线", "",
             "每个任务/先验独立绘图，每个 β 一个子图，每条线为一个训练 seed；不平滑、无均值阴影、不补点、不外推。× 为数值失败 run 的最后已测点，失败停止步数写在图例中，未到首次评估的 run 只有图例。", ""]
    for env, prior in PAIRS:
        fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), squeeze=False)
        pair = [r for r in rows if (r["env"], r["prior"]) == (env, prior)]
        for ax, beta in zip(axes.ravel(), BETAS):
            selected = sorted([r for r in pair if r["beta"] == beta], key=lambda r: r["seed"])
            for row, color in zip(selected, ("#2563eb", "#16a34a", "#dc2626")):
                label = f"seed={row['seed']}"
                failed = row["status"] != "completed"
                if failed:
                    label += f", failed @{row['actual_train_steps']}"
                xs = [p["step"] / 1000 for p in row["curve"]]
                ys = [p["return"] for p in row["curve"]]
                ax.plot(xs, ys, marker="o", color=color, label=label, ls=":" if failed else "-")
                if failed and xs:
                    ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=65, zorder=4)
            ax.axhline(pair[0]["prior_return_mean"], color="#7c3aed", ls="--", lw=1, label="Direct prior")
            ax.set(title=f"beta={beta:g} ({sum(r['status'] == 'completed' for r in selected)}/3 completed)",
                   xlabel="Environment steps (thousands)", ylabel="Stochastic return", xlim=(8, 31), xticks=[10, 20, 30])
            ax.grid(alpha=.2)
            ax.legend(fontsize=7, loc="best")
        axes.ravel()[-1].axis("off")
        fig.suptitle(f"{ident}: {NAMES[env]} / {prior.upper()} — individual training seeds")
        fig.tight_layout(rect=(0, 0, 1, .95))
        figure = asset / f"{env}_{prior}_seeds.png"
        fig.savefig(figure, dpi=150)
        plt.close(fig)
        text += [f"### {NAMES[env]} / {prior.upper()}", "", f"![{env}/{prior} 逐 seed 曲线]({figure.name})", ""]

    text += ["## 数值失败与不可误读项", "",
             "21 个失败均 β=1、flow_training 阶段、`non-finite values in critic loss`；这只是直接退出检查，未定位首个异常根因。失败时间/实际步数不能替代完成预算；原服务器汇总失败行的 final_tail_return 可能只是中途评估。", ""]
    text += table(["环境 / 先验", "seed", "停止步数", "证据"],
                  [[f"{r['env']} / {r['prior']}", r["seed"], f"{r['actual_train_steps']:,}",
                    link(ROOT / r["run_dir"] / "numerical_failure.json")] for r in rows if r["status"] == "failed"])
    text += ["", "## 下一位 Agent 必读", "",
             "1. 本批结果已结束，不要为补记录重新启动、重试或续训。数值失败保留；以后新实验另建编号和目录。旧 EXP-001–009 不覆盖。",
             "2. 同一先验的 15 次直接评估复用相同种子，不能当作 15 个独立先验样本。Reacher/PID 与 MPC 的 β0 曲线相同，不能合并成 6 个独立高斯 seed。",
             "3. PPO 包装器构造先于 Agent，会消耗全局 Torch RNG；本轮只按任务/先验组内 β 配对，不宣称跨先验初始权重相同。",
             "4. 147 个 run 源码 commit=4e154ad 且 dirty=true，3 个未记录 Git；结果数据同步 commit=e8f7036。dirty 可能包含被跟踪的 outputs，缺少源码 diff，不能断言是干净的算法快照。具体 provenance 见 run_index。",
             "5. 已核对每个成功 run 的 30k 随机回报与 JSONL、最终十回合 CSV、服务器汇总一致；server verification 整体 false 来自 21 个失败，129 个成功 run 的记录验收均 true。",
             "6. 最优 β 为本轮事后选择；仅三训练 seed、固定十个评估环境 seed，不做显著性或通用优越性结论。全部 PPO 任务最佳组均值低于直接先验；Reacher/MPC β=.75 为较一致正例，双摆 MPC β=1 高方差。",
             "7. eval/failure_rate 是环境 terminated，不是训练数值失败率，也不通用于安全性/任务成功率。β1 仍有先验随机性，不能写成无噪声。",
             "8. vGPU 额度与物理设备名称不同；日志显示 A100-SXM4-80GB，不能据此写成独占 80GB。未采集 GPU 利用率时间序列，不能推导 GPU 加速倍数。",
             "9. Flow checkpoint/replay 按 Git 规则未同步。本地可统计日志、核对冻结先验，不能据此宣称已重评训练策略。原始 YAML 保留服务器路径，恢复前要另取 checkpoint 并处理路径。",
             "10. 优先诊断 β1 首个异常、隔离初始化 RNG，再在固定 β 的独立 seeds/评估种子上确认正例；不要把先验制备或 MPC 模型访问成本漏出样本效率核算。", "",
             "## 离线更新", "", "在项目根目录运行；只读原始结果并重建派生文档，不训练、无网络：", "",
             "```bash", "PYTHONPATH=src .venv/bin/python scripts/summarize_mujoco_30k.py",
             ".venv/bin/python scripts/update_experiment_records.py", "```", "",
             "本批次专项生成器为 scripts/experiment_record_mujoco_30k.py，由总记录生成器调用。run_index 中 SHA-256 为完整配置文件字节哈希，recorded_config_hash 为训练系统记录的配置身份，两者不可混为同一个算法。", ""]
    (asset / "实验记录.md").write_text("\n".join(text))
    # Paths in this compact section are relative to docs/experiment_records/实验记录.md.
    return ["", f"## {ident} · {source.name}", "",
            "服务器正式 30k 扫描：129/150 完成，21 个数值失败全部 β=1；已结束，无待运行项。", "",
            f"完整逐 run/逐 seed 记录、150 份配置快照及 10 张独立曲线见 [{ident}/实验记录.md]({ident}/实验记录.md)。",
            "人读总结与 5 张综合图见 [实验总结](../experiment_reports/mujoco_30k_a100_v1/实验总结.md)；原始目录为 [mujoco_30k_a100_v1](../../outputs/experiments/mujoco_30k_a100_v1/)。", "",
            "不要按任务名混合不同先验、β 或失败 run 的终点；失败分数只列最后已测值。统计和解释边界见专项记录与 notes.md。", ""]
