"""
论文图 2 · 效应不稳定性 (viz_instability.py)
============================================
十五组受控比较的效应量 ± 95% CI。

2026-09-13 重绘（第四份评审意见：「十五个比较的置信区间重叠严重，读者难以快速识别关键对比；
建议拆分为两个子图并标注关键对比」）：
  - 由单面板 15 行改为**按模型分双面板**（M1 6 行 / M2 9 行）——
    行距放宽，且 M1 偏负、M2 偏正的整体对照一眼可见
  - **标注三处关键对比**：两个单次运行方向相反、池大小比较、同配置第二次运行
  - 两面板共享 x 轴与图例，配色沿用（单次=红 / 三次重复=蓝 / 第二次运行=橙）

输出 jaamas_journal/fig_instability.pdf / .png

运行：cd 代码 && D:/python/python.exe scripts/paper/viz_instability.py
"""

from __future__ import annotations
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
# 坑（2026-09-12）：原输出到已废弃的 docs/aamas/，而论文用的是 jaamas_journal/ 下的同名文件
# —— 重跑本脚本**不会**更新论文里的图，改动会静默丢失。
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "jaamas_journal")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10.5,
    "axes.linewidth": 0.9,
    "axes.labelsize": 11,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
})

# (标签, 文件, 条件, 样式)；样式: single / rep / run2
# 分面板：M1 六行，M2 九行
PANELS = [
    ("(a) M1: Lingshu-32B", [
        ("single run           random",  "ab_results.json",                None,      "single"),
        ("single run           matched", "ab_semantic_results.json",       None,      "single"),
        ("3 reps  pool 15  random",      "ab_repeated_results.json",       "random",  "rep"),
        ("3 reps  pool 15  matched",     "ab_repeated_results.json",       "semantic", "rep"),
        ("3 reps  pool 45  random",      "lingshu_bigpool_results.json",   "random",  "rep"),
        ("3 reps  pool 45  matched",     "lingshu_bigpool_results.json",   "semantic", "rep"),
    ]),
    ("(b) M2: DeepSeek chat", [
        ("single run           random",  "deepseek_ab_random.json",        None,      "single"),
        ("single run           matched", "deepseek_ab_semantic.json",      None,      "single"),
        ("3 reps  pool 15  random",      "deepseek_repeated_results.json", "random",  "rep"),
        ("3 reps  pool 15  matched",     "deepseek_repeated_results.json", "semantic", "rep"),
        ("3 reps  pool 45  random",      "deepseek_bigpool_results.json",  "random",  "rep"),
        ("3 reps  pool 45  matched",     "deepseek_bigpool_results.json",  "semantic", "rep"),
        ("3 reps  pool 45  run 2  random",  "deepseek_diversity_results.json", "random",    "run2"),
        ("3 reps  pool 45  run 2  matched", "deepseek_diversity_results.json", "similar",   "run2"),
        ("3 reps  pool 45  run 2  other-tier", "deepseek_diversity_results.json", "different", "run2"),
    ]),
]

COLOR = {"single": "#C44E52", "rep": "#4C72B0", "run2": "#DD8452"}


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as f:
        return json.load(f)


def paired_stats(path, cond=None):
    """返回 (mean, ci95_halfwidth, p, n)。cond=None 表示单次 A/B 格式。"""
    d = load(path)
    rs = d["results"]
    if cond is None:
        pairs = [(r["A"]["mean"], r["B"]["mean"]) for r in rs
                 if r["A"]["mean"] is not None and r["B"]["mean"] is not None]
    else:
        pairs = [(r["baseline_mean"], r[f"{cond}_mean"]) for r in rs
                 if r.get("baseline_mean") is not None and r.get(f"{cond}_mean") is not None]
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    diff = b - a
    n = len(diff)
    se = diff.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    try:
        _, p = stats.wilcoxon(diff)            # 双尾；2026-09-15 起不再 /2
    except Exception:
        p = float("nan")
    return diff.mean(), t * se, p, n


def main():
    # ⚠️ 两面板必须**等宽且 x 轴一致**：否则同样长度的误差条在两边的视觉长度不同，
    # 读者会误判效应量大小。（首版用 width_ratios=[6,9]，正是这个问题。）
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharex=True)

    for ax, (title, rows) in zip(axes, PANELS):
        ys = np.arange(len(rows))[::-1]
        for y, (label, f, cond, style) in zip(ys, rows):
            m, ci, p, n = paired_stats(os.path.join(DATA, f), cond)
            ax.errorbar(m, y, xerr=ci, fmt="o", capsize=2.5,
                        color=COLOR[style], ecolor="0.35",
                        elinewidth=0.9, markersize=5.0, zorder=3)
            if p < 0.05:
                ax.text(m + ci + 0.014, y, "*", ha="left", va="center",
                        fontsize=12, color=COLOR[style], zorder=4)

        ax.axvline(0, color="black", linestyle="--", linewidth=0.9, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[0] for r in rows], fontsize=9)
        ax.set_ylim(-1.5, len(rows) - 0.45)
        ax.set_xlim(-0.35, 0.35)
        ax.grid(axis="x", alpha=0.22, linewidth=0.5, zorder=0)
        ax.set_title(title, fontsize=11, pad=7)
        ax.set_xlabel("$\\Delta$ in report quality vs. no-replay baseline")


    # ---- 标注三处关键对比（评审要求「标注关键对比」）----
    # 全部放在坐标区**内部**，且不进入顶部留白 —— 首版把注释放在 y 超出轴范围处，
    # 结果与面板标题叠在一起。
    a0, a1 = axes

    # ① 两个单次运行方向相反（各自面板第一行）
    a0.annotate("single run:\nnegative", xy=(-0.172, 5), xytext=(-0.33, 3.95),
                fontsize=8.0, color="#C44E52", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#C44E52", linewidth=0.8,
                                shrinkA=0, shrinkB=3))
    a1.annotate("single run:\npositive", xy=(0.147, 8), xytext=(0.185, 5.55),
                fontsize=8.0, color="#C44E52", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#C44E52", linewidth=0.8,
                                shrinkA=0, shrinkB=3))

    # ② 池大小：复现后无差别（放在 M1 面板下半部的空白处）
    a0.text(-0.33, -0.90, "pool 15 vs pool 45 indistinguishable\n"
                          "after replication (Mann–Whitney\n$p$ = 0.42, 1.00)",
            fontsize=7.6, color="0.30", ha="left", va="center")

    # ③ 同配置第二次运行
    a1.annotate("same configuration,\nsecond run", xy=(0.092, 2), xytext=(0.155, 4.05),
                fontsize=8.0, color="#DD8452", ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color="#DD8452", linewidth=0.8,
                                shrinkA=0, shrinkB=3))

    handles = [plt.Line2D([], [], color=COLOR[k], marker="o", linestyle="none", label=v)
               for k, v in [("single", "single run"), ("rep", "3 repetitions"),
                            ("run2", "3 repetitions (run 2)")]]
    fig.legend(handles=handles, frameon=False, fontsize=8.8, ncol=3,
               loc="lower center", bbox_to_anchor=(0.5, -0.035))
    fig.text(0.5, -0.115, "* $p<0.05$ (two-tailed Wilcoxon signed-rank);  "
                          "error bars: 95% CI",
             ha="center", va="top", fontsize=8.5, style="italic")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(OUT, f"Fig2.{ext}")   # 官方要求图片按 "Fig<编号>" 命名
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"[viz] {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
