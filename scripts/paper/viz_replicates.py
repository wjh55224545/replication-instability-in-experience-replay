"""
论文图 · 同配置多次运行的效应分布 (viz_replicates.py)
==================================================
双面板森林图：M1（Lingshu-32B）与 M2（DeepSeek chat）各 5 次独立运行，
配置完全相同（池 45 / 每格重复 3 次 / random 条件），仅随机种子不同。

2026-09-12 重绘（审稿意见：图 4 清晰度不足、坐标轴标注混乱）：
  - 由单面板改为**双面板**，两个模型并排，直接呈现「M1 符号翻转、M2 符号恒定」
  - y 轴标签简化（原来每个刻度都重复 "pool 45, 3 reps"，冗长且拥挤）
  - p 值统一右对齐于固定列，避免与误差条重叠
  - 提高分辨率（400 dpi）、放大字号、统一网格与边框

输出 figures/fig_replicates.pdf / .png

运行（在本仓库根目录）：python scripts/paper/viz_replicates.py
"""

from __future__ import annotations
import os
import json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "figures")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.linewidth": 0.9,
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# (面板标题, 模型说明, 文件列表)
PANELS = [
    ("(a) M1: Lingshu-32B", "Lingshu-32B",
     [f"lingshu_replicate_run{i}.json" for i in range(1, 6)]),
    # M2 的前两次运行保存在另外两个文件名里（见 analyze_replicates.py 的 RUNS 表），
    # 必须一并列出，否则面板只显示 3 次，与 Table 2 的 5 次不一致。
    ("(b) M2: DeepSeek chat", "DeepSeek chat",
     ["deepseek_bigpool_results.json",      # run 1
      "deepseek_diversity_results.json",    # run 2
      "deepseek_replicate_run2.json",       # run 3
      "deepseek_replicate_run3.json",       # run 4
      "deepseek_replicate_run4.json"]),     # run 5
]


def stats_for(path: str, key: str = "random"):
    """返回 (Δ, 95%CI 半宽, p)。"""
    d = json.load(open(path, encoding="utf-8"))
    rs = [r for r in d["results"]
          if r.get("baseline_mean") is not None and r.get(f"{key}_mean") is not None]
    if len(rs) < 4:
        return None
    B = np.array([r["baseline_mean"] for r in rs])
    X = np.array([r[f"{key}_mean"] for r in rs])
    df = X - B
    try:
        _, p = stats.wilcoxon(df)
        p /= 2
    except Exception:                                       # noqa: BLE001
        p = float("nan")
    se = df.std(ddof=1) / np.sqrt(len(df))
    t = stats.t.ppf(0.975, len(df) - 1)
    return df.mean(), t * se, p


def main():
    # 收集两个面板的数据
    panels = []
    for title, model, files in PANELS:
        rows = []
        for i, f in enumerate(files, 1):
            p_ = os.path.join(DATA, f)
            if not os.path.exists(p_):
                print(f"  (缺失 {f})")
                continue
            s = stats_for(p_)
            if s:
                rows.append((f"Run {i}", *s))
        if rows:
            panels.append((title, model, rows))

    if not panels:
        print("无可用数据。")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(7.6, 3.5), sharex=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (title, model, rows) in zip(axes, panels):
        y = np.arange(len(rows))[::-1]
        for i, (label, m, ci, pv) in enumerate(rows):
            sig = pv < 0.05
            color = "#C44E52" if sig else "#4C72B0"
            ax.errorbar(m, y[i], xerr=ci, fmt="o", capsize=3.5, markersize=6,
                        color=color, ecolor="0.35", elinewidth=1.1,
                        markeredgewidth=0.8)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_title(title, fontsize=11, pad=7)
        ax.grid(axis="x", alpha=0.22, linewidth=0.5)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

        # 统一 x 轴范围，两面板可直接对照。
        # 右侧留够空间给 p 值列（否则最右侧的 p 值会压到误差条上）。
        ax.set_xlim(-0.17, 0.30)
        # 顶部留白，供汇总注释使用（放在底部会与 Run 5 的误差条重叠）
        ax.set_ylim(-0.8, len(rows) - 0.1)

        # p 值右对齐在固定列
        PX = 0.295
        for i, (label, m, ci, pv) in enumerate(rows):
            sig = pv < 0.05
            ax.text(PX, y[i], f"{pv:.3f}" + ("*" if sig else ""),
                    fontsize=8.5, va="center", ha="right",
                    color="#C44E52" if sig else "0.35",
                    fontweight="bold" if sig else "normal")

        # 汇总注释放在顶部空白处
        deltas = [r[1] for r in rows]
        n_sig = sum(1 for r in rows if r[3] < 0.05)
        ax.text(0.02, 0.03,
                f"range = {max(deltas) - min(deltas):.3f}   "
                f"significant {n_sig}/{len(rows)}",
                transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
                color="0.32")

        ax.set_xlabel("$\\Delta$ quality vs. no-replay baseline")

    axes[0].set_ylabel("Independent run")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_replicates.{ext}"), dpi=400)
        print(f"[viz] fig_replicates.{ext}")

    for title, model, rows in panels:
        d = [r[1] for r in rows]
        print(f"  {model}: Δ 范围 {min(d):+.3f} ~ {max(d):+.3f} "
              f"(极差 {max(d)-min(d):.3f})，显著 {sum(1 for r in rows if r[3]<0.05)}/{len(rows)}")
    plt.close(fig)


if __name__ == "__main__":
    main()
