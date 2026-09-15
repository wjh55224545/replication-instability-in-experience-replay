"""
JAAMAS 期刊版补充图 (viz_jaamas.py)
===================================
生成两张图：
  Fig1.pdf  —— 五智能体流水线与反馈回路示意图（官方要求图片按 "Fig<编号>" 命名）
  fig_poolsize.pdf  —— 池子大小与重复次数的效应对比

输出到 jaamas_journal/

运行：cd 代码 && D:/python/python.exe scripts/paper/viz_jaamas.py
"""

from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "jaamas_journal")

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 5.6); ax.axis("off")

    def box(x, y, w, h, label, fc, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    linewidth=1.0, edgecolor="0.25", facecolor=fc))
        ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2, style="-|>", color="0.25", ls="-", lw=1.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=11, linewidth=lw,
                                     color=color, linestyle=ls))

    # ---- 顶层：四个执行智能体 ----
    agents = [("Perception", "#DCE9F7"), ("Analysis", "#DCE9F7"),
              ("Report", "#F7E3D0"), ("Alert", "#E2F0DC")]
    w, h, y_top = 1.85, 0.80, 4.35
    xs = [0.35, 2.50, 4.65, 6.80]
    for (name, fc), x in zip(agents, xs):
        box(x, y_top, w, h, name, fc)
    for i in range(len(xs) - 1):
        arrow(xs[i] + w, y_top + h/2, xs[i+1], y_top + h/2)
    arrow(8.65, y_top + h/2, 9.55, y_top + h/2)
    ax.text(9.62, y_top + h/2, "output", fontsize=8.5, va="center", color="0.35")

    # ---- 中层：编排器 ----
    y_mid = 3.25
    box(0.35, y_mid, 9.20, 0.62, "Orchestrator  (sequencing  +  human-facing trace)", "#EDE3F5", fs=8.5)
    for x in xs:
        arrow(x + w/2, y_mid + 0.62, x + w/2, y_top, color="0.62", ls=":", lw=0.9)

    # ---- 底层：经验库 + 人工评分 ----
    y_bot = 1.85
    box(0.35, y_bot, 2.30, 0.85, "Experience\nstore", "#FBE9E7", fs=8.5)
    box(3.30, y_bot, 2.30, 0.85, "Human rating\n(1–10)", "#FBE9E7", fs=8.5)

    # 生成后记录：从 **Report 方框正下方** 出发 → 经验库
    # 坑（2026-09-12）：原实现起点 x=4.45 落在 Analysis(2.50–4.35) 与 Report(4.65–6.50)
    # 之间的空隙里，画出来像是从 Human rating 方框出发 —— 暗示「人工评分记录报告」，
    # 与图意（报告生成后入库）相反。改为从 Report 中心出发，拐一道进入经验库。
    x_record = 4.65 + w / 2                 # Report 方框中心
    # 垂直预算：编排器下沿 3.25，底排方框上沿 y_bot+0.85=2.70，空档仅 0.55。
    # 箭头线放 3.12，标签紧跟其下（va=top，占约 0.12），到 2.70 仍有余量。
    y_rec = y_bot + 1.27
    ax.add_patch(FancyArrowPatch((x_record, y_rec), (1.50, y_rec),
                                 arrowstyle="-", linewidth=1.1, color="#B5651D"))
    arrow(1.50, y_rec, 1.50, y_bot + 0.85, color="#B5651D")
    ax.text((x_record + 1.50) / 2, y_rec - 0.05, "record generated report",
            fontsize=8, color="#B5651D", ha="center", va="top")

    # 评分进入经验库
    arrow(3.30, y_bot + 0.42, 2.65, y_bot + 0.42, color="#B5651D")

    # ---- 回放回路：经验库 → Report 智能体（从上方绕行，不穿过方框）----
    x_loop = 0.12
    y_above = y_top + h + 0.32          # 顶层方框上方
    x_report = 4.65 + w / 2             # Report 方框中心
    # 经验库左出
    ax.add_patch(FancyArrowPatch((0.35, y_bot + 0.42), (x_loop, y_bot + 0.42),
                                 arrowstyle="-", linewidth=1.1, color="#B5651D",
                                 linestyle="--"))
    # 左侧上行
    ax.add_patch(FancyArrowPatch((x_loop, y_bot + 0.42), (x_loop, y_above),
                                 arrowstyle="-", linewidth=1.1, color="#B5651D",
                                 linestyle="--"))
    # 顶部横穿到 Report 上方
    ax.add_patch(FancyArrowPatch((x_loop, y_above), (x_report, y_above),
                                 arrowstyle="-", linewidth=1.1, color="#B5651D",
                                 linestyle="--"))
    # 向下注入 Report
    arrow(x_report, y_above, x_report, y_top + h, color="#B5651D", ls="--", lw=1.1)
    ax.text(x_report + 0.12, y_above, "retrieve & inject exemplars",
            fontsize=8, color="#B5651D", ha="left", va="center")

    # 底部说明
    ax.text(0.35, 1.30,
            "Report generation is the artifact under study; perception, analysis and alert\n"
            "components are held fixed and are not evaluated here.",
            fontsize=7.8, color="0.35", ha="left", va="top")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"Fig1.{ext}"), dpi=300)   # 官方命名规范 Fig<编号>
    print(f"[viz] Fig1.{'pdf'}")
    plt.close(fig)


def fig_poolsize():
    """池子大小 × 模型的效应对比。"""
    data = {
        ("M1", 15): (-0.048, 0.2209, -0.062, 0.0648),
        ("M1", 45): (-0.106, 0.0221, -0.090, 0.0084),
        ("M2", 15): (0.147, 0.0003, 0.046, 0.1205),
        ("M2", 45): (0.030, 0.1364, 0.012, 0.3733),
    }
    labels = ["M1\npool 15", "M1\npool 45", "M2\npool 15", "M2\npool 45"]
    keys = [("M1", 15), ("M1", 45), ("M2", 15), ("M2", 45)]
    rand = [data[k][0] for k in keys]
    rp = [data[k][1] for k in keys]
    mat = [data[k][2] for k in keys]
    mp = [data[k][3] for k in keys]

    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    b1 = ax.bar(x - w/2, rand, w, label="Random replay", color="#C44E52", alpha=0.85)
    b2 = ax.bar(x + w/2, mat, w, label="Matched replay", color="#4C72B0", alpha=0.85)
    for bars, ps in [(b1, rp), (b2, mp)]:
        for bar, p in zip(bars, ps):
            if p < 0.05:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + (0.012 if bar.get_height() >= 0 else -0.022),
                        "*", ha="center", fontsize=13)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("$\\Delta$ quality vs. no-replay baseline")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_ylim(-0.16, 0.20)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"fig_poolsize.{ext}"), dpi=300)
    print(f"[viz] fig_poolsize.{'pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    fig_pipeline()
    fig_poolsize()
