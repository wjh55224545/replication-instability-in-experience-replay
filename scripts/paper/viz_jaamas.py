"""
JAAMAS 期刊版补充图 (viz_jaamas.py)
===================================
生成两张图：
  Fig1.pdf  —— 五智能体流水线架构（2026-09-22 拆图后只保留架构）
  Fig2.pdf  —— 经验回放回路，并标注三个失效模式 F1/F2/F3
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

# 输出目录：项目内写到 jaamas_journal/；复现包内写回自己的 figures/。
# 坑（2026-09-15）：复现包位于项目**内部**，故不能用「三层上跳是否存在」判别
# ——那条路径在包里同样成立，图会被写到包**外面**去。改用「同级是否存在 figures/」判别。
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_FIGS = os.path.join(_HERE, "..", "..", "figures")
_PROJ = os.path.join(_HERE, "..", "..", "..", "jaamas_journal")
OUT = _PKG_FIGS if os.path.isdir(_PKG_FIGS) else _PROJ

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


def fig_loop():
    """图 2 · 经验回放回路，并把 F1/F2/F3 三个失效点标在发生的位置。

    第八份评审建议把原图 1 拆成「系统架构 + 回放回路」两幅，理由是信息密度偏高。
    拆分后第二幅不只是简化版，而是**承载新信息**：三个协调失效模式各自发生在
    回路的哪一段，读者一眼能看到「失败出在组件的接缝处」。
    """
    # 布局刻度（xlim 0–12.4）：
    #   Report 方框  x 3.8–6.3, y 3.70–4.55
    #   store  方框  x 0.5–3.1, y 0.70–1.65
    #   rating 方框  x 4.3–6.9, y 0.70–1.65
    #   回放虚线走 y=5.05（在 Report 上方），记录实线走 y=2.50（在两者之间）
    #   —— 首版把回放线放在 y=4.37，**横穿了 Report 方框**，且图例压在 rating 方框上。
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 5.6); ax.axis("off")

    def box(x, y, w, h, label, fc, fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.09",
                                    linewidth=1.0, edgecolor="0.25", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs)

    REP, STO, RAT = "#F7E3D0", "#FBE9E7", "#FBE9E7"
    box(3.8, 3.70, 2.5, 0.85, "Report agent", REP)
    box(0.5, 0.70, 2.6, 0.95, "Experience\nstore", STO)
    box(4.3, 0.70, 2.6, 0.95, "Human rating\n(1–10 scale)", RAT)

    ORANGE = "#B5651D"

    def seg(p, q, ls="-", lw=1.15, head=False):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle=("-|>" if head else "-"),
                                     mutation_scale=11, linewidth=lw,
                                     color=ORANGE, linestyle=ls))

    # ① 记录生成的报告：Report 底部(x=4.3) → 下 → 左 → 下进 store 顶(x=1.8)
    seg((4.3, 3.70), (4.3, 2.50))
    seg((4.3, 2.50), (1.8, 2.50))
    seg((1.8, 2.50), (1.8, 1.65), head=True)
    ax.text(3.08, 2.58, "record generated report", fontsize=8.2, color=ORANGE,
            ha="center", va="bottom")

    # ② 评分：Human rating 左 → store 右
    seg((4.30, 1.18), (3.10, 1.18), head=True)
    ax.text(3.70, 1.30, "rating", fontsize=8.2, color=ORANGE, ha="center", va="bottom")

    # ③ 回放：store 左 → 绕左侧上行 → 横穿 y=5.05 → 下进 Report 顶(x=5.05)
    seg((0.50, 1.18), (0.15, 1.18), ls="--", lw=1.2)
    seg((0.15, 1.18), (0.15, 5.05), ls="--", lw=1.2)
    seg((0.15, 5.05), (5.05, 5.05), ls="--", lw=1.2)
    seg((5.05, 5.05), (5.05, 4.55), ls="--", lw=1.2, head=True)
    ax.text(2.60, 5.13, "retrieve & inject exemplars", fontsize=8.2, color=ORANGE,
            ha="center", va="bottom")

    # ---- 三个失效点 ----
    def mark(x, y, tag):
        ax.text(x, y, tag, fontsize=8.0, color="#B03A2E", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.22", facecolor="white",
                          edgecolor="#B03A2E", linewidth=0.9))

    mark(4.30, 3.08, "F1")     # 写键与读键由不同组件决定 —— 就在这条记录路径的接缝上
    mark(3.70, 1.18, "F2")     # 评分进入库时的过滤语义
    mark(0.15, 3.10, "F3")     # 检索对「示例为何有用」的那个属性视而不见

    ax.text(7.15, 2.55,
            "F1  the key a record is written\n"
            "       under is not the key it is\n"
            "       read by\n"
            "F2  a missing rating is admitted\n"
            "       as unknown quality\n"
            "F3  retrieval ranks by nothing\n"
            "       that makes an exemplar useful",
            fontsize=7.8, color="0.28", ha="left", va="center")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"Fig2.{ext}"), dpi=300, bbox_inches="tight")
    print(f"[viz] Fig2.{'pdf'}")
    plt.close(fig)


def fig_pipeline():
    """图 1 · 五智能体流水线的系统架构（不含回放回路，回路见 Fig2）。"""
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

    # ---- 经验库与回放回路移到 Fig2 ----
    # 第八份评审：「Fig. 1 协议图的信息密度偏高，可考虑拆分为『系统架构 + 回放环路』两幅」。
    # 拆分后本图只画流水线本身；回路连同三个失效点见 Fig2（fig_loop）。
    ax.text(0.35, 2.30,
            "Report generation is the artifact under study; perception, analysis and alert\n"
            "components are held fixed and are not evaluated here.\n\n"
            "The experience-replay loop that closes through the store, and the three\n"
            "coordination failure modes it contains, are shown separately in Figure 2.",
            fontsize=8.2, color="0.35", ha="left", va="top")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"Fig1.{ext}"), dpi=300)   # 官方命名规范 Fig<编号>
    print(f"[viz] Fig1.{'pdf'}")
    plt.close(fig)


def fig_poolsize():
    """池子大小 × 模型的效应对比。"""
    # p 值为**双尾**（2026-09-15 全稿由单尾改双尾，此处同步）
    data = {
        ("M1", 15): (-0.048, 0.4419, -0.062, 0.1297),
        ("M1", 45): (-0.106, 0.0441, -0.090, 0.0168),
        ("M2", 15): (0.147, 0.0005, 0.046, 0.2410),
        ("M2", 45): (0.030, 0.2729, 0.012, 0.7465),
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
    fig_pipeline()   # 图 1 · 流水线架构
    fig_loop()       # 图 2 · 回放回路 + 三个失效点
    fig_poolsize()
