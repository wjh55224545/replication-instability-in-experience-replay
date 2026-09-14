"""
论文实验 · 图表生成（期刊规范：95% CI + 显著性图例 + Times New Roman）(viz.py)
==============================================================================
生成符合学术期刊出版标准的英文图表（输出到 data/figures/）：
1. A/B 对照（random vs semantic 的 Δ(B−A)，95% CI 误差棒 + 显著性图例）
2. 收敛曲线（95% CI 误差棒）
3. 消融结果（95% CI 误差棒）

运行（在本仓库根目录）：python scripts/paper/viz.py
"""

from __future__ import annotations
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
FIG_DIR = os.path.join(DATA_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 学术期刊规范：Times New Roman
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.0

C_RANDOM = "#C44E52"
C_SEMANTIC = "#4C72B0"

SIG_NOTE = "Error bars: 95% CI.  ns: p \u2265 .05,  *: p < .05,  **: p < .01,  ***: p < .001"


def load(name):
    with open(os.path.join(DATA_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def ci95(x):
    """95% 置信区间半宽（t 分布）。"""
    x = np.array(x, dtype=float)
    n = len(x)
    if n < 2:
        return 0.0
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.975, n - 1)
    return t * se


def delta_mean_ci(results):
    """从结果列表提取 Δ(B−A) 的 mean 与 95% CI 半宽。"""
    d = [r["B"]["mean"] - r["A"]["mean"] for r in results
         if r["A"]["mean"] is not None and r["B"]["mean"] is not None]
    return np.mean(d), ci95(d), d


def pval_delta(deltas):
    """单样本 t 检验（H0: delta=0），返回单边 p 值。"""
    if len(deltas) < 2:
        return 1.0
    t, p = stats.ttest_1samp(deltas, 0.0)
    return p / 2  # 单边


def sig_marker(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def annotate_sig(ax, x, y, err, p, h=0.04):
    top = y + err + h
    ax.text(x, top, sig_marker(p), ha="center", va="bottom", fontsize=12, fontweight="bold")


def fig_ab():
    rand = load("ab_results.json")["results"]
    semi = load("ab_semantic_results.json")["results"]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))

    # 左图：总体 Δ(B−A)
    ax = axes[0]
    m_r, e_r, d_r = delta_mean_ci(rand)
    m_s, e_s, d_s = delta_mean_ci(semi)
    ax.bar([0, 1], [m_r, m_s], yerr=[e_r, e_s], capsize=5, error_kw=dict(lw=1.2),
           color=[C_RANDOM, C_SEMANTIC], alpha=0.85, width=0.5)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random\nsampling", "Semantic\nretrieval"])
    ax.set_ylabel("\u0394(B\u2212A) quality score")
    ax.set_title("(a) Overall effect")
    ax.set_ylim(-0.8, 0.35)
    annotate_sig(ax, 0, m_r, e_r, pval_delta(d_r))
    annotate_sig(ax, 1, m_s, e_s, pval_delta(d_s))

    # 右图：分风险等级
    ax = axes[1]
    levels = ["green", "yellow", "red"]
    x = np.arange(len(levels))
    w = 0.36
    for i, lv in enumerate(levels):
        rr = [r for r in rand if r["risk"] == lv]
        ss = [r for r in semi if r["risk"] == lv]
        m_r, e_r, d_r = delta_mean_ci(rr)
        m_s, e_s, d_s = delta_mean_ci(ss)
        ax.bar(x[i] - w/2, m_r, w, yerr=e_r, capsize=4, error_kw=dict(lw=1.1),
               color=C_RANDOM, alpha=0.85, label="Random" if i == 0 else None)
        ax.bar(x[i] + w/2, m_s, w, yerr=e_s, capsize=4, error_kw=dict(lw=1.1),
               color=C_SEMANTIC, alpha=0.85, label="Semantic" if i == 0 else None)
        annotate_sig(ax, x[i] - w/2, m_r, e_r, pval_delta(d_r), h=0.02)
        annotate_sig(ax, x[i] + w/2, m_s, e_s, pval_delta(d_s), h=0.02)
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Green", "Yellow", "Red"])
    ax.set_xlabel("Risk level")
    ax.set_ylabel("\u0394(B\u2212A) quality score")
    ax.set_title("(b) Effect by risk level")
    ax.legend(frameon=False, fontsize=11)
    ax.set_ylim(-1.05, 0.45)

    # 显著性图例
    fig.text(0.99, 0.01, SIG_NOTE, ha="right", va="bottom", fontsize=9, style="italic")

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(FIG_DIR, "fig_ab_comparison.png")
    fig.savefig(out, dpi=300)
    print(f"[viz] Figure 1 -> {out}")
    plt.close(fig)


def fig_convergence():
    if not os.path.exists(os.path.join(DATA_DIR, "convergence_results.json")):
        print("[viz] skip Figure 2 (no convergence data)")
        return
    conv = load("convergence_results.json")
    rounds = conv["rounds"]
    xs = [r["round"] for r in rounds]
    ys = [r["mean"] for r in rounds]
    errs = [ci95(r.get("scores", [])) for r in rounds]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=5, ecolor="black", elinewidth=1.2,
                color=C_SEMANTIC, linewidth=1.5)
    ax.set_xlabel("Feedback round")
    ax.set_ylabel("Mean report quality")
    ax.set_title("Convergence over feedback rounds")
    ax.set_xticks(xs)
    ax.grid(alpha=0.3, linewidth=0.5)
    fig.text(0.99, 0.01, "Error bars: 95% CI", ha="right", va="bottom", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(FIG_DIR, "fig_convergence.png")
    fig.savefig(out, dpi=300)
    print(f"[viz] Figure 2 -> {out}")
    plt.close(fig)


def fig_ablation():
    if not os.path.exists(os.path.join(DATA_DIR, "ablation_results.json")):
        print("[viz] skip Figure 3 (no ablation data)")
        return
    abl = load("ablation_results.json")["results"]

    groups = {
        "top-k": [r for r in abl if r["name"].startswith("top_k")],
        "min_rating": [r for r in abl if r["name"].startswith("min_rating")],
        "injection position": [r for r in abl if r["name"].startswith("position")],
    }

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
    for ax, (gname, items) in zip(axes, groups.items()):
        labels = []
        means = []
        errs = []
        for r in items:
            labels.append(r["name"].split("=")[1])
            means.append(r["mean"])
            errs.append(ci95(r.get("scores", [])))
        ax.bar(labels, means, yerr=errs, capsize=4, error_kw=dict(lw=1.1), color=C_SEMANTIC, alpha=0.85)
        ax.set_xlabel(gname)
        ax.set_ylabel("Mean quality")
        ax.set_ylim(7.3, 8.5)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.text(0.99, 0.01, "Error bars: 95% CI", ha="right", va="bottom", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(FIG_DIR, "fig_ablation.png")
    fig.savefig(out, dpi=300)
    print(f"[viz] Figure 3 -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    fig_ab()
    fig_convergence()
    fig_ablation()
