# -*- coding: utf-8 -*-
"""人类专家评估分析 (analyze_expert_eval.py)
==========================================
数据来源：湖北大学心理学系熊猛教授团队 5 份评分表
    `人类评估材料/“心境”项目评分结果（...）/N. 评分表（...N）.xlsx`

设计：12 份报告（R01–R12，条件对评分者**盲**）× 8 维度 × 5 位评分者。
其中前 5 个维度与论文 LLM judge 的 5 维度**一一对应**：

    专业性 → clinical appropriateness    相关性 → relevance
    共情性 → empathy                     建议可操作性 → actionability
    结构完整性 → completeness

后 3 个（决策支持度 / 信任度 / 认知负荷）是人因维度，LLM 未评，本脚本不使用。

本脚本回答两个问题：
  1. **人类评估者之间**的一致性与水平差异有多大？
     ——若与 LLM 评估者之间的表现同量级，则「评估者不可靠」不是 LLM 特有的缺陷
  2. LLM judge 的打分与人类专家**是否一致**？

⚠️ 诚实性要求：任何结论都必须来自实际计算，不得选择性报告。
    详见 CLAUDE.md / 用户长期指令「不得隐瞒实验结果」。

运行：cd 代码 && D:/python/python.exe scripts/paper/analyze_expert_eval.py
"""

from __future__ import annotations
import glob
import io
import json
import os
import sys

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # noqa: BLE001
    pass

import openpyxl

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
_HERE = os.path.dirname(os.path.abspath(__file__))
# 两种布局都要能跑：项目内（素材放在 人类评估材料/）与复现包（放在 data/expert_eval/）
_CANDS = [
    (os.path.join(ROOT, "人类评估材料",
                  "“心境”项目评分结果（湖北大学心理学系熊猛教授团队评估）"),
     os.path.join(ROOT, "人类评估材料", "llm_scores.json")),
    (os.path.join(_HERE, "..", "..", "data", "expert_eval"),
     os.path.join(_HERE, "..", "..", "data", "expert_eval", "llm_scores.json")),
]
SHEETS, LLM = next(((s, l) for s, l in _CANDS if os.path.isdir(s)),
                   (_CANDS[0][0], _CANDS[0][1]))

# 前 5 个维度与 LLM judge 对齐（顺序即列顺序）
DIMS = ["专业性", "相关性", "共情性", "建议可操作性", "结构完整性"]
LLM_DIMS = ["专业性", "相关性", "共情性", "建议可操作性", "结构完整性"]


def load_raters():
    """返回 (raters, report_ids)，raters 形状 (R, N, D) 的 float 数组。"""
    files = sorted(glob.glob(os.path.join(SHEETS, "*.xlsx")))
    raters, reports = [], None
    for f in files:
        wb = openpyxl.load_workbook(f, data_only=True)
        ws = wb.worksheets[0]
        hdr, data = None, {}
        for row in ws.iter_rows(values_only=True):
            cells = ["" if c is None else str(c).strip() for c in row]
            if cells and cells[0] == "报告编号":
                hdr = cells
                continue
            if hdr and cells and len(cells[0]) == 3 and cells[0].startswith("R"):
                vals = []
                for n, d in enumerate(DIMS, 1):
                    key = f"F{n}.{d}"
                    try:
                        vals.append(float(cells[hdr.index(key)]))
                    except (ValueError, IndexError):
                        vals.append(np.nan)
                data[cells[0]] = vals
        ids = sorted(data)
        reports = reports or ids
        raters.append([data[i] for i in ids])
    return np.array(raters, dtype=float), reports   # (R, N, D)


def icc21(m):
    """ICC(2,1)：双向随机效应、绝对一致性、单评分者。

    m 形状 (n_targets, k_raters)。
    """
    n, k = m.shape
    grand = m.mean()
    ms_r = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)          # 目标间
    ms_c = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)          # 评分者间
    resid = m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return (ms_r - ms_e) / denom if denom else float("nan")


def icc31_consistency(m):
    """ICC(3,1)：双向混合、一致性、单评分者。只反映排序一致，不惩罚水平差。"""
    n, k = m.shape
    grand = m.mean()
    ms_r = k * ((m.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_c = n * ((m.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    resid = m - m.mean(axis=1, keepdims=True) - m.mean(axis=0, keepdims=True) + grand
    ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
    denom = ms_r + (k - 1) * ms_e
    return (ms_r - ms_e) / denom if denom else float("nan")


def main():
    r, reports = load_raters()
    R, N, D = r.shape
    print("=" * 84)
    print("人类专家评估分析（湖北大学心理学系熊猛教授团队）")
    print("=" * 84)
    print(f"评分者 {R} 位 × 报告 {N} 份 × 维度 {D} 个（与 LLM judge 对齐的前 5 维）")

    llm = json.load(io.open(LLM, encoding="utf-8"))

    # ---------------- 1. 评分者的整体水平 ----------------
    print("\n【1】每位评分者的总分均值（水平差异）")
    print(f"{'评分者':<8}{'均值':>8}{'SD':>8}{'最小':>6}{'最大':>6}   （1–10 量表）")
    print("-" * 84)
    means = []
    for i in range(R):
        v = r[i][~np.isnan(r[i])]
        means.append(v.mean())
        print(f"Rater {i+1:<2}{v.mean():>8.2f}{v.std(ddof=1):>8.2f}"
              f"{v.min():>6.0f}{v.max():>6.0f}")
    means = np.array(means)
    print(f"\n  评分者均值的极差 = {means.max() - means.min():.2f} 分"
          f"（最高 {means.max():.2f} / 最低 {means.min():.2f}）")
    print("  对照：论文 §5.5 报告两个 LLM 评分者之间的水平位移为 1.61 分")

    # ---------------- 2. 评分者间一致性 ----------------
    print("\n【2】评分者间一致性")
    # 维度级：把 (报告 × 维度) 展平成 60 个观测
    dim_level = r.transpose(1, 2, 0).reshape(N * D, R)
    pairs = []
    for i in range(R):
        for j in range(i + 1, R):
            a, b = dim_level[:, i], dim_level[:, j]
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() > 3:
                pairs.append(stats.pearsonr(a[ok], b[ok])[0])
    pairs = np.array(pairs)
    print(f"  维度级（n={N*D} 个 报告×维度 观测）："
          f"两两 Pearson r 均值 = {pairs.mean():+.3f}，范围 {pairs.min():+.3f} ~ {pairs.max():+.3f}")
    print(f"  对照：论文 §5.5 报告两个 LLM 评分者维度级 r = +0.55")

    # 报告级：先对每个评分者、每份报告取 5 维均值
    rep_level = np.nanmean(r, axis=2).T          # (N, R)
    rp = []
    for i in range(R):
        for j in range(i + 1, R):
            rp.append(stats.pearsonr(rep_level[:, i], rep_level[:, j])[0])
    rp = np.array(rp)
    print(f"  报告级（n={N} 份报告）：两两 Pearson r 均值 = {rp.mean():+.3f}，"
          f"范围 {rp.min():+.3f} ~ {rp.max():+.3f}")
    print(f"  对照：论文 §5.5 报告两个 LLM 评分者报告级 r = −0.70")

    icc = icc21(rep_level)
    print(f"  ICC(2,1)（报告级、绝对一致性、单评分者）= {icc:+.3f}")

    # ICC(3,1)：一致性（consistency）口径——只问排序是否一致，不惩罚水平差。
    # 论文 §5.5 关心的"rank agreement"对应这个口径，故两者都要报。
    icc31 = icc31_consistency(rep_level)
    print(f"  ICC(3,1)（报告级、一致性、单评分者）  = {icc31:+.3f}")

    # Spearman：秩相关，对水平位移不敏感
    sp = []
    for i in range(R):
        for j in range(i + 1, R):
            sp.append(stats.spearmanr(rep_level[:, i], rep_level[:, j])[0])
    sp = np.array(sp)
    print(f"  报告级 Spearman（两两均值）= {sp.mean():+.3f}，"
          f"范围 {sp.min():+.3f} ~ {sp.max():+.3f}")

    print("\n  各对评分者的报告级一致性明细（Pearson / Spearman）")
    k = 0
    for i in range(R):
        for j in range(i + 1, R):
            a, b = rep_level[:, i], rep_level[:, j]
            print(f"    Rater{i+1}–Rater{j+1}: r = {stats.pearsonr(a, b)[0]:+.3f}   "
                  f"rho = {stats.spearmanr(a, b)[0]:+.3f}   "
                  f"水平差 = {a.mean() - b.mean():+.2f}")
            k += 1

    # 留一法：去掉任一位评分者后 ICC(2,1) 的变化（检查是否由个别极端评分者驱动）
    print("\n  留一敏感性（去掉一位后，其余 4 位的 ICC(2,1)）")
    for i in range(R):
        keep = [j for j in range(R) if j != i]
        print(f"    去掉 Rater{i+1}: ICC = {icc21(rep_level[:, keep]):+.3f}")

    # ---------------- 3. 人类 vs LLM ----------------
    print("\n【3】人类专家 vs 系统自带 LLM judge")
    llm_rep = np.array([np.mean([llm[i]["llm_scores"][d] for d in LLM_DIMS])
                        for i in reports])
    human_rep = np.nanmean(r, axis=(0, 2))       # 每份报告的人类均值
    print(f"{'报告':<6}{'条件':<10}{'人类均值':>10}{'LLM':>8}{'差':>8}")
    print("-" * 84)
    for k, rid in enumerate(reports):
        print(f"{rid:<6}{llm[rid]['condition']:<10}{human_rep[k]:>10.2f}"
              f"{llm_rep[k]:>8.2f}{human_rep[k] - llm_rep[k]:>+8.2f}")

    rr = stats.pearsonr(human_rep, llm_rep)
    print(f"\n  人类均值 vs LLM：r = {rr[0]:+.3f}  (p = {rr[1]:.4f}, n = {N})")
    print(f"  人类评分范围 {human_rep.min():.2f}–{human_rep.max():.2f}"
          f"（SD {human_rep.std(ddof=1):.3f}）")
    print(f"  LLM  评分范围 {llm_rep.min():.2f}–{llm_rep.max():.2f}"
          f"（SD {llm_rep.std(ddof=1):.3f}）")

    # ---------------- 4. 按条件（若人类能分辨条件） ----------------
    print("\n【4】按条件的均值（人类 vs LLM）")
    conds = sorted({llm[i]["condition"] for i in reports})
    for c in conds:
        idx = [k for k, i in enumerate(reports) if llm[i]["condition"] == c]
        print(f"  {c:<10} n={len(idx):<3} 人类 {human_rep[idx].mean():.3f}"
              f"   LLM {llm_rep[idx].mean():.3f}")

    out = {
        "n_raters": int(R), "n_reports": int(N), "dims_used": DIMS,
        "rater_means": means.tolist(),
        "rater_mean_spread": float(means.max() - means.min()),
        "interrater_dim_level_mean_r": float(pairs.mean()),
        "interrater_report_level_mean_r": float(rp.mean()),
        "icc21_report_level": float(icc),
        "human_vs_llm_r": float(rr[0]), "human_vs_llm_p": float(rr[1]),
        "human_rep_scores": {i: float(human_rep[k]) for k, i in enumerate(reports)},
        "llm_rep_scores": {i: float(llm_rep[k]) for k, i in enumerate(reports)},
    }
    outp = os.path.join(_HERE, "..", "..", "data", "expert_eval_analysis.json")
    json.dump(out, io.open(outp, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[保存] {outp}")


if __name__ == "__main__":
    main()
