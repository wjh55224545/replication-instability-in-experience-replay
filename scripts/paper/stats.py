"""
论文实验 · 统计检验 (stats.py)
==============================

读取 ab_results.json，对 A/B 两条件的平均分做：
- 配对 t 检验（scipy.stats.ttest_rel）
- Wilcoxon 符号秩检验（scipy.stats.wilcoxon）
- Cohen's d 效应量（配对）

并可按风险等级分组输出。
运行（在本仓库根目录）：python scripts/paper/stats.py
"""

from __future__ import annotations
import os
import json
import sys

import numpy as np
from scipy import stats

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
AB_PATH = os.path.join(OUT_DIR, "ab_results.json")


def cohens_d_paired(a, b):
    """配对样本 Cohen's d = mean(diff) / std(diff)。"""
    diff = np.array([x - y for x, y in zip(a, b)], dtype=float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def analyze(rows):
    """对一组 {A_mean, B_mean} 行做配对检验，返回统计摘要。"""
    a = [r["A"]["mean"] for r in rows if r["A"]["mean"] is not None]
    b = [r["B"]["mean"] for r in rows if r["B"]["mean"] is not None]
    # 只保留成对的
    pairs = [(r["A"]["mean"], r["B"]["mean"]) for r in rows
             if r["A"]["mean"] is not None and r["B"]["mean"] is not None]
    if len(pairs) < 3:
        return {"n": len(pairs), "note": "样本不足"}
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    t_stat, p_ttest = stats.ttest_rel(b, a)  # 单边备择：B > A
    try:
        w_stat, p_wilcoxon = stats.wilcoxon(b, a)
    except ValueError:
        w_stat, p_wilcoxon = float("nan"), float("nan")
    d = cohens_d_paired(b, a)
    return {
        "n": len(pairs),
        "A_mean": round(float(np.mean(a)), 3),
        "B_mean": round(float(np.mean(b)), 3),
        "delta": round(float(np.mean(b) - np.mean(a)), 3),
        "ttest_rel_p": round(float(p_ttest / 2), 4),  # 单边
        "wilcoxon_p": round(float(p_wilcoxon / 2), 4) if not np.isnan(p_wilcoxon) else None,
        "cohens_d": round(d, 3),
    }


def main():
    if not os.path.exists(AB_PATH):
        print(f"[stats] 找不到 {AB_PATH}，请先运行 run_ab_experiment.py")
        return
    with open(AB_PATH, encoding="utf-8") as f:
        data = json.load(f)
    results = data["results"]
    meta = data.get("meta", {})

    print("=" * 60)
    print(f"[stats] 数据集: n={len(results)}, 种子={meta.get('seed_n')}, 经验池={meta.get('pool_size')}")
    print("=" * 60)

    overall = analyze(results)
    print("[总体]")
    print(json.dumps(overall, ensure_ascii=False, indent=2))

    print("\n[按风险等级分组]")
    for lv in ["green", "yellow", "red"]:
        sub = [r for r in results if r["risk"] == lv]
        if sub:
            s = analyze(sub)
            print(f"  {lv}: {json.dumps(s, ensure_ascii=False)}")

    # 显著性判定
    if overall.get("ttest_rel_p") is not None:
        sig = "显著" if overall["ttest_rel_p"] < 0.05 else "不显著"
        print(f"\n[结论] B(进化) vs A(基线) 差异{sig} "
              f"(t 检验单边 p={overall['ttest_rel_p']}, Cohen's d={overall.get('cohens_d')})")


if __name__ == "__main__":
    main()
