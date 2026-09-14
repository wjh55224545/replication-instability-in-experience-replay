# -*- coding: utf-8 -*-
"""复核论文 §5.7 的功效数字（recheck_mdes.py）
================================================
论文称：s=0.373、n=12 时 MDES=0.43 点；检出 0.10 需 ~218、0.03 需 ~2400。

反推发现这三个数都来自**两样本（非配对）公式**：
    2 × ((z_{α/2}+z_β) / dz)² ，dz=0.268 → 218.0（精确吻合）
但实验是**配对**设计（within-item），且配对检验该用**差值**的 SD，不是分数的 SD。

本脚本用真实数据核实：配对差值的 SD 是多少？正确的 MDES 是多少？

运行（在本仓库根目录）：python scripts/paper/recheck_mdes.py
"""
from __future__ import annotations
import json
import os

import numpy as np
from scipy import stats

D = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def mdes(sd, n, alpha=0.05, power=0.80):
    """单尾配对 t 检验的最小可检测效应（原始单位）。"""
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df)
    lo, hi = 0.0, 60.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if 1 - stats.nct.cdf(t_crit, df, mid) < power:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2) * sd / np.sqrt(n)


def n_for(delta, sd, alpha=0.05, power=0.80):
    n = 2
    while n < 100000:
        if mdes(sd, n, alpha, power) <= delta:
            return n
        n += 1
    return None


def main() -> None:
    # ---- 主实验：配对差值的 SD ----
    print("=== 从数据文件取配对差值 ===")
    for fn in ("ab_repeated_results.json", "ab_results.json"):
        p = os.path.join(D, fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        rs = d.get("results", d if isinstance(d, list) else [])
        diffs = []
        for r in rs:
            b = r.get("baseline_mean")
            for cond in ("random_mean", "semantic_mean"):
                v = r.get(cond)
                if b is not None and v is not None:
                    diffs.append(v - b)
        if diffs:
            a = np.array(diffs)
            print(f"  {fn}: n={len(a)}  SD(差值)={a.std(ddof=1):.4f}  "
                  f"mean={a.mean():.4f}")

    print("\n=== 论文 §5.7 的数字复核 ===")
    s_paper, n_ab = 0.373, 12
    print(f"  论文称 s={s_paper}, n={n_ab} → MDES=0.43 点")
    print(f"  正确（配对、单尾）MDES = {mdes(s_paper, n_ab):.4f} 点")
    print(f"  论文称检出 0.10 需 ~218；配对公式需要 {n_for(0.10, s_paper)}")
    print(f"  论文称检出 0.03 需 ~2400；配对公式需要 {n_for(0.03, s_paper)}")
    print()
    print("  反推论文用的公式（两样本）:")
    for delta, claimed in ((0.10, 218), (0.03, 2400)):
        dz = delta / s_paper
        z = 1.959964 + 0.8416212
        n2 = 2 * (z / dz) ** 2
        print(f"    Δ={delta}: 2×((z_α/2+z_β)/dz)² = {n2:.0f}   （论文写 {claimed}）")

    print("\n=== 结论 ===")
    print("  论文 §5.7 用的是两样本公式，而设计是配对的；")
    print("  且应当使用配对**差值**的 SD，而不是分数的 SD。")


if __name__ == "__main__":
    main()
