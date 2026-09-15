# -*- coding: utf-8 -*-
"""算出 §5.7 改写要用的全部功效数字（power_numbers.py）
=========================================================
口径 C：同时报「组内 pooled SD」与「配置间配对差值 SD」两个口径，
证明换哪个口径结论都不变。

运行：cd 代码 && D:/python/python.exe scripts/paper/power_numbers.py
"""
from __future__ import annotations
import json
import os

import numpy as np
from scipy import stats

D = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def mdes(sd, n, alpha=0.05, power=0.80, tail=2):
    """配对 t 检验的最小可检测效应（原始单位），双尾默认。

    2026-09-15：由 tail=1 改为 tail=2，与全文改双尾检验保持一致。
    口径统一后 MDES 变大（约 ×1.22），即「可检出的最小效应更高」——
    这与论文的论点同向：该效应量下本研究本就不足以分辨。
    """
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / tail, df)
    lo, hi = 0.0, 80.0
    for _ in range(300):
        mid = (lo + hi) / 2
        if 1 - stats.nct.cdf(t_crit, df, mid) < power:
            lo = mid
        else:
            hi = mid
    return ((lo + hi) / 2) * sd / np.sqrt(n)


def n_for(delta, sd, alpha=0.05, power=0.80, tail=2):
    n = 2
    while n < 200000:
        if mdes(sd, n, alpha, power, tail) <= delta:
            return n
        n += 1
    return None


def main() -> None:
    ab = json.load(open(os.path.join(D, "ablation_results.json"), encoding="utf-8"))["results"]
    sds = [np.std(r["scores"], ddof=1) for r in ab]
    pooled = float(np.sqrt(np.mean([np.var(r["scores"], ddof=1) for r in ab])))
    base = np.array(ab[0]["scores"])
    diffs = np.concatenate([np.array(r["scores"]) - base for r in ab[1:]])
    sd_diff = float(diffs.std(ddof=1))

    print("=== 消融（n=12/配置，8 个配置）===")
    print(f"  组内 pooled SD          = {pooled:.4f}")
    print(f"  配置间配对差值 SD        = {sd_diff:.4f}")
    print(f"  各配置组内 SD 的均值      = {np.mean(sds):.4f}")
    print(f"  （论文原写 s=0.373，数据文件里找不到此值）")
    print()
    print(f"  观测到的最大配置间差异    = "
          f"{max(abs(r['mean'] - ab[0]['mean']) for r in ab):.4f} 点")
    print()
    for label, sd in (("组内 pooled", pooled), ("配置间差值", sd_diff)):
        print(f"  --- 口径：{label} (SD={sd:.3f}) ---")
        print(f"      n=12 时 MDES（配对·双尾）= {mdes(sd,12):.4f} 点")
        for d in (0.03, 0.05, 0.10):
            print(f"      检出 {d:.2f} 点需要 {n_for(d,sd)} 份档案/条件")
        print()

    # 主实验
    v1 = []
    for r in json.load(open(os.path.join(D, "ab_repeated_results.json"),
                            encoding="utf-8"))["results"]:
        for k in ("baseline", "random", "semantic"):
            if isinstance(r.get(k), list):
                v1.extend(r[k])
    v2 = []
    for r in json.load(open(os.path.join(D, "deepseek_repeated_results.json"),
                            encoding="utf-8"))["results"]:
        for k in ("baseline", "random", "semantic"):
            if isinstance(r.get(k), list):
                v2.extend(r[k])
    a1, a2 = np.array(v1), np.array(v2)
    print("=== 主实验单次生成判定分 ===")
    print(f"  M1: n={len(a1)}  SD={a1.std(ddof=1):.4f}  众数占比 "
          f"{100*max(np.bincount((a1*10).astype(int)))/len(a1):.1f}%")
    print(f"  M2: n={len(a2)}  SD={a2.std(ddof=1):.4f}")
    both = np.concatenate([a1, a2])
    print(f"  合并: n={len(both)}  SD={both.std(ddof=1):.4f}")

    # 配对差值 SD（主实验）
    dd = []
    for fn in ("ab_repeated_results.json", "deepseek_repeated_results.json"):
        for r in json.load(open(os.path.join(D, fn), encoding="utf-8"))["results"]:
            b = r.get("baseline_mean")
            for c in ("random_mean", "semantic_mean"):
                if b is not None and r.get(c) is not None:
                    dd.append(r[c] - b)
    dd = np.array(dd)
    print(f"\n  主实验配对差值（三次重复取均值后）: n={len(dd)}  SD={dd.std(ddof=1):.4f}")
    print(f"  n=29 份档案时 MDES（配对·双尾）= {mdes(dd.std(ddof=1),29):.4f} 点")


if __name__ == "__main__":
    main()
