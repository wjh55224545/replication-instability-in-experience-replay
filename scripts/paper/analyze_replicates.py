"""
论文 · 同配置多次运行分析 (analyze_replicates.py)
==============================================
把同一配置（M2 / 池45 / 重复3次 / random 条件）的多次独立运行汇总，
刻画效应估计的分布 —— 论文核心论点的决定性证据。

运行（在本仓库根目录）：python scripts/paper/analyze_replicates.py
"""

from __future__ import annotations
import os
import json
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# (标签, 文件, random 的键名, matched 的键名)
RUNS = [
    ("run 1", "deepseek_bigpool_results.json",    "random", "semantic"),
    ("run 2", "deepseek_diversity_results.json",  "random", "similar"),
    ("run 3", "deepseek_replicate_run2.json",     "random", "semantic"),
    ("run 4", "deepseek_replicate_run3.json",     "random", "semantic"),
    ("run 5", "deepseek_replicate_run4.json",     "random", "semantic"),
]


def one_run(path, rk, mk):
    d = json.load(open(path, encoding="utf-8"))
    rs = [r for r in d["results"]
          if r.get("baseline_mean") is not None and r.get(f"{rk}_mean") is not None]
    B = np.array([r["baseline_mean"] for r in rs])
    R = np.array([r[f"{rk}_mean"] for r in rs])
    M = np.array([r[f"{mk}_mean"] for r in rs if r.get(f"{mk}_mean") is not None])
    out = {"n": len(rs), "pool": d["meta"].get("pool_size")}
    for name, arr in [("random", R), ("matched", M)]:
        df = arr - B[:len(arr)]
        try:
            _, p = stats.wilcoxon(df); p /= 2
        except Exception:
            p = float("nan")
        dz = df.mean() / df.std(ddof=1) if df.std(ddof=1) > 0 else 0.0
        se = df.std(ddof=1) / np.sqrt(len(df))
        t = stats.t.ppf(0.975, len(df) - 1)
        out[name] = {"delta": df.mean(), "ci": t * se, "p": p, "dz": dz}
    return out


def main():
    print("=" * 74)
    print("同配置多次独立运行：M2 / 经验池 45 / 每格重复 3 次")
    print("=" * 74)
    print(f"{'运行':6s} {'n':>3s} {'池':>4s} | {'random Δ':>10s} {'p':>7s} {'d_z':>6s}"
          f" | {'matched Δ':>10s} {'p':>7s}")
    print("-" * 74)

    res = {}
    for label, f, rk, mk in RUNS:
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            print(f"{label:6s}  (文件缺失: {f})")
            continue
        r = one_run(p, rk, mk)
        res[label] = r
        sig_r = "*" if r["random"]["p"] < 0.05 else " "
        print(f"{label:6s} {r['n']:3d} {r['pool']:4} | "
              f"{r['random']['delta']:+10.3f} {r['random']['p']:7.4f}{sig_r}"
              f" {r['random']['dz']:+6.2f} | "
              f"{r['matched']['delta']:+10.3f} {r['matched']['p']:7.4f}")

    if not res:
        return

    print("-" * 74)
    deltas = np.array([r["random"]["delta"] for r in res.values()])
    ps = np.array([r["random"]["p"] for r in res.values()])
    sig = int((ps < 0.05).sum())

    print(f"\n【random 条件的效应估计分布】")
    print(f"  运行次数      : {len(deltas)}")
    print(f"  Δ 范围        : {deltas.min():+.3f} ～ {deltas.max():+.3f}")
    print(f"  Δ 均值 ± SD   : {deltas.mean():+.3f} ± {deltas.std(ddof=1):.3f}")
    print(f"  Δ 为正的比例  : {int((deltas > 0).sum())}/{len(deltas)}")
    print(f"  达到 p<0.05 的: {sig}/{len(ps)} 次")

    md = np.array([r["matched"]["delta"] for r in res.values()])
    mp = np.array([r["matched"]["p"] for r in res.values()])
    print(f"\n【matched 条件的效应估计分布】")
    print(f"  Δ 范围        : {md.min():+.3f} ～ {md.max():+.3f}")
    print(f"  达到 p<0.05 的: {int((mp < 0.05).sum())}/{len(mp)} 次")

    print(f"\n【结论】同一配置的 {len(deltas)} 次独立运行中，")
    print(f"  random  效应有 {sig} 次显著、{len(ps)-sig} 次不显著，Δ 跨越 {deltas.min():+.3f} 到 {deltas.max():+.3f}；")
    print(f"  matched 效应有 {int((mp<0.05).sum())} 次显著。")
    print("  → 单次实验的显著性判断基本由随机性决定。")

    # 保存
    out = os.path.join(DATA, "replicate_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({k: {kk: {kkk: (None if (isinstance(vv, float) and np.isnan(vv)) else vv)
                            for kkk, vv in vv.items()} if isinstance(vv, dict) else vv
                       for kk, vv in v.items()}
                   for k, v in res.items()}, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main()
