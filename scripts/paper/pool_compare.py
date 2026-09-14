"""
池子大小对照 · 5-vs-5 分布比较 (pool_compare.py)
===============================================
检验论文 §5.3 的结论：「池 15→45 使 M1 上有害效应出现、M2 上有益效应消失」。

问题：原结论基于**每个池子大小各一次运行**，而单次运行的波动幅度（实测极差 0.10–0.14）
与所报告的池子间差异（≈0.06）同量级 —— 按论文自己的标准，该结论不可靠。

本脚本用**每个池子大小各 5 次独立运行**做正当的分布比较：
  1. 两个池子的 Δ 分布是否有实质分离（而非仅比均值）
  2. 用 Mann–Whitney U 检验（独立样本，非配对）看差异是否显著
  3. 报告效应量与区间重叠度

注意：池 15 与池 45 是**不同的配置**，各自独立运行，故用独立样本检验而非配对检验。

运行（在本仓库根目录）：python scripts/paper/pool_compare.py
"""

from __future__ import annotations
import os
import json
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# (标签, 池15 文件模板, 池45 文件模板)
MODELS = [
    ("M1 (Lingshu-32B)", "lingshu_pool15_run{}.json", "lingshu_replicate_run{}.json"),
    ("M2 (DeepSeek chat)", "deepseek_pool15_run{}.json", None),
]

# M2 池45 的五次分散在三个文件名里（与 analyze_replicates.py 一致）
M2_POOL45 = ["deepseek_bigpool_results.json", "deepseek_diversity_results.json",
             "deepseek_replicate_run2.json", "deepseek_replicate_run3.json",
             "deepseek_replicate_run4.json"]

N_RUNS = 5


def run_delta(path: str, key: str = "random") -> float | None:
    if not os.path.exists(path):
        return None
    d = json.load(open(path, encoding="utf-8"))
    rs = [r for r in d["results"]
          if r.get("baseline_mean") is not None and r.get(f"{key}_mean") is not None]
    if len(rs) < 4:
        return None
    B = np.array([r["baseline_mean"] for r in rs])
    X = np.array([r[f"{key}_mean"] for r in rs])
    return float((X - B).mean())


def collect(template: str | None, files: list | None = None) -> list:
    if files:
        return [v for v in (run_delta(os.path.join(DATA, f)) for f in files) if v is not None]
    if not template:
        return []
    return [v for v in (run_delta(os.path.join(DATA, template.format(i)))
                        for i in range(1, N_RUNS + 1)) if v is not None]


def report(model: str, p15: list, p45: list):
    print(f"\n{'='*74}")
    print(f"{model}")
    print(f"{'='*74}")
    print(f"  池 15（{len(p15)} 次）: " + "  ".join(f"{v:+.4f}" for v in p15))
    print(f"  池 45（{len(p45)} 次）: " + "  ".join(f"{v:+.4f}" for v in p45))
    if len(p15) < 2 or len(p45) < 2:
        print("  （数据不足，无法比较）")
        return None

    a, b = np.array(p15), np.array(p45)
    print(f"\n  池15 均值 {a.mean():+.4f}  SD {a.std(ddof=1):.4f}  范围 {a.min():+.3f}~{a.max():+.3f}")
    print(f"  池45 均值 {b.mean():+.4f}  SD {b.std(ddof=1):.4f}  范围 {b.min():+.3f}~{b.max():+.3f}")
    print(f"  均值之差        {b.mean()-a.mean():+.4f}")
    print(f"  池15 的极差     {a.max()-a.min():.4f}")
    print(f"  池45 的极差     {b.max()-b.min():.4f}")

    # 独立样本检验（池大小是不同配置，各次运行独立）
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    except Exception:                                        # noqa: BLE001
        u, p = float("nan"), float("nan")
    # 效应量：秩二列相关
    r_rb = 1 - (2 * u) / (len(a) * len(b)) if u == u else float("nan")

    # 区间重叠
    overlap = max(0.0, min(a.max(), b.max()) - max(a.min(), b.min()))
    span = max(a.max(), b.max()) - min(a.min(), b.min())
    print(f"\n  Mann–Whitney U 检验（双侧）: U = {u:.1f}, p = {p:.3f}")
    print(f"  秩二列相关 r          : {r_rb:+.3f}")
    print(f"  两分布的区间重叠      : {overlap:.4f} / {span:.4f} = "
          f"{overlap/span:.0%}" if span > 0 else "  （无跨度）")

    print(f"\n  【判定】")
    if p >= 0.05:
        print(f"    两个池子大小的效应估计**无法区分**（p = {p:.3f}）。")
        print(f"    原 §5.3 的结论建立在每个池子各一次运行之上，")
        print(f"    而单次波动幅度（极差 {max(a.max()-a.min(), b.max()-b.min()):.3f}）"
              f"大于所报告的池间差异（{abs(b.mean()-a.mean()):.4f}）。")
        print(f"    → 应改写为「未能与运行间波动区分」，论证重心移到机制证据。")
    else:
        print(f"    两分布存在统计差异（p = {p:.3f}），原结论方向得到支持。")

    return {"model": model, "pool15": p15, "pool45": p45,
            "delta_of_means": float(b.mean() - a.mean()), "p": float(p),
            "r_rb": float(r_rb)}


def main():
    print("=" * 74)
    print("池子大小对照：每档 5 次独立运行（替代原论文的单次比较）")
    print("=" * 74)

    out = {}
    for model, t15, t45 in MODELS:
        files45 = M2_POOL45 if t45 is None else None
        p15 = collect(t15)
        p45 = collect(t45, files45)
        r = report(model, p15, p45)
        if r:
            out[model] = r

    op = os.path.join(DATA, "pool_compare.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {op}")


if __name__ == "__main__":
    main()
