# -*- coding: utf-8 -*-
"""为附录表 6/7 计算标准误（table_stats.py）
=============================================
第四份评审：「附录中的表格（表 6、表 7）缺乏必要的统计信息，建议增加标准误或置信区间列。」

原始数据支持计算：
  - 表 6 消融：ablation_results.json 每个配置带 scores(n=12)；
                ablation_results_n5_original.json 带 scores(n=5)
  - 表 7 收敛：convergence_repeated.json 带 runs（每轮各次运行的分值）

运行（在本仓库根目录）：python scripts/paper/table_stats.py
"""
from __future__ import annotations
import json
import os

import numpy as np

D = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def se(a):
    a = np.asarray(a, dtype=float)
    return a.std(ddof=1) / np.sqrt(len(a))


def main() -> None:
    print("=" * 70)
    print("表 6（消融）—— 增加 SE 列")
    print("=" * 70)
    new = json.load(open(os.path.join(D, "ablation_results.json"),
                         encoding="utf-8"))["results"]
    old = json.load(open(os.path.join(D, "ablation_results_n5_original.json"),
                         encoding="utf-8"))["results"]
    om = {r["name"]: r for r in old}
    print(f"{'name':<24}{'mean12':>8}{'SE12':>8}   {'mean5':>7}{'SE5':>8}")
    for r in new:
        s12 = se(r["scores"])
        o = om.get(r["name"])
        s5 = se(o["scores"]) if o and "scores" in o else float("nan")
        m5 = o["mean"] if o else float("nan")
        print(f"{r['name']:<24}{r['mean']:>8.3f}{s12:>8.3f}   {m5:>7.2f}{s5:>8.3f}")

    print()
    print("=" * 70)
    print("表 7（收敛）—— 增加各轮 SD/SE")
    print("=" * 70)
    cv = json.load(open(os.path.join(D, "convergence_repeated.json"),
                        encoding="utf-8"))
    runs = cv["runs"]
    print(f"  n_runs={cv['n_runs']} n_rounds={cv['n_rounds']} "
          f"n_profiles={cv['n_profiles']}")
    print(f"  runs[0] 类型: {type(runs[0]).__name__}")
    print(f"  runs[0] = {runs[0]}")
    # round_means: 5 次运行 × 5 轮 的每轮均值
    mat = np.array(cv["round_means"], dtype=float)
    print(f"\n  矩阵 {mat.shape}（行=5 次重复，列=5 轮）")
    print(f"\n  {'轮次':<6}{'合并均值':>10}{'跨运行SD':>10}{'跨运行SE':>10}"
          f"{'轮内SE(8份)':>14}")
    for j in range(mat.shape[1]):
        col = mat[:, j]
        within = se(runs[0]["rounds"][j]["scores"])
        print(f"  第{j+1}轮{col.mean():>10.4f}{col.std(ddof=1):>10.4f}"
              f"{se(col):>10.4f}{within:>14.4f}")

    print("\n  LaTeX 行（供表 7 增加一行）：")
    sd_row = " & ".join(f"{v:.3f}" for v in mat.std(axis=0, ddof=1))
    print(f"    SD across repetitions & {sd_row} \\\\")


if __name__ == "__main__":
    main()
