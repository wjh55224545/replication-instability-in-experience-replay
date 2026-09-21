# -*- coding: utf-8 -*-
"""轨迹型任务分析 (traj_analyze.py)
=================================
回答两个问题：
  1. replay 在本任务上的**效应有多大**？（与主实验、setselect 对照）
  2. 跨独立运行，**结论是否稳定**？（论文的核心主张：不稳定）

判定口径与主实验一致：配对（同一案例内比较），报告 Δ、双尾 Wilcoxon、d_z，
并看跨运行的方向与显著性是否一致。

运行：cd 代码 && D:/python/python.exe scripts/paper/traj_analyze.py
"""

from __future__ import annotations

import io
import json
import os
import re
import statistics as st
import sys

import numpy as np
from scipy import stats

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:                                          # noqa: BLE001
    pass

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def load(model: str, tag: str):
    suffix = f"_{tag}" if tag else ""
    p = os.path.join(DATA, f"traj_experiment_{model}{suffix}.json")
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def paired(rows, cond_a: str, cond_b: str):
    """同案例内的配对差值 b - a。"""
    d = []
    for r in rows:
        a, b = r.get(cond_a + "_mean"), r.get(cond_b + "_mean")
        if a is not None and b is not None:
            d.append(b - a)
    return np.array(d)


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    print("=" * 84)
    print(f"轨迹型客观任务 · 结果分析（tag={tag}）")
    print("=" * 84)

    for model in ("lingshu", "deepseek"):
        d = load(model, tag)
        if not d:
            continue
        runs = d["runs"]
        print(f"\n{'#' * 84}\n模型 {model}｜{len(runs)} 次独立运行"
              f"｜{d['meta']['n_cases']} 案例 × {d['meta']['n_repeat']} 重复\n{'#' * 84}")

        # ---- 1. 各条件水平 ----
        print(f"\n【1】各条件得分（跨运行汇总）")
        print(f"  {'条件':<10}{'均分':>9}{'SD':>8}{'达标率':>9}   （跨全部运行的案例均值）")
        print("  " + "-" * 58)
        for c in ("baseline", "random", "matched", "filler"):
            vals, grn = [], []
            for r in runs:
                for row in r["results"]:
                    if row.get(c + "_mean") is not None:
                        vals.append(row[c + "_mean"])
                    if row.get(c + "_green") is not None:
                        grn.append(row[c + "_green"])
            if vals:
                print(f"  {c:<10}{st.mean(vals):>9.3f}{st.pstdev(vals):>8.3f}{st.mean(grn):>9.1%}")

        # ---- 2. 每次运行的效应量 ----
        print(f"\n【2】每次独立运行的效应（配对，同案例内）")
        print(f"  {'运行':<6}{'Δ(matched−base)':>18}{'p':>9}{'d_z':>8}{'Δ(filler−base)':>18}{'p':>9}")
        print("  " + "-" * 72)
        deltas, ps, greens = [], [], []
        for r in runs:
            rows = r["results"]
            dm = paired(rows, "baseline", "matched")
            df_ = paired(rows, "baseline", "filler")
            pm = stats.wilcoxon(dm).pvalue if len(dm) > 3 else float("nan")
            pf = stats.wilcoxon(df_).pvalue if len(df_) > 3 else float("nan")
            dz = dm.mean() / dm.std(ddof=1) if dm.std(ddof=1) > 0 else 0
            deltas.append(dm.mean())
            ps.append(pm)
            print(f"  run{r['run']:<3}{dm.mean():>+18.4f}{pm:>9.4f}{dz:>+8.2f}"
                  f"{df_.mean():>+18.4f}{pf:>9.4f}")

        # ---- 3. 稳定性判定 ----
        print(f"\n【3】跨运行的稳定性（论文的核心判据）")
        d_arr, p_arr = np.array(deltas), np.array(ps)
        print(f"  Δ 范围       {d_arr.min():+.4f} ~ {d_arr.max():+.4f}   极差 {d_arr.max() - d_arr.min():.4f}")
        print(f"  符号         {'全部同向 ✓' if (d_arr > 0).all() or (d_arr < 0).all() else '✗ 出现翻转'}"
              f"（{'正' if d_arr.mean() > 0 else '负'}）")
        n_sig = int((p_arr < 0.05).sum())
        print(f"  显著性       {n_sig}/{len(p_arr)} 次达到 p<0.05 "
              f"{'→ 判决一致 ✓' if n_sig in (0, len(p_arr)) else '→ ✗ 判决翻转'}")
        print(f"  Δ/SE 比      {np.abs(d_arr).min() / max(d_arr.std(ddof=1), 1e-9):.1f} "
              f"~ {np.abs(d_arr).max() / max(d_arr.std(ddof=1), 1e-9):.1f}")
        print()
        print("  对照（论文 §5.9 setselect 客观任务）：Δ 极差 0.022 上，判定仍翻转 3/5")
        print("  对照（论文 §5.4 主观任务）：Δ 极差 0.138，判定翻转 2/5，M1 符号也翻转")


if __name__ == "__main__":
    main()
