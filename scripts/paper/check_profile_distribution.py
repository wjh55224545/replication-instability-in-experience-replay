# -*- coding: utf-8 -*-
"""核对论文 §4.1 的说法是否成立（check_profile_distribution.py）
=================================================================
论文 §4.1 原话：
  "...we synthesize additional profiles by drawing scores, valences, and emotions
   from the **empirical distribution of records within each risk band**..."

但 make_dataset.py 的 `synth_profile()` 实际用的是**每个风险带内的均匀分布**
（`rng.uniform(0.70, 0.95)` 等硬编码区间），并非经验分布抽样。

第四份评审建议再加一句「KS 检验 p>0.05 证明边缘分布一致」——**若无实测支撑即为编造**。
本脚本实测到底能不能这么说。

运行（在本仓库根目录）：python scripts/paper/check_profile_distribution.py
"""
from __future__ import annotations
import json
import os
import sqlite3
import sys
from collections import defaultdict, Counter

import numpy as np
from scipy import stats

D = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DB = os.path.join(D, "psych.db")
DS = os.path.join(D, "paper_dataset.json")


def risk_level(s: float) -> str:
    return "green" if s >= 0.7 else ("yellow" if s >= 0.4 else "red")


def real_scores():
    """从 psych.db 按 (学生, 日期) 聚合出真实档案的 fused_score，并标注风险带。"""
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("SELECT * FROM emotion_records ORDER BY recorded_at"))
    conn.close()
    g = defaultdict(list)
    for r in rows:
        g[(r["student_id"], (r["recorded_at"] or "")[:10])].append(
            r["fused_score"] if r["fused_score"] is not None else 0.5)
    out = []
    for _, s in g.items():
        m = float(np.mean(s))
        out.append((m, risk_level(m)))
    return out


def main() -> None:
    real = real_scores()
    print(f"psych.db 聚合出的真实档案: {len(real)} 份")
    print("  风险分布:", dict(Counter(lv for _, lv in real)))

    ds = json.load(open(DS, encoding="utf-8"))
    if isinstance(ds, dict):
        ds = ds.get("profiles", ds.get("data", []))

    print(f"\npaper_dataset.json: {len(ds)} 份")
    print("  风险分布:", dict(Counter(p["risk_level"] for p in ds)))

    # 逐风险带做 KS 检验：真实档案 score 分布 vs 论文档案 score 分布
    print("\n=== 逐风险带 KS 检验（fused_score）===")
    print("  H0: 两个分布相同。p 小 = 分布不同。")
    for band in ("green", "yellow", "red"):
        a = np.array([s for s, lv in real if lv == band])
        b = np.array([p["emotion_data"]["fused_score"] for p in ds
                      if p["risk_level"] == band])
        if len(a) < 3 or len(b) < 3:
            print(f"  {band:<7} 样本不足 (真实 {len(a)} / 论文 {len(b)})，无法检验")
            continue
        k = stats.ks_2samp(a, b)
        verdict = "分布一致" if k.pvalue > 0.05 else "**分布不同**"
        print(f"  {band:<7} 真实 n={len(a):<3} mean={a.mean():.3f}  |  "
              f"论文 n={len(b):<3} mean={b.mean():.3f}  |  "
              f"D={k.statistic:.3f}  p={k.pvalue:.3f}  → {verdict}")

    # 全池
    a = np.array([s for s, _ in real])
    b = np.array([p["emotion_data"]["fused_score"] for p in ds])
    k = stats.ks_2samp(a, b)
    print(f"\n  全池   真实 n={len(a)}  | 论文 n={len(b)}  |  "
          f"D={k.statistic:.3f}  p={k.pvalue:.3f}")

    # 情绪标签分布
    print("\n=== 情绪标签分布（前 6）===")
    conn = sqlite3.connect(DB)
    emo_real = Counter(r[0] for r in conn.execute(
        "SELECT fused_emotion FROM emotion_records WHERE fused_emotion IS NOT NULL"))
    conn.close()
    emo_pap = Counter(p["emotion_data"]["fused_emotion"] for p in ds)
    keys = [k for k, _ in emo_real.most_common(6)]
    print(f"  {'情绪':<6} {'真实占比':>10} {'论文占比':>10}")
    for k in keys:
        print(f"  {k:<6} {emo_real[k]/sum(emo_real.values()):>10.3f} "
              f"{emo_pap.get(k,0)/sum(emo_pap.values()):>10.3f}")


if __name__ == "__main__":
    main()
