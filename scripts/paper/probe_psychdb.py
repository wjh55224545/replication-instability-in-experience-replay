# -*- coding: utf-8 -*-
"""穷举 psych.db 的所有取数口径，看能否得到高风险（红带）真实档案
====================================================================
背景：目前聚合出 19 份真实档案（16 黄 / 3 绿 / **0 红**），故论文里的红带全靠合成。
本脚本穷举各种口径，确认"确实拿不到红带真实数据"还是"取数方式不对"。

运行（在本仓库根目录）：python scripts/paper/probe_psychdb.py
"""
from __future__ import annotations
import os
import sqlite3
from collections import defaultdict, Counter

import numpy as np

DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "psych.db")


def risk_level(s: float) -> str:
    return "green" if s >= 0.7 else ("yellow" if s >= 0.4 else "red")


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    print("=== 表结构 ===")
    tabs = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")]
    for t in tabs:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<22} {n:>6} 行   {cols}")

    # ---- 口径 1：逐条记录（不聚合）----
    print("\n=== 口径 1：单条记录 fused_score 的风险分布 ===")
    vals = [r[0] for r in conn.execute(
        "SELECT fused_score FROM emotion_records") if r[0] is not None]
    print(f"  记录数 {len(vals)}  范围 [{min(vals):.3f}, {max(vals):.3f}]  "
          f"均值 {np.mean(vals):.3f}")
    print("  ", dict(Counter(risk_level(v) for v in vals)))

    # ---- 口径 2：按学生（跨所有日期平均）----
    print("\n=== 口径 2：按学生平均（不按天）===")
    g = defaultdict(list)
    for r in conn.execute("SELECT student_id, fused_score FROM emotion_records"):
        if r[1] is not None:
            g[r[0]].append(r[1])
    per_student = [float(np.mean(v)) for v in g.values()]
    print(f"  学生数 {len(per_student)}  范围 "
          f"[{min(per_student):.3f}, {max(per_student):.3f}]")
    print("  ", dict(Counter(risk_level(s) for s in per_student)))

    # ---- 口径 3：学生×天（论文口径）----
    print("\n=== 口径 3：学生×天（论文口径）===")
    g2 = defaultdict(list)
    for r in conn.execute(
            "SELECT student_id, recorded_at, fused_score FROM emotion_records"):
        if r[2] is not None:
            g2[(r[0], (r[1] or "")[:10])].append(r[2])
    per_day = [float(np.mean(v)) for v in g2.values()]
    print(f"  档案数 {len(per_day)}  范围 "
          f"[{min(per_day):.3f}, {max(per_day):.3f}]")
    print("  ", dict(Counter(risk_level(s) for s in per_day)))

    # ---- 口径 4：最低的单条 / 最低的日均 ----
    print("\n=== 最低的若干值 ===")
    print("  单条最低 5 个:", [round(v, 3) for v in sorted(vals)[:5]])
    print("  日均最低 5 个:", [round(v, 3) for v in sorted(per_day)[:5]])
    print(f"  红带阈值 = 0.4（即 < 0.4 才算 red）")

    # ---- 口径 5：是否存在别的风险等级字段 ----
    print("\n=== 是否存在现成的风险/等级字段 ===")
    for t in tabs:
        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
        hit = [c for c in cols if any(k in c.lower()
                                      for k in ("risk", "level", "score", "severity"))]
        if hit:
            print(f"  {t}: {hit}")

    # ---- 口径 6：recorded_at 的时间跨度 ----
    print("\n=== 时间跨度 ===")
    r = conn.execute("SELECT MIN(recorded_at), MAX(recorded_at) FROM emotion_records").fetchone()
    print(f"  {r[0]}  →  {r[1]}")

    conn.close()


if __name__ == "__main__":
    main()
