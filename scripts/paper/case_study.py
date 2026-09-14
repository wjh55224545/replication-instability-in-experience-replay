"""
论文 · Case Study 定性分析材料生成 (case_study.py)
================================================

生成「黄色风险」学生档案的三版报告全文，用于定性对比：
- baseline : 无进化上下文
- random   : 注入随机采样的历史经验（可能来自 green/red，不匹配 yellow）
- semantic : 注入语义检索（匹配风险/情绪）的历史经验

输出 data/case_study_reports.json，供论文 Results/Discussion 的 Case Study 段落使用。

运行（在本仓库根目录）：python scripts/paper/case_study.py
"""

from __future__ import annotations
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backend.evolution.memory as mem  # noqa: E402
from common import (  # noqa: E402
    load_profiles,
    generate_report,
    generate_report_semantic,
    judge,
    mean_score,
    record_feedback,
    build_evolution_context,
    CATEGORY,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(DATA_DIR, "case_study_reports.json")
CS_EXP = os.path.join(DATA_DIR, "case_study_experience.jsonl")


def warmup(profiles):
    """用 green/red 档案预热经验库，使 random 注入能采到不匹配的经验。"""
    for p in profiles:
        report = generate_report(p, use_evolution=False)
        if not report:
            continue
        s = judge(report, p)
        m = mean_score(s)
        if m is not None and m >= 4.0:
            record_feedback(
                CATEGORY, p["student_name"], rating=int(round(m)), outcome=report[:200],
                risk_level=p["risk_level"],
                emotion=p.get("emotion_data", {}).get("fused_emotion"),
            )


def main(n_warmup_green=3, n_warmup_red=3, n_case=2):
    mem.EXPERIENCE_PATH = CS_EXP
    if os.path.exists(CS_EXP):
        os.remove(CS_EXP)

    profiles = load_profiles()
    greens = [p for p in profiles if p["risk_level"] == "green"][:n_warmup_green]
    reds = [p for p in profiles if p["risk_level"] == "red"][:n_warmup_red]
    yellows = [p for p in profiles if p["risk_level"] == "yellow"][:n_case]

    print(f"[case_study] 预热 green={len(greens)}, red={len(reds)}; 案例 yellow={len(yellows)}")
    warmup(greens + reds)

    random_ctx = build_evolution_context(CATEGORY)  # 随机采样（可能含 green/red）
    print(f"[case_study] 随机注入上下文长度: {len(random_ctx)} 字符")

    results = []
    for p in yellows:
        base = generate_report(p, use_evolution=False)
        rand = generate_report(p, use_evolution=True, evolution_context=random_ctx)
        semi = generate_report_semantic(p)
        s_base = judge(base, p) if base else None
        s_rand = judge(rand, p) if rand else None
        s_semi = judge(semi, p) if semi else None
        results.append({
            "student_name": p["student_name"],
            "risk_level": p["risk_level"],
            "emotion": p.get("emotion_data", {}).get("fused_emotion"),
            "baseline": {"report": base, "judge": s_base, "mean": mean_score(s_base)},
            "random": {"report": rand, "judge": s_rand, "mean": mean_score(s_rand)},
            "semantic": {"report": semi, "judge": s_semi, "mean": mean_score(s_semi)},
        })
        print(f"[case_study] {p['student_name']}: baseline={mean_score(s_base)}, "
              f"random={mean_score(s_rand)}, semantic={mean_score(s_semi)}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[case_study] 已保存: {OUT}")


if __name__ == "__main__":
    main()
