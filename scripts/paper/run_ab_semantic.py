"""
论文实验 1b · A/B 对照（语义检索增强版）(run_ab_semantic.py)
============================================================

与 run_ab_experiment.py（随机采样 baseline）流程一致，区别在于：
- 经验条目携带 risk_level / emotion 特征
- B 条件用「语义检索」（按当前档案的风险/情绪匹配最相似经验）注入

用于对比「随机采样 vs 语义检索」对进化效果的影响（论文的技术贡献 + 消融）。

输出 data/ab_semantic_results.json；使用独立经验库 semantic_experience.jsonl。
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
    CATEGORY,
)
from run_ab_experiment import split_balanced  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(DATA_DIR, "ab_semantic_results.json")
SEMANTIC_EXP = os.path.join(DATA_DIR, "semantic_experience.jsonl")


def main(seed_n: int = 15, max_test: int = 29):
    # 独立经验库
    mem.EXPERIENCE_PATH = SEMANTIC_EXP
    if os.path.exists(SEMANTIC_EXP):
        os.remove(SEMANTIC_EXP)

    profiles = load_profiles()
    seed, test = split_balanced(profiles, seed_n)
    print(f"[A/B-语义] 种子组 {len(seed)} 个, 测试组 {len(test)} 个")

    # ---- 1. 预热：带特征记录高分经验 ----
    print("[A/B-语义] 预热...")
    pool = 0
    for p in seed:
        report = generate_report(p, use_evolution=False)
        if not report:
            continue
        scores = judge(report, p)
        m = mean_score(scores)
        if m is not None and m >= 4.0:
            record_feedback(
                CATEGORY, p["student_name"], rating=int(round(m)), outcome=report[:200],
                risk_level=p["risk_level"],
                emotion=p.get("emotion_data", {}).get("fused_emotion"),
            )
            pool += 1
    print(f"[A/B-语义] 预热完成，高分经验入池 {pool} 条")

    # ---- 2. A/B：A=无进化，B=语义检索进化 ----
    results = []
    for i, p in enumerate(test[:max_test]):
        rA = generate_report(p, use_evolution=False)
        rB = generate_report_semantic(p)
        sA = judge(rA, p) if rA else None
        sB = judge(rB, p) if rB else None
        row = {
            "profile": p["student_name"],
            "risk": p["risk_level"],
            "emotion": p.get("emotion_data", {}).get("fused_emotion"),
            "A": {"mean": mean_score(sA), "scores": sA},
            "B": {"mean": mean_score(sB), "scores": sB},
        }
        results.append(row)
        print(f"[{i+1}/{min(len(test), max_test)}] {p['student_name']}({p['risk_level']}) "
              f"A={row['A']['mean']} B={row['B']['mean']}")

    out = {
        "meta": {"seed_n": len(seed), "test_n": len(results), "pool_size": pool,
                 "retrieval": "semantic"},
        "results": results,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[A/B-语义] 结果已保存: {OUT}")


if __name__ == "__main__":
    main()
