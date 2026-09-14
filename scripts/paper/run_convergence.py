"""
论文实验 2 · 进化收敛 (run_convergence.py)
==========================================

验证 RQ3：反馈闭环在稀疏/累积反馈下是否收敛（质量随反馈轮次提升并稳定）。

流程：
1. 取 n_profiles 个测试档案，使用独立的经验库（避免污染 A/B 实验）。
2. 多轮循环：每轮对每个档案「生成 → LLM-as-judge 打分 → 高分经验入库」，
   下一轮生成时注入累积的高分经验。
3. 记录每轮平均分，输出收敛曲线数据到 data/convergence_results.json。

运行（在本仓库根目录）：python scripts/paper/run_convergence.py
"""

from __future__ import annotations
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backend.evolution.memory as mem  # noqa: E402
from common import load_profiles, generate_report, judge, mean_score, CATEGORY  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(DATA_DIR, "convergence_results.json")
# 独立经验库，避免污染主 experience.jsonl
CONV_EXP = os.path.join(DATA_DIR, "convergence_experience.jsonl")


def main(n_profiles: int = 8, n_rounds: int = 5):
    # 切到独立经验库
    mem.EXPERIENCE_PATH = CONV_EXP
    if os.path.exists(CONV_EXP):
        os.remove(CONV_EXP)

    profiles = load_profiles()[:n_profiles]
    rounds = []

    for r in range(n_rounds):
        scores = []
        for p in profiles:
            evo_ctx = mem.build_evolution_context(CATEGORY)
            report = generate_report(p, use_evolution=True, evolution_context=evo_ctx)
            s = judge(report, p)
            m = mean_score(s)
            if m is not None:
                scores.append(m)
                if m >= 4.0:
                    mem.record_feedback(
                        CATEGORY, p["student_name"], rating=int(round(m)), outcome=(report or "")[:200]
                    )
        round_mean = round(sum(scores) / len(scores), 3) if scores else 0.0
        pool = sum(1 for _ in open(CONV_EXP, encoding="utf-8")) if os.path.exists(CONV_EXP) else 0
        rounds.append({"round": r + 1, "mean": round_mean, "pool_size": pool, "scores": scores})
        print(f"[收敛] 轮次 {r+1}: 平均分={round_mean}, 经验池={pool}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": {"n_profiles": n_profiles, "n_rounds": n_rounds}, "rounds": rounds},
                  f, ensure_ascii=False, indent=2)
    print(f"[收敛] 结果已保存: {OUT}")


if __name__ == "__main__":
    main()
