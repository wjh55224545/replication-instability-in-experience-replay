"""
论文实验 1 · A/B 对照 (run_ab_experiment.py)
============================================

验证 RQ1：人在回路反馈驱动的经验注入（进化上下文）能否显著提升生成质量。

流程：
1. 分层抽样：把 44 个档案分成「种子组」与「测试组」（风险等级均衡）。
2. 预热：种子组生成报告 → LLM-as-judge 打分 → 高分(≥4)经验入经验库。
3. A/B：测试组每个档案各生成两份报告——
   - 条件 A（baseline）：无进化上下文
   - 条件 B（evolution）：注入预热得到的高分经验
4. 双条件各做 judge 打分，输出配对结果到 data/ab_results.json。

运行（在本仓库根目录）：python scripts/paper/run_ab_experiment.py
"""

from __future__ import annotations
import os
import json
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from common import (  # noqa: E402
    load_profiles,
    generate_report,
    judge,
    mean_score,
    build_evolution_context,
    record_feedback,
    CATEGORY,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "ab_results.json")


def split_balanced(profiles, seed_n):
    """按风险等级分层，种子组与测试组都保持绿/黄/红均衡（测试组轮询交错）。"""
    by_level = {"green": [], "yellow": [], "red": []}
    for p in profiles:
        by_level.setdefault(p["risk_level"], []).append(p)

    rng = random.Random(42)
    for lv in by_level:
        rng.shuffle(by_level[lv])

    seed, test = [], []
    per_seed = seed_n // 3  # 每层种子数
    for lv in ["green", "yellow", "red"]:
        seed.extend(by_level[lv][:per_seed])
        by_level[lv] = by_level[lv][per_seed:]
    # 剩余补足 seed（若 seed_n 不能被 3 整除）
    while len(seed) < seed_n:
        for lv in ["green", "yellow", "red"]:
            if len(seed) >= seed_n:
                break
            if by_level[lv]:
                seed.append(by_level[lv].pop(0))
    # 测试组轮询交错，保证 test[:k] 覆盖所有风险等级
    while any(by_level.values()):
        for lv in ["green", "yellow", "red"]:
            if by_level[lv]:
                test.append(by_level[lv].pop(0))
    return seed, test


def main(seed_n: int = 15, max_test: int = 29):
    profiles = load_profiles()
    seed, test = split_balanced(profiles, seed_n)
    print(f"[A/B] 种子组 {len(seed)} 个, 测试组 {len(test)} 个")

    # ---- 1. 预热：种子组生成 + 打分 + 高分入经验库 ----
    print("[A/B] 预热：种子组生成经验池...")
    pool = 0
    for p in seed:
        report = generate_report(p, use_evolution=False)
        if not report:
            continue
        scores = judge(report, p)
        if scores is None:
            continue
        m = mean_score(scores)
        if m is not None and m >= 4.0:
            record_feedback(CATEGORY, p["student_name"], rating=int(round(m)), outcome=report[:200])
            pool += 1
    print(f"[A/B] 预热完成，高分经验入池 {pool} 条")

    evo_ctx = build_evolution_context(CATEGORY)
    print(f"[A/B] 进化上下文长度 {len(evo_ctx)} 字符（{'有' if evo_ctx else '无'}历史经验）")

    # ---- 2. A/B 对比 ----
    results = []
    for i, p in enumerate(test[:max_test]):
        rA = generate_report(p, use_evolution=False)
        rB = generate_report(p, use_evolution=True, evolution_context=evo_ctx)
        sA = judge(rA, p) if rA else None
        sB = judge(rB, p) if rB else None
        row = {
            "profile": p["student_name"],
            "risk": p["risk_level"],
            "A": {"mean": mean_score(sA), "scores": sA},
            "B": {"mean": mean_score(sB), "scores": sB},
        }
        results.append(row)
        print(f"[{i+1}/{min(len(test), max_test)}] {p['student_name']}({p['risk_level']}) "
              f"A={row['A']['mean']} B={row['B']['mean']}")

    out = {
        "meta": {"seed_n": len(seed), "test_n": len(results), "pool_size": pool,
                 "evolution_context_len": len(evo_ctx)},
        "results": results,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[A/B] 结果已保存: {OUT}")


if __name__ == "__main__":
    main()
