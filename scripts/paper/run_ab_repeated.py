"""
论文实验 · A/B 对照（重复 3 次 + 共享 baseline）(run_ab_repeated.py)
====================================================================
回应评审 A3：每个 profile × condition 重复生成 3 次，random 与 semantic 共享
同一个 no-injection baseline，消除单次随机解码噪声，使效应估计更稳健。

输出 data/ab_repeated_results.json：
- 每个 profile 的 baseline/random/semantic 各 3 次分数 + 均值

运行（在本仓库根目录）：python scripts/paper/run_ab_repeated.py
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
    build_evolution_context,
    record_feedback,
    CATEGORY,
)
from run_ab_experiment import split_balanced  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(DATA_DIR, "ab_repeated_results.json")
REP_EXP = os.path.join(DATA_DIR, "ab_repeated_experience.jsonl")


def warmup(profiles):
    """预热经验池（用 seed 组生成 + 打分 + 记录高分经验）。"""
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


def repeat_generate(profile, condition, random_ctx, n_repeat):
    """对某个 condition 重复生成 n_repeat 次，返回 [mean_score, ...]。"""
    scores = []
    for _ in range(n_repeat):
        if condition == "baseline":
            report = generate_report(profile, use_evolution=False)
        elif condition == "random":
            report = generate_report(profile, use_evolution=True, evolution_context=random_ctx)
        else:  # semantic
            report = generate_report_semantic(profile)
        s = judge(report, profile)
        m = mean_score(s)
        if m is not None:
            scores.append(m)
    return scores


def main(seed_n=15, max_test=29, n_repeat=3):
    mem.EXPERIENCE_PATH = REP_EXP
    if os.path.exists(REP_EXP):
        os.remove(REP_EXP)

    profiles = load_profiles()
    seed, test = split_balanced(profiles, seed_n)
    test = test[:max_test]
    print(f"[重复A/B] 预热 {len(seed)} 个, 测试 {len(test)} 个, 重复 {n_repeat} 次")

    warmup(seed)
    random_ctx = build_evolution_context(CATEGORY)
    print(f"[重复A/B] random 注入上下文长度: {len(random_ctx)}")

    results = []
    for i, p in enumerate(test):
        base = repeat_generate(p, "baseline", random_ctx, n_repeat)
        rand = repeat_generate(p, "random", random_ctx, n_repeat)
        semi = repeat_generate(p, "semantic", random_ctx, n_repeat)
        row = {
            "profile": p["student_name"],
            "risk": p["risk_level"],
            "baseline": base,
            "random": rand,
            "semantic": semi,
            "baseline_mean": round(sum(base) / len(base), 3) if base else None,
            "random_mean": round(sum(rand) / len(rand), 3) if rand else None,
            "semantic_mean": round(sum(semi) / len(semi), 3) if semi else None,
        }
        results.append(row)
        print(f"[{i+1}/{len(test)}] {p['student_name']}({p['risk_level']}) "
              f"B={row['baseline_mean']} R={row['random_mean']} S={row['semantic_mean']}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {"seed_n": len(seed), "test_n": len(results), "n_repeat": n_repeat},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[重复A/B] 结果已保存: {OUT}")


if __name__ == "__main__":
    main()
