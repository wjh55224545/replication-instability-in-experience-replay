"""
论文实验 3 · 机制消融 (run_ablation.py)
========================================

验证 RQ2：进化机制的关键设计参数如何影响生成质量。

三个子消融（固定其它参数，单一变量）：
- top-k        ∈ {1, 2, 3}   （min_rating=4, position=system）
- min_rating   ∈ {3, 4, 5}   （top_k=2, position=system）
- 注入位置      ∈ {system, user}（top_k=2, min_rating=4）

流程：预热经验库 → 对每个配置用同一批档案生成 + judge 打分 → 输出 data/ablation_results.json。

运行（在本仓库根目录）：python scripts/paper/run_ablation.py
"""

from __future__ import annotations
import os
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backend.evolution.memory as mem  # noqa: E402
from common import load_profiles, generate_report, judge, mean_score, CATEGORY  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(DATA_DIR, "ablation_results.json")
ABL_EXP = os.path.join(DATA_DIR, "ablation_experience.jsonl")


def custom_evolution_context(category: str, top_k: int, min_rating: int) -> str:
    """可调参数的进化上下文构造（等价于 build_evolution_context 但参数可调）。"""
    exps = mem.retrieve_experience(category=category, min_rating=min_rating, limit=top_k)
    if not exps:
        return ""
    lines = ["\n## 历史成功案例参考（系统从反馈中自动学习）"]
    for i, exp in enumerate(exps, 1):
        lines.append(
            f"{i}. 对象: {exp.get('student_name', '未知')} | "
            f"评分: {exp.get('rating', '?')}/5\n"
            f"   生成效果: {exp.get('outcome_preview', '暂无详情')[:120]}"
        )
    lines.append("\n请参考以上案例的风格和深度，为当前学生生成同样高质量的内容。")
    return "\n".join(lines)


def warmup(profiles):
    """预热经验库：生成 + 打分 + 记录，使经验库有不同评分段的样本。"""
    for p in profiles:
        report = generate_report(p, use_evolution=False)
        if not report:
            continue
        s = judge(report, p)
        m = mean_score(s)
        if m is not None:
            mem.record_feedback(CATEGORY, p["student_name"], rating=int(round(m)), outcome=report[:200])


def run_config(name, profiles, top_k, min_rating, position):
    """对某个参数配置，用同一批档案生成并打分，返回平均分。"""
    scores = []
    for p in profiles:
        ctx = custom_evolution_context(CATEGORY, top_k, min_rating)
        report = generate_report(p, use_evolution=True, evolution_context=ctx, inject_position=position)
        s = judge(report, p)
        m = mean_score(s)
        if m is not None:
            scores.append(m)
    avg = round(sum(scores) / len(scores), 3) if scores else 0.0
    print(f"[消融] {name}: 平均分={avg} (n={len(scores)})")
    return {"name": name, "top_k": top_k, "min_rating": min_rating,
            "position": position, "mean": avg, "n": len(scores), "scores": scores}


def main(n_warmup: int = 8, n_per_config: int = 5):
    mem.EXPERIENCE_PATH = ABL_EXP
    if os.path.exists(ABL_EXP):
        os.remove(ABL_EXP)

    profiles = load_profiles()
    warmup(profiles[:n_warmup])
    test = profiles[n_warmup:n_warmup + n_per_config]
    print(f"[消融] 预热 {n_warmup} 个, 每个配置测试 {len(test)} 个")

    results = []
    # 子消融 1：top-k
    for k in [1, 2, 3]:
        results.append(run_config(f"top_k={k}", test, top_k=k, min_rating=4, position="system"))
    # 子消融 2：min_rating
    for mr in [3, 4, 5]:
        results.append(run_config(f"min_rating={mr}", test, top_k=2, min_rating=mr, position="system"))
    # 子消融 3：注入位置
    for pos in ["system", "user"]:
        results.append(run_config(f"position={pos}", test, top_k=2, min_rating=4, position=pos))

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": {"n_warmup": n_warmup, "n_per_config": n_per_config}, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"[消融] 结果已保存: {OUT}")


if __name__ == "__main__":
    # 用法: run_ablation.py [n_per_config]
    # 增大 n_per_config 可提升统计效力（审稿意见：n=5 效力不足）。
    # 注意：扩大样本后结论可能改变（任一方向），须如实报告。
    import sys as _sys
    n = int(_sys.argv[1]) if len(_sys.argv) > 1 else 5
    main(n_per_config=n)
