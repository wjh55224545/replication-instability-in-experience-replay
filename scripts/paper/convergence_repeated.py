"""
收敛实验 · 5 次独立运行 (convergence_repeated.py)
===============================================
论文 §5.7 报的收敛结果（8.200 → 7.850 → 8.050 → 8.025 → 8.175，"无单调趋势"）
是**一次运行**的结果，而它涉及 temperature=0.7 的随机生成 —— 属于会翻转的那类证据。

本脚本跑 5 次独立运行，报告每轮均值的分布，检验「无单调趋势」是否稳定。

并行化说明
----------
原脚本在每个档案生成后**立即入库**，于是同一轮内后面的档案会看到前面档案的记录。
这与论文「每轮使用累积经验池」的描述略有出入。本脚本在**每轮开始时**统一取上下文，
轮内在并行执行，轮末统一入库 —— 池子仍然每轮 +8（8 → 16 → 24 → 32 → 40），
与论文报告的池子增长完全一致，但消除了轮内的顺序依赖（该依赖在并行下无法保留，
且未在论文中描述）。

运行（在本仓库根目录）：python scripts/paper/convergence_repeated.py
"""

from __future__ import annotations
import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backend.evolution.memory as mem                       # noqa: E402
from common import load_profiles, generate_report, judge, mean_score, CATEGORY  # noqa: E402
from concurrency import pmap, DEFAULT_WORKERS                # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
N_PROFILES = 8
N_ROUNDS = 5
N_RUNS = 5


def one_run(run_id: int, profiles: list) -> dict:
    exp = os.path.join(DATA, f"convergence_run{run_id}_exp.jsonl")
    mem.EXPERIENCE_PATH = exp
    if os.path.exists(exp):
        os.remove(exp)

    rounds = []
    for r in range(N_ROUNDS):
        # 轮开始：统一取上下文（此时池子状态对本轮所有档案一致）
        ctx = mem.build_evolution_context(CATEGORY)

        def cell(p):
            rep = generate_report(p, use_evolution=True, evolution_context=ctx)
            m = mean_score(judge(rep, p))
            return p, rep, m

        results = pmap(cell, profiles, label=f"run{run_id} 轮{r+1}")

        # 轮末：统一入库（串行写，避免并发写 JSONL 交错）
        scores = []
        for item in results:
            if not item:
                continue
            p, rep, m = item
            if m is not None:
                scores.append(m)
                if m >= 4.0:
                    mem.record_feedback(CATEGORY, p["student_name"],
                                        rating=int(round(m)), outcome=(rep or "")[:200])

        pool = sum(1 for _ in open(exp, encoding="utf-8")) if os.path.exists(exp) else 0
        rm = round(sum(scores) / len(scores), 3) if scores else 0.0
        rounds.append({"round": r + 1, "mean": rm, "pool_size": pool, "scores": scores})
        print(f"[run{run_id}] 轮{r+1}: 均值={rm}　池={pool}", flush=True)
    return {"run": run_id, "rounds": rounds}


def is_monotone(means: list) -> str:
    """判断趋势类型。"""
    d = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    if all(x > 0 for x in d):
        return "严格上升"
    if all(x < 0 for x in d):
        return "严格下降"
    if all(x >= 0 for x in d):
        return "非降"
    if all(x <= 0 for x in d):
        return "非增"
    return "无单调趋势"


def main():
    profiles = load_profiles()[:N_PROFILES]
    print(f"收敛实验复现：{N_RUNS} 次独立运行 × {N_ROUNDS} 轮 × {N_PROFILES} 档案")
    print(f"并发 {DEFAULT_WORKERS}\n")

    t0 = time.time()
    runs = [one_run(i, profiles) for i in range(1, N_RUNS + 1)]
    total = (time.time() - t0) / 60

    print(f"\n{'='*72}")
    print("各次运行的逐轮均值")
    print(f"{'='*72}")
    print(f"{'运行':<6}" + "".join(f"{'轮'+str(r):>10}" for r in range(1, N_ROUNDS + 1))
          + f"{'趋势':>14}")
    print("-" * 72)
    for run in runs:
        m = [x["mean"] for x in run["rounds"]]
        print(f"run{run['run']:<3}" + "".join(f"{v:>10.3f}" for v in m)
              + f"{is_monotone(m):>14}")

    # 汇总
    allm = [[x["mean"] for x in run["rounds"]] for run in runs]
    print("-" * 72)
    avg = [sum(m[i] for m in allm) / len(allm) for i in range(N_ROUNDS)]
    print(f"{'均值':<6}" + "".join(f"{v:>10.3f}" for v in avg) + f"{is_monotone(avg):>14}")

    trends = [is_monotone(m) for m in allm]
    print(f"\n【判定】")
    print(f"  各次运行的趋势：{trends}")
    n_mono = sum(1 for t in trends if t != "无单调趋势")
    print(f"  出现单调趋势的运行：{n_mono}/{N_RUNS}")
    rng = [max(m) - min(m) for m in allm]
    print(f"  各次运行内的极差：{[round(x,3) for x in rng]}")
    print(f"  跨次运行的同轮极差："
          f"{[round(max(m[i] for m in allm) - min(m[i] for m in allm), 3) for i in range(N_ROUNDS)]}")

    out = os.path.join(DATA, "convergence_repeated.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_runs": N_RUNS, "n_rounds": N_ROUNDS, "n_profiles": N_PROFILES,
                   "minutes": round(total, 1), "runs": runs, "round_means": allm,
                   "trends": trends}, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {out}　总用时 {total:.1f} 分钟")


if __name__ == "__main__":
    main()
