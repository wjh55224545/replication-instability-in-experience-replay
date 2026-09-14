"""
13b 试点 · 客观分诊任务难度校准 (triage_pilot.py)
================================================
正式实验前必须先确认任务**有难度梯度**：
- 若 baseline 准确率 ≈ 100% → 方差为零，实验构造性失败，需加难
- 若 65–85% → 正好，开跑正式实验
- 若 < 50% → 太难，噪声主导，需减难度

只跑 baseline（无注入），不调 judge，用逐字段精确匹配。

运行：
    python scripts/paper/triage_pilot.py
    python scripts/paper/triage_pilot.py 20    # 只用前 20 个档案
"""

from __future__ import annotations

import os
import sys
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import OpenAI                                   # noqa: E402
from backend.config import get_settings                     # noqa: E402
from concurrency import pmap, DEFAULT_WORKERS               # noqa: E402
from triage_task import (                                   # noqa: E402
    build_user_prompt, ground_truth, parse_pred, score_one,
    SYSTEM_PROMPT, FIELDS,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
N_REPEAT = 3        # 每个档案重复几次，用于估计「单次生成的方差」


def make_caller():
    s = get_settings()
    client = OpenAI(base_url=s.lingshu_base_url, api_key=s.lingshu_api_key)
    model = s.lingshu_model

    def call(system: str, user: str, temperature: float) -> str | None:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=256,
        )
        return r.choices[0].message.content
    return call


def main(n_profiles: int | None = None):
    profiles = json.load(open(os.path.join(DATA, "paper_dataset.json"), encoding="utf-8"))
    if n_profiles:
        profiles = profiles[:n_profiles]
    call = make_caller()

    # 展开成 (档案, 第几次重复) 的扁平任务列表，交给 pmap 并行
    jobs = [(p, r) for p in profiles for r in range(N_REPEAT)]

    def run_one(job):
        p, r = job
        raw = call(SYSTEM_PROMPT, build_user_prompt(p), 0.7)
        pred = parse_pred(raw)
        return {"name": p["student_name"], "rep": r,
                "truth": ground_truth(p), "pred": pred,
                "score": score_one(pred, ground_truth(p))}

    t0 = time.time()
    print(f"试点：{len(profiles)} 档案 × {N_REPEAT} 次重复 = {len(jobs)} 次调用")
    print(f"并发 {DEFAULT_WORKERS}，预计 {len(jobs)*35/DEFAULT_WORKERS/60:.1f} 分钟\n")
    results = pmap(run_one, jobs, label="试点")
    ok = [r for r in results if r]
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟，成功 {len(ok)}/{len(jobs)}\n")

    if not ok:
        print("全部失败，检查 API 配置。")
        return

    # 逐字段准确率
    print("=" * 58)
    print("逐字段准确率")
    print("-" * 58)
    for f in FIELDS:
        acc = sum(r["score"][f] for r in ok) / len(ok)
        print(f"  {f:16s}: {acc:6.1%}")
    full = sum(r["score"]["all"] for r in ok) / len(ok)
    print(f"  {'全部字段正确':14s}: {full:6.1%}")
    print("=" * 58)

    # 同一档案多次重复之间是否一致 → 反映「单次生成的方差」
    by_name: dict = {}
    for r in ok:
        by_name.setdefault(r["name"], []).append(r["score"]["all"])
    consistent = sum(1 for v in by_name.values() if len(set(v)) == 1)
    print(f"\n同档案 {N_REPEAT} 次重复结果完全一致: "
          f"{consistent}/{len(by_name)} ({consistent/len(by_name):.0%})")

    # 与主观任务的量程对比
    print("\n=== 量程对比（决定噪声/信号比）===")
    print(f"  本任务：依赖变量 = 每档案 0–4 个字段正确（45% 最大类占比）")
    print(f"  主观任务：96 个分数里 73 个恰好是 8.2，全域 7.2–8.6")

    print("\n【判定】")
    if full >= 0.95:
        print("  [!!] baseline 准确率过高，方差趋零 —— 任务太简单，需要加难")
        print("       建议：增加字段、嵌套规则，或改为「从原始序列推导指标」")
    elif full >= 0.65:
        print("  [OK] 难度合适，可以开跑正式实验")
    else:
        print("  [!!] 准确率偏低 —— 任务过难，噪声可能主导，建议降低难度")

    out = os.path.join(DATA, "triage_pilot_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"n_profiles": len(profiles), "n_repeat": N_REPEAT,
                   "field_acc": {f: sum(r["score"][f] for r in ok) / len(ok) for f in FIELDS},
                   "full_acc": full, "results": ok}, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
