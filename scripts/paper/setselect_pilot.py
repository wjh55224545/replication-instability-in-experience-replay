"""
13b 路线A 试点 · 集合选择任务难度校准 (setselect_pilot.py)
========================================================
正式实验前确认任务有可用量程。

判定标准（与主观任务对照）：
- 随机基线 0.450
- 若准确率 ≈ 1.00 → 天花板，方差归零，任务不可用
- 若 0.65–0.90 → 可用
- 若 ≤ 0.50 → 模型在做错，噪声主导，不可用

运行：
    python scripts/paper/setselect_pilot.py
    python scripts/paper/setselect_pilot.py deepseek
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
from setselect_task import (                                # noqa: E402
    assign_ids, make_groups, ground_truth, build_user_prompt,
    parse_pred, score_one, SYSTEM_PROMPT, GROUP_SIZE, SELECT_K,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
N_GROUPS = 30
N_REPEAT = 3


def make_caller(model_name: str):
    s = get_settings()
    if model_name == "deepseek":
        base, key, model = s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model
    else:
        base, key, model = s.lingshu_base_url, s.lingshu_api_key, s.lingshu_model
    client = OpenAI(base_url=base, api_key=key)

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


def main(model_name: str = "lingshu"):
    profiles = assign_ids(json.load(open(os.path.join(DATA, "paper_dataset.json"),
                                         encoding="utf-8")))
    groups = make_groups(profiles, n_groups=N_GROUPS)
    call = make_caller(model_name)

    jobs = [(gi, g, r) for gi, g in enumerate(groups) for r in range(N_REPEAT)]

    def run_one(job):
        gi, g, r = job
        raw = call(SYSTEM_PROMPT, build_user_prompt(g), 0.7)
        pred = parse_pred(raw)
        truth = ground_truth(g)
        return {"group": gi, "rep": r, "pred": pred, "truth": truth,
                "score": score_one(pred, truth)}

    t0 = time.time()
    print(f"试点[{model_name}]：{N_GROUPS} 组 × {N_REPEAT} 次 = {len(jobs)} 次调用，"
          f"并发 {DEFAULT_WORKERS}\n")
    results = pmap(run_one, jobs, label="试点")
    ok = [r for r in results if r and r["score"] is not None]
    parsed_fail = sum(1 for r in results if r and r["score"] is None)
    print(f"\n完成，用时 {(time.time()-t0)/60:.1f} 分钟，"
          f"有效 {len(ok)}/{len(jobs)}，解析失败 {parsed_fail}\n")

    if not ok:
        print("全部失败。")
        return

    import statistics
    scores = [r["score"] for r in ok]
    mean = statistics.mean(scores)
    sd = statistics.pstdev(scores)

    print("=" * 58)
    print(f"平均重合度      : {mean:.3f}   (随机基线 0.450)")
    print(f"标准差          : {sd:.3f}")
    print(f"范围            : {min(scores):.2f} ~ {max(scores):.2f}")
    from collections import Counter
    print(f"取值分布        : {dict(sorted(Counter(scores).items()))}")
    print("=" * 58)

    # 同组重复之间的一致性
    by_g: dict = {}
    for r in ok:
        by_g.setdefault(r["group"], []).append(r["score"])
    consistent = sum(1 for v in by_g.values() if len(set(v)) == 1)
    within_sd = statistics.mean(
        [statistics.pstdev(v) if len(v) > 1 else 0 for v in by_g.values()])
    print(f"\n同组 {N_REPEAT} 次重复完全一致 : {consistent}/{len(by_g)} "
          f"({consistent/len(by_g):.0%})")
    print(f"组内重复的标准差(平均)     : {within_sd:.3f}")

    # 逐组均值
    gmeans = {g: statistics.mean(v) for g, v in by_g.items()}
    vals = list(gmeans.values())
    print(f"跨组均值分布               : 均值 {statistics.mean(vals):.3f}, "
          f"标准差 {statistics.pstdev(vals):.3f}, "
          f"范围 {min(vals):.2f}~{max(vals):.2f}")

    print("\n【判定】")
    if mean >= 0.97:
        print("  [!!] 天花板 —— 方差趋零，任务不可用")
    elif 0.65 <= mean <= 0.95:
        print("  [OK] 量程可用，可以开跑正式实验")
    elif mean < 0.52:
        print("  [!!] 接近或低于随机 —— 模型在做错，噪声主导，不可用")
    else:
        print("  [?] 偏低但有量程，需判断噪声来源")

    out = os.path.join(DATA, f"setselect_pilot_{model_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "n_groups": N_GROUPS, "n_repeat": N_REPEAT,
                   "mean": mean, "sd": sd, "parsed_fail": parsed_fail,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lingshu")
