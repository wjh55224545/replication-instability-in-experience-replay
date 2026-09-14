"""
论文实验 · DeepSeek 重复 A/B（共享 baseline + 重复 3 次）(deepseek_repeated.py)
============================================================================
验证「强模型 + 语义检索获益」是否在重复生成下稳健（对应 Lingshu 的
run_ab_repeated.py，两者设计完全一致，便于跨模型对比）。

输出 data/deepseek_repeated_results.json

运行（在本仓库根目录）：python scripts/paper/deepseek_repeated.py
"""

from __future__ import annotations
import os
import re
import json
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import OpenAI  # noqa: E402
from backend.config import get_settings  # noqa: E402
import backend.evolution.memory as mem  # noqa: E402
from common import (  # noqa: E402
    load_profiles,
    build_user_prompt,
    build_system_prompt,
    build_evolution_context,
    build_evolution_context_semantic,
    record_feedback,
    JUDGE_SYS,
    DIMENSIONS,
    mean_score,
    CATEGORY,
)
from run_ab_experiment import split_balanced  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(DATA_DIR, "deepseek_repeated_results.json")
EXP = os.path.join(DATA_DIR, "deepseek_repeated_experience.jsonl")


def ds_call(system_prompt, user_prompt, temperature=0.7, max_tokens=2048):
    s = get_settings()
    client = OpenAI(base_url=s.deepseek_base_url, api_key=s.deepseek_api_key)
    resp = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def ds_judge(report, profile):
    up = (f"## 待评估报告\n\n{report}\n\n"
          f"## 输入数据\n\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n请打分。")
    raw = ds_call(JUDGE_SYS, up, temperature=0.0, max_tokens=256)
    if not raw:
        return None
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        sc = json.loads(m.group(0))
        return {k: int(v) for k, v in sc.items() if k in DIMENSIONS}
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def main(seed_n=15, max_test=29, n_repeat=3):
    mem.EXPERIENCE_PATH = EXP
    if os.path.exists(EXP):
        os.remove(EXP)

    profiles = load_profiles()
    seed, test = split_balanced(profiles, seed_n)
    test = test[:max_test]
    print(f"[DS重复] 预热 {len(seed)}, 测试 {len(test)}, 重复 {n_repeat}")

    # 预热
    for p in seed:
        sp = build_system_prompt(False, "", "system")
        report = ds_call(sp, build_user_prompt(p))
        if not report:
            continue
        s = ds_judge(report, p)
        m = mean_score(s)
        if m is not None and m >= 4.0:
            record_feedback(CATEGORY, p["student_name"], rating=int(round(m)), outcome=report[:200],
                            risk_level=p["risk_level"], emotion=p.get("emotion_data", {}).get("fused_emotion"))
    random_ctx = build_evolution_context(CATEGORY)
    print(f"[DS重复] random 上下文长度: {len(random_ctx)}")

    results = []
    for i, p in enumerate(test):
        row = {"profile": p["student_name"], "risk": p["risk_level"]}
        for cond in ["baseline", "random", "semantic"]:
            scores = []
            for _ in range(n_repeat):
                if cond == "baseline":
                    ctx, use = "", False
                elif cond == "random":
                    ctx, use = random_ctx, True
                else:
                    ctx = build_evolution_context_semantic(CATEGORY, p)
                    use = bool(ctx)
                sp = build_system_prompt(use, ctx, "system")
                report = ds_call(sp, build_user_prompt(p))
                s = ds_judge(report, p) if report else None
                m = mean_score(s)
                if m is not None:
                    scores.append(m)
            row[cond] = scores
            row[f"{cond}_mean"] = round(sum(scores) / len(scores), 3) if scores else None
        results.append(row)
        print(f"[{i+1}/{len(test)}] {p['student_name']}({p['risk_level']}) "
              f"B={row['baseline_mean']} R={row['random_mean']} S={row['semantic_mean']}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": "deepseek", "seed_n": len(seed),
                            "test_n": len(results), "n_repeat": n_repeat},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[DS重复] 结果已保存: {OUT}")


if __name__ == "__main__":
    main()
