"""
论文实验 · 同配置重复运行 (replicate_runs.py)
==========================================
回应论文核心论点的证据缺口：同一配置目前只跑了 2 次。
本脚本以固定配置（模型 / 池子大小 / 重复次数 / 条件完全相同，仅随机种子不同）
再运行 N 次，用于刻画效应估计的分布。

用法：
    python scripts/paper/replicate_runs.py deepseek 2
    python scripts/paper/replicate_runs.py deepseek 3
    python scripts/paper/replicate_runs.py deepseek 4

输出 data/deepseek_replicate_run<N>.json
"""

from __future__ import annotations
import os
import re
import json
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import OpenAI  # noqa: E402
from backend.config import get_settings  # noqa: E402
import backend.evolution.memory as mem  # noqa: E402
from common import (  # noqa: E402
    load_profiles, build_user_prompt, build_system_prompt, build_evolution_context,
    build_evolution_context_semantic, record_feedback, JUDGE_SYS, DIMENSIONS,
    mean_score, CATEGORY,
)
from run_ab_experiment import split_balanced  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def make_caller(model_name):
    s = get_settings()
    if model_name == "deepseek":
        base, key, model = s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model
    elif model_name == "lingshu":
        base, key, model = s.lingshu_base_url, s.lingshu_api_key, s.lingshu_model
    else:
        raise ValueError(model_name)

    def call(sp, up, temperature):
        client = OpenAI(base_url=base, api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sp}, {"role": "user", "content": up}],
            temperature=temperature,
            max_tokens=2048 if temperature > 0 else 256,
        )
        return resp.choices[0].message.content
    return call


def parse_scores(raw):
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


def main(model_name="deepseek", run_tag="2", seed_n=15, max_test=29,
         n_repeat=3, pool_mult=3, seed_offset=0):
    OUT = os.path.join(DATA_DIR, f"{model_name}_replicate_run{run_tag}.json")
    EXP = os.path.join(DATA_DIR, f"{model_name}_replicate_run{run_tag}_exp.jsonl")
    mem.EXPERIENCE_PATH = EXP
    if os.path.exists(EXP):
        os.remove(EXP)

    call = make_caller(model_name)

    # 用不同的随机偏移构造略有差异的档案顺序 / 采样
    rng = random.Random(1000 + int(run_tag) + seed_offset)
    profiles = load_profiles()
    rng.shuffle(profiles)
    seed, test = split_balanced(profiles, seed_n)
    test = test[:max_test]
    print(f"[replicate-{run_tag}] 模型={model_name}, 预热={len(seed)}, 测试={len(test)}, 重复={n_repeat}")

    # 预热建池（同样 3 倍，使池子大小 = 45）
    for p in seed:
        for _ in range(pool_mult):
            sp = build_system_prompt(False, "", "system")
            rep = call(sp, build_user_prompt(p), 0.7)
            sc = parse_scores(call(JUDGE_SYS,
                                   f"## 待评估报告\n\n{rep}\n\n## 输入数据\n\n{json.dumps(p, ensure_ascii=False)}\n\n请打分。",
                                   0.0)) if rep else None
            m = mean_score(sc)
            if m is not None and m >= 4.0:
                record_feedback(CATEGORY, p["student_name"], rating=int(round(m)), outcome=rep[:200],
                                risk_level=p["risk_level"],
                                emotion=p.get("emotion_data", {}).get("fused_emotion"))

    pool_size = sum(1 for _ in open(EXP, encoding="utf-8"))
    print(f"[replicate-{run_tag}] 经验池: {pool_size}")

    results = []
    for i, p in enumerate(test):
        row = {"profile": p["student_name"], "risk": p["risk_level"]}
        for cond in ["baseline", "random", "semantic"]:
            scores = []
            for _ in range(n_repeat):
                if cond == "baseline":
                    ctx, use = "", False
                elif cond == "random":
                    ctx, use = build_evolution_context(CATEGORY), True
                else:
                    ctx = build_evolution_context_semantic(CATEGORY, p)
                    use = bool(ctx)
                sp = build_system_prompt(use, ctx, "system")
                rep = call(sp, build_user_prompt(p), 0.7)
                sc = parse_scores(call(JUDGE_SYS,
                                       f"## 待评估报告\n\n{rep}\n\n## 输入数据\n\n{json.dumps(p, ensure_ascii=False)}\n\n请打分。",
                                       0.0)) if rep else None
                m = mean_score(sc)
                if m is not None:
                    scores.append(m)
            row[cond] = scores
            row[f"{cond}_mean"] = round(sum(scores) / len(scores), 3) if scores else None
        results.append(row)
        if (i + 1) % 5 == 0 or i == 0:
            print(f"[{i+1}/{len(test)}] {p['student_name']}({p['risk_level']}) "
                  f"B={row['baseline_mean']} R={row['random_mean']} S={row['semantic_mean']}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": model_name, "run_tag": run_tag,
                            "pool_size": pool_size, "n_repeat": n_repeat},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[replicate-{run_tag}] 已保存: {OUT}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    tag = sys.argv[2] if len(sys.argv) > 2 else "2"
    main(model_name=model, run_tag=tag)
