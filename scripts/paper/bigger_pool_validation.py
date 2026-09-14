"""
论文实验 · 大经验池验证 (bigger_pool_validation.py)
================================================
回应「池子太小导致 semantic 退化为重复展示」的混淆：
用同一个 15-profile 预热组、但每个 profile 生成 3 次，把经验池从 15 扩到 ~45 条，
使语义检索有更多匹配候选。其余设计与 run_ab_repeated.py 完全一致。

用法：
    python scripts/paper/bigger_pool_validation.py deepseek
    python scripts/paper/bigger_pool_validation.py lingshu

输出 data/<model>_bigpool_results.json
"""

from __future__ import annotations
import os
import re
import json
import time
import sys

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


def make_caller(model_name, judge_name=None):
    """返回 call(system, user, temperature)。

    model_name: 生成模型；judge_name: 评判模型（默认与生成同模型自评；
    glm 默认改用 DeepSeek 评判，避免 GLM 天花板的区分度问题）。
    """
    s = get_settings()
    judge_name = judge_name or ("deepseek" if model_name == "glm" else model_name)
    _cfg = {
        "deepseek": (s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model),
        "lingshu": (s.lingshu_base_url, s.lingshu_api_key, s.lingshu_model),
        "glm": (s.zhipu_base_url, s.zhipu_api_key, "glm-4.5-flash"),
    }
    base, key, model = _cfg[model_name]
    jbase, jkey, jmodel = _cfg[judge_name]

    def call(system_prompt, user_prompt, temperature):
        """temperature > 0 视为生成（用生成模型），= 0 视为评判（用评判模型）。"""
        if temperature > 0:
            client, m = OpenAI(base_url=base, api_key=key), model
        else:
            client, m = OpenAI(base_url=jbase, api_key=jkey), jmodel
        resp = client.chat.completions.create(
            model=m,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
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


def main(model_name="deepseek", seed_n=15, max_test=29, n_repeat=3, pool_mult=3):
    OUT = os.path.join(DATA_DIR, f"{model_name}_bigpool_results.json")
    EXP = os.path.join(DATA_DIR, f"{model_name}_bigpool_experience.jsonl")
    mem.EXPERIENCE_PATH = EXP
    if os.path.exists(EXP):
        os.remove(EXP)

    call = make_caller(model_name)
    profiles = load_profiles()
    seed, test = split_balanced(profiles, seed_n)
    test = test[:max_test]
    print(f"[大池子-{model_name}] 预热 {len(seed)}×{pool_mult} 次, 测试 {len(test)}, 重复 {n_repeat}")

    # 预热：每个 seed profile 生成 pool_mult 次，扩大经验池
    for i, p in enumerate(seed):
        for k in range(pool_mult):
            sp = build_system_prompt(False, "", "system")
            report = call(sp, build_user_prompt(p), 0.7)
            sc = parse_scores(call(JUDGE_SYS,
                                   f"## 待评估报告\n\n{report}\n\n## 输入数据\n\n{json.dumps(p, ensure_ascii=False)}\n\n请打分。",
                                   0.0)) if report else None
            m = mean_score(sc)
            if m is not None and m >= 4.0:
                record_feedback(CATEGORY, p["student_name"], rating=int(round(m)), outcome=report[:200],
                                risk_level=p["risk_level"],
                                emotion=p.get("emotion_data", {}).get("fused_emotion"))
        print(f"  预热进度 {i+1}/{len(seed)}，池子当前 {sum(1 for _ in open(EXP, encoding='utf-8'))} 条")

    pool_size = sum(1 for _ in open(EXP, encoding="utf-8"))
    random_ctx = build_evolution_context(CATEGORY)
    print(f"[大池子-{model_name}] 经验池大小: {pool_size}")

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
                report = call(sp, build_user_prompt(p), 0.7)
                sc = parse_scores(call(JUDGE_SYS,
                                       f"## 待评估报告\n\n{report}\n\n## 输入数据\n\n{json.dumps(p, ensure_ascii=False)}\n\n请打分。",
                                       0.0)) if report else None
                m = mean_score(sc)
                if m is not None:
                    scores.append(m)
            row[cond] = scores
            row[f"{cond}_mean"] = round(sum(scores) / len(scores), 3) if scores else None
        results.append(row)
        print(f"[{i+1}/{len(test)}] {p['student_name']}({p['risk_level']}) "
              f"B={row['baseline_mean']} R={row['random_mean']} S={row['semantic_mean']}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": model_name, "pool_size": pool_size,
                            "seed_n": len(seed), "test_n": len(results), "n_repeat": n_repeat},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[大池子-{model_name}] 结果已保存: {OUT}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    main(model_name=model)
