"""
论文实验 · 显式多样性验证 (diversity_validation.py)
=================================================
直接检验「多样性 vs 相似性」是否为因果机制：
新增两个条件——
  similar   : 同风险 + 同情绪（最相似，即原 semantic 条件）
  different : 不同风险 + 不同情绪（最不相似，但仍为高质量样本）
  random    : 随机（原 random 条件）
  baseline  : 无注入

若 different ≈ random > similar  → 相似性本身有害（过度锚定）
若 random > different ≈ similar  → 效应来自随机性而非相似性

用法：
    python scripts/paper/diversity_validation.py deepseek

输出 data/<model>_diversity_results.json
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
    record_feedback, JUDGE_SYS, DIMENSIONS, mean_score, CATEGORY,
)
from run_ab_experiment import split_balanced  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def make_caller(model_name):
    s = get_settings()
    if model_name == "deepseek":
        base, key, model = s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model
    elif model_name == "lingshu":
        base, key, model = s.lingshu_base_url, s.lingshu_api_key, s.lingshu_model
    elif model_name == "glm":
        base, key, model = s.zhipu_base_url, s.zhipu_api_key, "glm-4.5-flash"
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


def load_pool():
    exps = []
    if os.path.exists(mem.EXPERIENCE_PATH):
        for line in open(mem.EXPERIENCE_PATH, encoding="utf-8"):
            try:
                exps.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return exps


def ctx_from(pool):
    if not pool:
        return ""
    lines = ["\n## 历史成功案例参考（系统从反馈中自动学习）"]
    for i, e in enumerate(pool, 1):
        lines.append(f"{i}. 对象: {e.get('student_name','未知')} | 评分: {e.get('rating','?')}/5 | "
                     f"风险: {e.get('risk_level','?')} | 情绪: {e.get('emotion','?')}\n"
                     f"   生成效果: {e.get('outcome_preview','')[:120]}")
    lines.append("\n请参考以上案例的风格和深度，为当前学生生成同样高质量的内容。")
    return "\n".join(lines)


def pick(pool, mode, profile):
    """按模式挑选 2 条经验。"""
    if not pool:
        return []
    risk = profile.get("risk_level")
    emo = profile.get("emotion_data", {}).get("fused_emotion")
    if mode == "random":
        return random.sample(pool, min(2, len(pool)))
    if mode == "similar":
        hits = [e for e in pool if e.get("risk_level") == risk and e.get("emotion") == emo]
        if not hits:
            hits = [e for e in pool if e.get("risk_level") == risk]
        return hits[:2] if hits else random.sample(pool, min(2, len(pool)))
    if mode == "different":
        hits = [e for e in pool if e.get("risk_level") != risk and e.get("emotion") != emo]
        if not hits:
            hits = [e for e in pool if e.get("risk_level") != risk]
        return hits[:2] if hits else random.sample(pool, min(2, len(pool)))
    return []


def main(model_name="deepseek", seed_n=15, max_test=29, n_repeat=3, pool_mult=3):
    OUT = os.path.join(DATA_DIR, f"{model_name}_diversity_results.json")
    EXP = os.path.join(DATA_DIR, f"{model_name}_diversity_experience.jsonl")
    mem.EXPERIENCE_PATH = EXP
    if os.path.exists(EXP):
        os.remove(EXP)

    call = make_caller(model_name)
    profiles = load_profiles()
    seed, test = split_balanced(profiles, seed_n)
    test = test[:max_test]
    print(f"[多样性-{model_name}] 预热 {len(seed)}x{pool_mult}, 测试 {len(test)}, 重复 {n_repeat}")

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

    pool = load_pool()
    print(f"[多样性-{model_name}] 经验池: {len(pool)} 条")

    conds = ["baseline", "random", "similar", "different"]
    results = []
    for i, p in enumerate(test):
        row = {"profile": p["student_name"], "risk": p["risk_level"]}
        for cond in conds:
            scores = []
            for _ in range(n_repeat):
                if cond == "baseline":
                    ctx, use = "", False
                else:
                    ctx, use = ctx_from(pick(pool, cond, p)), True
                sp = build_system_prompt(use, ctx, "system")
                rep = call(sp, build_user_prompt(p), 0.7)
                sc = parse_scores(call(JUDGE_SYS,
                                       f"## 待评估报告\n\n{rep}\n\n## 输入数据\n\n{json.dumps(p, ensure_ascii=False)}\n\n请打分。",
                                       0.0)) if rep else None
                m = mean_score(sc)
                if m is not None:
                    scores.append(m)
            row[cond] = scores
            row[f"{cond}_mean"] = round(sum(scores)/len(scores), 3) if scores else None
        results.append(row)
        print(f"[{i+1}/{len(test)}] {p['student_name']}({p['risk_level']}) "
              f"B={row['baseline_mean']} Ran={row['random_mean']} "
              f"Sim={row['similar_mean']} Dif={row['different_mean']}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": model_name, "pool_size": len(pool), "n_repeat": n_repeat},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[多样性-{model_name}] 结果已保存: {OUT}")


if __name__ == "__main__":
    main(model_name=sys.argv[1] if len(sys.argv) > 1 else "deepseek")
