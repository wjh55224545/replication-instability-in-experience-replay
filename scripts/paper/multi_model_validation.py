"""
论文 · 多模型验证 (multi_model_validation.py)
==========================================

用指定模型重跑核心 A/B 实验（random vs semantic），验证「随机反馈有害、
语义检索消除危害」这一发现是否跨模型稳健。

用法：
    python scripts/paper/multi_model_validation.py deepseek
    python scripts/paper/multi_model_validation.py zhipu

输出 data/<model>_ab_random.json / data/<model>_ab_semantic.json。
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
    build_evolution_context_semantic,
    build_evolution_context,
    record_feedback,
    JUDGE_SYS,
    DIMENSIONS,
    mean_score,
    CATEGORY,
)
from run_ab_experiment import split_balanced  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def get_model_config(name):
    s = get_settings()
    return {
        "deepseek": (s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model),
        "zhipu": (s.zhipu_base_url, s.zhipu_api_key, s.zhipu_model),
        "lingshu": (s.lingshu_base_url, s.lingshu_api_key, s.lingshu_model),
    }[name]


def make_generate(model_name, delay=0.0):
    base_url, api_key, model = get_model_config(model_name)

    def _gen(system_prompt, user_prompt, temperature=0.7, max_tokens=2048):
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=temperature, max_tokens=max_tokens,
        )
        if delay:
            time.sleep(delay)
        return resp.choices[0].message.content

    return _gen


def main(model_name="deepseek", seed_n=12, max_test=15, delay=0.5):
    gen = make_generate(model_name, delay=delay)
    mem.EXPERIENCE_PATH = os.path.join(DATA_DIR, f"{model_name}_experience.jsonl")
    if os.path.exists(mem.EXPERIENCE_PATH):
        os.remove(mem.EXPERIENCE_PATH)

    profiles = load_profiles()
    seed, test = split_balanced(profiles, seed_n)
    test = test[:max_test]
    print(f"[多模型验证] 模型={model_name}, 种子={len(seed)}, 测试={len(test)}")

    def report(profile, use_evolution=True, evolution_context="", inject_position="system"):
        sp = build_system_prompt(use_evolution, evolution_context, inject_position)
        up = build_user_prompt(profile)
        if use_evolution and evolution_context and inject_position == "user":
            up += "\n" + evolution_context
        return gen(sp, up)

    def report_semantic(profile):
        ctx = build_evolution_context_semantic(CATEGORY, profile)
        return report(profile, use_evolution=bool(ctx), evolution_context=ctx)

    def judge(report_text, profile):
        if not report_text:
            return None
        up = (f"## 待评估报告\n\n{report_text}\n\n"
              f"## 输入数据\n\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n请打分。")
        raw = gen(JUDGE_SYS, up, temperature=0.0, max_tokens=256)
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

    # 预热
    for p in seed:
        r = report(p, use_evolution=False)
        if not r:
            continue
        s = judge(r, p)
        m = mean_score(s)
        if m is not None and m >= 4.0:
            record_feedback(CATEGORY, p["student_name"], rating=int(round(m)), outcome=r[:200],
                            risk_level=p["risk_level"], emotion=p.get("emotion_data", {}).get("fused_emotion"))
    random_ctx = build_evolution_context(CATEGORY)
    print(f"[多模型验证] random 注入上下文长度: {len(random_ctx)}")

    def run_ab(name, semantic_mode):
        results = []
        for p in test:
            rA = report(p, use_evolution=False)
            rB = report_semantic(p) if semantic_mode else report(p, use_evolution=True, evolution_context=random_ctx)
            sA = judge(rA, p); sB = judge(rB, p)
            results.append({"profile": p["student_name"], "risk": p["risk_level"],
                            "A": {"mean": mean_score(sA)}, "B": {"mean": mean_score(sB)}})
            print(f"  [{name}] {p['student_name']}({p['risk_level']}) A={mean_score(sA)} B={mean_score(sB)}")
        return results

    print(f"\n[1/2] {model_name} random A/B ...")
    res_random = run_ab("random", semantic_mode=False)
    print(f"\n[2/2] {model_name} semantic A/B ...")
    res_semantic = run_ab("semantic", semantic_mode=True)

    with open(os.path.join(DATA_DIR, f"{model_name}_ab_random.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": model_name}, "results": res_random}, f, ensure_ascii=False, indent=2)
    with open(os.path.join(DATA_DIR, f"{model_name}_ab_semantic.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": model_name}, "results": res_semantic}, f, ensure_ascii=False, indent=2)

    import numpy as np
    def summ(results):
        pairs = [(r["A"]["mean"], r["B"]["mean"]) for r in results
                 if r["A"]["mean"] is not None and r["B"]["mean"] is not None]
        if not pairs:
            return None
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs]); d = b - a
        return {"n": len(pairs), "M_A": round(a.mean(), 3), "M_B": round(b.mean(), 3),
                "delta": round(d.mean(), 3),
                "cohens_d": round(d.mean() / d.std(ddof=1), 3) if d.std(ddof=1) > 0 else 0.0}

    print(f"\n=== {model_name} 结果 ===")
    print("random:  ", json.dumps(summ(res_random), ensure_ascii=False))
    print("semantic:", json.dumps(summ(res_semantic), ensure_ascii=False))


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek"
    main(model_name=model)
