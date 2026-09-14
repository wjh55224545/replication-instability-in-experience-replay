"""
论文 · 反向交叉评估 (cross_eval_reverse.py)
==========================================
回应 C16：补上「Lingshu 评 DeepSeek 生成的报告」，形成完整的
generator × judge 交叉评分矩阵。

现状：
- Lingshu 生成 → Lingshu 评（自评，主实验已有）
- Lingshu 生成 → DeepSeek 评（cross_eval.py 已有，r=0.868）
- DeepSeek 生成 → DeepSeek 评（自评，multi_model_validation 已有）
- DeepSeek 生成 → Lingshu 评（本脚本补上）

运行（在本仓库根目录）：python scripts/paper/cross_eval_reverse.py
"""

from __future__ import annotations
import os
import re
import json
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import OpenAI  # noqa: E402
from backend.config import get_settings  # noqa: E402
from common import load_profiles, build_user_prompt, build_system_prompt, JUDGE_SYS, DIMENSIONS, mean_score  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
OUT = os.path.join(DATA_DIR, "cross_eval_reverse.json")


def deepseek_generate(system_prompt, user_prompt, temperature=0.7, max_tokens=2048):
    s = get_settings()
    client = OpenAI(base_url=s.deepseek_base_url, api_key=s.deepseek_api_key)
    resp = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return resp.choices[0].message.content


def lingshu_judge(report_text, profile):
    """用 Lingshu-32B 评 DeepSeek 生成的报告。"""
    from backend.aigc.llm_client import llm_generate
    user_prompt = (f"## 待评估报告\n\n{report_text}\n\n"
                   f"## 输入数据\n\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n请打分。")
    raw = llm_generate(JUDGE_SYS, user_prompt, temperature=0.0, max_tokens=256)
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


def main(n_profiles=12):
    profiles = load_profiles()[:n_profiles]
    results = []
    for p in profiles:
        sys_prompt = build_system_prompt(False, "", "system")
        user_prompt = build_user_prompt(p)
        report = deepseek_generate(sys_prompt, user_prompt)
        if not report:
            continue
        # DeepSeek 自评
        ds_self = deepseek_judge(report, p)
        # Lingshu 评 DeepSeek
        ling_on_ds = lingshu_judge(report, p)
        results.append({
            "profile": p["student_name"],
            "risk": p["risk_level"],
            "ds_self": mean_score(ds_self),
            "ling_on_ds": mean_score(ling_on_ds),
        })
        print(f"[C16] {p['student_name']}({p['risk_level']}) DeepSeek自评={mean_score(ds_self)}, Lingshu评={mean_score(ling_on_ds)}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 计算 DeepSeek 自评 vs Lingshu 评 的相关系数
    import numpy as np
    from scipy import stats
    a = [r["ds_self"] for r in results if r["ds_self"] is not None and r["ling_on_ds"] is not None]
    b = [r["ling_on_ds"] for r in results if r["ds_self"] is not None and r["ling_on_ds"] is not None]
    if len(a) >= 3:
        r, p = stats.pearsonr(a, b)
        print(f"\nDeepSeek自评 vs Lingshu评: Pearson r = {r:.3f} (p={p:.4f}, n={len(a)})")
    print(f"结果已保存: {OUT}")


def deepseek_judge(report_text, profile):
    """用 DeepSeek 评报告（自评）。"""
    s = get_settings()
    client = OpenAI(base_url=s.deepseek_base_url, api_key=s.deepseek_api_key)
    user_prompt = (f"## 待评估报告\n\n{report_text}\n\n"
                   f"## 输入数据\n\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n请打分。")
    resp = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": user_prompt}],
        temperature=0.0, max_tokens=256,
    )
    raw = resp.choices[0].message.content
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL) if raw else None
    if not m:
        return None
    try:
        sc = json.loads(m.group(0))
        return {k: int(v) for k, v in sc.items() if k in DIMENSIONS}
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


if __name__ == "__main__":
    main()
