"""
论文 · 交叉评估 (cross_eval.py)
==============================

打破「自评估偏差」：用 DeepSeek 作为评审模型，对 Lingshu-32B 生成的 20 份报告打分，
计算 DeepSeek 评分与 Lingshu 自身评分的 Pearson 相关系数，验证 LLM-as-judge 的
评分是否受评审模型偏好的系统性影响。

若两模型评分高度一致 → 自评估偏差有限，结论可靠。
输出 data/cross_eval_results.json。

运行（在本仓库根目录）：python scripts/paper/cross_eval.py
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
from common import JUDGE_SYS, DIMENSIONS, mean_score  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
MATERIAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "human_eval")
OUT = os.path.join(DATA_DIR, "cross_eval_results.json")


def deepseek_judge(report_text, delay=0.5):
    s = get_settings()
    client = OpenAI(base_url=s.deepseek_base_url, api_key=s.deepseek_api_key)
    user_prompt = (f"## 待评估报告\n\n{report_text}\n\n请打分。")
    resp = client.chat.completions.create(
        model=s.deepseek_model,
        messages=[{"role": "system", "content": JUDGE_SYS},
                  {"role": "user", "content": user_prompt}],
        temperature=0.0, max_tokens=256,
    )
    time.sleep(delay)
    raw = resp.choices[0].message.content
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


def parse_reports():
    """解析 reports.md 的 R01-R20 报告全文。"""
    path = os.path.join(MATERIAL_DIR, "reports.md")
    text = open(path, encoding="utf-8").read()
    reports = {}
    parts = re.split(r"\n## (R\d+)\n", text)
    # parts: ['', 'R01', '报告内容', 'R02', '报告内容', ...]
    for i in range(1, len(parts), 2):
        rid = parts[i].strip()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        reports[rid] = content
    return reports


def main():
    reports = parse_reports()
    llm_ref = json.load(open(os.path.join(MATERIAL_DIR, "llm_scores.json"), encoding="utf-8"))

    import numpy as np
    from scipy import stats

    per_dim_ling, per_dim_ds = [], []
    ling_means, ds_means = [], []
    for rid, report in sorted(reports.items()):
        if rid not in llm_ref:
            continue
        lsc = llm_ref[rid].get("llm_scores") or {}
        dsc = deepseek_judge(report)
        if not dsc:
            continue
        l_mean = mean_score(lsc)
        d_mean = mean_score(dsc)
        print(f"[cross_eval] {rid}: Lingshu评={l_mean}, DeepSeek评={d_mean}")
        ling_means.append(l_mean)
        ds_means.append(d_mean)
        for dim in DIMENSIONS:
            if dim in lsc and dim in dsc:
                per_dim_ling.append(lsc[dim])
                per_dim_ds.append(dsc[dim])

    results = {}
    if len(per_dim_ling) >= 3:
        r, p = stats.pearsonr(per_dim_ling, per_dim_ds)
        results["dimension_pearson_r"] = round(r, 3)
        results["dimension_p"] = round(p, 4)
        results["n_dims"] = len(per_dim_ling)
    if len(ling_means) >= 3:
        r, p = stats.pearsonr(ling_means, ds_means)
        results["report_pearson_r"] = round(r, 3)
        results["report_p"] = round(p, 4)
        results["n_reports"] = len(ling_means)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n=== 交叉评估结果 ===")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
