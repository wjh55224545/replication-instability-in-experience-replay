"""
13b 正式实验 · 集合选择任务上的复制不稳定性 (setselect_experiment.py)
==================================================================
把主实验的设计原样搬到客观任务上，检验论文的边界预测：

    主观任务（写日报）：效应≈0，SE 与效应同量级 → 5 次运行结论翻转
    客观任务（集合选择）：效应≈0，SE 应显著更小 → 5 次运行结论应一致

条件（与原实验一致）
------------------
- baseline : 不注入
- random   : 从经验库随机取 2 条 worked example 注入
- matched  : 取与当前组「学生重合度」最高的 2 条注入

replay 的对应物
--------------
原机制注入「历史高分报告」。本任务中一次「输出」= 一份分组的选择结果，
故经验库存 (分组列表 + 正确选择)，注入即「过往的解题示范」。这是演示式提示
的最忠实对应。

设计
----
- 30 个测试组 × 10 名学生，组内固定（条件间配对）
- 每格重复 3 次
- 独立运行 5 次（每次重新预热经验库、重跑全部条件）
- 2 个模型

运行：
    python scripts/paper/setselect_experiment.py lingshu
    python scripts/paper/setselect_experiment.py deepseek
"""

from __future__ import annotations

import os
import sys
import json
import time
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import OpenAI                                   # noqa: E402
from backend.config import get_settings                     # noqa: E402
from concurrency import pmap, DEFAULT_WORKERS               # noqa: E402
from setselect_task import (                                # noqa: E402
    assign_ids, make_groups, ground_truth, build_user_prompt,
    build_group_block, parse_pred, score_one, SYSTEM_PROMPT,
    GROUP_SIZE, SELECT_K, RULES_TEXT, OUTPUT_SPEC, make_filler_context,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

N_TEST_GROUPS = 30
N_WARMUP_GROUPS = 8
N_REPEAT = 3
N_RUNS = 5
STORE_MIN_SCORE = 0.75      # 经验库只收「做得好」的示范（对应原机制的高分筛选）

# 四个条件。filler 是混淆对照：与 random/matched 等长同格式，但信息无效。
CONDITIONS = ["baseline", "random", "matched", "filler"]


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


# ---------------------------------------------------------------- 经验库
def build_context(entries: list) -> str:
    """把 worked example 拼成注入上下文（格式对应原机制的『历史成功案例参考』）。"""
    if not entries:
        return ""
    lines = ["\n## 历史分诊案例参考（系统从反馈中自动学习）"]
    for i, e in enumerate(entries, 1):
        lines.append(
            f"\n【案例 {i}】评分: {e['rating']}/5\n"
            f"{e['listing']}\n"
            f"正确答案: {'、'.join(e['answer'])}"
        )
    lines.append("\n请参考以上案例的判断方式，为当前分组给出同样准确的答案。")
    return "\n".join(lines)


def retrieve_random(store: list, rng: random.Random, k: int = 2) -> list:
    return rng.sample(store, min(k, len(store))) if store else []


def retrieve_matched(store: list, group: list, k: int = 2) -> list:
    """按学生重合度检索最相似的过往案例（对应原机制的相关性检索）。"""
    if not store:
        return []
    cur = {p["sid"] for p in group}
    scored = [(len(cur & set(e["sids"])) / len(cur | set(e["sids"])), e) for e in store]
    scored.sort(key=lambda x: -x[0])
    return [e for _, e in scored[:k]]


# ---------------------------------------------------------------- 单次运行
def one_run(run_id: int, test_groups: list, warmup_groups: list,
            call, verbose: bool = True) -> dict:
    rng = random.Random(90000 + run_id)

    # --- 预热经验库（用独立的分组，避免与测试组泄漏）---
    def warm_one(g):
        raw = call(SYSTEM_PROMPT, build_user_prompt(g), 0.7)
        pred = parse_pred(raw)
        sc = score_one(pred, ground_truth(g))
        if sc is not None and sc >= STORE_MIN_SCORE:
            return {"listing": build_group_block(g), "answer": ground_truth(g),
                    "rating": int(round(sc * 5)), "sids": [p["sid"] for p in g]}
        return None

    store = [e for e in pmap(warm_one, warmup_groups, label=f"run{run_id} 预热")
             if e is not None]
    if verbose:
        print(f"[run{run_id}] 经验库 {len(store)}/{len(warmup_groups)} 条", flush=True)

    # --- 测试条件 ---
    # filler：等长同格式但无有效信息的伪样例，用于分离「replay 内容」与「上下文长度」
    filler_ctx = make_filler_context(2)
    jobs = []
    for gi, g in enumerate(test_groups):
        for cond in CONDITIONS:
            for rep in range(N_REPEAT):
                jobs.append((gi, g, cond, rep))

    def run_one(job):
        gi, g, cond, rep = job
        if cond == "baseline":
            ctx = ""
        elif cond == "random":
            ctx = build_context(retrieve_random(store, rng))
        elif cond == "matched":
            ctx = build_context(retrieve_matched(store, g))
        else:                                   # filler
            ctx = filler_ctx
        raw = call(SYSTEM_PROMPT, build_user_prompt(g, ctx), 0.7)
        return {"group": gi, "cond": cond, "rep": rep,
                "score": score_one(parse_pred(raw), ground_truth(g))}

    res = pmap(run_one, jobs, label=f"run{run_id} 测试")
    ok = [r for r in res if r and r["score"] is not None]

    rows = []
    for gi, g in enumerate(test_groups):
        row = {"group": gi}
        for cond in CONDITIONS:
            vals = [r["score"] for r in ok if r["group"] == gi and r["cond"] == cond]
            row[cond] = vals
            row[f"{cond}_mean"] = (sum(vals) / len(vals)) if vals else None
        rows.append(row)

    return {"run": run_id, "store_size": len(store), "results": rows}


# ---------------------------------------------------------------- 主流程
def main(model_name: str = "lingshu"):
    profiles = assign_ids(json.load(open(os.path.join(DATA, "paper_dataset.json"),
                                         encoding="utf-8")))
    test_groups = make_groups(profiles, n_groups=N_TEST_GROUPS, seed=20260911)
    warmup_groups = make_groups(profiles, n_groups=N_WARMUP_GROUPS, seed=777)

    call = make_caller(model_name)
    n_calls = N_RUNS * (N_WARMUP_GROUPS + N_TEST_GROUPS * len(CONDITIONS) * N_REPEAT)
    print(f"模型 {model_name}｜{N_TEST_GROUPS} 组 × {len(CONDITIONS)} 条件 × {N_REPEAT} 重复 × {N_RUNS} 次运行")
    print(f"总调用约 {n_calls} 次，并发 {DEFAULT_WORKERS}\n")

    t0 = time.time()
    runs = []
    for r in range(1, N_RUNS + 1):
        runs.append(one_run(r, test_groups, warmup_groups, call))
        print(f"[run{r}] 完成，累计 {(time.time()-t0)/60:.1f} 分钟\n", flush=True)

    out = os.path.join(DATA, f"setselect_experiment_{model_name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": model_name, "n_groups": N_TEST_GROUPS,
                            "n_repeat": N_REPEAT, "n_runs": N_RUNS},
                   "runs": runs}, f, ensure_ascii=False, indent=2)
    print(f"[保存] {out}")
    print(f"总用时 {(time.time()-t0)/60:.1f} 分钟")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lingshu")
