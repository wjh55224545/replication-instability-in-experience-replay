"""
论文实验 · 同配置重复运行（并行版）(replicate_runs_parallel.py)
==============================================================
补审稿意见指出的缺口：「五轮重复实验仅针对 M2 的随机回放条件，
没有覆盖所有配置，复制不稳定性的普适性验证不够全面。」

本脚本把原 replicate_runs.py 的**设计完全保留**（池 45 / 每格重复 3 次 /
条件 baseline-random-semantic / 仅随机种子不同），只做两件事：

  1. 测试阶段改为**并行**（每次调用约 22–40 秒，串行 5 次要 4 小时以上）
  2. 一次跑完 N 次运行，输出与原脚本相同的 JSON 格式

两处并行化的边界（重要）
----------------------
- **预热**：生成+打分**并行**，文件写入**串行**。
  预热期间 use_evolution=False，不读经验库，各次生成彼此独立；
  但 record_experience 是「打开→写一行→关闭」，并发写会交错损坏 JSONL，
  故并行段只含生成与打分，写入回主线程完成。
- **测试**：整段并行。该阶段只读经验库（build_evolution_context），无写冲突；
  每次抽样独立，执行顺序不影响结果。

（首版把预热整个写成串行，导致每次运行光预热就要约 1 小时、5 次要 5 小时。
 改为并行后总时长降到约 1 小时。）

用法：
    python scripts/paper/replicate_runs_parallel.py lingshu 5
    python scripts/paper/replicate_runs_parallel.py deepseek 2

输出 data/{model}_replicate_run{1..N}.json（与旧格式一致，可直接被 analyze_replicates.py 读取）
"""

from __future__ import annotations
import os
import re
import sys
import json
import random
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import OpenAI                                   # noqa: E402
from backend.config import get_settings                     # noqa: E402
import backend.evolution.memory as mem                      # noqa: E402
from common import (                                        # noqa: E402
    load_profiles, build_user_prompt, build_system_prompt, build_evolution_context,
    build_evolution_context_semantic, record_feedback, JUDGE_SYS, DIMENSIONS,
    mean_score, CATEGORY,
)
from run_ab_experiment import split_balanced                # noqa: E402
from concurrency import pmap, DEFAULT_WORKERS               # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

SEED_N = 15
MAX_TEST = 29
N_REPEAT = 3
POOL_MULT = 3


def make_caller(model_name: str):
    s = get_settings()
    if model_name == "deepseek":
        base, key, model = s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model
    else:
        base, key, model = s.lingshu_base_url, s.lingshu_api_key, s.lingshu_model
    client = OpenAI(base_url=base, api_key=key)

    def call(sp: str, up: str, temperature: float) -> str | None:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": sp},
                      {"role": "user", "content": up}],
            temperature=temperature,
            max_tokens=2048 if temperature > 0 else 256,
        )
        return r.choices[0].message.content
    return call


def parse_scores(raw: str | None):
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


def judge_one(call, report: str, profile: dict):
    """生成一份报告并打分，返回均分或 None。"""
    if not report:
        return None
    raw = call(JUDGE_SYS,
               f"## 待评估报告\n\n{report}\n\n## 输入数据\n\n"
               f"{json.dumps(profile, ensure_ascii=False)}\n\n请打分。",
               0.0)
    return mean_score(parse_scores(raw))


def one_run(model_name: str, run_tag: int, pool_mult: int = POOL_MULT,
            verbose: bool = True) -> dict:
    """跑一次完整运行。

    pool_mult: 每个预热档案生成几次。3 → 池 45（主实验配置）；1 → 池 15。
    输出文件名随池子大小区分，避免覆盖主实验数据。
    """
    pool_size_target = SEED_N * pool_mult
    stem = (f"{model_name}_replicate_run{run_tag}" if pool_mult == POOL_MULT
            else f"{model_name}_pool{pool_size_target}_run{run_tag}")
    OUT = os.path.join(DATA_DIR, f"{stem}.json")
    EXP = os.path.join(DATA_DIR, f"{stem}_exp.jsonl")
    mem.EXPERIENCE_PATH = EXP
    if os.path.exists(EXP):
        os.remove(EXP)

    call = make_caller(model_name)
    t0 = time.time()

    rng = random.Random(1000 + run_tag)
    profiles = load_profiles()
    rng.shuffle(profiles)
    seed, test = split_balanced(profiles, SEED_N)
    test = test[:MAX_TEST]
    print(f"\n[{stem}] 模型={model_name}, 预热={len(seed)}×{pool_mult}={pool_size_target}, "
          f"测试={len(test)}, 重复={N_REPEAT}", flush=True)

    # ---- 预热：并行生成 + 串行写入 ----
    # 预热期间 use_evolution=False，不读经验库，故各次生成彼此独立，可并行。
    # 但 record_experience 是「打开→写一行→关闭」，多线程并发写可能交错损坏 JSONL，
    # 因此只把**生成+打分**并行，文件写入仍在主线程串行完成。
    warm_jobs = [(p, k) for p in seed for k in range(pool_mult)]

    def warm_one(job):
        p, _ = job
        sp = build_system_prompt(False, "", "system")
        rep = call(sp, build_user_prompt(p), 0.7)
        return (p, rep, judge_one(call, rep, p))

    warm_out = pmap(warm_one, warm_jobs, label=f"run{run_tag} 预热")
    kept = 0
    for item in warm_out:
        if not item:
            continue
        p, rep, m = item
        if rep and m is not None and m >= 4.0:
            record_feedback(CATEGORY, p["student_name"], rating=int(round(m)),
                            outcome=rep[:200], risk_level=p["risk_level"],
                            emotion=p.get("emotion_data", {}).get("fused_emotion"))
            kept += 1

    pool_size = sum(1 for _ in open(EXP, encoding="utf-8"))
    print(f"[{stem}] 经验池: {pool_size} 条（入库 {kept}/{len(warm_jobs)}）"
          f"　预热用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    # ---- 测试：并行（只读经验库，无写冲突）----
    jobs = [(p, cond, rep)
            for p in test
            for cond in ["baseline", "random", "semantic"]
            for rep in range(N_REPEAT)]

    def run_cell(job):
        p, cond, rep = job
        if cond == "baseline":
            ctx, use = "", False
        elif cond == "random":
            ctx, use = build_evolution_context(CATEGORY), True
        else:
            ctx = build_evolution_context_semantic(CATEGORY, p)
            use = bool(ctx)
        sp = build_system_prompt(use, ctx, "system")
        rep_text = call(sp, build_user_prompt(p), 0.7)
        return judge_one(call, rep_text, p)

    scores = pmap(run_cell, jobs, label=f"run{run_tag}")

    # ---- 汇总（保持与旧格式一致）----
    out_cells: dict = {}
    for (p, cond, rep), sc in zip(jobs, scores):
        if sc is not None:
            out_cells.setdefault((p["student_name"], cond), []).append(sc)

    results = []
    for p in test:
        row = {"profile": p["student_name"], "risk": p["risk_level"]}
        for cond in ["baseline", "random", "semantic"]:
            vals = out_cells.get((p["student_name"], cond), [])
            row[cond] = vals
            row[f"{cond}_mean"] = round(sum(vals) / len(vals), 3) if vals else None
        results.append(row)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": model_name, "run_tag": run_tag,
                            "pool_size": pool_size, "n_repeat": N_REPEAT,
                            "parallel": True},
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"[{stem}] 已保存 {OUT}　用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
    return {"run_tag": run_tag, "pool_size": pool_size, "path": OUT}


def main(model_name: str = "lingshu", n_runs: int = 5, pool_mult: int = POOL_MULT):
    print(f"模型 {model_name}｜{n_runs} 次独立运行｜设计: 池 {SEED_N*pool_mult} "
          f"/ 每格重复 {N_REPEAT} / 条件 baseline+random+semantic")
    print(f"并发 {DEFAULT_WORKERS}\n")
    t0 = time.time()
    summary = []
    for tag in range(1, n_runs + 1):
        summary.append(one_run(model_name, tag, pool_mult=pool_mult))
    total = (time.time() - t0) / 60
    print(f"\n全部完成，总用时 {total:.1f} 分钟")
    tag_name = (f"{model_name}_replicate_runs_summary.json" if pool_mult == POOL_MULT
                else f"{model_name}_pool{SEED_N*pool_mult}_runs_summary.json")
    with open(os.path.join(DATA_DIR, tag_name), "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "n_runs": n_runs,
                   "pool_size": SEED_N * pool_mult,
                   "minutes": round(total, 1), "runs": summary},
                  f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 用法: replicate_runs_parallel.py <model> <n_runs> [pool_mult]
    #   pool_mult=3 → 池 45（主实验）；1 → 池 15（§5.3 池子大小对照）
    m = sys.argv[1] if len(sys.argv) > 1 else "lingshu"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    pm = int(sys.argv[3]) if len(sys.argv) > 3 else POOL_MULT
    main(m, n, pm)
