"""
13c · 轨迹型客观任务上的复制不稳定性 (traj_experiment.py)
=========================================================
把主实验的设计搬到**轨迹型**任务上：逐日决策、状态累积、早期一步选错会让
后续动作非法（详见 traj_task.py 的说明）。这是评审第八份 P1-5 点名要的第二类
客观任务——「success is a property of a trajectory rather than of an answer」。

条件（与主实验、setselect 完全一致）
----------------------------------
- baseline : 不注入
- random   : 从经验库随机取 2 条成功方案
- matched  : 取起点最接近的 2 条（按 (s,n) 的欧氏距离）
- filler   : 等长同格式但内容无效的伪方案（分离「内容」与「长度」）

replay 的对应物
--------------
主机制注入「历史高分报告」；本任务一次「输出」= 一条完整的 5 日干预序列，
故经验库存 (起点状态 + 成功顺序)，注入即「过往做成的排程示范」。

规模可调（本脚本同时用于小试与正式跑）
------------------------------------
    # 小试（约 200 次调用，5 分钟）
    python scripts/paper/traj_experiment.py lingshu --cases 10 --repeat 1 --runs 1 --tag pilot
    # 正式（30 案例 × 4 条件 × 3 重复 × 5 运行）
    python scripts/paper/traj_experiment.py lingshu --cases 30 --repeat 3 --runs 5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import OpenAI                                   # noqa: E402
from backend.config import get_settings                     # noqa: E402
from concurrency import pmap, DEFAULT_WORKERS               # noqa: E402
from traj_task import (                                     # noqa: E402
    ACTIONS, HORIZON, SYSTEM_PROMPT, build_day_prompt, make_cases,
    parse_action, reached_green, rollout, score_state, tier_of,
)

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
CONDITIONS = ["baseline", "random", "matched", "filler"]


def make_caller(model_name: str):
    s = get_settings()
    if model_name == "deepseek":
        base, key, model = s.deepseek_base_url, s.deepseek_api_key, s.deepseek_model
    else:
        base, key, model = s.lingshu_base_url, s.lingshu_api_key, s.lingshu_model
    client = OpenAI(base_url=base, api_key=key)

    def call(system: str, messages: list, temperature: float) -> str | None:
        """messages 为完整对话历史（多轮轨迹任务必须带历史，
        否则后续轮次看不到规则与经验库，等于每轮从零开始）。"""
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system}] + messages,
                temperature=temperature,
                max_tokens=24,                  # 只需回一个动作名
            )
            return r.choices[0].message.content
        except Exception as exc:                                # noqa: BLE001
            print(f"    [call 失败] {exc}", flush=True)
            return None
    return call


# ---------------------------------------------------------------- 经验库
def describe(case: dict) -> str:
    return (f"初始状态：压力累积指数 {case['s0']:.3f}，负面情绪占比 {case['n0']:.3f}"
            f"（{case['tier0']} 档）")


def build_context(entries: list) -> str:
    if not entries:
        return ""
    lines = ["\n## 历史成功干预方案参考（系统从过往反馈中积累）"]
    for i, e in enumerate(entries, 1):
        lines.append(
            f"\n【方案 {i}】{describe(e)}\n"
            f"采用顺序：{' → '.join(e['answer'])}\n"
            f"结果：5 天后达到 green"
        )
    lines.append("\n请参考以上方案，为当前学生安排同样有效的排程。")
    return "\n".join(lines)


def retrieve_random(store, rng, k=2):
    return rng.sample(store, min(k, len(store))) if store else []


def retrieve_matched(store, case, k=2):
    """按起点状态的距离取最相似的历史方案（对应原机制的相关性检索）。"""
    if not store:
        return []
    scored = sorted(store, key=lambda e: (e["s0"] - case["s0"]) ** 2 + (e["n0"] - case["n0"]) ** 2)
    return scored[:k]


def make_filler(n_examples=2, seed=131313) -> str:
    """等长同格式、但动作顺序随机（无效）的伪方案。"""
    rng = random.Random(seed)
    names = [a[0] for a in ACTIONS]
    lines = ["\n## 历史成功干预方案参考（系统从过往反馈中积累）"]
    for i in range(1, n_examples + 1):
        seq = rng.sample(names, HORIZON)
        lines.append(
            f"\n【方案 {i}】初始状态：压力累积指数 {rng.uniform(0.5, 0.9):.3f}，"
            f"负面情绪占比 {rng.uniform(0.4, 0.7):.3f}（red 档）\n"
            f"采用顺序：{' → '.join(seq)}\n"
            f"结果：5 天后达到 green"
        )
    lines.append("\n请参考以上方案，为当前学生安排同样有效的排程。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 一条轨迹
def run_episode(case: dict, ctx: str, call, rng) -> dict:
    """多轮跑完一条轨迹：每天一次调用，**带完整对话历史**，状态确定性推进。"""
    state = (case["s0"], case["n0"])
    used, actions, log, msgs = [], [], [], []
    for day in range(1, HORIZON + 1):
        prompt = build_day_prompt(case, state, day, used, ctx if day == 1 else "")
        msgs.append({"role": "user", "content": prompt})
        raw = call(SYSTEM_PROMPT, msgs, 0.7 + 0.1 * rng.random())
        msgs.append({"role": "assistant", "content": raw or ""})
        act = parse_action(raw)
        state, log_ = rollout(state, [act if act else ""], horizon=1)
        rec = log_[0]
        rec["day"] = day
        log.append(rec)
        actions.append(act)
        if rec["ok"]:
            used.append(act)
    return {"score": score_state(state), "green": reached_green(state),
            "final": [round(state[0], 4), round(state[1], 4)],
            "actions": actions, "log": log}


# ---------------------------------------------------------------- 单次运行
def one_run(run_id: int, cases: list, warmup_cases: list, call,
            n_repeat: int, verbose: bool = True) -> dict:
    rng = random.Random(70000 + run_id)

    # 预热：跑 baseline，把达标的轨迹收进经验库
    def warm(c):
        ep = run_episode(c, "", call, rng)
        if ep["green"]:
            return {"s0": c["s0"], "n0": c["n0"], "tier0": c["tier0"],
                    "answer": [a for a in ep["actions"] if a]}
        return None
    store = [e for e in pmap(warm, warmup_cases, label=f"run{run_id} 预热") if e]
    if verbose:
        print(f"[run{run_id}] 经验库 {len(store)}/{len(warmup_cases)} 条成功方案", flush=True)

    filler_ctx = make_filler(2)
    jobs = [(ci, c, cond, rep) for ci, c in enumerate(cases)
            for cond in CONDITIONS for rep in range(n_repeat)]

    def job(j):
        ci, c, cond, rep = j
        if cond == "baseline":
            ctx = ""
        elif cond == "random":
            ctx = build_context(retrieve_random(store, rng))
        elif cond == "matched":
            ctx = build_context(retrieve_matched(store, c))
        else:
            ctx = filler_ctx
        ep = run_episode(c, ctx, call, random.Random(hash((run_id, ci, cond, rep)) & 0xFFFF))
        return {"case": ci, "cond": cond, "rep": rep, "score": ep["score"],
                "green": ep["green"], "final": ep["final"]}

    res = pmap(job, jobs, label=f"run{run_id} 测试")
    ok = [r for r in res if r]

    rows = []
    for ci in range(len(cases)):
        row = {"case": ci, "s0": cases[ci]["s0"], "n0": cases[ci]["n0"]}
        for cond in CONDITIONS:
            vals = [r["score"] for r in ok if r["case"] == ci and r["cond"] == cond]
            grn = [r["green"] for r in ok if r["case"] == ci and r["cond"] == cond]
            row[cond] = vals
            row[f"{cond}_mean"] = (sum(vals) / len(vals)) if vals else None
            row[f"{cond}_green"] = (sum(grn) / len(grn)) if grn else None
        rows.append(row)
    return {"run": run_id, "store_size": len(store), "results": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="lingshu")
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    all_cases = make_cases(a.cases + a.warmup)
    cases = all_cases[:a.cases]
    warmup_cases = all_cases[a.cases:a.cases + a.warmup]

    call = make_caller(a.model)
    per_run = (a.warmup + a.cases * len(CONDITIONS) * a.repeat) * HORIZON
    print(f"模型 {a.model}｜{a.cases} 案例 × {len(CONDITIONS)} 条件 × {a.repeat} 重复 × {a.runs} 次运行")
    print(f"每次运行约 {per_run} 次调用（多轮，每天一次），并发 {DEFAULT_WORKERS}\n")

    t0 = time.time()
    runs = []
    for r in range(1, a.runs + 1):
        runs.append(one_run(r, cases, warmup_cases, call, a.repeat))
        print(f"[run{r}] 完成，累计 {(time.time() - t0) / 60:.1f} 分钟\n", flush=True)

    suffix = f"_{a.tag}" if a.tag else ""
    out = os.path.join(DATA, f"traj_experiment_{a.model}{suffix}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": {"model": a.model, "n_cases": a.cases, "n_repeat": a.repeat,
                            "n_runs": a.runs, "horizon": HORIZON}, "runs": runs},
                  f, ensure_ascii=False, indent=2)
    print(f"[保存] {out}")
    print(f"总用时 {(time.time() - t0) / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
