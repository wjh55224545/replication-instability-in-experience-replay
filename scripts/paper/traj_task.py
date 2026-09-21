"""
13c 路线B · 轨迹型客观任务 (traj_task.py)
=========================================
给模型一个学生的初始状态，要求在 **5 天内每天选一个干预动作**，
使该生在结束时达到「绿色」档。状态按确定性规则演化，动作有前置条件。

为什么是「轨迹型」
----------------
现有客观任务（setselect）是**一次性选择**：成功是「答案」的属性。
本任务的成功是**轨迹**的属性——
  逐日决策 → 状态累积 → 早期一步选错会让后续动作非法，最终目标不可达。

具体陷阱（本任务的核心）
----------------------
- 「家校沟通」是唯一能把负面情绪压下去的重大动作，但**只在 red 档可用**；
- 「心理辅导」减压最猛，但它会**一步把学生带出 red 档**（s 从 0.75 → 0.50）。
→ 先做心理辅导，家校沟通就永久失去，目标不可达。
→ 正确顺序是**先用家校沟通**，再减压。

这条依赖在单步里看不出来，只有把整条轨迹放在一起才成立——正是评审要的
"success is a property of a trajectory rather than of an answer"。

确定性
------
状态转移是纯函数，无随机、无 LLM 参与：给定动作序列，结果唯一。
故评分是**确定性答案键**，不含任何评估者噪声。

自检：cd 代码 && D:/python/python.exe scripts/paper/traj_task.py
"""

from __future__ import annotations

import json
import os
import random
import re
import sys

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

HORIZON = 5                     # 天数（每天一个动作）
GREEN_S = 0.30                  # 绿色阈值（与 setselect 的分档规则一致）
GREEN_N = 0.30

# 动作表：(名称, Δs, Δn, 前置条件说明, 前置条件函数)
# 每个动作**整条轨迹内至多使用一次**。
ACTIONS = [
    ("心理辅导", -0.18, -0.12, "无", lambda s, n: True),
    ("减压训练", -0.16, -0.05, "无", lambda s, n: True),
    ("家校沟通", -0.30, -0.06, "当前为 red 档", lambda s, n: tier_of(s, n) == "red"),
    ("同伴支持", -0.03, -0.08, "无", lambda s, n: True),
    ("日常关注", -0.01, -0.01, "无", lambda s, n: True),
]
ACTION_NAMES = [a[0] for a in ACTIONS]


def tier_of(s: float, n: float) -> str:
    """风险分档。阈值与 setselect_task 保持一致。"""
    if s > 0.60 or n > 0.50:
        return "red"
    if s > 0.30 or n > 0.30:
        return "yellow"
    return "green"


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def step(state, action: str, used: set):
    """执行一步。返回 (新状态, 是否合法, 原因)。

    非法或重复的动作 = 浪费一天（状态不变），这是轨迹型任务里错误的代价。
    """
    if action not in ACTION_NAMES:
        return state, False, "动作名不在菜单内"
    if action in used:
        return state, False, "该动作本次干预中已使用过"
    _, ds, dn, _, pre = next(a for a in ACTIONS if a[0] == action)
    s, n = state
    if not pre(s, n):
        return state, False, "前置条件不满足"
    return (clamp(s + ds), clamp(n + dn)), True, ""


def rollout(state, actions, horizon: int = HORIZON):
    """给定动作序列，确定性地推出最终状态与逐日记录。"""
    s, n = state
    used, log = set(), []
    for i, a in enumerate(actions[:horizon]):
        (s, n), ok, why = step((s, n), a, used)
        if ok:
            used.add(a)
        log.append({"day": i + 1, "action": a, "ok": ok, "why": why,
                    "s": round(s, 4), "n": round(n, 4), "tier": tier_of(s, n)})
    return (s, n), log


def score_state(state) -> float:
    """评分：最终状态的「绿色程度」。1 = 完全达标，0 = 最差。确定性。"""
    s, n = state
    return round(1.0 - (s + n) / 2.0, 4)


def reached_green(state) -> bool:
    s, n = state
    return s <= GREEN_S and n <= GREEN_N


def best_possible(start):
    """穷举所有合法序列，返回（最优分数, 达到绿色的最短天数）。用于自检。"""
    from itertools import permutations
    best, best_days = -1.0, None
    for r in range(0, HORIZON + 1):
        for seq in permutations(ACTION_NAMES, r):
            (s, n), log = rollout(start, list(seq) + [""] * (HORIZON - r))
            if len(log) < HORIZON:
                continue
            sc = score_state((s, n))
            if sc > best:
                best = sc
            if reached_green((s, n)):
                d = len([x for x in log if x["ok"]])
                best_days = d if best_days is None else min(best_days, d)
    return best, best_days


# ---------------------------------------------------------------- 案例构造
def make_cases(n_cases: int = 30, seed: int = 20260921) -> list:
    """构造 n 个起点。

    每个起点都满足两条，保证任务对每个案例都是**真问题**：
      ① 用对顺序（家校沟通在 red 时用）→ 可达绿色
      ② 丢掉家校沟通               → 不可达绿色
    """
    rng = random.Random(seed)
    out = []
    tries = 0
    while len(out) < n_cases and tries < 200000:
        tries += 1
        s = round(rng.uniform(0.70, 0.76), 3)
        n = round(rng.uniform(0.45, 0.60), 3)
        if tier_of(s, n) != "red":
            continue
        # ① 家校沟通优先 → 可达 green
        good = ["家校沟通", "心理辅导", "减压训练", "同伴支持", "日常关注"]
        (fs, fn), _ = rollout((s, n), good)
        if not reached_green((fs, fn)):
            continue
        # ② 先用大动作（把学生带出 red 档）→ 家校沟通被挡 → 不可达
        #    这才是真实的失败路径：不是"忘了用"，而是"顺序错了导致用不了"。
        bad = ["心理辅导", "家校沟通", "减压训练", "同伴支持", "日常关注"]
        (bs, bn), _ = rollout((s, n), bad)
        if reached_green((bs, bn)):
            continue
        # ③ 确认陷阱确实来自档位变化，而非动作被重复使用
        _, bad_log = rollout((s, n), bad)
        if bad_log[1]["ok"]:
            continue
        out.append({"case": len(out), "s0": s, "n0": n,
                    "tier0": tier_of(s, n),
                    "best_score": score_state((fs, fn))})
    return out


# ---------------------------------------------------------------- prompt
RULES_TEXT = """请为该学生安排接下来 5 天的干预（每天一个动作）。

【状态与分档】
- red：压力 > 0.60 或 负面情绪 > 0.50
- yellow：不满足 red，但 压力 > 0.30 或 负面情绪 > 0.30
- green：以上都不满足

【可选动作】每个动作在本次 5 天中**至多使用一次**：
| 动作 | 压力变化 | 负面情绪变化 | 前置条件 |
|---|---|---|---|
| 心理辅导 | -0.25 | -0.12 | 无 |
| 减压训练 | -0.15 | -0.05 | 无 |
| 家校沟通 | -0.02 | -0.22 | **执行时学生须处于 red 档** |
| 同伴支持 | -0.06 | -0.10 | 无 |
| 日常关注 | -0.02 | -0.02 | 无 |

【目标】5 天结束时学生达到 **green**（压力 ≤ 0.30 且 负面情绪 ≤ 0.30）。
不满足前置条件、或重复使用某个动作，该天视为**浪费**，状态不变。

【回复格式】只回一个动作名，不要任何解释。例如：
心理辅导"""

SYSTEM_PROMPT = (
    "你是一个学生心理健康干预排程系统。"
    "你会在每一天收到学生当前的状态，需要选出一个干预动作。"
    "必须严格只回复一个动作名，不要输出解释或标点。"
)


def build_day_prompt(case: dict, state, day: int, used: list, evolution_context: str = "") -> str:
    s, n = state
    parts = []
    if day == 1:
        parts.append(RULES_TEXT)
        if evolution_context:
            parts.append(evolution_context)
    parts.append(
        f"【第 {day} 天】\n"
        f"学生当前状态：压力累积指数 {s:.3f}，负面情绪占比 {n:.3f}"
        f"（当前档位：{tier_of(s, n)}）\n"
        f"已使用过的动作：{'、'.join(used) if used else '无'}"
    )
    return "\n\n".join(parts)


def parse_action(raw: str | None) -> str | None:
    """从回复里取动作名。取第一个命中的菜单项。"""
    if not raw:
        return None
    for name in ACTION_NAMES:
        if name in raw:
            return name
    return None


# ---------------------------------------------------------------- 自检
def _self_check():
    cases = make_cases(30)
    print(f"构造 {len(cases)} 个案例（全部为 red 起点）\n")
    print(f"{'案例':<5}{'s0':>7}{'n0':>7}{'档':>6}{'最优分':>9}{'最优天数':>9}")
    print("-" * 48)
    for c in cases[:8]:
        b, d = best_possible((c["s0"], c["n0"]))
        print(f"{c['case']:<5}{c['s0']:>7.3f}{c['n0']:>7.3f}{c['tier0']:>6}"
              f"{b:>9.3f}{str(d):>9}")

    print("\n=== 陷阱验证（第 0 号案例）===")
    c = cases[0]
    start = (c["s0"], c["n0"])
    for label, seq in [
        ("正确顺序（家校沟通优先）", ["家校沟通", "心理辅导", "减压训练", "同伴支持", "日常关注"]),
        ("陷阱顺序（心理辅导优先）", ["心理辅导", "家校沟通", "减压训练", "同伴支持", "日常关注"]),
    ]:
        (fs, fn), log = rollout(start, seq)
        print(f"\n{label}：终态 s={fs:.3f} n={fn:.3f} "
              f"档={tier_of(fs, fn)} 达标={reached_green((fs, fn))} 分={score_state((fs, fn))}")
        for e in log:
            mark = " " if e["ok"] else "×"
            print(f"   {mark} 第{e['day']}天 {e['action']:<6} → s={e['s']:.3f} n={e['n']:.3f} "
                  f"{e['tier']:<7}{('(' + e['why'] + ')') if e['why'] else ''}")

    # 全部案例都要满足两条前提
    n_good = sum(1 for c in cases
                 if reached_green(rollout((c["s0"], c["n0"]),
                                          ["家校沟通", "心理辅导", "减压训练", "同伴支持", "日常关注"])[0]))
    n_bad = sum(1 for c in cases
                if reached_green(rollout((c["s0"], c["n0"]),
                                         ["心理辅导", "减压训练", "同伴支持", "日常关注", "日常关注"])[0]))
    print(f"\n【前提自检】{len(cases)} 个案例中：")
    print(f"  正确顺序可达 green：{n_good}/{len(cases)}  （应为全部）")
    print(f"  丢掉家校沟通仍可达：{n_bad}/{len(cases)}  （应为 0）")


if __name__ == "__main__":
    _self_check()
