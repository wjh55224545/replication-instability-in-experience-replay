"""
验证「检索退化」机制 (verify_degeneration.py)
============================================
论文 §5.3 现在把论证重心放在一条机制观察上：

    「池子只有 15 条时，同一风险等级内的每个档案，相关性检索返回的是**同样的两条**样本。」

这条主张是**代码级**的（不依赖统计比较），因此必须**可复现地验证**，不能只凭断言。
本脚本直接调用检索函数，逐档案记录返回的经验条目 ID，检查同一风险等级内是否重复。

无 LLM 调用，纯本地计算，秒级完成。

运行（在本仓库根目录）：python scripts/paper/verify_degeneration.py
"""

from __future__ import annotations
import os
import sys
import json
import random
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import backend.evolution.memory as mem                       # noqa: E402
from common import load_profiles, CATEGORY                   # noqa: E402
from run_ab_experiment import split_balanced                 # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")
N_REPEAT = 3


def entries_of(path: str) -> list:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def key_of(e: dict) -> str:
    """条目标识。retrieve_* 返回的是原始条目（不含行号），
    故用条目自带的稳定字段组合成键。"""
    return f"{e.get('timestamp','')}|{e.get('student_name','')}|{e.get('rating','')}"


def probe(pool_path: str, label: str, seed_n: int = 15, max_test: int = 29):
    """把经验库指向 pool_path，对每个测试档案调用检索，记录返回的条目 ID。"""
    pool = entries_of(pool_path)
    if not pool:
        print(f"  [{label}] 经验库为空（{os.path.basename(pool_path)}），跳过")
        return None
    mem.EXPERIENCE_PATH = pool_path

    rng = random.Random(1000 + 1)
    profiles = load_profiles()
    rng.shuffle(profiles)
    _, test = split_balanced(profiles, seed_n)
    test = test[:max_test]

    by_risk = defaultdict(list)
    details = []
    for p in test:
        risk = p.get("risk_level")
        emo = p.get("emotion_data", {}).get("fused_emotion")
        got = mem.retrieve_experience_semantic(
            category=CATEGORY, min_rating=4, limit=2,
            risk_level=risk, emotion=emo)
        ids = tuple(sorted(key_of(e) for e in got))
        by_risk[risk].append(ids)
        details.append({"profile": p["student_name"], "risk": risk,
                        "emotion": emo, "returned_ids": list(ids)})

    print(f"\n{'='*70}")
    print(f"{label}　经验库 {len(pool)} 条，测试 {len(test)} 个档案")
    print(f"{'='*70}")
    all_same = True
    for risk in ("green", "yellow", "red"):
        rows = by_risk.get(risk) or []
        if not rows:
            continue
        uniq = set(rows)
        n = len(rows)
        # 有多少档案返回的是「该等级内最常见的那个组合」
        from collections import Counter
        c = Counter(rows)
        top, top_n = c.most_common(1)[0]
        same = len(uniq) == 1
        all_same &= same
        print(f"  {risk:7s}: {n:2d} 个档案 → {len(uniq)} 种不同的返回组合"
              f"　最常见组合出现在 {top_n}/{n} 个档案上"
              f"{'　← 全部相同' if same else ''}")
        print(f"           最常见组合 = {[t.split(chr(124))[1] for t in top] if top else '(空)'}")

    n_uniq_total = len({r for v in by_risk.values() for r in v})
    print(f"\n  跨全部 {len(test)} 个档案，共出现 {n_uniq_total} 种不同的返回组合")
    print(f"  经验库有 {len(pool)} 条，理论上限 = C({len(pool)},2) = "
          f"{len(pool)*(len(pool)-1)//2} 种")

    return {"label": label, "pool_size": len(pool), "n_test": len(test),
            "unique_combos": n_uniq_total, "all_same_within_tier": bool(all_same),
            "details": details}


def main():
    print("=" * 70)
    print("验证「小池子导致检索退化」")
    print("=" * 70)

    runs = [
        ("池 15", [f"lingshu_pool15_run{i}_exp.jsonl" for i in (1, 2, 3)]),
        ("池 45", [f"lingshu_replicate_run{i}_exp.jsonl" for i in (1, 2, 3)]),
    ]

    out = {}
    for label, files in runs:
        for fn in files:
            r = probe(os.path.join(DATA, fn), f"{label} · {fn[:34]}")
            if r:
                out[f"{label}:{fn}"] = r

    print(f"\n{'='*70}")
    print("结论")
    print(f"{'='*70}")
    if not out:
        print("  无可用经验库文件。先跑 replicate_runs_parallel.py 生成。")
        return
    p15 = [v for k, v in out.items() if k.startswith("池 15")]
    p45 = [v for k, v in out.items() if k.startswith("池 45")]
    if p15:
        deg = sum(1 for v in p15 if v["all_same_within_tier"])
        avg = sum(v["unique_combos"] for v in p15) / len(p15)
        print(f"  池 15：{deg}/{len(p15)} 次运行中，**每个风险等级内返回完全相同的组合**")
        print(f"         平均只有 {avg:.1f} 种不同的返回组合（跨 29 个档案）")
    if p45:
        avg45 = sum(v["unique_combos"] for v in p45) / len(p45)
        print(f"  池 45：平均 {avg45:.1f} 种不同的返回组合")
    print()
    print("  若池 15 的组合数远少于池 45，则「检索退化」主张成立。")

    op = os.path.join(DATA, "degeneration_check.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {op}")


if __name__ == "__main__":
    main()
