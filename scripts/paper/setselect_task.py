"""
13b 路线A · 集合选择型客观任务 (setselect_task.py)
================================================
给模型一组学生（10 名）的指标，要求挑出**最需要立即干预的 4 名**。

为什么这样设计
------------
前三次尝试（argmax 式字段抽取）失败的原因：任务要求逐步数值比较，
32B 模型做不到，测出来的是「模型执行规则的能力」而非 replay 效应。
本任务把难点从「数值比较」移到「跨对象聚合判断」：

  底层规则 = 分等级（简单，模型做得来）
  难点     = 从 10 个里挑出最紧急的 4 个（聚合 + 排序）

打分用**重合度**，天然是 0 / 0.25 / 0.5 / 0.75 / 1 的渐变分数，
量程由构造保证，不依赖模型表现。

配对设计
--------
组内成员固定，同一组分别跑 baseline / random / matched，
保证条件间可比（配对）。不同的组是独立分析单元。

自检：python scripts/paper/setselect_task.py
"""

from __future__ import annotations

import os
import re
import sys
import json
import random

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

GROUP_SIZE = 10       # 每组学生数
SELECT_K = 4          # 需挑出的紧急人数

TIER_ORDER = {"red": 0, "yellow": 1, "green": 2}

RULES_TEXT = """请依据以下规则找出最需要**立即干预**的学生：

【第一步：判定风险等级】
- red：情绪稳定指数 < 0.45，或 负面情绪占比 > 0.50，或 压力累积指数 > 0.60
- yellow：不满足 red，但 情绪稳定指数 < 0.70，或 负面情绪占比 > 0.30，或 压力累积指数 > 0.30
- green：以上都不满足

【第二步：按紧急度排序】依次比较，前一项相同才看后一项：
1. 风险等级：red 最紧急，其次 yellow，最后 green
2. 压力累积指数，**越大越紧急**
3. 负面情绪占比，**越大越紧急**
4. 学生编号，按字典序（仅在以上全部相同时使用）

【第三步】按上述顺序取**前 4 名**，即为最需要立即干预的学生。
"""

OUTPUT_SPEC = (
    '输出格式（严格一行 JSON，按紧急度从高到低排列）：\n'
    '{"urgent": ["学生编号1", "学生编号2", "学生编号3", "学生编号4"]}'
)

SYSTEM_PROMPT = (
    "你是一个学生心理健康分诊系统的决策模块。"
    "你的任务是根据一组学生的情绪指标，按规则找出最需要立即干预的学生。"
    "必须严格只输出一行 JSON，不要输出任何解释或 Markdown 代码块。"
)


# ---------------------------------------------------------------- 组与答案
SID_KEY = "sid"


def assign_ids(profiles: list) -> list:
    """给每个档案分配唯一编号。

    坑：数据集里 student_name 大量重复（44 个档案只有 28 个不同姓名，
    张三出现 7 次）。若用姓名作标识，模型无法区分同一组内的重名档案，
    标准答案也不唯一。改用「学生NN」编号，顺带去掉姓名带来的无关线索。
    """
    out = []
    for i, p in enumerate(profiles, 1):
        q = dict(p)
        q[SID_KEY] = f"学生{i:02d}"
        out.append(q)
    return out


def make_groups(profiles: list, n_groups: int = 30, seed: int = 20260911) -> list:
    """随机抽样成组。组间可重叠，但组内不重复。"""
    rng = random.Random(seed)
    return [rng.sample(profiles, GROUP_SIZE) for _ in range(n_groups)]


def tier_of(p: dict) -> str:
    ind = p["analysis_result"]["indicators"]
    if (ind["emotional_stability_index"] < 0.45
            or ind["negative_emotion_ratio"] > 0.50
            or ind["stress_accumulation_index"] > 0.60):
        return "red"
    if (ind["emotional_stability_index"] < 0.70
            or ind["negative_emotion_ratio"] > 0.30
            or ind["stress_accumulation_index"] > 0.30):
        return "yellow"
    return "green"


def urgency_key(p: dict):
    """紧急度排序键（越小越紧急）。与 prompt 中的规则严格对应。"""
    ind = p["analysis_result"]["indicators"]
    return (TIER_ORDER[tier_of(p)],
            -ind["stress_accumulation_index"],
            -ind["negative_emotion_ratio"],
            p[SID_KEY])


def ground_truth(group: list) -> list:
    """返回按紧急度排序的前 SELECT_K 个学生编号。"""
    return [p[SID_KEY] for p in sorted(group, key=urgency_key)[:SELECT_K]]


# ---------------------------------------------------------------- prompt
def build_group_block(group: list) -> str:
    lines = []
    for p in group:
        ind = p["analysis_result"]["indicators"]
        emo = p["emotion_data"]
        lines.append(
            f"- {p[SID_KEY]}｜融合情绪: {emo.get('fused_emotion')}｜"
            f"情绪稳定指数: {ind['emotional_stability_index']}｜"
            f"负面情绪占比: {ind['negative_emotion_ratio']}｜"
            f"情绪趋势: {ind['trend']}｜"
            f"压力累积指数: {ind['stress_accumulation_index']}｜"
            f"情绪恢复速度: {ind['emotion_recovery_speed']}"
        )
    return "【学生列表】\n" + "\n".join(lines)


def build_user_prompt(group: list, evolution_context: str = "") -> str:
    parts = [build_group_block(group)]
    if evolution_context:
        parts.append(evolution_context)
    parts.append(RULES_TEXT)
    parts.append(OUTPUT_SPEC)
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 混淆对照
# 本任务的 worked example 每条约 10 行，远长于主观任务的约 3 行。
# 因此「replay 有害」可能只是上下文变长造成的，而非 replay 语义本身。
# filler 条件注入**等长、同格式、但无任何有效信息**的伪样例，用于分离这两种解释：
#   filler ≈ baseline  → 有害来自 replay 的**信息内容**
#   filler ≈ random    → 有害来自**上下文长度/格式**
_EMO_POOL = ["开心", "平静", "中性", "焦虑", "紧张", "愤怒", "难过", "惊讶"]


def make_filler_context(n_examples: int = 2, group_size: int = GROUP_SIZE,
                        seed: int = 424242) -> str:
    """生成等长同格式、但信息无效的伪样例（随机编号 + 随机指标 + 随机答案）。"""
    rng = random.Random(seed)
    lines = ["\n## 历史分诊案例参考（系统从反馈中自动学习）"]
    for i in range(1, n_examples + 1):
        sids = [f"学生{rng.randint(1, 44):02d}" for _ in range(group_size)]
        rows = []
        for sid in sids:
            rows.append(
                f"- {sid}｜融合情绪: {rng.choice(_EMO_POOL)}｜"
                f"情绪稳定指数: {rng.uniform(0.2, 1.0):.3f}｜"
                f"负面情绪占比: {rng.uniform(0.0, 0.7):.3f}｜"
                f"情绪趋势: {rng.choice(['稳定', '上升', '下降'])}｜"
                f"压力累积指数: {rng.uniform(0.0, 0.6):.3f}｜"
                f"情绪恢复速度: {rng.uniform(0.2, 1.0):.3f}"
            )
        answer = rng.sample(sids, SELECT_K)
        lines.append(
            f"\n【案例 {i}】评分: {rng.randint(3, 5)}/5\n"
            + "【学生列表】\n" + "\n".join(rows)
            + f"\n正确答案: {'、'.join(answer)}"
        )
    lines.append("\n请参考以上案例的判断方式，为当前分组给出同样准确的答案。")
    return "\n".join(lines)


# ---------------------------------------------------------------- 解析与评分
def parse_pred(raw: str | None) -> list | None:
    if not raw:
        return None
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    v = d.get("urgent")
    if not isinstance(v, list):
        return None
    return [str(x).strip() for x in v]


def score_one(pred: list | None, truth: list) -> float | None:
    """重合度 = |预测 ∩ 标准| / K。返回 0~1；无法解析返回 None。"""
    if pred is None:
        return None
    return len(set(pred) & set(truth)) / len(truth)


# ---------------------------------------------------------------- 自检
def _self_check():
    profiles = assign_ids(json.load(open(os.path.join(DATA, "paper_dataset.json"), encoding="utf-8")))
    groups = make_groups(profiles)

    from collections import Counter
    print(f"档案 {len(profiles)} 个 → 生成 {len(groups)} 组 × {GROUP_SIZE} 人\n")

    # 随机基线：随便挑 4 个的期望重合度
    rng = random.Random(0)
    rand_scores = []
    for g in groups:
        truth = set(ground_truth(g))
        pick = set(rng.sample([p[SID_KEY] for p in g], SELECT_K))
        rand_scores.append(len(pick & truth) / SELECT_K)
    print(f"随机选择基线: 平均重合度 {sum(rand_scores)/len(rand_scores):.3f}")
    print(f"  （若模型低于此值，说明它在系统性地做错）\n")

    # 标准答案的等级构成
    print("=== 各组标准答案的等级构成 ===")
    comp = Counter()
    for g in groups:
        truth = set(ground_truth(g))
        for p in g:
            if p[SID_KEY] in truth:
                comp[tier_of(p)] += 1
    print(f"  {dict(comp)}")
    print(f"  （理想情况：red 占多数，yellow 少量，green 极少）\n")

    # 是否有组内并列导致的排序不稳定
    ties = 0
    for g in groups:
        keys = [urgency_key(p)[:3] for p in g]
        if len(set(keys)) < len(keys):
            ties += 1
    print(f"组内存在同 (等级,压力,负面) 的组: {ties}/{len(groups)}")

    print("\n=== 样例（第 1 组）===")
    g = groups[0]
    for p in g:
        print(f"  {p[SID_KEY]:10s} {tier_of(p):7s} "
              f"stress={p['analysis_result']['indicators']['stress_accumulation_index']:.3f}")
    print("\n标准答案:", ground_truth(g))
    print("\n标准答案的紧急度键:")
    for p in sorted(g, key=urgency_key)[:SELECT_K]:
        print(f"  {p[SID_KEY]:10s} {urgency_key(p)[:3]}")


if __name__ == "__main__":
    _self_check()
