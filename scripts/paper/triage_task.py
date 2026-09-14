"""
13b 跨任务验证 · 客观分诊任务定义 (triage_task.py)
=================================================
为「复制不稳定性是否只存在于主观质量评估任务」提供客观任务与标准答案。

设计要点
--------
原任务（写心理日报）的依赖变量是 LLM 主观评分，量程被压缩（96 个分数里 73 个是 8.2）。
本模块定义一个**有确定标准答案**的任务：

    输入：学生的 6 个情绪指标 + 趋势 + 融合情绪
    输出：JSON 分诊卡（风险等级 / 首要关注项 / 建议动作 / 随访天数）

评分用逐字段精确匹配 → 准确率。不需要 LLM judge。

关键说明（诚实记录）
------------------
本任务必须把规则**完全写死**才能有客观标准答案；而规则写死意味着样例的
「约定示范」作用被削弱，replay 的效应可能接近零。

这**不影响实验成立**：论文的论点是「效应估计的稳定性取决于 噪声/信号 比」，
而不是「效应必须非零」。预测的对比是：

    主观任务：效应≈0，但 SE 巨大 → 5 次运行结论翻转
    客观任务：效应≈0，但 SE 很小 → 5 次运行结论一致

即使两边效应都为零，这个对比依然是对论点的直接检验。

运行自检：python scripts/paper/triage_task.py
"""

from __future__ import annotations

import os
import re
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# ---------------------------------------------------------------- 规则定义
# 规则同时用于：(a) 生成 prompt 中的规则说明；(b) 计算标准答案。
# 两者必须由同一份代码产生，否则会出现「标准答案与题目不符」的致命 bug。

RULES_TEXT = """请依据以下规则生成分诊卡：

【风险等级 risk_tier】
- red：情绪稳定指数 < 0.45，或 负面情绪占比 > 0.50，或 压力累积指数 > 0.60
- yellow：不满足 red，但 情绪稳定指数 < 0.70，或 负面情绪占比 > 0.30，或 压力累积指数 > 0.30
- green：以上都不满足

【首要关注项 primary_concern】先算四个偏离度：
- stability = 1 - 情绪稳定指数
- negative  = 负面情绪占比
- stress    = 压力累积指数
- recovery  = 1 - 情绪恢复速度

取偏离度**最大**的那一项。若多项并列最大（与最大值的差 < 0.05），
按此固定顺序取最先出现者：**stability → stress → negative → recovery**。
注意 recovery 排在最后：只有当其余三项都不在并列范围内时，才会选中 recovery。

【建议动作 action】由 趋势 × 风险等级 决定：
- 趋势为「下降」：red→"立即约谈"   yellow→"三日内约谈"   green→"加强观察"
- 趋势为「上升」：red→"三日内约谈" yellow→"加强观察"     green→"常规跟进"
- 趋势为「稳定」：red→"立即约谈"   yellow→"加强观察"     green→"常规跟进"

【随访天数 followup_days】基础值：red=7, yellow=3, green=1；
若趋势为「下降」则 +3；上限 14。

风险等级、首要关注项、建议动作必须使用上述给定的取值，不得自创。
"""

FIELDS = ["risk_tier", "primary_concern", "action", "followup_days"]
TIERS = ["red", "yellow", "green"]
CONCERNS = ["stability", "negative", "stress", "recovery"]
ACTIONS = ["立即约谈", "三日内约谈", "加强观察", "常规跟进"]


# ---------------------------------------------------------------- 标准答案
def ground_truth(p: dict) -> dict:
    """由规则确定性计算标准答案。"""
    ind = p["analysis_result"]["indicators"]
    stab = ind["emotional_stability_index"]
    neg = ind["negative_emotion_ratio"]
    stress = ind["stress_accumulation_index"]
    rec = ind["emotion_recovery_speed"]
    trend = ind["trend"]

    # 风险等级
    if stab < 0.45 or neg > 0.50 or stress > 0.60:
        tier = "red"
    elif stab < 0.70 or neg > 0.30 or stress > 0.30:
        tier = "yellow"
    else:
        tier = "green"

    # 首要关注项（含 0.05 并列带与固定优先级）
    # 必须用与 prompt 中展示的同一份（四舍五入后）偏离度，否则题目与答案不符
    devs = deviations(p)
    top = max(devs.values())
    tied = [k for k, v in devs.items() if top - v < 0.05]
    for k in ["stability", "stress", "negative", "recovery"]:
        if k in tied:
            concern = k
            break

    # 动作
    table = {
        "下降": {"red": "立即约谈", "yellow": "三日内约谈", "green": "加强观察"},
        "上升": {"red": "三日内约谈", "yellow": "加强观察", "green": "常规跟进"},
        "稳定": {"red": "立即约谈", "yellow": "加强观察", "green": "常规跟进"},
    }
    action = table.get(trend, table["稳定"])[tier]

    # 随访天数
    days = {"red": 7, "yellow": 3, "green": 1}[tier]
    if trend == "下降":
        days += 3
    days = min(days, 14)

    return {"risk_tier": tier, "primary_concern": concern,
            "action": action, "followup_days": days}


# ---------------------------------------------------------------- prompt
SYSTEM_PROMPT = (
    "你是一个学生心理健康分诊系统的结构化输出模块。"
    "你的任务是根据给定的情绪指标，按规则生成一张分诊卡。"
    "必须严格只输出一行 JSON，不要输出任何解释或 Markdown 代码块。"
)

OUTPUT_SPEC = (
    '输出格式（严格一行 JSON）：\n'
    '{"risk_tier": "red|yellow|green", '
    '"primary_concern": "stability|negative|stress|recovery", '
    '"action": "立即约谈|三日内约谈|加强观察|常规跟进", '
    '"followup_days": <整数>}'
)


def deviations(p: dict) -> dict:
    """四个偏离度。预先算好放进输入，把算术负担移出模型
    （试点显示 32B 模型做多步数值比较不可靠，会把噪声注进测量）。"""
    ind = p["analysis_result"]["indicators"]
    return {
        "stability": round(1 - ind["emotional_stability_index"], 3),
        "negative": round(ind["negative_emotion_ratio"], 3),
        "stress": round(ind["stress_accumulation_index"], 3),
        "recovery": round(1 - ind["emotion_recovery_speed"], 3),
    }


def build_input_block(p: dict) -> str:
    """只给原始指标与偏离度，**不给** risk_level（那是答案）。"""
    ind = p["analysis_result"]["indicators"]
    emo = p["emotion_data"]
    dev = deviations(p)
    return (
        f"学生编号: {p['student_name']}\n"
        f"融合情绪: {emo.get('fused_emotion')}（效价 {emo.get('fused_valence')}，"
        f"唤醒 {emo.get('fused_arousal')}）\n"
        f"情绪稳定指数: {ind['emotional_stability_index']}\n"
        f"正面情绪占比: {ind['positive_emotion_ratio']}\n"
        f"负面情绪占比: {ind['negative_emotion_ratio']}\n"
        f"情绪趋势: {ind['trend']}\n"
        f"压力累积指数: {ind['stress_accumulation_index']}\n"
        f"情绪恢复速度: {ind['emotion_recovery_speed']}\n"
        f"\n【已算好的四项偏离度】\n"
        f"stability={dev['stability']}  negative={dev['negative']}  "
        f"stress={dev['stress']}  recovery={dev['recovery']}"
    )


def build_user_prompt(p: dict, evolution_context: str = "") -> str:
    parts = [build_input_block(p)]
    if evolution_context:
        parts.append(evolution_context)
    parts.append(RULES_TEXT)
    parts.append(OUTPUT_SPEC)
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 解析与评分
def parse_pred(raw: str | None) -> dict | None:
    """从模型输出里抽出 JSON。"""
    if not raw:
        return None
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {}
    for f in FIELDS:
        v = d.get(f)
        if f == "followup_days":
            try:
                out[f] = int(v)
            except (TypeError, ValueError):
                return None
        elif isinstance(v, str):
            out[f] = v.strip()
        else:
            return None
    return out


def score_one(pred: dict | None, truth: dict) -> dict:
    """逐字段比对，返回 {字段: 0/1} 及是否全对。"""
    if pred is None:
        return {f: 0 for f in FIELDS} | {"all": 0}
    r = {f: int(pred.get(f) == truth[f]) for f in FIELDS}
    r["all"] = int(all(r[f] for f in FIELDS))
    return r


# ---------------------------------------------------------------- 自检
def _self_check():
    profiles = json.load(open(os.path.join(DATA, "paper_dataset.json"), encoding="utf-8"))
    print(f"档案数: {len(profiles)}\n")

    from collections import Counter
    truths = [ground_truth(p) for p in profiles]
    print("=== 标准答案分布（决定任务是否有区分度）===")
    for f in FIELDS:
        c = Counter(t[f] for t in truths)
        print(f"  {f:16s}: {dict(c)}")

    tiers = Counter(t["risk_tier"] for t in truths)
    print(f"\n等级分布: {dict(tiers)}")
    top = max(tiers.values()) / len(truths)
    print(f"最大类占比: {top:.0%}")
    if top > 0.7:
        print("  [!!] 类别过于集中 —— 全猜多数类就能得高分，任务区分度不足")
    else:
        print("  [OK] 类别分布可用于评分")

    print("\n=== 样例（第 1 条）===")
    p = profiles[0]
    print(build_user_prompt(p)[:600])
    print("\n标准答案:", json.dumps(ground_truth(p), ensure_ascii=False))


if __name__ == "__main__":
    _self_check()
