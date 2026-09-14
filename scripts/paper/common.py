"""
论文实验 · 共享工具 (common.py)
===============================

提供三个核心能力，供 A/B / 收敛 / 消融实验复用：

1. load_profiles()      — 加载 paper_dataset.json
2. generate_report()    — 生成心理日报，进化上下文可开关、可控制注入位置
3. judge()              — LLM-as-judge 多维打分（1-5 分）

运行方式（在 代码/ 目录下）：
    python scripts/paper/run_ab_experiment.py
"""

from __future__ import annotations
import os
import re
import json
import sys

# 确保能 import backend 模块（脚本在 代码/scripts/paper/ 下）
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.aigc.llm_client import llm_generate  # noqa: E402
from backend.evolution.memory import (  # noqa: E402
    build_evolution_context,
    record_experience,
    record_feedback,
    retrieve_experience_semantic,
)

DATASET = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paper_dataset.json")
CATEGORY = "daily_report"

# 评估维度（LLM-as-judge）
DIMENSIONS = ["专业性", "相关性", "共情性", "建议可操作性", "结构完整性"]


# ========================
# 数据加载
# ========================

def load_profiles() -> list:
    with open(DATASET, encoding="utf-8") as f:
        return json.load(f)


# ========================
# 报告生成（复用 report_generator 的 prompt 结构，进化上下文可控）
# ========================

def build_system_prompt(use_evolution: bool, evolution_context: str, inject_position: str) -> str:
    sys_prompt = (
        "你是一位专业的学校心理辅导老师，负责撰写学生心理健康评估日报。"
        "请使用专业的心理学语言，同时保持报告对教师和家长友好可读。"
        "用 Markdown 格式输出结构化报告，适当使用 emoji 增强可读性。"
    )
    if use_evolution and evolution_context and inject_position == "system":
        sys_prompt += "\n" + evolution_context
    return sys_prompt


def build_user_prompt(profile: dict) -> str:
    emo = profile.get("emotion_data", {})
    ana = profile.get("analysis_result", {})
    ind = ana.get("indicators", {})
    pred = ana.get("llm_prediction", {})
    risk_factors = ana.get("risk_factors", [])
    risk_str = "\n".join(f"- {f}" for f in risk_factors) if risk_factors else "- 无明显风险因素"

    return f"""请为 {profile.get('student_name')} 同学生成 {profile.get('date')} 的心理健康评估日报。

## 数据概览
- 综合情绪评分：{emo.get('fused_score', 0.7):.2f}/1.00
- 主要情绪：{emo.get('fused_emotion', '未检测')}
- 风险等级：{ana.get('risk_level', 'green')}
- 情绪稳定性指数：{ind.get('emotional_stability_index', 'N/A')}
- 积极情绪占比：{ind.get('positive_emotion_ratio', 'N/A')}
- 负面情绪占比：{ind.get('negative_emotion_ratio', 'N/A')}
- 情绪趋势：{ind.get('trend', '稳定')}
- 压力累积指数：{ind.get('stress_accumulation_index', 'N/A')}
- 情绪恢复速度：{ind.get('emotion_recovery_speed', 'N/A')}

## 风险因素
{risk_str}

## 明日预测
- 趋势预测：{pred.get('trend_prediction', '稳定') if pred else '稳定'}

请按以下结构输出完整报告：
1. **情绪概况** — 1段自然语言概述
2. **关键指标表格** — Markdown表格，含数值和状态评估
3. **关键发现** — 2-3条要点
4. **风险分析** — 根据风险等级具体分析
5. **明日预测** — 简要预测说明
6. **建议措施** — 3-5条可操作的具体建议

请用中文撰写，语言专业、温暖、有建设性。"""


def generate_report(
    profile: dict,
    use_evolution: bool = True,
    evolution_context: str = "",
    inject_position: str = "system",
) -> str | None:
    """
    生成心理日报。use_evolution 控制是否注入进化上下文；inject_position 控制注入位置。
    """
    sys_prompt = build_system_prompt(use_evolution, evolution_context, inject_position)
    user_prompt = build_user_prompt(profile)
    if use_evolution and evolution_context and inject_position == "user":
        user_prompt += "\n" + evolution_context
    return llm_generate(sys_prompt, user_prompt)


def build_evolution_context_semantic(
    category: str, profile: dict, top_k: int = 2, min_rating: int = 4
) -> str:
    """语义检索构造进化上下文：按 profile 的风险等级/情绪检索最相似的高分经验。"""
    risk = profile.get("risk_level") or profile.get("analysis_result", {}).get("risk_level")
    emotion = profile.get("emotion_data", {}).get("fused_emotion")
    exps = retrieve_experience_semantic(
        category=category, min_rating=min_rating, limit=top_k,
        risk_level=risk, emotion=emotion,
    )
    if not exps:
        return ""
    lines = ["\n## 历史成功案例参考（系统从反馈中自动学习，语义相似度优先）"]
    for i, exp in enumerate(exps, 1):
        lines.append(
            f"{i}. 对象: {exp.get('student_name', '未知')} | "
            f"评分: {exp.get('rating', '?')}/5 | 风险: {exp.get('risk_level', '?')} | 情绪: {exp.get('emotion', '?')}\n"
            f"   生成效果: {exp.get('outcome_preview', '暂无详情')[:120]}"
        )
    lines.append("\n请参考以上案例的风格和深度，为当前学生生成同样高质量的内容。")
    return "\n".join(lines)


def generate_report_semantic(
    profile: dict, top_k: int = 2, min_rating: int = 4, inject_position: str = "system"
) -> str | None:
    """语义检索版生成：根据当前档案特征检索相似经验并注入。"""
    ctx = build_evolution_context_semantic(CATEGORY, profile, top_k, min_rating)
    return generate_report(
        profile, use_evolution=bool(ctx), evolution_context=ctx, inject_position=inject_position
    )


# ========================
# LLM-as-judge 评估
# ========================

JUDGE_SYS = """你是一位严格的心理报告评审专家。请对下面这份学生心理健康评估日报打分。

评分维度（每项 1-10 分，1=极差，10=极好）：
1. 专业性：心理学表述是否准确、恰当，是否使用专业但不过度学术化的语言
2. 相关性：报告内容是否忠实反映输入的情绪数据（评分、情绪、趋势）
3. 共情性：措辞是否温暖、有建设性，不引起不必要的恐慌
4. 建议可操作性：建议是否具体、可执行、有针对性
5. 结构完整性：报告是否包含情绪概况、指标表格、关键发现、风险分析、预测、建议

评分要求：
- 充分利用 1-10 的完整范围，不要都打高分
- 普通质量的报告应在 5-7 分，有明显缺陷应低于 5 分，特别优秀才能给 8-10 分
- 严格区分不同报告之间的细微差异

请严格只输出一行 JSON，不要输出任何其他文字、解释或 Markdown 代码块：
{"专业性": X, "相关性": X, "共情性": X, "建议可操作性": X, "结构完整性": X}
其中 X 为 1 到 10 的整数。"""


def judge(report_text: str, profile: dict | None = None) -> dict | None:
    """对一份报告做多维打分，返回 {维度: 分数} 或 None（调用/解析失败）。"""
    if not report_text:
        return None
    ctx = profile if profile else {}
    user_prompt = (
        f"## 待评估报告\n\n{report_text}\n\n"
        f"## 输入的情绪数据（用于判断相关性）\n\n"
        f"{json.dumps(ctx, ensure_ascii=False, indent=2)}\n\n请打分。"
    )
    raw = llm_generate(JUDGE_SYS, user_prompt, temperature=0.0, max_tokens=256)
    if not raw:
        return None
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        scores = json.loads(m.group(0))
        return {k: int(v) for k, v in scores.items() if k in DIMENSIONS}
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def mean_score(scores: dict | None) -> float | None:
    """五维平均分，缺维度则用已有维度平均。"""
    if not scores:
        return None
    vals = [v for v in scores.values() if isinstance(v, (int, float))]
    return round(sum(vals) / len(vals), 3) if vals else None
