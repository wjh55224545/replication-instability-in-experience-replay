"""
论文实验 · 数据构造脚本 (make_dataset.py)
==========================================

从 psych.db 的真实情绪记录，按「学生 × 日期」聚合构造学生情绪档案，
作为 ReportGenerator 生成心理日报的输入。每个档案包含：

- emotion_data:     融合情绪 + 分数 + 效价 + 唤醒度
- analysis_result:  overall_score / risk_level / 12 项指标的子集

输出 JSON 到 data/paper_dataset.json，供 A/B / 收敛 / 消融实验复用。
"""

from __future__ import annotations
import os
import json
import sqlite3
from collections import defaultdict, Counter

DB = os.path.join(os.path.dirname(__file__), "..", "..", "data", "psych.db")
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paper_dataset.json")


def load_records():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("SELECT * FROM emotion_records ORDER BY recorded_at"))
    students = {r["id"]: r["name"] for r in conn.execute("SELECT id, name FROM students")}
    conn.close()
    return rows, students


def group_by_student_day(rows):
    groups = defaultdict(list)
    for r in rows:
        day = (r["recorded_at"] or "")[:10]  # YYYY-MM-DD
        groups[(r["student_id"], day)].append(r)
    return groups


def risk_level(score: float) -> str:
    if score >= 0.7:
        return "green"
    if score >= 0.4:
        return "yellow"
    return "red"


def build_profile(student_name: str, day: str, recs) -> dict:
    scores = [r["fused_score"] if r["fused_score"] is not None else 0.5 for r in recs]
    valences = [r["fused_valence"] if r["fused_valence"] is not None else 0.0 for r in recs]
    arousals = [r["fused_arousal"] if r["fused_arousal"] is not None else 0.0 for r in recs]
    emotions = [r["fused_emotion"] for r in recs if r["fused_emotion"]]
    n = len(scores)

    overall = sum(scores) / n
    dom_emotion = Counter(emotions).most_common(1)[0][0] if emotions else "平静"

    # 方差 → 稳定性
    var = sum((s - overall) ** 2 for s in scores) / n
    stability = max(0.0, min(1.0, 1 - var * 4))
    positive_ratio = sum(1 for v in valences if v > 0.1) / n
    negative_ratio = sum(1 for v in valences if v < -0.1) / n

    # 线性趋势
    trend = "稳定"
    if n >= 2:
        xs = list(range(n))
        mx = sum(xs) / n
        denom = sum((x - mx) ** 2 for x in xs) or 1.0
        slope = sum((xs[i] - mx) * (scores[i] - overall) for i in range(n)) / denom
        trend = "上升" if slope > 0.03 else ("下降" if slope < -0.03 else "稳定")

    level = risk_level(overall)
    return {
        "student_name": student_name,
        "date": day,
        "risk_level": level,
        "emotion_data": {
            "fused_emotion": dom_emotion,
            "fused_score": round(overall, 3),
            "fused_valence": round(sum(valences) / n, 3),
            "fused_arousal": round(sum(arousals) / n, 3),
        },
        "analysis_result": {
            "overall_score": round(overall, 3),
            "risk_level": level,
            "indicators": {
                "emotional_stability_index": round(stability, 3),
                "positive_emotion_ratio": round(positive_ratio, 3),
                "negative_emotion_ratio": round(negative_ratio, 3),
                "trend": trend,
                "stress_accumulation_index": round(negative_ratio * (1 - stability), 3),
                "emotion_recovery_speed": round(1 - negative_ratio, 3),
            },
            "risk_factors": [] if level == "green" else ["情绪波动偏大"],
            "suggestions": [],
            "llm_prediction": {"trend_prediction": trend},
        },
    }


POS_EMOTIONS = ["开心", "平静", "惊讶", "中性"]
NEG_EMOTIONS = ["悲伤", "愤怒", "焦虑"]


def synth_profile(level: str, idx: int, rng) -> dict:
    """合成一个指定风险等级的档案，参数范围基于真实数据分布（fused_score 0.18~0.95）。"""
    if level == "green":
        score = rng.uniform(0.70, 0.95)
        valence = rng.uniform(0.20, 0.80)
        arousal = rng.uniform(0.20, 0.90)
        emotion = rng.choice(["开心", "平静", "惊讶"])
        negative_ratio = rng.uniform(0.05, 0.20)
        stability = rng.uniform(0.70, 0.95)
        trend = rng.choice(["上升", "稳定"])
        risk_factors = []
    elif level == "yellow":
        score = rng.uniform(0.40, 0.70)
        valence = rng.uniform(-0.10, 0.20)
        arousal = rng.uniform(0.20, 0.70)
        emotion = rng.choice(["中性", "平静", "惊讶"])
        negative_ratio = rng.uniform(0.20, 0.40)
        stability = rng.uniform(0.40, 0.75)
        trend = rng.choice(["下降", "稳定"])
        risk_factors = ["情绪波动偏大"]
    else:  # red
        score = rng.uniform(0.10, 0.40)
        valence = rng.uniform(-0.80, -0.20)
        arousal = rng.uniform(0.40, 0.95)
        emotion = rng.choice(["悲伤", "愤怒", "焦虑"])
        negative_ratio = rng.uniform(0.45, 0.75)
        stability = rng.uniform(0.20, 0.55)
        trend = "下降"
        risk_factors = ["持续负面情绪", "情绪恢复缓慢"]

    positive_ratio = round(1 - negative_ratio, 3)
    return {
        "student_name": f"合成学生{idx}",
        "date": f"2026-08-{10 + (idx % 20):02d}",
        "risk_level": level,
        "emotion_data": {
            "fused_emotion": emotion,
            "fused_score": round(score, 3),
            "fused_valence": round(valence, 3),
            "fused_arousal": round(arousal, 3),
        },
        "analysis_result": {
            "overall_score": round(score, 3),
            "risk_level": level,
            "indicators": {
                "emotional_stability_index": round(stability, 3),
                "positive_emotion_ratio": positive_ratio,
                "negative_emotion_ratio": round(negative_ratio, 3),
                "trend": trend,
                "stress_accumulation_index": round(negative_ratio * (1 - stability), 3),
                "emotion_recovery_speed": round(1 - negative_ratio, 3),
            },
            "risk_factors": risk_factors,
            "suggestions": [],
            "llm_prediction": {"trend_prediction": trend},
        },
    }


def main():
    import random
    rng = random.Random(42)

    rows, students = load_records()
    groups = group_by_student_day(rows)
    profiles = []
    for (sid, day), recs in sorted(groups.items()):
        profiles.append(build_profile(students.get(sid, f"学生{sid}"), day, recs))

    # 合成补齐：目标绿/黄/红各约 14 个
    cnt = Counter(p["risk_level"] for p in profiles)
    target = 14
    idx = 0
    for level in ["green", "yellow", "red"]:
        need = max(0, target - cnt.get(level, 0))
        for _ in range(need):
            profiles.append(synth_profile(level, idx, rng))
            idx += 1

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)

    lv = Counter(p["risk_level"] for p in profiles)
    print(f"[make_dataset] 生成 {len(profiles)} 个档案, 风险分布={dict(lv)}")
    print(f"[make_dataset] 输出: {OUT}")


if __name__ == "__main__":
    main()
