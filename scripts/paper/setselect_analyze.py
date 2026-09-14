"""
13b 分析 · 客观任务上的估计稳定性 (setselect_analyze.py)
======================================================
把 5 次独立运行的效应估计汇总，与主观任务对照，检验论文的边界预测。

对照的核心指标不是「有没有显著」，而是**信号噪声比**：
    Δ / SE   ——  主观任务该比值 0.6~3.5，横跨显著阈值，故结论翻转
    若客观任务该比值显著更大且集中在阈值同侧，边界预测成立。

运行（在本仓库根目录）：python scripts/paper/setselect_analyze.py
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

MODELS = ["lingshu", "deepseek"]


def analyze_model(path: str, label: str) -> dict | None:
    if not os.path.exists(path):
        print(f"  ({label} 文件缺失)")
        return None
    d = json.load(open(path, encoding="utf-8"))
    runs = d["runs"]

    print(f"\n{'='*72}")
    print(f"模型 {label}｜{len(runs)} 次独立运行")
    print(f"{'='*72}")
    print(f"{'运行':<6}{'n':>4} | {'random Δ':>9}{'p':>7}{'t':>6}"
          f" | {'matched Δ':>9}{'p':>7}{'t':>6}"
          f" | {'filler Δ':>9}{'p':>7}{'t':>6}")
    print("-" * 84)

    out = {"label": label, "runs": []}
    for run in runs:
        rows = [r for r in run["results"] if r.get("baseline_mean") is not None]
        B = np.array([r["baseline_mean"] for r in rows])
        rec = {"run": run["run"], "n": len(rows)}
        line = f"{run['run']:<6}{len(rows):>4} |"
        for cond in ["random", "matched", "filler"]:
            X = np.array([r[f"{cond}_mean"] for r in rows
                          if r.get(f"{cond}_mean") is not None])
            if len(X) != len(B) or len(X) < 4:
                line += f"{'--':>9}{'--':>7}{'--':>6} |"
                continue
            df = X - B
            try:
                _, p = stats.wilcoxon(df)
                p /= 2
            except Exception:                               # noqa: BLE001
                p = float("nan")
            se = df.std(ddof=1) / np.sqrt(len(df))
            ratio = df.mean() / se if se > 0 else float("nan")
            rec[cond] = {"delta": float(df.mean()), "se": float(se),
                         "p": float(p), "ratio": float(ratio)}
            star = "*" if p < 0.05 else " "
            line += f"{df.mean():>+9.4f}{p:>6.3f}{star}{ratio:>6.2f} |"
        print(line)
        out["runs"].append(rec)

    # --- 汇总 ---
    for cond in ["random", "matched", "filler"]:
        vals = [r[cond] for r in out["runs"] if cond in r]
        if not vals:
            continue
        deltas = np.array([v["delta"] for v in vals])
        ratios = np.array([v["ratio"] for v in vals])
        ps = np.array([v["p"] for v in vals])
        out[f"{cond}_summary"] = {
            "delta_range": [float(deltas.min()), float(deltas.max())],
            "delta_mean": float(deltas.mean()), "delta_sd": float(deltas.std(ddof=1)),
            "ratio_range": [float(np.nanmin(ratios)), float(np.nanmax(ratios))],
            "n_sig": int((ps < 0.05).sum()), "n": len(ps)}
        s = out[f"{cond}_summary"]
        print(f"\n  [{cond}] Δ 范围 {deltas.min():+.4f} ~ {deltas.max():+.4f}"
              f"  (极差 {deltas.max()-deltas.min():.4f})")
        print(f"          Δ/SE 范围 {np.nanmin(ratios):.2f} ~ {np.nanmax(ratios):.2f}")
        print(f"          显著次数 {s['n_sig']}/{s['n']}")
    return out


def main():
    print("="*72)
    print("13b 客观任务（集合选择）· 效应估计的稳定性")
    print("="*72)

    summaries = {}
    for m in MODELS:
        r = analyze_model(os.path.join(DATA, f"setselect_experiment_{m}.json"), m)
        if r:
            summaries[m] = r

    # --- 与主观任务对照 ---
    print(f"\n{'='*72}")
    print("与主观任务对照（论文 §5.4）")
    print(f"{'='*72}")
    subj = os.path.join(DATA, "heterogeneity_results.json")
    if os.path.exists(subj):
        h = json.load(open(subj, encoding="utf-8"))
        th = h["thetas"]
        print(f"\n主观任务（日报生成，DeepSeek，同配置 5 次运行）:")
        print(f"  Δ 范围        {min(th):+.4f} ~ {max(th):+.4f}"
              f"  (极差 {max(th)-min(th):.4f})")
        print(f"  Δ/SE 范围     {min(t/s for t,s in zip(th,h['ses'])):.2f} ~ "
              f"{max(t/s for t,s in zip(th,h['ses'])):.2f}")
        print(f"  显著次数      {'2/5（结论翻转）'}")
        print(f"  I²            {h['I2_percent']:.1f}%")

    print(f"\n客观任务（集合选择）:")
    for m, s in summaries.items():
        if "random_summary" in s:
            rs = s["random_summary"]
            print(f"  {m:10s} Δ 极差 {rs['delta_range'][1]-rs['delta_range'][0]:.4f}"
                  f"｜Δ/SE {rs['ratio_range'][0]:.2f}~{rs['ratio_range'][1]:.2f}"
                  f"｜显著 {rs['n_sig']}/{rs['n']}")

    # --- 混淆判读：有害来自 replay 内容，还是来自上下文变长？---
    print(f"\n{'='*72}")
    print("混淆判读：有害来自「replay 内容」还是「上下文长度」？")
    print(f"{'='*72}")
    print("  baseline   无额外文本")
    print("  filler     等长同格式但信息无效的伪样例（长度混淆的对照）")
    print("  random     真实过往示范，随机取")
    print("  matched    真实过往示范，按相似度取")
    print()
    for m, s in summaries.items():
        if "filler_summary" not in s or "random_summary" not in s:
            continue
        fs, rs = s["filler_summary"], s["random_summary"]
        b = s.get("baseline_sd")
        print(f"  {m}:")
        print(f"    random  Δ 均值 {rs['delta_mean']:+.4f}  显著 {rs['n_sig']}/{rs['n']}")
        print(f"    filler  Δ 均值 {fs['delta_mean']:+.4f}  显著 {fs['n_sig']}/{fs['n']}")
        diff = abs(rs["delta_mean"] - fs["delta_mean"])
        print(f"    两者之差 {diff:.4f}")
        if diff < 0.01:
            print("    → filler 与 random 表现接近：有害主要来自**上下文长度/格式**")
        elif fs["n_sig"] == 0 and rs["n_sig"] > 0:
            print("    → filler 不显著而 random 显著：有害来自 **replay 的信息内容**")
        else:
            print("    → 两者都有一定效应，需结合具体数值判断")

    print(f"\n【总判读】")
    print("  若客观任务的 Δ/SE 明显更远离阈值、且 5 次运行结论一致")
    print("  → 论文边界预测成立：不稳定性随 噪声/信号 比 下降")
    print("  反之 → 边界声明需修改（现象比我们声称的更普遍）")

    out = os.path.join(DATA, "setselect_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {out}")


if __name__ == "__main__":
    main()
