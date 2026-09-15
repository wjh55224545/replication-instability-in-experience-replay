"""
论文 · 同配置多次运行分析（通用版）(analyze_replicates_v2.py)
==========================================================
原 analyze_replicates.py 把 DeepSeek 的文件名写死了。本版按模型名通用：

    {model}_replicate_run{1..N}.json   （由 replicate_runs_parallel.py 产出）

对每个模型汇总「同一配置 N 次独立运行」的效应估计分布，并给出跨模型对照。

用法：
    cd 代码 && D:/python/python.exe scripts/paper/analyze_replicates_v2.py
    D:/python/python.exe scripts/paper/analyze_replicates_v2.py lingshu 5
"""

from __future__ import annotations
import os
import sys
import json
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def one_run(path: str, rk: str = "random", mk: str = "semantic"):
    d = json.load(open(path, encoding="utf-8"))
    rs = [r for r in d["results"]
          if r.get("baseline_mean") is not None and r.get(f"{rk}_mean") is not None]
    B = np.array([r["baseline_mean"] for r in rs])
    out = {"n": len(rs), "pool": d["meta"].get("pool_size")}
    for name, key in [("random", rk), ("matched", mk)]:
        X = np.array([r[f"{key}_mean"] for r in rs if r.get(f"{key}_mean") is not None])
        if len(X) != len(B):
            continue
        df = X - B
        try:
            # 双尾（scipy 默认）。2026-09-15 起不再手工折半为单尾——论文的主张
            # 「效应不稳定」无方向，单尾会放大 I 类错误。
            _, p = stats.wilcoxon(df)
        except Exception:                                   # noqa: BLE001
            p = float("nan")
        se = df.std(ddof=1) / np.sqrt(len(df))
        out[name] = {"delta": float(df.mean()), "se": float(se), "p": float(p),
                     "dz": float(df.mean() / df.std(ddof=1)) if df.std(ddof=1) > 0 else 0.0,
                     "ratio": float(df.mean() / se) if se > 0 else float("nan")}
    return out


def analyze_model(model: str, n_runs: int):
    print(f"\n{'='*78}")
    print(f"模型 {model}｜同一配置 {n_runs} 次独立运行（池 45 / 每格重复 3 次）")
    print(f"{'='*78}")
    print(f"{'运行':<6}{'n':>4}{'池':>5} | {'random Δ':>10}{'p':>8}{'Δ/SE':>7}"
          f" | {'matched Δ':>10}{'p':>8}{'Δ/SE':>7}")
    print("-" * 78)

    res = {}
    for tag in range(1, n_runs + 1):
        path = os.path.join(DATA, f"{model}_replicate_run{tag}.json")
        if not os.path.exists(path):
            print(f"run{tag:<3} (缺失)")
            continue
        r = one_run(path)
        res[f"run {tag}"] = r
        sig = "*" if r["random"]["p"] < 0.05 else " "
        print(f"run{tag:<3}{r['n']:>4}{r['pool']:>5} | "
              f"{r['random']['delta']:>+10.4f}{r['random']['p']:>7.4f}{sig}"
              f"{r['random']['ratio']:>7.2f} | "
              f"{r['matched']['delta']:>+10.4f}{r['matched']['p']:>7.4f} "
              f"{r['matched']['ratio']:>6.2f}")

    if not res:
        print("  无数据。")
        return None

    summary = {"model": model, "runs": {}}
    for cond in ["random", "matched"]:
        vals = [r[cond] for r in res.values() if cond in r]
        if not vals:
            continue
        deltas = np.array([v["delta"] for v in vals])
        ratios = np.array([v["ratio"] for v in vals])
        ps = np.array([v["p"] for v in vals])
        summary["runs"][cond] = {
            "delta_range": [float(deltas.min()), float(deltas.max())],
            "delta_span": float(deltas.max() - deltas.min()),
            "delta_mean": float(deltas.mean()), "delta_sd": float(deltas.std(ddof=1)),
            "ratio_range": [float(np.nanmin(ratios)), float(np.nanmax(ratios))],
            "n_sig": int((ps < 0.05).sum()), "n": len(ps),
            "signs": [("+" if d > 0 else "-") for d in deltas]}
    print("-" * 78)
    for cond, s in summary["runs"].items():
        print(f"\n  [{cond}] Δ 范围 {s['delta_range'][0]:+.4f} ~ {s['delta_range'][1]:+.4f}"
              f"　极差 {s['delta_span']:.4f}")
        print(f"          Δ/SE 范围 {s['ratio_range'][0]:.2f} ~ {s['ratio_range'][1]:.2f}")
        print(f"          显著次数 {s['n_sig']}/{s['n']}　符号 {' '.join(s['signs'])}")
    return summary


def main():
    models = [(sys.argv[1], int(sys.argv[2]))] if len(sys.argv) > 2 else \
             [("lingshu", 5), ("deepseek", 5)]

    print("=" * 78)
    print("同配置多次独立运行：效应估计的稳定性")
    print("=" * 78)

    out = {}
    for m, n in models:
        s = analyze_model(m, n)
        if s:
            out[m] = s

    # 跨模型对照
    if len(out) > 1:
        print(f"\n{'='*78}")
        print("跨模型对照（random 条件）")
        print(f"{'='*78}")
        print(f"{'模型':<12}{'Δ 极差':>10}{'Δ/SE 范围':>18}{'显著':>8}{'符号':>10}")
        print("-" * 78)
        for m, s in out.items():
            r = s["runs"].get("random")
            if not r:
                continue
            lo, hi = r["ratio_range"]
            signs = " ".join(r["signs"])
            print(f"{m:<12}{r['delta_span']:>10.4f}"
                  f"{lo:>10.2f}~{hi:<7.2f}"
                  f"{r['n_sig']}/{r['n']:>6}"
                  f"{signs:>10}")

    outp = os.path.join(DATA, "replicate_summary_all_models.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {outp}")


if __name__ == "__main__":
    main()
