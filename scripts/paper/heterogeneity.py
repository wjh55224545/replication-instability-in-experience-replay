"""
论文 · 效应量异质性分析 (heterogeneity.py)
=========================================
对同一配置的 5 次独立运行计算 meta 分析式的异质性统计量：

  Q  = Σ w_i (θ_i − θ̄_w)²        （Cochran's Q，权重 w_i = 1/SE_i²）
  I² = max(0, (Q − df) / Q) × 100  （异质性占总变异的比例）
  τ² = max(0, (Q − df) / (Σw − Σw²/Σw))   （between-run 方差分量）

解读：
  I² ≈ 0  → 各次运行的估计差异可由「运行内抽样误差」完全解释
  I² 高   → 存在超出抽样误差的 between-run 变异（即经验池重抽样带来的额外方差）
            这正是论文「同一配置结论不可复现」的量化证据。

运行（在本仓库根目录）：python scripts/paper/heterogeneity.py
"""

from __future__ import annotations
import os
import json
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# (标签, 文件, random 的键名)  —— 5 次同配置运行
RUNS = [
    ("run 1", "deepseek_bigpool_results.json",   "random"),
    ("run 2", "deepseek_diversity_results.json", "random"),
    ("run 3", "deepseek_replicate_run2.json",    "random"),
    ("run 4", "deepseek_replicate_run3.json",    "random"),
    ("run 5", "deepseek_replicate_run4.json",    "random"),
]


def effect_and_se(path, key):
    """返回配对的平均差及其标准误。"""
    d = json.load(open(path, encoding="utf-8"))
    rs = [r for r in d["results"]
          if r.get("baseline_mean") is not None and r.get(f"{key}_mean") is not None]
    B = np.array([r["baseline_mean"] for r in rs])
    X = np.array([r[f"{key}_mean"] for r in rs])
    df = X - B
    n = len(df)
    return df.mean(), df.std(ddof=1) / np.sqrt(n), n


def main():
    thetas, ses, labels = [], [], []
    for label, f, key in RUNS:
        p = os.path.join(DATA, f)
        if not os.path.exists(p):
            print(f"  (缺失: {f})")
            continue
        m, se, n = effect_and_se(p, key)
        thetas.append(m); ses.append(se); labels.append(label)
        print(f"{label}: Δ = {m:+.4f}, SE = {se:.4f}, n = {n}")

    thetas = np.array(thetas); ses = np.array(ses)
    k = len(thetas)
    if k < 2:
        print("运行数不足，无法计算异质性。")
        return

    w = 1.0 / ses**2                      # 固定效应权重
    theta_bar = (w * thetas).sum() / w.sum()   # 加权均值
    Q = (w * (thetas - theta_bar) ** 2).sum()
    df = k - 1

    I2 = max(0.0, (Q - df) / Q) * 100 if Q > 0 else 0.0
    tau2 = max(0.0, (Q - df) / (w.sum() - (w**2).sum() / w.sum()))
    p_q = 1 - stats.chi2.cdf(Q, df)       # Q 检验的 p 值

    print()
    print("=" * 52)
    print(f"加权均值 (pooled): {theta_bar:+.4f}")
    print(f"Cochran's Q     : {Q:.3f}  (df = {df})")
    print(f"Q 检验 p 值     : {p_q:.4f}"
          f"  {'<- 存在显著异质性' if p_q < 0.05 else ''}")
    print(f"I^2             : {I2:.1f}%")
    print(f"tau^2 (between-run): {tau2:.6f}")
    print(f"tau   (SD)      : {np.sqrt(tau2):.4f}")
    print(f"运行间 Δ 极差   : {thetas.max() - thetas.min():.4f}"
          f"  ({thetas.min():+.3f} ~ {thetas.max():+.3f})")
    print("=" * 52)

    out = os.path.join(DATA, "heterogeneity_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "thetas": thetas.tolist(), "ses": ses.tolist(),
                   "theta_bar": float(theta_bar), "Q": float(Q), "df": df,
                   "Q_p": float(p_q), "I2_percent": float(I2), "tau2": float(tau2)},
                  f, ensure_ascii=False, indent=2)
    print(f"[保存] {out}")


if __name__ == "__main__":
    main()
