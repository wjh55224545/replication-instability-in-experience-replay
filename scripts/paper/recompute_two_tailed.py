"""
双尾检验重算 (recompute_two_tailed.py)
======================================
背景（2026-09-15，第五份评审意见）：
    全文使用**单尾** Wilcoxon 符号秩检验（`stats.wilcoxon(df)` 后手工 `p /= 2`），
    而论文的核心主张是「不稳定」——一个无方向的论断。评审指出这在统计上不自洽：
    单尾会放大 I 类错误，在主张「无效应可复现」的论文中使用尤其不当，应全部改为双尾。

本脚本的作用：**在改稿之前，先算出改双尾会带来什么后果**。
    对论文中每一个报告了 p 值的比较，同时计算单尾与双尾 p，逐行对照，
    并回答一个决定性问题——**有没有哪个「显著」的格子会失去显著性？**

    Wilcoxon 符号秩检验的零分布关于 0 对称，故 exact 方法下
        p_two-sided = 2 × p_one-sided
    因此预期是「全部 p 值近似翻倍、显著性集合不变」。但这必须**从原始数据实测**，
    而不是靠这条性质推断——尤其要确认 scipy 的实际取值路径（exact / normal approx）。

同时核对 Table 3 标题里的一处交叉引用编号（见 `check_table3_caption`）。

运行：cd 代码 && D:/python/python.exe scripts/paper/recompute_two_tailed.py
"""

from __future__ import annotations
import os
import json
import numpy as np
from scipy import stats

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data")

# ---------------------------------------------------------------- 数据来源
# Table 1 / 图 2 的 15 个对比。与 viz_instability.py 的 PANELS 逐行对应。
# (编号, 模型, 标签, 文件, 条件；cond=None 表示 A/B 单次运行格式)
TABLE1 = [
    (1,  "M1", "single run            random",   "ab_results.json",                None),
    (2,  "M1", "single run            matched",  "ab_semantic_results.json",       None),
    (3,  "M1", "3 reps  pool 15  random",        "ab_repeated_results.json",       "random"),
    (4,  "M1", "3 reps  pool 15  matched",       "ab_repeated_results.json",       "semantic"),
    (5,  "M1", "3 reps  pool 45  random",        "lingshu_bigpool_results.json",   "random"),
    (6,  "M1", "3 reps  pool 45  matched",       "lingshu_bigpool_results.json",   "semantic"),
    (7,  "M2", "single run            random",   "deepseek_ab_random.json",        None),
    (8,  "M2", "single run            matched",  "deepseek_ab_semantic.json",      None),
    (9,  "M2", "3 reps  pool 15  random",        "deepseek_repeated_results.json", "random"),
    (10, "M2", "3 reps  pool 15  matched",       "deepseek_repeated_results.json", "semantic"),
    (11, "M2", "3 reps  pool 45  random",        "deepseek_bigpool_results.json",  "random"),
    (12, "M2", "3 reps  pool 45  matched",       "deepseek_bigpool_results.json",  "semantic"),
    (13, "M2", "3 reps  pool 45  run 2  random", "deepseek_diversity_results.json", "random"),
    (14, "M2", "3 reps  pool 45  run 2  matched", "deepseek_diversity_results.json", "similar"),
    (15, "M2", "3 reps  pool 45  run 2  other-tier", "deepseek_diversity_results.json", "different"),
]

# Table 3 / 图 3 的 10 次独立运行。与 viz_replicates.py 的 PANELS 逐行对应。
# M2 的前两次运行保存在另外的文件名下（历史原因），必须一并列出。
TABLE3 = [
    ("M1", "lingshu_replicate_run1.json"),
    ("M1", "lingshu_replicate_run2.json"),
    ("M1", "lingshu_replicate_run3.json"),
    ("M1", "lingshu_replicate_run4.json"),
    ("M1", "lingshu_replicate_run5.json"),
    ("M2", "deepseek_bigpool_results.json"),      # M2 run 1
    ("M2", "deepseek_diversity_results.json"),    # M2 run 2
    ("M2", "deepseek_replicate_run2.json"),       # M2 run 3
    ("M2", "deepseek_replicate_run3.json"),       # M2 run 4
    ("M2", "deepseek_replicate_run4.json"),       # M2 run 5
]


def load_diffs(fname: str, cond):
    """取出配对差值数组；cond=None 时按 A/B 单次格式读。"""
    with open(os.path.join(DATA, fname), encoding="utf-8") as f:
        d = json.load(f)
    rs = d["results"]
    if cond is None:
        pairs = [(r["A"]["mean"], r["B"]["mean"]) for r in rs
                 if r["A"]["mean"] is not None and r["B"]["mean"] is not None]
    else:
        pairs = [(r["baseline_mean"], r[f"{cond}_mean"]) for r in rs
                 if r.get("baseline_mean") is not None and r.get(f"{cond}_mean") is not None]
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return b - a


def both_tails(diff: np.ndarray):
    """返回 (Δ, 单尾 p, 双尾 p, dz, n, 实际使用的方法)。"""
    n = len(diff)
    # 显式请求双尾，并记录 scipy 实际选用的方法（exact / approx / asymp）
    try:
        res = stats.wilcoxon(diff, alternative="two-sided")
        p_two = float(res.pvalue)
        method = getattr(res, "method", "n/a")
    except Exception as exc:                                  # noqa: BLE001
        return float(diff.mean()), float("nan"), float("nan"), float("nan"), n, f"ERR {exc}"
    # 单尾：论文现行做法（等价于 alternative="greater" / "less" 中较大的那个）
    p_one = p_two / 2
    sd = diff.std(ddof=1)
    dz = float(diff.mean() / sd) if sd > 0 else 0.0
    return float(diff.mean()), p_one, p_two, dz, n, method


def main():
    print("=" * 92)
    print("双尾检验重算：改双尾会改变什么？")
    print("=" * 92)

    out = {"table1": [], "table3": [], "flips": []}

    # ------------------------------------------------ Table 1
    print("\n【Table 1】15 个受控比较")
    print(f"{'#':>3} {'模型':<4}{'对比':<34}{'Δ':>9}{'单尾p':>9}{'双尾p':>9}"
          f"{'d_z':>8}  {'单尾显著':<9}{'双尾显著':<9}")
    print("-" * 92)

    n_flip = 0
    for idx, model, label, fname, cond in TABLE1:
        diff = load_diffs(fname, cond)
        delta, p1, p2, dz, n, method = both_tails(diff)
        s1, s2 = p1 < 0.05, p2 < 0.05
        if s1 != s2:
            n_flip += 1
            out["flips"].append({"where": f"Table1 #{idx}", "label": label,
                                 "p_one": p1, "p_two": p2})
        mark = "  <-- 改变!" if s1 != s2 else ""
        print(f"{idx:>3} {model:<4}{label:<34}{delta:>+9.4f}{p1:>9.4f}{p2:>9.4f}"
              f"{dz:>+8.2f}  {'YES' if s1 else 'no':<9}{'YES' if s2 else 'no':<9}{mark}")
        out["table1"].append({"idx": idx, "model": model, "label": label, "file": fname,
                              "cond": cond, "n": n, "delta": delta, "dz": dz,
                              "p_one_tailed": p1, "p_two_tailed": p2,
                              "method": method, "sig_one": s1, "sig_two": s2})

    # ------------------------------------------------ Table 3
    print("\n【Table 3】同一配置的 10 次独立运行（池 45 / 3 reps / random）")
    print(f"{'模型':<4}{'运行':<7}{'文件':<36}{'Δ':>9}{'单尾p':>9}{'双尾p':>9}"
          f"  {'单尾显著':<9}{'双尾显著':<9}")
    print("-" * 92)

    for i, (model, fname) in enumerate(TABLE3, 1):
        diff = load_diffs(fname, "random")
        delta, p1, p2, dz, n, method = both_tails(diff)
        s1, s2 = p1 < 0.05, p2 < 0.05
        if s1 != s2:
            n_flip += 1
            out["flips"].append({"where": f"Table3 {model} run{i}", "label": fname,
                                 "p_one": p1, "p_two": p2})
        mark = "  <-- 改变!" if s1 != s2 else ""
        print(f"{model:<4}{'run ' + str(i):<7}{fname:<36}{delta:>+9.4f}{p1:>9.4f}{p2:>9.4f}"
              f"  {'YES' if s1 else 'no':<9}{'YES' if s2 else 'no':<9}{mark}")
        out["table3"].append({"run": i, "model": model, "file": fname, "n": n,
                              "delta": delta, "dz": dz,
                              "p_one_tailed": p1, "p_two_tailed": p2,
                              "method": method, "sig_one": s1, "sig_two": s2})

    # ------------------------------------------------ 结论
    print("\n" + "=" * 92)
    print(f"汇总：单尾→双尾后**显著性判定改变**的格子数 = {n_flip}")
    if n_flip == 0:
        print("  → 没有任何比较改变显著性。改双尾是纯保守化，论文结论不受影响。")
    else:
        print("  → 有格子失去显著性，需逐条核对这些行在正文中的表述。")
        for f in out["flips"]:
            print(f"     {f['where']}  {f['label']}  单尾 {f['p_one']:.4f} → 双尾 {f['p_two']:.4f}")

    # 双尾下的最大值（用于正文写「最大 p = ...」这句）
    sig_two = [r["p_two_tailed"] for r in out["table1"] + out["table3"] if r["sig_two"]]
    if sig_two:
        print(f"\n  双尾下仍显著者共 {len(sig_two)} 个，其中最大 p = {max(sig_two):.4f}")
        out["max_sig_p_two"] = max(sig_two)
        out["n_sig_two"] = len(sig_two)

    # ------------------------------------------------ FDR（Benjamini–Hochberg）
    # 论文以「刻画不稳定性，而非确立单一效应」为由不做多重比较校正。评审要求补校正。
    # 关键问题不是「校正后还剩几个显著」，而是**剩下的是哪些**：
    # 若存活者本身互相矛盾，则校正反而强化论文的论点。
    print("\n" + "=" * 92)
    print("多重比较校正：Benjamini–Hochberg FDR（对全部 25 个比较，α = 0.05）")
    print("=" * 92)

    rows = [(r["p_two_tailed"], f"Table 1 #{r['idx']:<2} {r['model']} {r['label']}")
            for r in out["table1"]]
    rows += [(r["p_two_tailed"], f"Table 3     {r['model']} run {r['run']}")
             for r in out["table3"]]
    rows.sort(key=lambda t: t[0])
    m = len(rows)
    k = 0
    for i, (p, _) in enumerate(rows, 1):
        crit = i / m * 0.05
        ok = p <= crit
        if ok:
            k = i
        print(f"  {i:>2}. p = {p:.4f}   阈值 {crit:.4f}   {'通过' if ok else '不通过':<7} {rows[i-1][1]}")

    print(f"\n  存活 {k} / {m} 个比较")
    if k:
        print("  存活者：")
        for p, lab in rows[:k]:
            print(f"    p = {p:.4f}   {lab}")
    out["fdr"] = {"m": m, "alpha": 0.05, "n_surviving": k,
                  "surviving": [{"p": p, "label": lab} for p, lab in rows[:k]]}

    outp = os.path.join(DATA, "two_tailed_recompute.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {outp}")


def check_table3_caption():
    """核对 Table 3 标题里的交叉引用编号是否与实际数据一致。

    标题写的是「Runs 3--5 of M2 are also reported in rows 13--15 of Table 1」，
    但按 viz_replicates.py 的映射，M2 的 run 2 才是 deepseek_diversity_results.json
    —— 也就是 Table 1 的 rows 13--15。此处实测确认。
    """
    print("\n" + "=" * 92)
    print("附带核对：Table 3 标题声称『Runs 3--5 of M2 即 Table 1 的 rows 13--15』")
    print("=" * 92)
    d_diversity = load_diffs("deepseek_diversity_results.json", "random")
    d_rep2 = load_diffs("deepseek_replicate_run2.json", "random")
    print(f"  deepseek_diversity_results.json     Δ = {d_diversity.mean():+.4f}  (Table 1 row 13 报 +0.092)")
    print(f"  deepseek_replicate_run2.json        Δ = {d_rep2.mean():+.4f}  (Table 3 run 3 报 +0.025)")
    if abs(d_diversity.mean() - 0.092) < 0.001 and abs(d_rep2.mean() - 0.025) < 0.001:
        print("  → 确认：与 Table 1 rows 13--15 对应的是 M2 的 **Run 2**，不是 Runs 3--5。")
        print("     Table 3 标题里的 “Runs 3--5” 应改为 “Run 2”。")
    else:
        print("  → 数值与预期不符，需人工检查。")


if __name__ == "__main__":
    main()
    check_table3_caption()
