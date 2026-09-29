# -*- coding: utf-8 -*-
"""拼写一致性检查：全文应统一为**美式**拼写。

起因（2026-09-13）：新写 §2.2/§2.4 时引入了 4 处英式拼写（behavioural / specialisation /
generalisation / behaviour），与全文美式不一致；另查出 1 处原有不一致。
评审建议 3「检查拼写一致性」被我评为"泛泛之谈"，结果自己在下一段就犯了。
把这个检查固化下来，避免再犯。

运行：cd 代码 && D:/python/python.exe scripts/paper/check_spelling_consistency.py
"""
from __future__ import annotations
import os
import re
import sys

TEX = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "jaamas_journal", "main.tex")
BIB = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                   "jaamas_journal", "references.bib")

# 英式 → 美式。
# 坑（2026-09-13）：首版用「词干裸匹配」，误报一堆 ——
#   `optimis` 命中 **optimism**（美式）、`characteris` 命中 **characteristics**（美式）、
#   `programme` 命中 **programmer**、`specialis` 命中 Zoethout 标题里的 **Specialists**。
# 故拆成两类：只有**确实带英式后缀**的才算，以及本身就拼写不同的独立词。
ISE_STEMS = {                       # 词干 + ise/ised/ises/ising/isation
    "organis": "organiz",
    "specialis": "specializ",       # 不含 specialist（英美同形）
    "generalis": "generaliz",       # 不含 generalist
    "characteris": "characteriz",   # 不含 characteristic
    "optimis": "optimiz",           # 不含 optimism
    "normalis": "normaliz",
    "utilis": "utiliz",
    "recognis": "recogniz",
    "catalyse": "catalyze",
    "analys": "analyz",             # 不含 analysis（英美同形）
    "minimis": "minimiz",
    "maximis": "maximiz",
    # 坑（2026-09-29）：本词表漏了 economis，于是 §6 里我新写的 "economise" 一路漏检。
    #        参照《论文修改准则》自查时才发现。**词表是枚举式的，永远会漏** ——
    #        所以下面 STANDALONE 也补了几个同类漏网词。
    "economis": "economiz",
    "emphasis": "emphasiz",
    "summaris": "summariz",
    "realis": "realiz",
    "apologis": "apologiz",
    "criticis": "criticiz",
}
STANDALONE = {                      # 本身拼写就不同
    "behaviour": "behavior",
    "colour": "color",
    "favour": "favor",
    "labour": "labor",
    "neighbour": "neighbor",
    "modelling": "modeling",
    "labelled": "labeled",
    "signalling": "signaling",
    "centre": "center",
    "fibre": "fiber",
    "defence": "defense",
    "licence": "license",
    "practise": "practice",
    "catalogue": "catalog",
    "analogue": "analog",           # 2026-09-29 补：§5.10 里我写的 analogue 曾漏检
    "judgement": "judgment",
    "acknowledgement": "acknowledgment",
    "whilst": "while",
    "amongst": "among",
    "grey": "gray",
    "metre": "meter",
    "litre": "liter",
    "theatre": "theater",
    "offence": "offense",
    "pretence": "pretense",
    "storey": "story",
    "mould": "mold",
    "smoulder": "smolder",
}
SUFFIX = r"(?:e|es|ed|ing|ation|ations)\b"
PROGRAMME = r"programme\b"


def british_hits(line: str):
    """返回该行命中的 (英式词, 美式词)。"""
    out = []
    for stem, amr in ISE_STEMS.items():
        for m in re.finditer(stem + SUFFIX, line, re.I):
            out.append((m.group(0), amr + m.group(0)[len(stem):]))
    for brt, amr in STANDALONE.items():
        for m in re.finditer(r"\b" + brt + r"\w*", line, re.I):
            out.append((m.group(0), amr + m.group(0)[len(brt):]))
    for m in re.finditer(PROGRAMME, line):
        out.append((m.group(0), "program"))
    return out


def scan(path: str, label: str) -> int:
    lines = open(path, encoding="utf-8").read().split("\n")
    is_bib = path == BIB
    hits = []
    in_title = False          # bib 的 title 字段是**引文原文**，拼写必须照抄
    for n, line in enumerate(lines, 1):
        if is_bib:
            if re.search(r"\btitle\s*=", line):
                in_title = True
            if in_title:
                if line.count("}") and not line.rstrip().endswith(("{", "and")) \
                        and re.search(r"\}\s*,?\s*$", line):
                    in_title = False
                continue
        else:
            if line.lstrip().startswith("%"):     # 注释掉的模板建议，不是正文
                continue
        for got, want in british_hits(line):
            hits.append((n, got, want, line.strip()[:88]))
    print(f"\n[{label}] {len(hits)} 处")
    for n, got, want, ctx in hits:
        print(f"  {os.path.basename(path)}:{n}  {got!r} → {want!r}")
        print(f"      {ctx}")
    return len(hits)


def main() -> int:
    total = scan(TEX, "main.tex") + scan(BIB, "references.bib")
    print()
    if total == 0:
        print("[OK] 全文拼写一致（美式），无英式残留")
    else:
        print(f"[!!] {total} 处英式拼写，与全文美式不一致")
    return total


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
