"""
并发阈值探测 (probe_concurrency.py)
===================================
在提高实验并发前，先测出 API 能承受多少并发请求。

做法：在每个并发档位上同时发出 N 个**短请求**（max_tokens 很小，几秒即回），
测量成功率、延迟与错误类型。遇到 429 / 超时即说明已触限。

输出建议的安全并发数，供 PAPER_WORKERS 使用。

运行：
    python scripts/paper/probe_concurrency.py
    python scripts/paper/probe_concurrency.py deepseek
"""

from __future__ import annotations

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openai import OpenAI                       # noqa: E402
from backend.config import get_settings         # noqa: E402

LEVELS = [2, 4, 8, 12, 16]
PER_LEVEL = 8           # 每档发出的请求总数

# 真实负载模式：max_tokens 小的时候请求 0.4 秒就返回，根本压不到服务端，
# 测出的"100% 成功"没有意义。必须用接近真实实验的负载才有参考价值。
REALISTIC_MAX_TOKENS = int(os.environ.get("PROBE_MAX_TOKENS", "0"))
if REALISTIC_MAX_TOKENS > 0:
    LEVELS = [4, 8, 12]
    PER_LEVEL = 12


def make_client(model_name: str):
    s = get_settings()
    if model_name == "deepseek":
        return OpenAI(base_url=s.deepseek_base_url, api_key=s.deepseek_api_key), s.deepseek_model
    return OpenAI(base_url=s.lingshu_base_url, api_key=s.lingshu_api_key), s.lingshu_model


PROBE_PROMPT = (
    "你是一位专业的学校心理辅导老师，负责撰写学生心理健康评估日报。"
    "请生成一份约 600 字的报告，包含情绪概况、关键发现、风险分析、建议措施四个部分。"
)


def one_call(client, model, idx, out, lock):
    t0 = time.time()
    mt = REALISTIC_MAX_TOKENS or 8
    prompt = PROBE_PROMPT if REALISTIC_MAX_TOKENS else f"回复数字 {idx}，不要任何其他内容。"
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=mt,
        )
        content = r.choices[0].message.content or ""
        usage = getattr(r, "usage", None)
        ct = usage.completion_tokens if usage else 0
        rec = {"ok": bool(content), "sec": time.time() - t0, "err": "", "tokens": ct}
    except Exception as e:                              # noqa: BLE001
        rec = {"ok": False, "sec": time.time() - t0, "tokens": 0,
               "err": f"{type(e).__name__}: {str(e)[:110]}"}
    with lock:
        out.append(rec)


def main(model_name: str = "lingshu"):
    client, model = make_client(model_name)
    print(f"探测目标: {model_name} ({model})")
    print(f"每档 {PER_LEVEL} 个请求，并发档位 {LEVELS}\n")
    print(f"{'并发':>4} {'成功':>6} {'成功率':>8} {'平均延迟':>10} {'最慢':>8}  错误")
    print("-" * 74)

    best = 0
    for w in LEVELS:
        out: list = []
        lock = threading.Lock()
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=w) as ex:
            for i in range(PER_LEVEL):
                ex.submit(one_call, client, model, i, out, lock)
        wall = time.time() - t0

        ok = sum(1 for r in out if r["ok"])
        rate = ok / len(out) if out else 0
        avg = sum(r["sec"] for r in out) / len(out) if out else 0
        mx = max((r["sec"] for r in out), default=0)
        errs = [r["err"] for r in out if r["err"]]
        err_show = errs[0][:44] if errs else "—"

        flag = ""
        if rate >= 0.99:
            best = w
            flag = "  <- 可用"
        print(f"{w:>4} {ok:>4}/{len(out):<2} {rate:>7.0%} {avg:>9.1f}s {mx:>7.1f}s  {err_show}{flag}")
        if rate < 0.9:
            print(f"\n并发 {w} 已出现失败，停止上探。")
            break
        time.sleep(1.0)

    print("-" * 74)
    mode = (f"真实负载 (max_tokens={REALISTIC_MAX_TOKENS})"
            if REALISTIC_MAX_TOKENS else "轻量探针（仅供参考，非真实压力）")
    print(f"\n[模式] {mode}")
    print(f"建议 PAPER_WORKERS = {best}（留一档余量可试 {max(2, best)}）")
    print(f"配套环境变量：")
    print(f"    LLM_MAX_CONCURRENCY={best}")
    print(f"    LLM_ACQUIRE_TIMEOUT=0   # 无限等待，避免并发饱和静默丢样本")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "lingshu")
