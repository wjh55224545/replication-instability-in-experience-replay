"""
论文实验 · 并行执行工具 (concurrency.py)
========================================
把串行的实验循环改成并行，把 5 小时压到 40 分钟。

背景：每次 LLM 调用约 22 秒（模型逐字吐 token，算术上快不了），
而原脚本是三到四层嵌套 for 循环串行发送，99% 的时间在等待。

用法：
    from concurrency import pmap, JsonlWriter, DEFAULT_WORKERS

    results = pmap(generate_one, profiles, label="生成")

环境变量：
    PAPER_WORKERS     并行线程数（默认见 DEFAULT_WORKERS）
    LLM_MAX_CONCURRENCY  底层 llm_client 的信号量上限（需 >= PAPER_WORKERS）
    LLM_ACQUIRE_TIMEOUT  建议设 0（无限等待），避免并发饱和静默丢样本

注意：提高并发前先用 probe_concurrency.py 探测 API 限流阈值。
"""

from __future__ import annotations

import os
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 默认并发线程数。moark.com 的限流阈值未公开，保守起步；
# 探测后可调高（环境变量 PAPER_WORKERS 覆盖）。
DEFAULT_WORKERS = int(os.environ.get("PAPER_WORKERS", "6"))


def pmap(fn, items, workers: int | None = None, label: str = "",
         retries: int = 2, retry_wait: float = 3.0):
    """
    并行执行 fn(item)，保持返回顺序与 items 一致。

    Args:
        fn:     单参数函数；返回 None 视为失败（会重试）
        items:  待处理列表
        workers: 并发数，默认 DEFAULT_WORKERS
        label:  进度输出前缀
        retries: 失败重试次数
        retry_wait: 重试前的等待秒数（线性递增）

    Returns:
        list，长度与 items 相同；失败项为 None
    """
    items = list(items)
    n = len(items)
    if n == 0:
        return []
    workers = workers or DEFAULT_WORKERS
    workers = max(1, min(workers, n))
    results: list = [None] * n
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def task(idx_item):
        idx, item = idx_item
        last = None
        for attempt in range(retries + 1):
            try:
                out = fn(item)
                if out is not None:
                    return idx, out
                last = "返回 None"
            except Exception as e:                      # noqa: BLE001
                last = f"{type(e).__name__}: {e}"
            if attempt < retries:
                time.sleep(retry_wait * (attempt + 1))
        return idx, None if last is None else None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(task, (i, it)): i for i, it in enumerate(items)}
        for fut in as_completed(futures):
            idx, out = fut.result()
            results[idx] = out
            with lock:
                done += 1
                if label and (done % 10 == 0 or done == n):
                    el = time.time() - t0
                    rate = done / el if el > 0 else 0
                    eta = (n - done) / rate if rate > 0 else 0
                    print(f"[{label}] {done}/{n}  "
                          f"用时 {el/60:.1f}min  剩余约 {eta/60:.1f}min", flush=True)

    fails = sum(1 for r in results if r is None)
    if fails:
        print(f"[{label}] 完成，{fails}/{n} 项失败（返回 None）", flush=True)
    return results


class JsonlWriter:
    """线程安全的 JSONL 追加写入器。

    并行写入时多线程同时打开同一文件会交错损坏内容，必须加锁。
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def append(self, obj: dict) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def count(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path, encoding="utf-8") as f:
            return sum(1 for _ in f)


def describe_plan(n_calls: int, workers: int | None = None,
                  sec_per_call: float = 22.0) -> str:
    """估算串行 vs 并行耗时，用于运行前提示。"""
    w = workers or DEFAULT_WORKERS
    serial = n_calls * sec_per_call / 60
    par = n_calls * sec_per_call / max(1, w) / 60
    return (f"共 {n_calls} 次调用｜串行约 {serial:.0f} 分钟｜"
            f"{w} 并发约 {par:.0f} 分钟")
