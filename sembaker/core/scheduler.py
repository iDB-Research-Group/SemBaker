"""Backend-agnostic compile-task scheduler.

This is the SINGLE place that owns compile parallelism. Backends (and harnesses)
hand it a flat list of zero-arg "compile tasks"; it runs them concurrently. The
same entry point serves both axes:

  * within-query parallelism — a query's filter/map/join compiles run together
    (a backend's eager path submits its node tasks here), and
  * cross-query parallelism — many queries' compiles run together (a warm phase
    submits one task per query here).

Keeping this external — not duplicated inside each backend adapter or each eval
harness — means every backend (PZ / Lotus / Nirvana / DocETL) gets identical
parallel/async behavior for free, matching the external-optimizer design.

Concurrency is bounded by CX_COMPILE_WORKERS (default 8). A task that raises is
captured (its result slot holds the Exception) so one bad compile never sinks
the whole batch.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

DEFAULT_WORKERS = int(os.environ.get("CX_COMPILE_WORKERS", "8"))

# Canonical warm/eager switch aliases: EAGER (Lotus/Nirvana) and PZ_WARM (PZ)
# predate the unified name and stay valid so old commands keep working.
_WARM_ENV = ("CX_WARM", "EAGER", "PZ_WARM")


def warm_enabled() -> bool:
    """True if the warm / eager pre-compile phase is on for THIS run.

    One switch across every backend harness: `CX_WARM=1`. `EAGER=1` and
    `PZ_WARM=1` are kept as backward-compatible aliases (any of the three
    turns warm on)."""
    return any(os.environ.get(k) == "1" for k in _WARM_ENV)


def _safe(task: Callable):
    try:
        return task()
    except Exception as e:  # keep the batch alive; caller can inspect
        return e


def run(tasks, *, workers: int | None = None, label: str = "compile") -> list:
    """Run zero-arg compile `tasks` concurrently; return results in order.

    Used for BOTH within-query (a query's operator compiles) and cross-query
    (many queries) — one flat pool, one code path. `workers=1` runs serially
    (useful for debugging / strict ordering)."""
    tasks = [t for t in (tasks or []) if t is not None]
    if not tasks:
        return []
    w = min(workers if workers is not None else DEFAULT_WORKERS, len(tasks))
    if w <= 1:
        return [_safe(t) for t in tasks]
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(_safe, tasks))


def run_map(fn: Callable, items, *, workers: int | None = None, label: str = "compile") -> list:
    """Convenience: run `fn(item)` over `items` concurrently."""
    return run([(lambda it=it: fn(it)) for it in items], workers=workers, label=label)


# ---- background (non-blocking) submission ------------------------------------
# For pipelined execution: submit compile tasks and return IMMEDIATELY —
# execution starts right away and rendezvouses per operator (compile_op's
# single-flight table makes an executing operator WAIT on an in-flight compile
# instead of duplicating it). Submission order = start order (FIFO pool), so
# callers should submit in execution order to keep compiles ahead of the run.

_BG_POOL: ThreadPoolExecutor | None = None


def submit(tasks, *, workers: int | None = None, label: str = "compile-bg") -> list:
    """Submit zero-arg tasks to a persistent background pool; returns the
    futures without waiting. Exceptions are captured per-task (same contract
    as run())."""
    global _BG_POOL
    tasks = [t for t in (tasks or []) if t is not None]
    if not tasks:
        return []
    if _BG_POOL is None:
        _BG_POOL = ThreadPoolExecutor(max_workers=workers or DEFAULT_WORKERS)
    return [_BG_POOL.submit(_safe, t) for t in tasks]
