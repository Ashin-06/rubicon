"""Sync calls reuse one event loop per thread instead of building a new one every call."""
import asyncio
import threading
import time

from rubicon import APPROVE_ALL, NOOP, Transaction, run, tool


def test_2000_sync_calls_in_a_row_stay_fast():
    reg = {}
    tool(lambda k: k, name="put", undo=NOOP, registry=reg)
    tx = Transaction(registry=reg, log=lambda m: None)
    started = time.monotonic()
    for i in range(2000):
        assert tx.call("put", i) == i
    tx.commit()
    assert time.monotonic() - started < 30  # a hang or a socket leak fails here, not in CI's timeout


def test_nothing_started_by_one_call_outlives_it():
    reg, seen = {}, []

    async def leaks():
        async def forever():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                seen.append("cancelled")
                raise
        asyncio.get_running_loop().create_task(forever())
        await asyncio.sleep(0)
        return "ok"

    tool(leaks, name="leaks", undo=NOOP, registry=reg)
    assert run([{"id": "a", "tool": "leaks"}], registry=reg, log=lambda m: None) == {"a": "ok"}
    assert seen == ["cancelled"]


def test_each_thread_gets_its_own_loop():
    reg = {}
    tool(lambda: id(asyncio.get_running_loop()), name="loop_id", undo=NOOP, registry=reg)
    ids, both_alive = [[], []], threading.Barrier(2)

    def worker(mine):
        for _ in range(2):
            mine.append(run([{"id": "a", "tool": "loop_id"}], registry=reg, log=lambda m: None)["a"])
        both_alive.wait()  # neither loop is freed yet, so their ids can't coincide

    threads = [threading.Thread(target=worker, args=(mine,)) for mine in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    (a1, a2), (b1, b2) = ids
    if hasattr(asyncio, "Runner"):  # Python 3.11+: one loop per thread, reused
        assert a1 == a2 and b1 == b2 and a1 != b1


def test_ctrl_c_still_rolls_back():
    state, reg = [], {}
    tool(lambda: state.append("held") or "h", name="hold", undo=lambda r: state.remove("held"), registry=reg)

    def interrupt():
        raise KeyboardInterrupt

    tool(interrupt, name="stop", undo=NOOP, registry=reg)
    try:
        run([{"id": "h", "tool": "hold"}, {"id": "s", "tool": "stop", "after": ["h"]}],
            registry=reg, log=lambda m: None, confirm=APPROVE_ALL)
        raise AssertionError("KeyboardInterrupt must propagate")
    except KeyboardInterrupt:
        pass
    assert state == []
    assert run([{"id": "h", "tool": "hold"}], registry=reg, log=lambda m: None) == {"h": "h"}


def test_sync_calls_see_the_callers_context_vars():
    import contextvars
    var = contextvars.ContextVar("caller", default="unset")
    reg = {}
    tool(lambda: var.get(), name="read", undo=NOOP, registry=reg)
    run([{"id": "a", "tool": "read"}], registry=reg, log=lambda m: None)  # the loop exists now
    for value in ("first", "second"):
        var.set(value)
        assert run([{"id": "a", "tool": "read"}], registry=reg, log=lambda m: None) == {"a": value}


def test_loops_close_when_their_thread_ends_and_at_exit():
    import subprocess
    import sys
    from pathlib import Path

    script = (
        "import asyncio, gc, sys, threading\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})\n"
        "from rubicon import NOOP, run, tool\n"
        "reg = {}\n"
        "tool(lambda: asyncio.get_running_loop(), name='loop', undo=NOOP, registry=reg)\n"
        "main_loop = run([{'id': 'a', 'tool': 'loop'}], registry=reg, log=lambda m: None)['a']\n"
        "seen = []\n"
        "t = threading.Thread(target=lambda: seen.append(\n"
        "    run([{'id': 'a', 'tool': 'loop'}], registry=reg, log=lambda m: None)['a']))\n"
        "t.start(); t.join(); del t; gc.collect()\n"
        "assert sys.version_info < (3, 11) or seen[0].is_closed(), 'worker loop left open'\n"
        "assert sys.version_info < (3, 11) or not main_loop.is_closed()\n"  # 3.10: a loop per call
        "print('ok')\n")
    done = subprocess.run([sys.executable, "-W", "error::ResourceWarning", "-c", script],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr
    assert done.stderr == "", done.stderr  # nothing printed while the interpreter shuts down
