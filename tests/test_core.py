"""Core behaviour: ordering, checks, the confirmation gate, rollback, journals and recovery."""
import asyncio
import contextlib
import datetime
import enum
import functools
import inspect
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import typing
from dataclasses import dataclass

import pytest

import rubicon
from rubicon import (APPROVE_ALL, FileJournal, JournalBusy, NOOP, PlanFailed, SQLiteJournal, TOOLS, Transaction, UNSET, _LIVE, _MARK, _has_mark, _jsonable, _logger, _lookup, _pid_alive, _prepare, _replay, _resolve, _schema, acknowledge, arun, compact, dead_letters, order, recover, run, tool, tools_for_llm)

@pytest.fixture(autouse=True, scope="module")
def _scratch(tmp_path_factory):
    """Every mkdtemp() in this module lands in one scratch folder that pytest cleans up."""
    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("scratch"))
    yield
    tempfile.tempdir = previous


def crashed(tx):
    """Simulate the process dying: the transaction stops being live in this process."""
    _LIVE.discard(tx.id)

@contextlib.contextmanager
def captured_warnings():
    """Collect warnings logged by the "rubicon" logger, whatever logging setup is active."""
    messages = []

    class Collect(logging.Handler):
        def emit(self, record):
            messages.append(record.getMessage())
    handler = Collect(logging.WARNING)
    _logger.addHandler(handler)
    try:
        yield messages
    finally:
        _logger.removeHandler(handler)

quiet = lambda m: None

def expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


# ---------------------------------------------------------------- shared worlds

db = {"bookings": {"OLD1"}, "holds": [], "seats": {"AI101": 1}, "outbox": [], "smtp_down": False}


@tool(undo=NOOP)

def search(route: str) -> list:
    """Find flights with free seats."""
    return [f for f, n in db["seats"].items() if n > 0]


def release(hold):
    db["holds"].remove(hold["flight"])
    db["seats"][hold["flight"]] += 1


@tool(undo=release)

def book_flight(flight: str, note: str = "") -> dict:
    """Hold a seat (can be released)."""
    if db["seats"].get(flight, 0) < 1:
        raise RuntimeError(f"no seats on {flight}")
    db["seats"][flight] -= 1
    db["holds"].append(flight)
    return {"flight": flight}


def must_exist(ref):
    if ref not in db["bookings"]:
        raise RuntimeError(f"no booking {ref}")
    return f"will cancel {ref}"


@tool(check=must_exist)

def cancel_booking(ref: str):
    """Cancel a booking. PERMANENT."""
    db["bookings"].remove(ref)


def valid_to(to, body):
    if "@" not in to:
        raise RuntimeError(f"bad address {to}")


@tool(check=valid_to)

def send_email(to: str, body: str):
    """Send an email. PERMANENT."""
    if db["smtp_down"]:
        raise ConnectionError("SMTP timeout")
    db["outbox"].append((to, body))


def plan(flight="AI101", to="me@example.com"):
    return [
        {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}},
        {"id": "find", "tool": "search", "args": {"route": "BLR-DEL"}},
        {"id": "hold", "tool": "book_flight", "args": {"flight": flight}},
        {"id": "mail", "tool": "send_email", "args": {"to": to, "body": "held $hold.flight"}},
    ]


def reset():
    db.update(bookings={"OLD1"}, holds=[], seats={"AI101": 1}, outbox=[], smtp_down=False)


def clean():
    return db["bookings"] == {"OLD1"} and db["seats"]["AI101"] == 1 and not db["holds"]


reg, log2 = {}, []


async def aundo(r):
    log2.append(("undo", r))


async def acheck(x):
    if x < 0:
        raise ValueError("negative")

live = {"now": 0, "peak": 0}


@tool(undo=aundo, registry=reg)

async def reserve(x: int) -> int:
    """Async undoable tool."""
    live["now"] += 1
    live["peak"] = max(live["peak"], live["now"])
    try:
        await asyncio.sleep(0.05)
        if x == 99:
            raise RuntimeError("boom")
        return x * 2
    finally:
        live["now"] -= 1


@tool(check=acheck, registry=reg)

async def charge(x: int):
    """Async permanent tool."""
    log2.append(("charge", x))


odd, got = {}, []


@tool(undo=NOOP, registry=odd)

def varargs(x, *items, **kw):
    got.append((x, items, kw))


def cc_check(to, cc):
    got.append(("check", to, cc))


@tool(check=cc_check, registry=odd)

def mail(to, cc="none"):
    pass
tool(len, name="length", undo=NOOP, registry=odd)


outage = {"down": True, "calls": []}


def undo_b(r):
    outage["calls"].append(r)
    if outage["down"]:
        raise ConnectionError("service down")


@tool(undo=lambda r: outage["calls"].append(r), registry=odd)

def part_a():
    return "a"


@tool(undo=undo_b, undo_retries=0, registry=odd)

def part_b():
    return "b"


# a real crash (the process dies) in the middle of a step
here = os.path.dirname(os.path.abspath(rubicon.__file__))


def crash_run(dies_in, journal_path):
    child = (
        "import os, sys\n"
        f"sys.path.insert(0, {here!r})\n"
        "from rubicon import APPROVE_ALL, tool, run, NOOP\n"
        "reg = {}\n"
        "tool(lambda flight: {'flight': flight}, name='hold', undo=NOOP, registry=reg)\n"
        f"tool(lambda: os._exit(3) if {dies_in!r} == 'probe' else 'x', name='probe', undo=NOOP, registry=reg)\n"
        f"tool(lambda ref: os._exit(3) if {dies_in!r} == 'cancel' else 'ok', name='cancel', registry=reg)\n"
        "run([{'id': 'h', 'tool': 'hold', 'args': {'flight': 'AI101'}},\n"
        "     {'id': 'p', 'tool': 'probe', 'after': ['h']},\n"
        "     {'id': 'c', 'tool': 'cancel', 'args': {'ref': 'OLD1'}}],\n"
        f"    registry=reg, journal={journal_path!r}, log=lambda m: None, confirm=APPROVE_ALL)\n")
    assert subprocess.run([sys.executable, "-c", child]).returncode == 3

released = []
after_crash = {}
tool(lambda flight: None, name="hold", undo=lambda r: released.append(r), registry=after_crash)
tool(lambda: None, name="probe", undo=NOOP, registry=after_crash)
tool(lambda ref: None, name="cancel", registry=after_crash)


owed = {"down": True, "undone": []}


def release_owed(r):
    if owed["down"]:
        raise ConnectionError("down")
    owed["undone"].append(r)


@tool(undo=release_owed, undo_retries=0, registry=odd)

def take():
    return "seat"


@tool(registry=odd)

def break_commit():
    raise RuntimeError("permanent step failed")


@tool(undo=NOOP, registry=odd)

def make_hold():
    return {"flight": "AI101"}


@tool(registry=odd)

def do_cancel():
    return "ok"


@tool(registry=odd)

def notify(flight: str, note: str = "", channel: str = "email"):
    return "sent"


def crash_with(journal):
    tx = Transaction(registry=odd, journal=journal, log=quiet, confirm=APPROVE_ALL)
    tx.call("part_a")
    tx.call("make_hold")
    crashed(tx)
    return tx.id


class MemoryJournal:  # what a Redis/Postgres-backed journal needs to provide
    def __init__(self):
        self.rows = []

    def append(self, records):
        self.rows.extend(json.loads(json.dumps(r, default=repr)) for r in records)

    def records(self):
        return list(self.rows)


class OtherProcess:
    """The journal as another worker sees it: this process is a live process it can't look
    inside (its pid is swapped for the parent's), and nothing is in that worker's _LIVE."""

    def __init__(self, path):
        self.j = FileJournal(path)

    def append(self, records):
        self.j.append(records)

    def records(self):
        return [{**r, "pid": os.getppid()} if "pid" in r else r for r in self.j.records()]

    def drop(self, tx_ids):
        self.j.drop(tx_ids)


@contextlib.contextmanager

def from_other_process(path):
    hidden = set(_LIVE)
    _LIVE.clear()
    try:
        yield OtherProcess(path)
    finally:
        _LIVE.update(hidden)

stage = {}
stage_undone = []
tool(lambda: "a", name="sa", undo=lambda r: stage_undone.append("a"), registry=stage)
tool(lambda: "b", name="sb", undo=lambda r: stage_undone.append("b"), registry=stage)
tool(lambda: 1 / 0, name="sbroken", undo=NOOP, registry=stage)
tool(lambda: "c", name="sc", undo=lambda r: 1 / 0, undo_retries=0, registry=stage)
tool(lambda: "p", name="sp", registry=stage)


world6 = set()
reg6 = {}
tool(lambda k: world6.add(k) or k, name="put", undo=lambda r: world6.discard(r), registry=reg6)
tool(lambda: "p", name="perm", registry=reg6)

@pytest.fixture(autouse=True)
def _fresh_world():
    """Every test starts from the same world, whatever ran before it."""
    reset()
    log2.clear()
    live.update(now=0, peak=0)
    got.clear()
    outage.update(down=True, calls=[])
    owed.update(down=True, undone=[])
    released.clear()
    stage_undone.clear()
    world6.clear()



def test_rebooking_scenario():
    # --- rebooking scenario: the classic agent mistake is cancelling first ---
    assert [s["id"] for s in order(plan())] == ["find", "hold", "cancel", "mail"]


def test_references_resolve_text_nested_and_paths():
    # references: text, nested, field paths, "$50" and unknown names left alone
    res = {"hold": {"flight": "AI101", "legs": [{"no": 7}]}, "file": {"name": "report"}}
    assert _resolve({"b": "held $hold.flight for $50", "w": "$hold", "n": [{"x": "$hold.legs.0.no"}],
                     "c": "on $hold.flight.", "d": "$hold.legs.-1", "f": "Saved to $file.name.txt",
                     "g": "$file.name.txt", "u": "$nobody here"}, res) == \
        {"b": "held AI101 for $50", "w": res["hold"], "n": [{"x": 7}], "c": "on AI101.",
         "d": {"no": 7}, "f": "Saved to report.txt", "g": "report.txt", "u": "$nobody here"}
    for typo in ("Flight: $hold.flihgt", "$hold.legs.5", "$hold.legs.x"):
        expect(ValueError, _resolve, {"a": typo}, res)  # typos fail loudly, never send a repr
    assert [s["id"] for s in order([
        {"id": "m", "tool": "send_email", "args": {"to": "a@b", "body": ["ref", "$h", "$50"]}},
        {"id": "h", "tool": "book_flight", "args": {"flight": "AI101"}}])] == ["h", "m"]


def test_malformed_plans_and_mismatched_arguments_are_rejected():
    # malformed plans and mismatched arguments are rejected before anything runs
    good = {"id": "a", "tool": "search", "args": {"route": "x"}}
    for bad in ["nope", [1], [{**good, "id": "a b"}], [{**good, "id": 3}], [{**good, "id": "1"}],
                [good, good], [{**good, "tool": "nope"}], [{**good, "tool": ["x"]}], [{**good, "args": [1]}],
                [{**good, "after": "b"}], [{**good, "after": ["ghost"]}], [{**good, "after": ["a"]}],
                [{**good, "args": {"route": "x", "extra": 1}}], [{**good, "args": {}}],
                [{"id": "c", "tool": "cancel_booking", "args": {"booking_id": "OLD1"}}]]:
        expect(ValueError, order, bad)
    # "$1" in text is money, not a reference (ids can't be purely numeric)
    assert run([{"id": "h", "tool": "book_flight", "args": {"flight": "AI101", "note": "Refunded $1"}}],
               log=quiet, confirm=APPROVE_ALL) == {"h": {"flight": "AI101"}}
    reset()


def test_no_seats():
    # 1. no seats -> old booking survives, nothing committed
    e = expect(PlanFailed, run, plan(flight="XX9"), log=quiet, confirm=APPROVE_ALL)
    assert e.step == "hold" and e.committed == [] and e.not_run == ["cancel", "mail"] and clean()


def test_dry_run_blocks():
    # 2. dry-run blocks -> seat released, nothing permanent happened
    e = expect(PlanFailed, run, plan(to="not-an-email"), log=quiet, confirm=APPROVE_ALL)
    assert e.step == "mail" and e.undone == ["hold", "find"] and clean()


def test_human_declines():
    # 3. human declines; confirm sees only permanent steps, resolved args and previews
    seen = []
    e = expect(PlanFailed, run, plan(), confirm=lambda st: seen.extend(st) or False, log=quiet)
    assert e.step == "confirm" and [s["id"] for s in seen] == ["cancel", "mail"] and clean()
    assert seen[1]["args"]["body"] == "held AI101" and seen[0]["preview"] == "will cancel OLD1"


def test_permanent_failure_after_commit_keeps_the_hold():
    # 4. CRITICAL: a permanent step fails after another committed -> the hold it relies on stays
    db["smtp_down"] = True
    e = expect(PlanFailed, run, plan(), log=quiet, confirm=APPROVE_ALL)
    assert e.step == "mail" and e.committed == ["cancel"] and e.undone == [] and e.not_run == ["mail"]
    assert db["bookings"] == set() and db["holds"] == ["AI101"]  # user still has the new flight
    reset()


def test_ctrl_c_mid_run_still_unwinds_then_propagates():
    # 5. Ctrl+C mid-run still unwinds, then propagates
    @tool(undo=NOOP)
    def interrupt():
        raise KeyboardInterrupt
    expect(KeyboardInterrupt, run, plan()[1:3] + [{"id": "x", "tool": "interrupt", "after": ["hold"]}], log=quiet, confirm=APPROVE_ALL)
    assert clean()


def test_an_undo_that_keeps_failing_is_retried_reported_and():
    # 6. an undo that keeps failing is retried, reported, and unwinding continues
    calls = []
    @tool(undo=lambda r: calls.append(1) or 1 / 0, undo_retries=1)
    def fragile():
        return "f"
    e = expect(PlanFailed, run, [{"id": "f", "tool": "fragile"}] + plan(to="bad")[1:], log=quiet, confirm=APPROVE_ALL)
    assert e.undo_failed == ["f"] and "hold" in e.undone and len(calls) == 2 and clean()


def test_dry_runs_wait_for_after_dependencies_not_just_references():
    # 7. dry runs wait for `after` dependencies, not just $references
    made = set()

    @tool
    def create(name: str):
        made.add(name)

    def exists(name):
        if name not in made:
            raise RuntimeError(f"{name} does not exist yet")

    @tool(check=exists)
    def publish(name: str):
        pass
    run([{"id": "c", "tool": "create", "args": {"name": "doc"}},
         {"id": "p", "tool": "publish", "args": {"name": "doc"}, "after": ["c"]}], log=quiet, confirm=APPROVE_ALL)

    # 8. a bad reference in a permanent step is blamed on that step
    e = expect(PlanFailed, run, [plan()[2], {"id": "mail", "tool": "send_email",
                                             "args": {"to": "a@b", "body": "$hold.nope"}}], log=quiet, confirm=APPROVE_ALL)
    assert e.step == "mail" and clean()


def test_happy_path_with_structured_events():
    # 9. happy path, with structured events
    events = []
    r = run(plan(), log=quiet, on_event=events.append, confirm=APPROVE_ALL)
    assert db["bookings"] == set() and db["outbox"] == [("me@example.com", "held AI101")]
    assert r["hold"] == {"flight": "AI101"}
    kinds = [ev["event"] for ev in events]
    assert kinds[:2] == ["step_started", "step_done"] and "gate_confirmed" in kinds and kinds[-1] == "step_committed"
    assert all("duration_ms" in ev for ev in events if ev["event"] in ("step_done", "step_committed"))
    reset()


def test_undo_receives_the_call_s_arguments_when_it_asks_for_them():
    # --- undo receives the call's arguments when it asks for them ---
    files = {}

    @tool(undo=lambda result, args: files.pop(args["path"]))
    def put_file(path: str, content: str) -> str:
        files[path] = content
        return "ok"
    e = expect(PlanFailed, run, [{"id": "p", "tool": "put_file", "args": {"path": "/a", "content": "x"}},
                                 {"id": "m", "tool": "send_email", "args": {"to": "bad", "body": "x"}}], log=quiet, confirm=APPROVE_ALL)
    assert files == {} and e.undone == ["p"]


def test_retries_and_timeouts():
    # --- retries and timeouts ---
    attempts = []

    @tool(undo=NOOP, retries=2)
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("blip")
        return "ok"
    assert run([{"id": "f", "tool": "flaky"}], log=quiet, confirm=APPROVE_ALL) == {"f": "ok"} and len(attempts) == 3

    @tool(undo=NOOP, timeout=0.05)
    async def slow():
        await asyncio.sleep(1)
    expect(PlanFailed, run, [{"id": "s", "tool": "slow"}], log=quiet, confirm=APPROVE_ALL)


def test_async_tools_async_undo_check_confirm_isolated_registry():
    # --- async tools, async undo/check/confirm, isolated registry, parallel waves ---
    async def aconfirm(steps):
        return True

    p = [{"id": "c", "tool": "charge", "args": {"x": "$r"}}, {"id": "r", "tool": "reserve", "args": {"x": 3}}]
    assert asyncio.run(arun(p, registry=reg, confirm=aconfirm, log=quiet)) == {"r": 6, "c": None}
    assert log2 == [("charge", 6)] and "reserve" not in TOOLS  # registries don't leak
    log2.clear()
    p[1]["args"]["x"] = -1  # charge(-2) fails its async check -> async undo runs
    expect(PlanFailed, lambda: asyncio.run(arun(p, registry=reg, log=quiet, confirm=APPROVE_ALL)))
    assert log2 == [("undo", -2)]

    wide = [{"id": f"r{i}", "tool": "reserve", "args": {"x": i}} for i in range(4)]
    live["peak"] = 0
    asyncio.run(arun(wide, registry=reg, log=quiet, confirm=APPROVE_ALL))
    assert live["peak"] == 1  # sequential by default
    assert asyncio.run(arun(wide, registry=reg, log=quiet, parallel=True, confirm=APPROVE_ALL)) == {f"r{i}": i * 2 for i in range(4)}
    assert live["peak"] == 4  # all four ran at the same time (counted, not timed)
    log2.clear()
    e = expect(PlanFailed, lambda: asyncio.run(arun(wide + [{"id": "bad", "tool": "reserve", "args": {"x": 99}}],
                                                    registry=reg, log=quiet, parallel=True, confirm=APPROVE_ALL)))
    assert e.step == "bad" and sorted(e.undone) == ["r0", "r1", "r2", "r3"] and len(log2) == 4

    async def inside_loop():
        return expect(RuntimeError, run, p, registry=reg, confirm=APPROVE_ALL)
    assert "await arun" in str(asyncio.run(inside_loop()))


def test_transaction_guards_an_agent_loop():
    # --- Transaction: guard an agent loop that calls tools one by one ---
    with Transaction(log=quiet, confirm=APPROVE_ALL) as tx:
        hold = tx.wrap("book_flight")
        cancel = tx.wrap("cancel_booking")
        assert hold.__name__ == "book_flight" and "flight" in str(inspect.signature(hold))
        assert hold("AI101") == {"flight": "AI101"}  # positional args work; runs now
        assert "will cancel OLD1" in cancel(ref="OLD1") and db["bookings"] == {"OLD1"}  # deferred, previewed
        expect(RuntimeError, cancel, ref="NOPE")  # dry run fails now, agent can react
    assert db["bookings"] == set() and db["holds"] == ["AI101"]  # committed on clean exit
    reset()

    tx = Transaction(log=quiet, confirm=APPROVE_ALL)
    try:
        with tx:
            tx.call("book_flight", flight="AI101")
            tx.call("cancel_booking", ref="OLD1")
            raise RuntimeError("agent crashed")
    except RuntimeError:
        pass  # expected: the simulated agent crash; the rollback is checked below
    assert clean() and tx.undone == ["book_flight-1"] and tx.undo_failed == []

    tx = Transaction(confirm=lambda st: False, log=quiet)
    tx.call("book_flight", flight="AI101")
    tx.call("cancel_booking", ref="OLD1")
    e = expect(PlanFailed, tx.commit)
    assert e.step == "confirm" and clean()


def test_transaction_failing_permanent_step_after_commit():
    # Transaction: a failing permanent step after a commit keeps the earlier state
    db["smtp_down"] = True
    tx = Transaction(log=quiet, confirm=APPROVE_ALL)
    tx.call("book_flight", flight="AI101")
    tx.call("cancel_booking", ref="OLD1")
    tx.call("send_email", to="a@b", body="x")
    e = expect(PlanFailed, tx.commit)
    assert e.committed == ["cancel_booking-2"] and e.undone == [] and db["holds"] == ["AI101"]
    reset()


def test_wrap_passes_args_kwargs_and_defaults_to_checks():
    # wrap(): *args, **kwargs, positional-only, defaults reaching the check
    with Transaction(registry=odd, log=quiet, confirm=APPROVE_ALL) as tx:
        tx.wrap("varargs")(1, 2, 3, y=4)
        assert tx.wrap("length")([1, 2]) == 2
        tx.wrap("mail")("a@example.com")
    assert got[:2] == [(1, (2, 3), {"y": 4}), ("check", "a@example.com", "none")]
    assert run([{"id": "n", "tool": "length", "args": {"obj": [1, 2, 3]}}], registry=odd, log=quiet, confirm=APPROVE_ALL) == {"n": 3}


def test_async_framework_runs_sync_tools_on_its_loop():
    # async framework: sync tools run on its loop thread; async checks get an async wrapper
    async def async_agent():
        async with Transaction(registry=reg, log=quiet, confirm=APPROVE_ALL) as atx:
            assert inspect.iscoroutinefunction(atx.wrap("charge"))
            assert await atx.wrap("reserve")(5) == 10
            assert "Scheduled" in await atx.acall("charge", x=1)
        sync_tx = Transaction(log=quiet, confirm=APPROVE_ALL)
        assert sync_tx.wrap("search")("BLR-DEL") == ["AI101"]
        assert sync_tx.call("book_flight", "AI101") == {"flight": "AI101"}
        await sync_tx.arollback()
    log2.clear()
    asyncio.run(async_agent())
    assert log2 == [("charge", 1)] and clean()


def test_rollback_keeps_unwinding_through_an_interruption_then_re():
    # rollback keeps unwinding through an interruption, then re-raises it
    order_undone = []

    @tool(undo=lambda r: order_undone.append(r), registry=odd)
    def step_a(n: int):
        return n

    def stop(r):
        raise KeyboardInterrupt

    @tool(undo=stop, undo_retries=0, registry=odd)
    def step_b():
        return "b"
    tx = Transaction(registry=odd, log=quiet, confirm=APPROVE_ALL)
    tx.call("step_a", 1)
    tx.call("step_b")
    tx.call("step_a", 2)
    expect(KeyboardInterrupt, tx.rollback)
    assert order_undone == [2, 1] and tx.undo_failed == ["step_b-2"]


def test_journal_and_crash_recovery():
    # --- journal and crash recovery ---
    path = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(journal=path, log=quiet, confirm=APPROVE_ALL)
    tx.call("book_flight", flight="AI101")
    tx.call("search", route="x")
    crashed(tx)  # the process dies here: no rollback, no commit
    assert db["holds"] == ["AI101"]
    report = recover(path, log=quiet)
    assert report[0]["undone"] == ["search-2", "book_flight-1"] and clean()
    assert recover(path, log=quiet) == []  # already finished
    run(plan(), journal=path, log=quiet, confirm=APPROVE_ALL)  # a finished run leaves nothing to recover
    assert recover(path, log=quiet) == []
    reset()


def test_cancelling_a_parallel_run_stops_the_wave_first_so_no():
    # cancelling a parallel run stops the wave first, so no step finishes after the rollback
    trail = []

    @tool(undo=lambda r: trail.append(("undo", r)), registry=reg)
    async def wave_step(n: int):
        await asyncio.sleep(0 if n == 0 else 0.3)
        trail.append(("done", n))
        return n

    async def cancel_midway():
        task = asyncio.ensure_future(arun([{"id": f"w{i}", "tool": "wave_step", "args": {"n": i}} for i in range(3)],
                                          registry=reg, log=quiet, parallel=True, confirm=APPROVE_ALL))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.4)  # an orphaned step would finish in this time
    asyncio.run(cancel_midway())
    assert trail == [("done", 0), ("undo", 0)]


def test_a_check_only_receives_the_parameters_it_asks_for():
    # a check only receives the parameters it asks for; UNSET parameters are left out entirely
    checked = []

    def ref_only(ref):
        checked.append(ref)

    @tool(check=ref_only, registry=odd)
    def cancel_with_reason(ref: str, reason: str = "user_request"):
        pass

    @tool(undo=NOOP, registry=odd)
    def remote(a: str, b: str = UNSET):
        got.append(("remote", a, b))
        return {"a": a}

    shown = []
    run([{"id": "c", "tool": "cancel_with_reason", "args": {"ref": "R1"}},
         {"id": "r", "tool": "remote", "args": {"a": "x"}}],
        registry=odd, confirm=lambda st: shown.extend(st) or True, log=quiet)
    assert checked == ["R1"] and shown[0]["args"] == {"ref": "R1", "reason": "user_request"}
    assert got[-1] == ("remote", "x", UNSET) and not UNSET
    tx = Transaction(registry=odd, log=quiet, confirm=APPROVE_ALL)
    tx.call("remote", "y")
    assert tx.done[-1]["args"] == {"a": "y"}  # not shown, not journaled


def test_journal_marks_only_unserializable_values():
    # journal: only the values that can't be serialized are marked
    assert _jsonable({"a": 1, "o": object, 2: [object]}) == {
        "a": 1, "o": {_MARK: repr(object)}, "2": [{_MARK: repr(object)}]}


def test_recovery_journals_each_undo_so_it_can_be_re_run_without():
    # recovery journals each undo, so it can be re-run without undoing anything twice
    path2 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=odd, journal=path2, log=quiet, confirm=APPROVE_ALL)
    tx.call("part_a")
    tx.call("part_b")
    crashed(tx)
    first = recover(path2, registry=odd, log=quiet)
    assert first[0]["undone"] == ["part_a-1"] and first[0]["undo_failed"] == ["part_b-2"]
    outage["down"] = False
    second = recover(path2, registry=odd, log=quiet)
    assert second[0]["undone"] == ["part_b-2"] and second[0]["undo_failed"] == []
    assert outage["calls"] == ["b", "a", "b"]  # part_a undone exactly once
    assert recover(path2, registry=odd, log=quiet) == []


def test_an_in_process_rollback_whose_undo_fails_leaves_the():
    # an in-process rollback whose undo fails leaves the journal open for recover() to retry
    path2 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    outage.update(down=True, calls=[])
    tx = Transaction(registry=odd, journal=path2, log=quiet, confirm=APPROVE_ALL)
    tx.call("part_b")
    tx.rollback()
    assert tx.undo_failed == ["part_b-1"]
    outage["down"] = False
    assert recover(path2, registry=odd, log=quiet)[0]["undone"] == ["part_b-1"]


def test_recover_also_covers_a_transaction_that_committed_in():
    # recover() also covers a Transaction that committed in stages and crashed in a later stage
    path3 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    outage.update(down=False, calls=[])
    tx = Transaction(registry=odd, journal=path3, log=quiet, confirm=APPROVE_ALL)
    tx.call("part_a")
    tx.commit()  # stage 1 done: part_a is final
    tx.call("part_b")  # stage 2...
    crashed(tx)
    staged = recover(path3, registry=odd, log=quiet)
    assert staged[0]["undone"] == ["part_b-2"] and outage["calls"] == ["b"]


def test_a_step_that_cancels_itself_inside_a_parallel_wave_is_a():
    # a step that cancels itself inside a parallel wave is a failure, not a finished step
    @tool(undo=NOOP, registry=reg)
    async def self_cancel():
        raise asyncio.CancelledError()
    e = expect(PlanFailed, lambda: asyncio.run(arun([{"id": "sc", "tool": "self_cancel"},
                                                    {"id": "r", "tool": "reserve", "args": {"x": 1}}],
                                                   registry=reg, log=quiet, parallel=True, confirm=APPROVE_ALL)))
    assert e.step == "sc" and "cancelled" in str(e.cause)


def test_a_sync_tool_s_retries_also_work_through_tx_call_inside_a():
    # a sync tool's retries also work through tx.call() inside a running event loop
    tries = []

    @tool(undo=NOOP, retries=2, registry=odd)
    def blip():
        tries.append(1)
        if len(tries) < 2:
            raise ConnectionError("blip")
        return "ok"

    async def in_framework():
        return Transaction(registry=odd, log=quiet, confirm=APPROVE_ALL).call("blip")
    assert asyncio.run(in_framework()) == "ok" and len(tries) == 2


def test_a_broken_on_event_or_log_callback_never_changes_what():
    # a broken on_event or log callback never changes what runs or gets undone
    def broken(*a):
        raise RuntimeError("logging backend down")
    with captured_warnings() as warned:
        assert run(plan(), on_event=broken, log=broken, confirm=APPROVE_ALL) == {"find": ["AI101"], "hold": {"flight": "AI101"},
                                                            "cancel": None, "mail": None}
    assert any("on_event callback failed" in w for w in warned) and any("log callback failed" in w for w in warned)
    reset()


def test_dies_during_the_permanent_cancel():
    # dies during the permanent cancel: it may have gone through, so the hold must be kept
    path4 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("cancel", path4)
    rep = recover(path4, registry=after_crash, log=quiet)[0]
    assert rep["needs_attention"] == ["c"] and rep["kept"] == ["h", "p"] and released == []
    assert recover(path4, registry=after_crash, log=quiet)[0]["needs_attention"] == ["c"]  # still flagged


def test_dies_during_an_undoable_step():
    # dies during an undoable step: completed steps are undone, the half-run one is reported
    path5 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("probe", path5)
    rep = recover(path5, registry=after_crash, log=quiet)[0]
    assert rep["undone"] == ["h"] and rep["uncertain"] == ["p"] and released == [{"flight": "AI101"}]
    assert recover(path5, registry=after_crash, log=quiet) == []


def test_a_failed_undo_stays_owed_to_recover():
    # a failed undo stays owed to recover(): through a second rollback and through later commits
    path6 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    try:
        with Transaction(registry=odd, journal=path6, log=quiet, confirm=APPROVE_ALL) as tx:
            tx.call("take")
            tx.call("break_commit")
            tx.commit()  # fails, its rollback can't undo "take", then __exit__ rolls back again
    except PlanFailed:
        pass  # expected: what matters is that the owed undo survives, checked below
    owed["down"] = False
    assert recover(path6, registry=odd, log=quiet)[0]["undone"] == ["take-1"] and owed["undone"] == ["seat"]

    owed.update(down=True, undone=[])
    path7 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=odd, journal=path7, log=quiet, confirm=APPROVE_ALL)
    tx.call("take")
    tx.rollback()  # stage 1: undo fails
    tx.call("part_a")
    tx.commit()  # stage 2 commits; part_a is final, "take" is still owed
    owed["down"] = False
    outage.update(calls=[])
    rep = recover(path7, registry=odd, log=quiet)[0]
    assert rep["undone"] == ["take-1"] and owed["undone"] == ["seat"] and outage["calls"] == []
    assert recover(path7, registry=odd, log=quiet) == []


def test_a_sync_tool_s_retry_backoff_inside_arun_doesn_t_block():
    # a sync tool's retry backoff inside arun() doesn't block the event loop
    loop_state = {"ticks": 0, "done": False, "seen": []}

    @tool(undo=NOOP, retries=1, registry=odd)
    def slow_retry():
        loop_state["seen"].append(loop_state["ticks"])
        if len(loop_state["seen"]) < 2:
            raise ConnectionError("blip")
        loop_state["done"] = True

    @tool(undo=NOOP, registry=odd)
    async def ticker():
        while not loop_state["done"]:
            loop_state["ticks"] += 1
            await asyncio.sleep(0.01)
    asyncio.run(arun([{"id": "s", "tool": "slow_retry"}, {"id": "t", "tool": "ticker"}],
                     registry=odd, log=quiet, parallel=True, confirm=APPROVE_ALL))
    assert loop_state["seen"][1] - loop_state["seen"][0] >= 3  # the ticker kept running during backoff


def test_the_confirmation_resolves_what_it_can_even_for_a_step():
    # the confirmation resolves what it can, even for a step that waits on another permanent step
    shown2 = []

    run([{"id": "h", "tool": "make_hold"}, {"id": "c", "tool": "do_cancel"},
         {"id": "n", "tool": "notify", "args": {"flight": "$h.flight", "note": "after $c"}, "after": ["c"]}],
        registry=odd, confirm=lambda st: shown2.extend(st) or True, log=quiet)
    assert shown2[1]["args"] == {"flight": "AI101", "note": "after $c", "channel": "email"}


def test_sqlite_custom_and_concurrent_journals():
    # --- journals: SQLite, custom stores, concurrent writers, recovery lock ---
    outage.update(down=False, calls=[])
    db_path = os.path.join(tempfile.mkdtemp(), "rubicon.db")
    tx_id = crash_with(SQLiteJournal(db_path))
    rep = recover(SQLiteJournal(db_path), registry=odd, log=quiet)
    assert rep[0]["tx"] == tx_id and rep[0]["undone"] == ["make_hold-2", "part_a-1"] and outage["calls"] == ["a"]
    assert recover(SQLiteJournal(db_path), registry=odd, log=quiet) == []

    mem = MemoryJournal()
    outage.update(calls=[])
    crash_with(mem)
    assert recover(mem, registry=odd, log=quiet)[0]["undone"] == ["make_hold-2", "part_a-1"]
    expect(TypeError, Transaction, journal=object())
    expect(TypeError, compact, mem)  # no drop(): says so instead of failing halfway

    for journal in (FileJournal(os.path.join(tempfile.mkdtemp(), "j.jsonl")), SQLiteJournal(db_path)):
        with journal.lock():
            expect(JournalBusy, recover, journal, registry=odd, log=quiet)  # one recovery at a time
        assert recover(journal, registry=odd, log=quiet) == []


def test_several_processes_writing_to_one_journal_at_once_lose():
    # several processes writing to one journal at once lose and corrupt nothing
    def hammer(journal_expr, n_procs=4, n_records=40):
        child = (f"import sys; sys.path.insert(0, {here!r})\n"
                 "from rubicon import FileJournal, SQLiteJournal\n"
                 f"j = {journal_expr}\n"
                 f"[j.append([{{'tx': sys.argv[1], 'event': 'begin', 'n': i, 'pad': 'x' * 2000}}]) for i in range({n_records})]\n")
        procs = [subprocess.Popen([sys.executable, "-c", child, f"p{i}"]) for i in range(n_procs)]
        assert all(p.wait() == 0 for p in procs)

    file_path = os.path.join(tempfile.mkdtemp(), "j.jsonl")
    hammer(f"FileJournal({file_path!r})")
    rows = FileJournal(file_path).records()
    assert len(rows) == 160 and sorted({r["tx"] for r in rows}) == ["p0", "p1", "p2", "p3"]
    db_path2 = os.path.join(tempfile.mkdtemp(), "j.db")
    SQLiteJournal(db_path2)
    hammer(f"SQLiteJournal({db_path2!r})")
    assert len(SQLiteJournal(db_path2).records()) == 160


def test_dead_letters():
    # --- dead letters ---
    owed.update(down=True, undone=[])
    path8 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=odd, journal=path8, log=quiet, confirm=APPROVE_ALL)
    tx.call("take")
    tx.rollback()  # undo fails: a dead letter, owed to recover()
    letters = dead_letters(path8)
    assert [(d["step"], d["kind"]) for d in letters] == [("take-1", "undo_failed")]
    notified = []
    recover(path8, registry=odd, log=quiet, on_dead_letter=notified.append)  # still down
    assert [d["step"] for d in notified] == ["take-1"]  # recorded by the rollback, reported here
    assert set(notified[0]) == {"tx", "step", "kind", "reason", "time"}
    recover(path8, registry=odd, log=quiet, on_dead_letter=notified.append)
    assert len(notified) == 1  # once, not on every run
    acknowledge(path8, tx.id, "take-1", note="released by hand")
    assert dead_letters(path8) == [] and recover(path8, registry=odd, log=quiet) == []

    notified.clear()
    path9 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("cancel", path9)  # crash during a permanent step
    released.clear()
    for _ in range(2):
        rep = recover(path9, registry=after_crash, log=quiet, on_dead_letter=notified.append)[0]
        assert rep["needs_attention"] == ["c"] and released == []
    assert [(n["step"], n["kind"]) for n in notified] == [("c", "in_flight_permanent")]  # once
    assert [d["step"] for d in dead_letters(path9)] == ["c"]
    acknowledge(path9, rep["tx"], note="checked: the cancel went through")
    assert dead_letters(path9) == [] and recover(path9, registry=after_crash, log=quiet) == [] and released == []

    path10 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("probe", path10)  # crash during an undoable step
    rep = recover(path10, registry=after_crash, log=quiet)[0]
    assert [d["kind"] for d in dead_letters(path10)] == ["uncertain"]  # survives the finished tx
    acknowledge(path10, rep["tx"], "p")
    assert dead_letters(path10) == []


def test_crash_during_a_permanent_step():
    # crash during a permanent step; a person says "it didn't happen, undo the rest"
    path11 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("cancel", path11)
    released.clear()
    tx_id = recover(path11, registry=after_crash, log=quiet)[0]["tx"]
    acknowledge(path11, tx_id, "c")
    assert recover(path11, registry=after_crash, log=quiet)[0]["undone"] == ["p", "h"] and len(released) == 1


def test_compact():
    # --- compact ---
    for journal in (FileJournal(os.path.join(tempfile.mkdtemp(), "j.jsonl")),
                    SQLiteJournal(os.path.join(tempfile.mkdtemp(), "j.db"))):
        for _ in range(3):
            run([{"id": "a", "tool": "part_a"}], registry=odd, journal=journal, log=quiet, confirm=APPROVE_ALL)  # settled
        owed.update(down=True)
        stuck = Transaction(registry=odd, journal=journal, log=quiet, confirm=APPROVE_ALL)
        stuck.call("take")
        stuck.rollback()  # owes an undo: must survive compaction
        open_tx = crash_with(journal)  # unfinished: must survive compaction
        assert compact(journal) == 3 and compact(journal) == 0
        assert {r["tx"] for r in journal.records()} == {stuck.id, open_tx}
        owed.update(down=False)
        outage.update(calls=[])
        reps = {r["tx"]: r for r in recover(journal, registry=odd, log=quiet)}
        assert reps[stuck.id]["undone"] == ["take-1"] and reps[open_tx]["undone"] == ["make_hold-2", "part_a-1"]
        assert compact(journal) == 2 and journal.records() == []


def test_recover_never_touches_a_transaction_that_may_still_be():
    # --- recover() never touches a transaction that may still be running ---
    path12 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    outage.update(down=False, calls=[])
    live_tx = Transaction(registry=odd, journal=path12, log=quiet, confirm=APPROVE_ALL)
    live_tx.call("part_a")  # still running in this process
    assert recover(path12, registry=odd, log=quiet) == [] and outage["calls"] == []
    crashed(live_tx)
    assert recover(path12, registry=odd, log=quiet)[0]["undone"] == ["part_a-1"]


def test_in_another_process_that_is_alive_then_killed():
    # ...in another process that is alive, then killed
    path13 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    child = (f"import sys, time; sys.path.insert(0, {here!r})\n"
             "from rubicon import APPROVE_ALL, Transaction, tool, NOOP\n"
             "reg = {}\n"
             "tool(lambda: 'a', name='part_a', undo=NOOP, registry=reg)\n"
             f"tx = Transaction(registry=reg, journal={path13!r}, confirm=APPROVE_ALL)\n"
             "tx.call('part_a')\n"
             "print('ready', flush=True)\n"
             "time.sleep(60)\n")
    worker = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True)
    try:
        assert worker.stdout.readline().strip() == "ready"
        outage.update(calls=[])
        assert recover(path13, registry=odd, log=quiet) == [] and outage["calls"] == []
    finally:
        worker.kill()
        worker.wait()
        worker.stdout.close()
    assert recover(path13, registry=odd, log=quiet)[0]["undone"] == ["part_a-1"] and outage["calls"] == ["a"]


def test_on_another_machine_can_t_be_checked_from_here_unless():
    # ...on another machine (can't be checked from here) unless told otherwise
    elsewhere = MemoryJournal()
    elsewhere.append([{"tx": "t9", "event": "begin", "pid": 1, "host": "another-machine"},
                      {"tx": "t9", "event": "started", "step": "part_a-1", "tool": "part_a", "permanent": False},
                      {"tx": "t9", "event": "done", "step": "part_a-1", "tool": "part_a", "args": {}, "result": "a"}])
    assert recover(elsewhere, registry=odd, log=quiet) == []
    assert recover(elsewhere, registry=odd, log=quiet, include_live=True)[0]["undone"] == ["part_a-1"]


def test_but_a_settled_transaction_that_owes_an_undo_is_retried():
    # ...but a settled transaction that owes an undo is retried even while its worker lives on
    path18 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    child = (f"import sys, time; sys.path.insert(0, {here!r})\n"
             "from rubicon import APPROVE_ALL, Transaction, tool\n"
             "reg = {}\n"
             "def release(r):\n"
             "    raise ConnectionError('service down')\n"
             "tool(lambda: 'seat', name='take', undo=release, undo_retries=0, registry=reg)\n"
             f"tx = Transaction(registry=reg, journal={path18!r}, log=lambda m: None, confirm=APPROVE_ALL)\n"
             "tx.call('take')\n"
             "tx.rollback()\n"
             "print('ready', flush=True)\n"
             "time.sleep(60)\n")
    worker = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, text=True)
    try:
        assert worker.stdout.readline().strip() == "ready"
        owed.update(down=False, undone=[])
        assert recover(path18, registry=odd, log=quiet)[0]["undone"] == ["take-1"] and owed["undone"] == ["seat"]
    finally:
        worker.kill()
        worker.wait()
        worker.stdout.close()


def test_a_crashed_transaction_whose_last_open_items_are():
    # a crashed transaction whose last open items are acknowledged can be compacted
    path19 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("probe", path19)
    hold_down = {}
    tool(lambda flight: None, name="hold", undo=lambda r: 1 / 0, undo_retries=0, registry=hold_down)
    tool(lambda: None, name="probe", undo=NOOP, registry=hold_down)
    tool(lambda ref: None, name="cancel", registry=hold_down)
    rep = recover(path19, registry=hold_down, log=quiet)[0]  # "h" undo fails, "p" uncertain
    acknowledge(path19, rep["tx"], "h")
    acknowledge(path19, rep["tx"], "p")
    assert compact(path19) == 1 and FileJournal(path19).records() == []


def test_a_standalone_call_whose_first_step_fails_leaves_nothing():
    # a standalone call whose first step fails leaves nothing marked live
    lone = Transaction(journal=os.path.join(tempfile.mkdtemp(), "j.jsonl"), log=quiet, confirm=APPROVE_ALL)
    expect(RuntimeError, lone.call, "book_flight", flight="XX9")
    assert lone.id not in _LIVE


def test_liveness_of_running_killed_and_zombie_processes():
    # liveness: running, killed, and (Linux) zombie processes
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    assert _pid_alive(sleeper.pid)
    sleeper.kill()
    if sys.platform.startswith("linux"):
        time.sleep(0.5)  # exited but not reaped yet: a zombie
        assert not _pid_alive(sleeper.pid)
    sleeper.wait()
    assert not _pid_alive(sleeper.pid) and _pid_alive(os.getpid()) and not _pid_alive(0)


def test_maintenance_while_a_staged_transaction_runs_never():
    # --- maintenance while a staged transaction runs never touches the running stage ---

    # compact() between stages, then a live stage 2
    path20 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path20, log=quiet, confirm=APPROVE_ALL)
    tx.call("sa")
    tx.commit()
    assert compact(path20) == 1  # stage 1 removed, "begin" included
    tx.call("sb")
    assert [r["event"] for r in FileJournal(path20).records()][0] == "begin"  # stage 2 has its own
    with from_other_process(path20) as other:
        assert recover(other, registry=stage, log=quiet) == [] and stage_undone == []
    assert recover(path20, registry=stage, log=quiet) == []  # and in this process
    tx.rollback()


def test_acknowledging_stage_1_s_failed_undo_while_stage_2_runs():
    # acknowledging stage 1's failed undo while stage 2 runs
    path21 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path21, log=quiet, confirm=APPROVE_ALL)
    tx.call("sc")
    tx.rollback()  # undo fails: owed
    tx.call("sb")  # stage 2, running
    acknowledge(path21, tx.id, "sc-1")
    stage_undone.clear()
    with from_other_process(path21) as other:
        assert recover(other, registry=stage, log=quiet) == [] and stage_undone == []
    tx.rollback()


def test_a_stage_whose_rollback_owed_an_undo_acknowledged_and():
    # a stage whose rollback owed an undo, acknowledged and compacted by another worker,
    # followed by a new stage (found by the randomised tests, seed 3079722)
    path24 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path24, log=quiet, confirm=APPROVE_ALL)
    tx.call("sc")
    tx.rollback()  # undo fails: "settled", owed
    with from_other_process(path24) as other:
        acknowledge(other, tx.id, "sc-1", note="fixed by hand")
        assert compact(other) == 1  # nothing left: removed, "begin" included
    tx.call("sb")  # the next stage
    stage_undone.clear()
    with from_other_process(path24) as other:
        assert recover(other, registry=stage, log=quiet) == [] and stage_undone == []
    tx.rollback()


def test_a_lone_failed_first_step_settles_itself_so_it_can_be():
    # a lone failed first step settles itself, so it can be compacted
    path22 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    lone = Transaction(registry=stage, journal=path22, log=quiet, confirm=APPROVE_ALL)
    expect(ZeroDivisionError, lone.call, "sbroken")
    assert compact(path22) == 1


def test_but_not_while_a_permanent_step_is_pending():
    # ...but not while a permanent step is pending: the journal doesn't know about it yet
    path23 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path23, log=quiet, confirm=APPROVE_ALL)
    tx.call("sp")  # scheduled, not journaled until commit
    expect(ZeroDivisionError, tx.call, "sbroken")
    with from_other_process(path23) as other:
        compact(other)  # must not remove the running transaction's "begin"
    tx.call("sb")
    stage_undone.clear()
    with from_other_process(path23) as other:
        assert recover(other, registry=stage, log=quiet) == [] and stage_undone == []
    tx.rollback()


def test_nor_when_an_earlier_stage_s_owed_undo_is_acknowledged():
    # ...nor when an earlier stage's owed undo is acknowledged and compacted: the records left
    # must still say who runs the transaction
    path25 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path25, log=quiet, confirm=APPROVE_ALL)
    tx.call("sc")
    tx.rollback()  # owed
    tx.call("sp")  # pending
    expect(ZeroDivisionError, tx.call, "sbroken")
    with from_other_process(path25) as other:
        acknowledge(other, tx.id, "sc-1")
        compact(other)
    tx.call("sb")
    stage_undone.clear()
    with from_other_process(path25) as other:
        assert recover(other, registry=stage, log=quiet) == [] and stage_undone == []
    tx.rollback()


def test_a_crash_inside_a_permanent_step_does_not_hold_back_an():
    # a crash inside a permanent step does not hold back an earlier stage's owed undo, and
    # acknowledging the whole transaction afterwards does not wipe it
    path26 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    FileJournal(path26).append([
        {"tx": "T", "event": "begin"},
        {"tx": "T", "event": "started", "step": "sc-1", "tool": "sc", "permanent": False},
        {"tx": "T", "event": "done", "step": "sc-1", "tool": "sc", "args": {}, "result": "c"},
        {"tx": "T", "event": "undo_failed", "step": "sc-1", "error": "down"},
        {"tx": "T", "event": "settled", "status": "rolled_back"},
        {"tx": "T", "event": "started", "step": "sp-2", "tool": "sp", "permanent": True}])
    owed_reg, owed_calls = {}, []
    tool(lambda: "c", name="sc", undo=owed_calls.append, registry=owed_reg)
    tool(lambda: "p", name="sp", registry=owed_reg)
    first = recover(path26, registry=owed_reg, log=quiet)
    assert owed_calls == ["c"] and first[0]["undone"] == ["sc-1"], first  # retried, not held
    assert first[0]["needs_attention"] == ["sp-2"]
    assert [d["kind"] for d in dead_letters(path26)] == ["in_flight_permanent"]
    path27 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    FileJournal(path27).append(FileJournal(path26).records()[:6])  # the same, with the undo still down
    down_reg = {}
    tool(lambda: "c", name="sc", undo=lambda r: 1 / 0, undo_retries=0, registry=down_reg)
    tool(lambda: "p", name="sp", registry=down_reg)
    recover(path27, registry=down_reg, log=quiet)
    assert sorted(d["kind"] for d in dead_letters(path27)) == ["in_flight_permanent", "undo_failed"]
    acknowledge(path27, "T", note="the permanent step went through")
    assert [d["kind"] for d in dead_letters(path27)] == ["undo_failed"]  # the owed undo survives
    owed_calls.clear()
    recover(path27, registry=owed_reg, log=quiet)
    assert owed_calls == ["c"] and not dead_letters(path27)


def test_a_failed_permanent_step_after_an_earlier_commit_leaves():
    # a failed permanent step after an earlier commit leaves one settle, from the plan's end
    path28 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    seq = {}
    tool(lambda: "1", name="q1", registry=seq)
    tool(lambda: 1 / 0, name="q2", registry=seq)
    expect(PlanFailed, run, [{"id": "a", "tool": "q1"}, {"id": "b", "tool": "q2", "after": ["a"]}],
           registry=seq, journal=path28, log=quiet, confirm=APPROVE_ALL)
    settles = [r.get("status") for r in FileJournal(path28).records() if r["event"] in ("finished", "settled")]
    assert "step_failed" not in settles, settles


def test_recover_retrying_an_owed_undo_while_the_worker_starts():
    # recover() retrying an owed undo while the worker starts its next stage: the new stage must
    # not be closed by a record recover() writes afterwards
    in_undo, release = threading.Event(), threading.Event()
    racing, undo_down = {}, [True]

    def slow_undo(result):
        if undo_down[0]:
            raise RuntimeError("down")
        in_undo.set()
        release.wait(10)

    tool(lambda: "c", name="sc", undo=slow_undo, undo_retries=0, registry=racing)
    tool(lambda: "b", name="sb", undo=NOOP, registry=racing)
    path29 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=racing, journal=path29, log=quiet, confirm=APPROVE_ALL)
    tx.call("sc")
    tx.rollback()  # owed
    undo_down[0] = False
    worker = threading.Thread(target=recover, args=(path29,), kwargs={"registry": racing, "log": quiet})
    worker.start()
    assert in_undo.wait(10)
    tx.call("sb")  # stage 2 starts while recover() is undoing stage 1
    release.set()
    worker.join()
    state = _replay(FileJournal(path29).records())[0][tx.id]
    assert "sb-2" in state["done"] and not state["owed"] and not state["finished"], state
    tx.rollback()
    assert _replay(FileJournal(path29).records())[0][tx.id]["finished"]


def test_an_undo_journaled_for_a_step_whose_done_was_never():
    # an undo journaled for a step whose "done" was never written closes that step too
    for outcome in ("undone", "undo_failed"):
        torn = _replay([{"tx": "T", "event": "started", "step": "s", "tool": "sa", "permanent": False},
                        {"tx": "T", "event": outcome, "step": "s"}])[0]["T"]
        assert not torn["open"], torn


def test_a_read_only_step_whose_result_can_t_be_stored_needs():
    # a read-only step whose result can't be stored needs nothing undone after a crash
    dated = {}
    tool(lambda: datetime.date(2026, 1, 1), name="today", undo=NOOP, registry=dated)
    path30 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=dated, journal=path30, log=quiet, confirm=APPROVE_ALL)
    tx.call("today")
    crashed(tx)
    rep = recover(path30, registry=dated, log=quiet)
    assert rep[0]["undone"] == ["today-1"] and not rep[0]["undo_failed"] and not dead_letters(path30), rep


def test_positional_only_parameters_given_by_name_bind_in_order():
    # positional-only parameters given by name bind in order, defaults filling any gap
    def two(a, b=2, /):
        return a, b

    def both_default(a=1, b=2, /):
        return a, b

    expect(TypeError, _prepare, two, (), {"b": 99})  # a is required and missing
    assert _prepare(both_default, (), {"b": 99})[2] == {"a": 1, "b": 99}
    assert _prepare(two, (), {"a": 5})[2] == {"a": 5, "b": 2}
    gap = {}

    @tool(undo=NOOP, registry=gap)
    def unset_gap(a, b=UNSET, c="d", /):  # b can't be left out while c is given
        return a

    expect(ValueError, order, [{"id": "s", "tool": "unset_gap", "args": {"a": "1", "c": "x"}}], gap)
    assert run([{"id": "s", "tool": "unset_gap", "args": {"a": "1"}}], registry=gap, log=quiet, confirm=APPROVE_ALL) == {"s": "1"}


def test_list_indexes():
    # list indexes: anything but plain ASCII digits is a bad reference, not a crash
    for part in ("--0", chr(0xB2), "-"):  # chr(0xB2) is a superscript two
        expect(LookupError, _lookup, {"s": [1, 2]}, "s", "." + part)
    assert _lookup({"s": [1, 2]}, "s", ".-1") == (2, "")


def test_settle_records():
    # --- settle records: one per settle, none for an empty transaction ---
    path14 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    with Transaction(registry=odd, journal=path14, log=quiet, confirm=APPROVE_ALL) as tx:
        tx.call("part_a")
        tx.commit()  # explicit commit inside the block; leaving it must not write again
    events = [r["event"] for r in FileJournal(path14).records()]
    assert events.count("finished") == 1 and events[0] == "begin"
    with Transaction(registry=odd, journal=path14, log=quiet, confirm=APPROVE_ALL):
        pass
    assert len(FileJournal(path14).records()) == len(events)  # nothing written


def test_acknowledge_rejects_ids_the_journal_doesn_t_have():
    # --- acknowledge() rejects ids the journal doesn't have ---
    path14 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    with Transaction(registry=odd, journal=path14, log=quiet, confirm=APPROVE_ALL) as tx:
        tx.call("part_a")
    expect(ValueError, acknowledge, path14, "no-such-tx")
    expect(ValueError, acknowledge, path14, FileJournal(path14).records()[0]["tx"], "no-such-step")


def test_compact_removes_a_transaction_once_its_last_dead_letter():
    # --- compact() removes a transaction once its last dead letter is acknowledged ---
    path15 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    owed.update(down=True)
    tx = Transaction(registry=odd, journal=path15, log=quiet, confirm=APPROVE_ALL)
    tx.call("take")
    tx.rollback()  # undo fails: owed, a dead letter
    acknowledge(path15, tx.id, "take-1", note="released by hand")
    assert compact(path15) == 1 and FileJournal(path15).records() == []
    owed.update(down=False)


def test_an_uncertain_step_is_reported_once_even_while_another():
    # --- an uncertain step is reported once, even while another undo keeps failing ---
    down_registry = {}
    tool(lambda flight: None, name="hold", undo=lambda r: 1 / 0, undo_retries=0, registry=down_registry)
    tool(lambda: None, name="probe", undo=NOOP, registry=down_registry)
    tool(lambda ref: None, name="cancel", registry=down_registry)
    path16 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("probe", path16)
    with captured_warnings() as warned:
        first = recover(path16, registry=down_registry)[0]
        second = recover(path16, registry=down_registry)[0]
    assert first["uncertain"] == ["p"] and second["uncertain"] == [] and second["undo_failed"] == ["h"]
    assert sum("started but never reported" in w for w in warned) == 1


def test_compaction_and_reads_while_other_processes_write_windows():
    # --- compaction and reads while other processes write (Windows: no PermissionError) ---
    path17 = os.path.join(tempfile.mkdtemp(), "j.jsonl")
    writer = (f"import sys; sys.path.insert(0, {here!r})\n"
              "from rubicon import FileJournal\n"
              f"j = FileJournal({path17!r})\n"
              "for i in range(60):\n"
              "    j.append([{'tx': sys.argv[1] + str(i), 'event': 'begin'},\n"
              "              {'tx': sys.argv[1] + str(i), 'event': 'finished'}])\n")
    writers = [subprocess.Popen([sys.executable, "-c", writer, f"w{i}-"]) for i in range(3)]
    removed = 0
    while any(w.poll() is None for w in writers):
        removed += compact(path17)  # rewrites the file while writers append
        FileJournal(path17).records()
    assert all(w.wait() == 0 for w in writers)
    removed += compact(path17)
    assert removed == 180 and FileJournal(path17).records() == []  # every record seen, none lost


def test_tool_descriptions_for_models():
    # --- tool descriptions for models ---
    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    @tool(undo=NOOP, registry=odd)
    def typed(a: str | None, b: typing.Literal["x", "y"], c: list[str], d: Color, e: typing.Optional[int] = None,
              f: dict = None, g: tuple[int, ...] = (), h: tuple[int, str] = (0, "")):
        """Typed tool."""
    props = {t["name"]: t for t in tools_for_llm(odd)}["typed"]["parameters"]["properties"]
    assert props == {"a": {"type": "string"}, "b": {"type": "string", "enum": ["x", "y"]},
                     "c": {"type": "array", "items": {"type": "string"}},
                     "d": {"type": "string", "enum": ["red", "blue"]}, "e": {"type": "integer"},
                     "f": {"type": "object"}, "g": {"type": "array", "items": {"type": "integer"}},
                     "h": {"type": "array"}}
    spec = {t["name"]: t for t in tools_for_llm()}
    assert spec["cancel_booking"]["reversible"] is False and spec["search"]["reversible"] is True
    assert spec["book_flight"]["parameters"] == {"type": "object", "required": ["flight"],
        "properties": {"flight": {"type": "string"}, "note": {"type": "string"}}}
    expect(ValueError, tool, functools.partial(print), registry={})
    tool(functools.partial(print), name="say", registry=odd)
    assert "say" in [t["name"] for t in tools_for_llm(odd)]


def test_findings_from_the_0_8_6_audits():
    # ---------------------------------------------------------------- independent audits, 0.8.6

    # compact() never removes records written after it read the journal (a new stage)
    for kind in (FileJournal, SQLiteJournal):
        class Racy(kind):
            hook = None

            def drop(self, tx_ids, upto=None):
                if self.hook:  # the live worker writes while compact() is between read and drop
                    hook, self.hook = self.hook, None
                    hook()
                return super().drop(tx_ids, upto)

        racy = Racy(os.path.join(tempfile.mkdtemp(), "rubicon.j"))
        tx = Transaction(registry=reg6, journal=racy, log=quiet, confirm=APPROVE_ALL)
        tx.call("put", "a")
        tx.commit()
        racy.hook = lambda: tx.call("put", "b")
        assert compact(racy) == 1
        assert [r["event"] for r in racy.records() if r["tx"] == tx.id] == ["begin", "started", "done"]
        crashed(tx)
        recover(racy, registry=reg6, log=quiet)
        assert world6 == {"a"}, (kind, world6)
        world6.clear()


def test_a_line_torn_by_a_power_cut_is_ended_so_the_next_record():
    # a line torn by a power cut is ended, so the next record is not lost with it
    path31 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    FileJournal(path31).append([{"tx": "T", "event": "begin"}])
    with open(path31, "a", encoding="utf-8") as f:
        f.write('{"tx": "other", "event": "done", "resu')
    FileJournal(path31).append([{"tx": "T", "event": "started", "step": "p-1", "tool": "perm", "permanent": True}])
    assert [r["event"] for r in FileJournal(path31).records()] == ["begin", "started"]


def test_a_transaction_running_in_this_process_is_never_touched():
    # a transaction running in this process is never touched by a recover() here, including
    # the moments its first and last records are being written
    class Watching(SQLiteJournal):
        when, found = None, None

        def append(self, records):
            if self.when and records[-1]["event"] == self.when:
                self.when = None
                worker = threading.Thread(target=lambda: setattr(
                    self, "found", recover(self, registry=reg6, log=quiet)))
                worker.start()
                worker.join()
            super().append(records)

    for moment in ("started", "finished"):
        watched = Watching(os.path.join(tempfile.mkdtemp(), "rubicon.db"))
        tx = Transaction(registry=reg6, journal=watched, log=quiet, confirm=APPROVE_ALL)
        watched.when = moment if moment == "started" else None
        tx.call("put", "seat")
        watched.when = moment if moment == "finished" else None
        tx.commit()
        assert watched.found == [] and world6 == {"seat"} and not dead_letters(watched), (moment, watched.found)
        world6.clear()


def test_acknowledging_a_whole_transaction_never_settles_a_stage():
    # acknowledging a whole transaction never settles a stage it started afterwards
    flaky_undo_down = {"down": True}

    def flaky_undo(r):
        if flaky_undo_down["down"]:
            raise RuntimeError("down")
        world6.discard(r)

    reg7 = {}
    tool(lambda k: world6.add(k) or k, name="put", undo=flaky_undo, undo_retries=0, registry=reg7)
    path32 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=reg7, journal=path32, log=quiet, confirm=APPROVE_ALL)
    tx.call("put", "a")
    tx.rollback()  # owed
    tx.call("put", "b")  # the next stage
    acknowledge(path32, tx.id, note="looked at it")
    crashed(tx)
    flaky_undo_down["down"] = False
    recover(path32, registry=reg7, log=quiet)
    assert world6 == set(), world6
    world6.clear()


def test_a_journal_that_fails_mid_rollback_never_hides_the_real():
    # a journal that fails mid-rollback never hides the real error or a step's true outcome
    class Failing(FileJournal):
        fail = set()

        def append(self, records):
            if records[-1]["event"] in self.fail:
                self.fail.discard(records[-1]["event"])
                raise OSError("journal unavailable")
            super().append(records)

    tool(lambda: 1 / 0, name="broken", registry=reg6)
    failing = Failing(os.path.join(tempfile.mkdtemp(), "rubicon.jsonl"))
    failing.fail = {"undone", "step_failed"}
    with captured_warnings():
        e = expect(PlanFailed, run, [{"id": "a", "tool": "put", "args": {"k": "a"}}, {"id": "x", "tool": "broken"}],
                   registry=reg6, journal=failing, log=quiet, confirm=APPROVE_ALL)
    assert isinstance(e.cause, ZeroDivisionError) and e.undone == ["a"] and e.undo_failed == [], e
    assert world6 == set()


def test_results_that_can_t_be_journaled_are_marked_never_a():
    # results that can't be journaled are marked, never a failure; without a journal, not even copied
    def cyclic():
        node = {"children": []}
        node["children"].append({"parent": node})
        return node

    tool(cyclic, name="cyclic", undo=NOOP, registry=reg6)
    tool(lambda: functools.reduce(lambda acc, _: [acc], range(2000), []), name="deep", undo=NOOP, registry=reg6)
    assert set(run([{"id": "c", "tool": "cyclic"}, {"id": "d", "tool": "deep"}], registry=reg6, log=quiet, confirm=APPROVE_ALL)) == {"c", "d"}
    path33 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    run([{"id": "c", "tool": "cyclic"}, {"id": "d", "tool": "deep"}], registry=reg6, journal=path33, log=quiet, confirm=APPROVE_ALL)
    done33 = {r["step"]: r["result"] for r in FileJournal(path33).records() if r["event"] == "done"}
    assert _has_mark(done33["c"]) and "d" in done33  # the cycle is marked; deep JSON is kept or marked
    # (whether 2000 levels serialize depends on the Python version's stack), never a failure


def test_sync_transaction_code_inside_a_running_event_loop():
    # sync Transaction code inside a running event loop: commit and rollback work like call()
    async def sync_code_in_loop(fail):
        with Transaction(registry=reg6, log=quiet, confirm=APPROVE_ALL) as tx:
            tx.call("put", "inside")
            if fail:
                raise KeyError("agent error")

    expect(KeyError, lambda: asyncio.run(sync_code_in_loop(True)))
    assert world6 == set()
    asyncio.run(sync_code_in_loop(False))
    assert world6 == {"inside"}
    world6.clear()


def test_an_async_tool_called_from_sync_code_inside_a_loop_is():
    # an async tool called from sync code inside a loop is refused before any of it runs
    async def put_async(k):
        world6.add(k)
        await asyncio.sleep(0)

    tool(put_async, name="put_async", undo=NOOP, registry=reg6)

    async def drive_async():
        Transaction(registry=reg6, log=quiet, confirm=APPROVE_ALL).call("put_async", "half")

    expect(RuntimeError, lambda: asyncio.run(drive_async()))
    assert world6 == set()


def test_a_step_is_marked_running_before_started_is_written_so_no():
    # a step is marked running before "started" is written, so no settle can miss it
    class Peeking(FileJournal):
        seen = None

        def append(self, records):
            if records[-1]["event"] == "started":
                self.seen = set(peek_tx._running)
            super().append(records)

    peeking = Peeking(os.path.join(tempfile.mkdtemp(), "rubicon.jsonl"))
    peek_tx = Transaction(registry=reg6, journal=peeking, log=quiet, confirm=APPROVE_ALL)
    peek_tx.call("put", "x")
    assert peeking.seen == {"put-1"}, peeking.seen
    peek_tx.rollback()


def test_a_step_that_finishes_on_another_thread_after_a_rollback():
    # a step that finishes on another thread after a rollback undoes itself
    gate = threading.Event()

    def slow_put(k):
        gate.wait(10)
        world6.add(k)
        return k

    tool(slow_put, name="slow_put", undo=lambda r: world6.discard(r), registry=reg6)
    tx = Transaction(registry=reg6, log=quiet, confirm=APPROVE_ALL)
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(expect(RuntimeError, tx.call, "slow_put", "late")))
    worker.start()
    time.sleep(0.05)
    tx.rollback()
    gate.set()
    worker.join()
    assert world6 == set() and "slow_put-1" in tx.undone and outcome, (world6, tx.undone)


def test_a_process_that_exited_with_code_259_still_active_s_value():
    # a process that exited with code 259 (STILL_ACTIVE's value on Windows) is not alive
    child = subprocess.Popen([sys.executable, "-c", "import os; os._exit(259)"])
    child.wait()
    assert not _pid_alive(child.pid)  # the Popen object still holds its handle


def test_plan_inputs_are_validated():
    # plan inputs: names must be strings, *args after a left-out argument, big plans stay fast
    expect(ValueError, order, [{"id": "s", "tool": "put", "args": {1: 2, "k": 3}}], reg6)

    @tool(undo=NOOP, registry=reg6)
    def star(a, b=UNSET, *rest):
        return a

    assert run([{"id": "s", "tool": "star", "args": {"a": 1}}], registry=reg6, log=quiet, confirm=APPROVE_ALL) == {"s": 1}
    started = time.perf_counter()
    order([{"id": f"s{i}", "tool": "put", "args": {"k": i}} for i in range(5000)], reg6)
    assert time.perf_counter() - started < 3, "order() must not be quadratic"


def test_a_path_into_an_object_that_isn_t_a_dict_or_list_is_a_bad():
    # a path into an object that isn't a dict or list is a bad reference, not dumped text
    @dataclass
    class Seat:
        seat: str

    tool(lambda: Seat("12A"), name="seat_obj", undo=NOOP, registry=reg6)
    tool(lambda body: body, name="say", undo=NOOP, registry=reg6)
    e = expect(PlanFailed, run, [{"id": "h", "tool": "seat_obj"},
                                 {"id": "m", "tool": "say", "args": {"body": "Seat $h.seat"}}], registry=reg6, log=quiet, confirm=APPROVE_ALL)
    assert "not a dict or list" in str(e.cause)


def test_an_undo_with_an_option_of_its_own_gets_just_the_result():
    # an undo with an option of its own gets just the result
    got = []
    tool(lambda k: k, name="opt", undo=lambda r, force=False: got.append((r, force)), registry=reg6)
    t = Transaction(registry=reg6, log=quiet, confirm=APPROVE_ALL)
    t.call("opt", "z")
    t.rollback()
    assert got == [("z", False)]


def test_tool_descriptions_see_the_hints_of_partials_and_callable():
    # tool descriptions see the hints of partials and callable objects
    class Typed:
        def __call__(self, x: int):
            return x

    assert _schema(Typed())["properties"] == {"x": {"type": "integer"}}

    def two_args(a: int, b: str = "b"):
        return a

    assert _schema(functools.partial(two_args, b="c"))["properties"] == {"a": {"type": "integer"}, "b": {"type": "string"}}
