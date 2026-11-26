"""checkpoint(value): an undoable step records how to undo it before it finishes."""
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rubicon import (APPROVE_ALL, NOOP, FileJournal, PlanFailed, Transaction, arun, checkpoint,
                     dead_letters, recover, run, tool)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def world():
    state = {"orders": set(), "undone": []}
    reg = {}

    def cancel(order):
        state["undone"].append(order)
        state["orders"].discard(order)

    def place(item: str, fail_after: bool = False):
        order = f"order-{item}"
        state["orders"].add(order)  # the remote call succeeded
        checkpoint(order)           # from here on, the order can be cancelled
        if fail_after:
            raise ConnectionError("lost the connection while confirming the order")
        return order

    async def aplace(item: str):
        order = f"order-{item}"
        state["orders"].add(order)
        checkpoint(order)
        await asyncio.sleep(0)
        raise TimeoutError("confirmation timed out")

    tool(place, undo=cancel, registry=reg)
    tool(aplace, undo=cancel, registry=reg)
    tool(lambda: checkpoint("x"), name="perm", registry=reg)
    tool(lambda: None, name="send", registry=reg)
    return state, reg


def test_a_step_that_fails_after_its_checkpoint_is_undone(world):
    state, reg = world
    with pytest.raises(PlanFailed) as e:
        run([{"id": "o", "tool": "place", "args": {"item": "a", "fail_after": True}}], registry=reg,
            log=lambda m: None)
    assert e.value.step == "o" and e.value.undone == ["o"]
    assert state == {"orders": set(), "undone": ["order-a"]}


def test_async_steps_and_transactions_too(world):
    state, reg = world
    with pytest.raises(PlanFailed):
        asyncio.run(arun([{"id": "o", "tool": "aplace", "args": {"item": "b"}}], registry=reg, log=lambda m: None))
    assert state["orders"] == set() and state["undone"] == ["order-b"]

    tx = Transaction(registry=reg, log=lambda m: None)
    with pytest.raises(ConnectionError):
        tx.call("place", "c", fail_after=True)
    assert state["orders"] == {"order-c"}  # still in place: the agent may carry on
    tx.rollback()
    assert state["orders"] == set() and tx.undone == ["place-1"]


def test_a_finished_step_is_undone_with_its_result(world):
    state, reg = world
    with pytest.raises(PlanFailed):  # the gate is declined after the step finished
        run([{"id": "o", "tool": "place", "args": {"item": "d"}}, {"id": "s", "tool": "send"}],
            registry=reg, confirm=lambda st: False, log=lambda m: None)
    assert state["undone"] == ["order-d"]


def test_outside_a_step_or_in_a_permanent_one_it_refuses(world):
    state, reg = world
    with pytest.raises(RuntimeError, match="only be called"):
        checkpoint("x")
    with pytest.raises(PlanFailed) as e:
        run([{"id": "p", "tool": "perm"}], registry=reg, confirm=APPROVE_ALL, log=lambda m: None)
    assert isinstance(e.value.cause, RuntimeError)


def test_the_checkpoint_is_journaled(world, tmp_path):
    state, reg = world
    path = tmp_path / "j.jsonl"
    with pytest.raises(PlanFailed):
        run([{"id": "o", "tool": "place", "args": {"item": "e", "fail_after": True}}], registry=reg,
            journal=str(path), log=lambda m: None)
    events = [(r["event"], r.get("result")) for r in FileJournal(str(path)).records() if r.get("step") == "o"]
    assert ("checkpoint", "order-e") in events and ("undone", None) in events
    assert recover(str(path), registry=reg, log=lambda m: None) == []


CHILD = """
import os, sys
sys.path.insert(0, {root!r})
from rubicon import NOOP, checkpoint, run, tool
reg = {{}}
def place(item: str):
    open({orders!r}, "a").write(f"order-{{item}}\\n")
    {mark}
    os._exit(3)  # the process dies before the step finishes
tool(place, undo=NOOP, registry=reg)
run([{{"id": "o", "tool": "place", "args": {{"item": "f"}}}}], registry=reg, journal={journal!r}, log=lambda m: None)
"""


@pytest.mark.parametrize("with_checkpoint", [True, False])
def test_a_crash_mid_step_is_undone_only_with_a_checkpoint(tmp_path, with_checkpoint):
    orders, journal = tmp_path / "orders.txt", tmp_path / "j.jsonl"
    code = CHILD.format(root=str(ROOT), orders=str(orders), journal=str(journal),
                        mark='checkpoint("order-f")' if with_checkpoint else "pass")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 3

    cancelled = []
    reg = {}
    tool(lambda item: None, name="place", undo=lambda order: cancelled.append(order), registry=reg)
    report = recover(str(journal), registry=reg, log=lambda m: None)[0]
    if with_checkpoint:
        assert report["undone"] == ["o"] and report["uncertain"] == [] and cancelled == ["order-f"]
        assert not dead_letters(str(journal))
    else:  # the old window: the order exists, but nothing says how to undo it
        assert report["uncertain"] == ["o"] and cancelled == []
    assert os.path.exists(orders)


def test_a_failed_try_s_checkpoint_is_undone_when_a_retry_succeeds(monkeypatch, tmp_path):
    import rubicon
    monkeypatch.setattr(rubicon, "RETRY_DELAY", 0)
    made, undone, reg = [], [], {}

    def place():
        made.append(f"order{len(made) + 1}")
        checkpoint(made[-1])
        if len(made) == 1:
            raise ConnectionError("flaky")
        return made[-1]

    tool(place, undo=undone.append, retries=1, registry=reg)
    journal = FileJournal(tmp_path / "j.jsonl")
    with Transaction(reg, log=lambda m: None, journal=journal) as tx:
        assert tx.call("place") == "order2"
    assert undone == ["order1"]  # the orphan of the failed try; order2 is the committed result
    assert recover(journal, registry=reg) == [] and undone == ["order1"]


def test_failed_tries_are_all_undone_when_the_step_fails(monkeypatch):
    import rubicon
    monkeypatch.setattr(rubicon, "RETRY_DELAY", 0)
    made, undone, reg = [], [], {}

    def place():
        made.append(f"order{len(made) + 1}")
        checkpoint(made[-1])
        raise ConnectionError("down")

    tool(place, undo=undone.append, retries=2, registry=reg)
    with pytest.raises(ConnectionError):
        with Transaction(reg, log=lambda m: None) as tx:
            tx.call("place")
    assert sorted(undone) == ["order1", "order2", "order3"]
