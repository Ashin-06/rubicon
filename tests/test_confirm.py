"""The confirmation gate fails closed: only a confirm that returns exactly True runs a permanent step."""
import asyncio

import pytest

from rubicon import APPROVE_ALL, NOOP, PlanFailed, Transaction, arun, run, tool
from rubicon_llm import aagent, agent


@pytest.fixture
def world():
    state = {"held": [], "sent": []}
    reg = {}
    tool(lambda flight: state["held"].append(flight) or flight, name="hold",
         undo=lambda r: state["held"].remove(r), registry=reg)
    tool(lambda to: state["sent"].append(to), name="send", registry=reg)  # permanent
    tool(lambda: "ok", name="look", undo=NOOP, registry=reg)
    return state, reg


PLAN = [{"id": "h", "tool": "hold", "args": {"flight": "AI101"}},
        {"id": "s", "tool": "send", "args": {"to": "me@example.com"}}]


def boom(steps):
    raise RuntimeError("reviewer unreachable")


async def aboom(steps):
    raise RuntimeError("reviewer unreachable")


async def ayes(steps):
    return "yes"


class Truthy:
    def __bool__(self):
        return True


REFUSING = {
    "missing": None,
    "raises": boom,
    "returns 1": lambda st: 1,
    "returns 'yes'": lambda st: "yes",
    "returns a list": lambda st: [True],
    "returns a truthy object": lambda st: Truthy(),
    "returns None": lambda st: None,
    "async raises": aboom,
    "async returns 'yes'": ayes,
    "lambda around async": lambda st: ayes(st),
}


def assert_refused(e, state, events=None):
    assert e.step == "confirm"
    assert e.committed == [] and e.not_run == ["s"] and e.undone == ["h"]
    assert state == {"held": [], "sent": []}
    if events is not None:
        assert [ev["event"] for ev in events if ev["event"].startswith("gate_")] == ["gate_declined"]


@pytest.mark.parametrize("confirm", REFUSING.values(), ids=REFUSING.keys())
def test_run_refuses(world, confirm):
    state, reg = world
    events = []
    with pytest.raises(PlanFailed) as e:
        run(PLAN, registry=reg, confirm=confirm, log=lambda m: None, on_event=events.append)
    assert_refused(e.value, state, events)


@pytest.mark.parametrize("confirm", REFUSING.values(), ids=REFUSING.keys())
def test_arun_refuses(world, confirm):
    state, reg = world
    with pytest.raises(PlanFailed) as e:
        asyncio.run(arun(PLAN, registry=reg, confirm=confirm, log=lambda m: None, parallel=True))
    assert_refused(e.value, state)


@pytest.mark.parametrize("confirm", REFUSING.values(), ids=REFUSING.keys())
def test_transaction_refuses(world, confirm):
    state, reg = world
    with pytest.raises(PlanFailed) as e:
        with Transaction(registry=reg, confirm=confirm, log=lambda m: None) as tx:
            tx.call("hold", "AI101")
            assert tx.call("send", "me@example.com").startswith("Scheduled")
    assert e.value.step == "confirm" and state == {"held": [], "sent": []}
    assert tx.undone == ["hold-1"]


@pytest.mark.parametrize("confirm", REFUSING.values(), ids=REFUSING.keys())
def test_async_transaction_refuses(world, confirm):
    state, reg = world

    async def go():
        async with Transaction(registry=reg, confirm=confirm, log=lambda m: None) as tx:
            await tx.acall("hold", "AI101")
            await tx.acall("send", "me@example.com")

    with pytest.raises(PlanFailed) as e:
        asyncio.run(go())
    assert e.value.step == "confirm" and state == {"held": [], "sent": []}


def test_agent_without_confirm_refuses(world):
    state, reg = world
    with pytest.raises(PlanFailed) as e:
        agent("move", planner=lambda goal, client, registry: PLAN, registry=reg, confirm=None,
              log=lambda m: None)
    assert_refused(e.value, state)


def test_async_agent_without_confirm_refuses(world):
    state, reg = world
    with pytest.raises(PlanFailed) as e:
        asyncio.run(aagent("move", planner=lambda goal, client, registry: PLAN, registry=reg,
                           confirm=None, log=lambda m: None))
    assert_refused(e.value, state)
    assert "confirm=APPROVE_ALL" in str(e.value.cause)


def test_missing_confirm_says_how_to_opt_in(world):
    state, reg = world
    with pytest.raises(PlanFailed) as e:
        run(PLAN, registry=reg, log=lambda m: None)
    assert "confirm=APPROVE_ALL" in str(e.value.cause)


def test_raising_confirm_keeps_the_cause(world):
    state, reg = world
    with pytest.raises(PlanFailed) as e:
        run(PLAN, registry=reg, confirm=boom, log=lambda m: None)
    assert "reviewer unreachable" in str(e.value.cause)
    assert isinstance(e.value.cause.__cause__, RuntimeError)


@pytest.mark.parametrize("confirm", [APPROVE_ALL, lambda st: True], ids=["APPROVE_ALL", "returns True"])
def test_true_approves(world, confirm):
    state, reg = world
    assert run(PLAN, registry=reg, confirm=confirm, log=lambda m: None) == {"h": "AI101", "s": None}
    assert state == {"held": ["AI101"], "sent": ["me@example.com"]}


def test_async_true_approves(world):
    state, reg = world

    async def yes(steps):
        return True

    asyncio.run(arun(PLAN, registry=reg, confirm=yes, log=lambda m: None))
    assert state["sent"] == ["me@example.com"]


def test_undoable_only_plan_needs_no_confirm(world):
    state, reg = world
    plan = [{"id": "h", "tool": "hold", "args": {"flight": "AI101"}}, {"id": "l", "tool": "look"}]
    assert run(plan, registry=reg, log=lambda m: None) == {"h": "AI101", "l": "ok"}
    assert asyncio.run(arun(plan, registry=reg, log=lambda m: None))["l"] == "ok"
    with Transaction(registry=reg, log=lambda m: None) as tx:
        tx.call("look")
    assert state["held"] == ["AI101", "AI101"]


def test_confirm_is_not_asked_without_permanent_steps(world):
    state, reg = world
    asked = []
    run([{"id": "l", "tool": "look"}], registry=reg, confirm=lambda st: asked.append(st) or False,
        log=lambda m: None)
    assert asked == []
