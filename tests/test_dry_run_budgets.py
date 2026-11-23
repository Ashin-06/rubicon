"""dry_run(): what run() would do, with nothing executed. max_calls / max_seconds: budgets that roll back."""
import asyncio

import pytest

from rubicon import NOOP, BudgetExceeded, PlanFailed, Transaction, dry_run, run, tool


def registry(ran):
    reg = {}
    tool(lambda flight: ran.append(("hold", flight)) or {"flight": flight}, undo=NOOP, name="hold", registry=reg)

    def check_pay(amount):
        if amount > 500:
            raise ValueError("over the card limit")
        return f"charge {amount}"

    tool(lambda amount: ran.append(("pay", amount)), check=check_pay, name="pay", registry=reg)
    tool(lambda flight: ran.append(("book", flight)), check=lambda flight: f"book {flight}", name="book", registry=reg)
    return reg


PLAN = [{"id": "pay", "tool": "pay", "args": {"amount": 300}},
        {"id": "hold", "tool": "hold", "args": {"flight": "AI101"}},
        {"id": "book", "tool": "book", "args": {"flight": "$hold.flight"}}]


def test_dry_run_orders_marks_the_point_of_no_return_and_runs_known_checks_only():
    ran = []
    out = dry_run(PLAN, registry(ran))
    assert ran == []  # nothing executed, checks are dry runs
    assert [s["id"] for s in out["steps"]] == ["hold", "pay", "book"]
    assert out["point_of_no_return"] == "pay" and out["undoable"] == ["hold"] and out["permanent"] == ["pay", "book"]
    assert out["steps"][1]["check"] == {"preview": "charge 300"}
    assert out["steps"][2]["check"] == {"waits_for": ["hold"]}
    assert out["would_fail"] == []


def test_dry_run_reports_a_failing_check_and_rejects_an_invalid_plan():
    ran = []
    out = dry_run([{"id": "pay", "tool": "pay", "args": {"amount": 900}}], registry(ran))
    assert out["would_fail"] == ["pay"] and "card limit" in out["steps"][0]["check"]["error"] and ran == []
    with pytest.raises(ValueError):
        dry_run([{"id": "x", "tool": "nope"}], registry(ran))


def test_max_calls_rejects_a_longer_plan_before_anything_runs():
    ran = []
    with pytest.raises(BudgetExceeded):
        run(PLAN, registry(ran), confirm=lambda s: True, max_calls=2, log=lambda m: None)
    assert ran == []


def test_an_agent_over_its_call_budget_is_rolled_back():
    ran, undone = [], []
    reg = {}
    tool(lambda n: ran.append(n) or n, undo=lambda r: undone.append(r), name="step", registry=reg)
    tx = Transaction(reg, log=lambda m: None, max_calls=2)
    tx.call("step", 1)
    tx.call("step", 2)
    with pytest.raises(BudgetExceeded):
        tx.call("step", 3)
    assert ran == [1, 2] and undone == [2, 1]


def test_max_seconds_stops_before_the_point_of_no_return_and_rolls_back():
    ran, undone = [], []
    reg = {}

    async def slow(n):
        await asyncio.sleep(0.2)
        ran.append(n)
        return n

    tool(slow, undo=lambda r: undone.append(r), name="slow", registry=reg)
    tool(lambda: ran.append("sent"), name="send", registry=reg)
    plan = [{"id": "a", "tool": "slow", "args": {"n": 1}}, {"id": "s", "tool": "send"}]
    with pytest.raises(PlanFailed) as e:
        run(plan, reg, confirm=lambda s: True, max_seconds=0.05, log=lambda m: None)
    assert isinstance(e.value.cause, BudgetExceeded)
    assert "sent" not in ran and undone == [1]
