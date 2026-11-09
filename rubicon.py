"""rubicon: reversibility-aware execution for AI agent plans.

Every tool declares whether it can be undone. Given a plan (e.g. written by an LLM),
rubicon:
  1. reorders it so undoable steps run first and permanent ones run last (respecting dependencies),
  2. runs every permanent step's `check` (dry-run) BEFORE the point of no return,
  3. asks `confirm` once for the whole commit phase,
  4. on any failure, undoes completed undoable steps in reverse order.

Plan format (JSON-friendly, so a model can emit it):
  [{"id": "hold", "tool": "book_flight", "args": {"flight": "AI101"}},
   {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}, "after": ["hold"]}]
An arg value "$<id>" is replaced by that step's result and adds an implicit dependency.
"""
import inspect
import re
from dataclasses import dataclass
from graphlib import TopologicalSorter
from typing import Callable

NOOP = lambda result: None  # undo for read-only tools


@dataclass
class Tool:
    fn: Callable
    undo: Callable | None  # None = permanent
    check: Callable | None  # dry-run for permanent tools; raise to block

    @property
    def reversible(self):
        return self.undo is not None


TOOLS: dict[str, Tool] = {}


def tool(undo=None, check=None):
    """Register a tool. undo(result) reverses it; leave undo=None for permanent actions."""
    def wrap(fn):
        TOOLS[fn.__name__] = Tool(fn, undo, check)
        return fn
    return wrap


def tools_for_llm():
    """Tool list with reversibility visible to the model (most frameworks hide this)."""
    return [{"name": n, "doc": (t.fn.__doc__ or "").strip(), "reversible": t.reversible,
             "args": getattr(t.fn, "input_schema", None) or str(inspect.signature(t.fn))}
            for n, t in TOOLS.items()]


class PlanFailed(Exception):
    def __init__(self, step, cause, undone, committed):
        self.step, self.cause, self.undone, self.committed = step, cause, undone, committed
        super().__init__(f"step {step!r} failed: {cause!r}; undone={undone}; "
                         f"already committed (cannot undo)={committed}")


_REF = re.compile(r"\$([\w-]+)")


def _refs(args, ids):
    """Step ids referenced as "$id" anywhere in string args ("$50" etc. are not refs)."""
    return [m for v in args.values() if isinstance(v, str) for m in _REF.findall(v) if m in ids]


def _resolve(args, results):
    """Whole-value "$id" -> the raw result; "$id" inside text -> str(result)."""
    def sub(v):
        if not isinstance(v, str):
            return v
        m = _REF.fullmatch(v)
        if m and m[1] in results:
            return results[m[1]]
        return _REF.sub(lambda m: str(results[m[1]]) if m[1] in results else m[0], v)
    return {k: sub(v) for k, v in args.items()}


def order(plan):
    """Topological order; among ready steps, undoable first, then original plan order."""
    by_id = {s["id"]: s for s in plan}
    unknown = [s["tool"] for s in plan if s["tool"] not in TOOLS]
    if unknown:
        raise ValueError(f"unknown tools: {unknown}")
    if len(by_id) != len(plan):
        raise ValueError("duplicate step ids")
    deps = {s["id"]: [*s.get("after", []), *_refs(s.get("args", {}), by_id)] for s in plan}
    dangling = {d for ds in deps.values() for d in ds} - by_id.keys()
    if dangling:
        raise ValueError(f"steps depend on unknown ids: {sorted(dangling)}")
    pos = {s["id"]: i for i, s in enumerate(plan)}
    ts = TopologicalSorter(deps)
    ts.prepare()  # raises CycleError on cycles
    out, ready = [], []
    while ts.is_active():
        ready += ts.get_ready()
        ready.sort(key=lambda i: (not TOOLS[by_id[i]["tool"]].reversible, pos[i]))
        i = ready.pop(0)
        out.append(by_id[i])
        ts.done(i)
    return out


def run(plan, confirm=lambda steps: True, log=print):
    steps = order(plan)
    ids = {s["id"] for s in steps}
    rev = lambda s: TOOLS[s["tool"]].reversible
    # point of no return = first permanent step; everything before it is fully undoable
    ponr = next((i for i, s in enumerate(steps) if not rev(s)), len(steps))
    results, done, committed = {}, [], []

    def fail(step, cause):
        undone = []
        for s in reversed(done):
            try:
                TOOLS[s["tool"]].undo(results[s["id"]])
                undone.append(s["id"])
            except Exception as e:  # keep unwinding; report what stuck
                log(f"undo of {s['id']} failed: {e!r}")
        raise PlanFailed(step, cause, undone, committed) from cause

    def call(s):
        try:
            results[s["id"]] = TOOLS[s["tool"]].fn(**_resolve(s.get("args", {}), results))
        except Exception as e:
            fail(s["id"], e)
        (done if rev(s) else committed).append(s if rev(s) else s["id"])
        log(f"{'ran' if rev(s) else 'COMMITTED'} {s['id']}")

    for s in steps[:ponr]:
        call(s)

    tail = steps[ponr:]
    # gate: dry-run every permanent step whose inputs are known now
    def check(s):
        try:
            TOOLS[s["tool"]].check(**_resolve(s.get("args", {}), results))
        except Exception as e:
            fail(s["id"], e)

    checked = set()
    for s in tail:
        if not rev(s) and TOOLS[s["tool"]].check and all(r in results for r in _refs(s.get("args", {}), ids)):
            check(s)
            checked.add(s["id"])
    if tail and not confirm(tail):
        fail("confirm", RuntimeError("commit phase not confirmed"))

    for s in tail:
        # NOTE: a permanent step that needs another permanent step's output can only be
        # checked just-in-time, after the point of no return; split the plan if that matters.
        if not rev(s) and TOOLS[s["tool"]].check and s["id"] not in checked:
            check(s)
        call(s)
    return results


if __name__ == "__main__":
    # Rebooking demo: the classic agent mistake is cancelling first, then finding no seats.
    db = {"bookings": {"OLD1"}, "holds": set(), "seats": {"AI101": 1}, "outbox": []}

    @tool(undo=NOOP)
    def search(route):
        """Find flights."""
        return [f for f, n in db["seats"].items() if n > 0]

    @tool(undo=lambda hold: (db["holds"].discard(hold), db["seats"].__setitem__(hold[1], db["seats"][hold[1]] + 1)))
    def book_flight(flight):
        """Hold a seat (can be released)."""
        if db["seats"].get(flight, 0) < 1:
            raise RuntimeError(f"no seats on {flight}")
        db["seats"][flight] -= 1
        hold = ("H", flight)
        db["holds"].add(hold)
        return hold

    def must_exist(ref):
        if ref not in db["bookings"]:
            raise RuntimeError(f"no booking {ref}")

    @tool(check=must_exist)
    def cancel_booking(ref):
        """Cancel a booking. PERMANENT."""
        db["bookings"].remove(ref)

    def valid_to(to, body):
        if "@" not in to:
            raise RuntimeError(f"bad address {to}")

    @tool(check=valid_to)
    def send_email(to, body):
        """Send an email. PERMANENT."""
        db["outbox"].append((to, body))

    # The plan a naive model writes: cancel first.
    def plan(flight="AI101", to="me@x.com"):
        return [
            {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}},
            {"id": "find", "tool": "search", "args": {"route": "BLR-DEL"}},
            {"id": "hold", "tool": "book_flight", "args": {"flight": flight}},
            {"id": "mail", "tool": "send_email", "args": {"to": to, "body": "$hold"}},
        ]

    assert [s["id"] for s in order(plan())] == ["find", "hold", "cancel", "mail"]
    # "$id" inside text is substituted and creates a dependency; "$50" is just text.
    assert _resolve({"b": "held $hold for $50", "w": "$hold"}, {"hold": ("H", "AI101")}) == \
        {"b": "held ('H', 'AI101') for $50", "w": ("H", "AI101")}
    assert [s["id"] for s in order([{"id": "m", "tool": "send_email", "args": {"to": "a@b", "body": "ref $h, $50"}},
                                    {"id": "h", "tool": "book_flight", "args": {"flight": "AI101"}}])] == ["h", "m"]
    assert {t["name"]: t["reversible"] for t in tools_for_llm()}["cancel_booking"] is False

    # 1. No seats: old booking must survive.
    try:
        run(plan(flight="XX999"), log=lambda m: None)
        raise AssertionError("should fail")
    except PlanFailed as e:
        assert e.step == "hold" and e.committed == []
    assert db["bookings"] == {"OLD1"}

    # 2. Bad email caught by dry-run: seat released, nothing permanent happened.
    try:
        run(plan(to="not-an-email"), log=lambda m: None)
        raise AssertionError("should fail")
    except PlanFailed as e:
        assert e.step == "mail" and e.undone == ["hold", "find"] and e.committed == []
    assert db["bookings"] == {"OLD1"} and db["seats"]["AI101"] == 1 and not db["holds"]

    # 3. Human says no at the gate: same full rollback.
    try:
        run(plan(), confirm=lambda steps: False, log=lambda m: None)
        raise AssertionError("should fail")
    except PlanFailed as e:
        assert e.step == "confirm"
    assert db["seats"]["AI101"] == 1 and db["bookings"] == {"OLD1"}

    # 4. Happy path.
    r = run(plan(), confirm=lambda steps: print("commit:", [s["id"] for s in steps]) or True)
    assert db["bookings"] == set() and db["outbox"] == [("me@x.com", ("H", "AI101"))]
    print("all checks passed")
