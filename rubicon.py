"""Rubicon: reversibility-aware execution for AI agent plans and tool calls.

Every tool declares once whether it can be undone. Rubicon then guarantees:
  * undoable steps run first and permanent steps last (dependencies respected),
  * every permanent step is dry-run (`check`) before the point of no return,
  * one confirmation covers all permanent steps,
  * any failure, refusal or interruption before commit undoes completed steps in reverse order.

Two ways to use it:
  run(plan) / await arun(plan)   execute a whole plan, e.g. one written by an LLM
  Transaction()                  guard any agent loop: undoable calls run immediately, permanent
                                 calls are dry-run immediately and deferred until commit()

Tools, undo, check and confirm functions may be sync or async.

Plan format (JSON-friendly, so a model can write it):
  [{"id": "hold", "tool": "hold_seat", "args": {"flight": "AI101"}},
   {"id": "mail", "tool": "send_email", "args": {"body": "Held: $hold"}, "after": ["hold"]}]
"$<id>" as a whole argument value passes that step's raw result; inside text it inserts it as
text. "$<id>.key.0" reaches into dict keys and list indexes. References work in nested
lists/objects and add a dependency automatically.
"""
import asyncio
import concurrent.futures
import functools
import inspect
import itertools
import re
import typing
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Callable


def NOOP(result):
    """Undo for read-only tools: there is nothing to reverse."""


@dataclass
class Tool:
    fn: Callable
    undo: Callable | None = None  # undo(result); None means the tool is permanent
    check: Callable | None = None  # check(**args): dry run for permanent tools; raise to block

    @property
    def reversible(self) -> bool:
        return self.undo is not None


TOOLS: dict[str, Tool] = {}  # default registry; pass registry={} anywhere for an isolated one


def tool(fn=None, *, undo=None, check=None, name=None, registry=None):
    """Register a tool.

    @tool                      permanent
    @tool(check=validate)      permanent, dry-run before the point of no return
    @tool(undo=reverse)        undoable; reverse(result) is called on rollback
    @tool(undo=NOOP)           read-only
    """
    def register(f):
        key = name or getattr(f, "__name__", None)
        if not key:
            raise ValueError(f"{f!r} has no __name__; pass tool(..., name='...')")
        (TOOLS if registry is None else registry)[key] = Tool(f, undo, check)
        return f
    return register(fn) if fn is not None else register


_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", tuple: "array", dict: "object"}


def _schema(fn):
    """JSON Schema for a tool's arguments: explicit `fn.input_schema`, else from the signature."""
    explicit = getattr(fn, "input_schema", None)
    if explicit:
        return explicit
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = {}
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):  # some builtins and C functions have no signature
        return {"type": "object"}
    props, required = {}, []
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        h = hints.get(p.name)
        t = _JSON_TYPES.get(typing.get_origin(h) or h)
        props[p.name] = {"type": t} if t else {}
        if p.default is p.empty:
            required.append(p.name)
    return {"type": "object", "properties": props, "required": required}


def tools_for_llm(registry=None):
    """Registered tools with description, JSON Schema parameters and reversibility."""
    reg = TOOLS if registry is None else registry
    return [{"name": n, "description": inspect.getdoc(t.fn) or "", "reversible": t.reversible,
             "parameters": _schema(t.fn)} for n, t in reg.items()]


class PlanFailed(Exception):
    """A step failed, a check blocked, or the commit was declined. `undone` steps were reversed."""

    def __init__(self, step, cause, undone, committed, undo_failed, results):
        self.step, self.cause, self.undone, self.committed = step, cause, undone, committed
        self.undo_failed, self.results = undo_failed, results
        msg = f"step {step!r} failed: {cause!r}; undone={undone}; committed (cannot undo)={committed}"
        if undo_failed:
            msg += f"; UNDO FAILED for {undo_failed}"
        super().__init__(msg)


_ID = re.compile(r"[\w-]+")
_REF = re.compile(r"\$([\w-]+)((?:\.[\w-]+)*)")  # $id, optionally followed by .key.0 path


def _walk(v, f):
    if isinstance(v, str):
        return f(v)
    if isinstance(v, dict):
        return {k: _walk(x, f) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_walk(x, f) for x in v]
    return v


def _refs(args, ids):
    """Step ids referenced as "$id" anywhere in args ("$50" etc. are not references)."""
    found = []
    _walk(args, lambda s: found.extend(m for m, _ in _REF.findall(s) if m in ids) or s)
    return found


def _dig(value, path):
    """Follow a ".key.0" path into dicts and lists. Raises LookupError if it doesn't exist."""
    for part in filter(None, path.split(".")):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, (list, tuple)) and part.lstrip("-").isdigit() and -len(value) <= int(part) < len(value):
            value = value[int(part)]
        else:
            raise LookupError(f"no {part!r} in {value!r}")
    return value


def _resolve(args, results):
    def embed(m):
        if m[1] not in results:
            return m[0]
        try:
            return str(_dig(results[m[1]], m[2]))
        except LookupError:  # e.g. "Booked $hold.Thanks": keep the text after the value
            return str(results[m[1]]) + m[2]

    def sub(s):
        m = _REF.fullmatch(s)
        if m and m[1] in results:
            try:
                return _dig(results[m[1]], m[2])
            except LookupError as e:
                raise ValueError(f"reference {s!r}: {e}") from None
        return _REF.sub(embed, s)
    return _walk(args, sub)


async def _call(fn, *args, **kwargs):
    r = fn(*args, **kwargs)
    return await r if inspect.isawaitable(r) else r


def _sync(coro, thread_ok=False):
    """Run a coroutine from sync code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if thread_ok:  # sync caller on an event-loop thread; the work has no loop affinity
        with concurrent.futures.ThreadPoolExecutor(1) as ex:
            return ex.submit(asyncio.run, coro).result()
    coro.close()
    raise RuntimeError("called from inside a running event loop; use the async API "
                       "(await arun(...), await tx.acall(...), async with Transaction())")


def order(plan, registry=None):
    """Validate a plan and return its execution order: undoable steps first, then plan order."""
    tools = TOOLS if registry is None else registry
    if not isinstance(plan, list):
        raise ValueError("plan must be a list of steps")
    by_id = {}
    for s in plan:
        if not isinstance(s, dict):
            raise ValueError(f"step is not an object: {s!r}")
        sid, name, after = s.get("id"), s.get("tool"), s.get("after", [])
        if not isinstance(sid, str) or not _ID.fullmatch(sid):
            raise ValueError(f"invalid step id {sid!r} (use letters, digits, _ or -)")
        if sid in by_id:
            raise ValueError(f"duplicate step id {sid!r}")
        if not isinstance(name, str) or name not in tools:
            raise ValueError(f"step {sid!r}: unknown tool {name!r}")
        if not isinstance(s.get("args", {}), dict):
            raise ValueError(f"step {sid!r}: args must be an object")
        if not isinstance(after, list) or not all(isinstance(a, str) for a in after):
            raise ValueError(f"step {sid!r}: after must be a list of step ids")
        by_id[sid] = s
    deps = {sid: [*s.get("after", []), *_refs(s.get("args", {}), by_id)] for sid, s in by_id.items()}
    dangling = sorted({d for ds in deps.values() for d in ds} - by_id.keys())
    if dangling:
        raise ValueError(f"steps depend on unknown ids: {dangling}")
    pos = {sid: i for i, sid in enumerate(by_id)}
    ts = TopologicalSorter(deps)
    try:
        ts.prepare()
    except CycleError as e:
        raise ValueError(f"plan has a dependency cycle: {e.args[1]}") from e
    out, ready = [], []
    while ts.is_active():
        ready += ts.get_ready()
        ready.sort(key=lambda i: (not tools[by_id[i]["tool"]].reversible, pos[i]))
        i = ready.pop(0)
        out.append(by_id[i])
        ts.done(i)
    return out


def _approve(steps):
    return True


class Transaction:
    """Guard tool calls from any agent loop.

    Undoable tools run immediately. Permanent tools are dry-run immediately (so the agent sees a
    check failure and can react) and deferred: the agent gets a "scheduled" message, and they run
    on commit() after one confirmation. Any exception inside the `with` block, a declined
    confirmation, or a failing permanent step rolls back every undoable call in reverse order.

        with Transaction(confirm=ask_human) as tx:        # or: async with
            tools = tx.wrap_all()                         # hand these to your agent framework
            ...                                           # commit on clean exit, rollback on error
    """

    def __init__(self, registry=None, confirm=None, log=print):
        self.tools = TOOLS if registry is None else registry
        self.confirm, self.log = confirm or _approve, log
        self.done = []  # (label, tool, result) of undoable calls, in execution order
        self.pending = []  # permanent steps waiting for commit
        self.committed, self.results = [], {}
        self._ids = itertools.count(1)

    def _tool(self, name):
        if name not in self.tools:
            raise ValueError(f"unknown tool {name!r}")
        return self.tools[name]

    async def _run_undoable(self, label, name, args):
        r = await _call(self._tool(name).fn, **args)
        self.done.append((label, name, r))
        self.results[label] = r
        self.log(f"ran {label}")
        return r

    async def _run_permanent(self, label, name, args):
        r = await _call(self._tool(name).fn, **args)
        self.committed.append(label)
        self.results[label] = r
        self.log(f"COMMITTED {label}")
        return r

    async def acall(self, name, /, **args):
        t = self._tool(name)
        label = f"{name}-{next(self._ids)}"
        if t.reversible:
            return await self._run_undoable(label, name, args)
        if t.check:
            await _call(t.check, **args)
        self.pending.append({"id": label, "tool": name, "args": args})
        return (f"Scheduled {name} ({label}): it is a permanent action and will run after "
                f"confirmation when the transaction commits.")

    async def arollback(self):
        """Undo every undoable call in reverse order and drop pending permanent steps."""
        undone, failed = [], []
        while self.done:
            label, name, r = self.done.pop()
            try:
                await _call(self.tools[name].undo, r)
                undone.append(label)
            except Exception as e:  # keep unwinding; report what stuck
                self.log(f"undo of {label} failed: {e!r}")
                failed.append(label)
        self.pending.clear()
        return undone, failed

    async def _fail(self, step, cause):
        undone, failed = await self.arollback()
        return PlanFailed(step, cause, undone, list(self.committed), failed, dict(self.results))

    async def acommit(self):
        """Confirm and run pending permanent steps. Returns all results."""
        steps, current = list(self.pending), "confirm"
        try:
            if steps and not await _call(self.confirm, steps):
                raise RuntimeError("commit not confirmed")
            for s in steps:
                current = s["id"]
                await self._run_permanent(s["id"], s["tool"], s["args"])
        except Exception as e:
            raise await self._fail(current, e) from e
        except BaseException:
            await self.arollback()
            raise
        self.pending.clear()
        self.done.clear()  # committed: earlier undoable calls are now final
        return dict(self.results)

    def call(self, name, /, **args):
        return _sync(self.acall(name, **args))

    def commit(self):
        return _sync(self.acommit())

    def rollback(self):
        return _sync(self.arollback())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is None:
            await self.acommit()
        else:
            await self.arollback()
        return False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
        return False

    def wrap(self, name):
        """The tool as a guarded function with the original name, docstring and signature."""
        fn = self._tool(name).fn
        sig = inspect.signature(fn)

        def kwargs(a, k):
            return dict(sig.bind(*a, **k).arguments) if a else k

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def guarded(*a, **k):
                return await self.acall(name, **kwargs(a, k))
        else:
            @functools.wraps(fn)
            def guarded(*a, **k):
                return _sync(self.acall(name, **kwargs(a, k)), thread_ok=True)
        return guarded

    def wrap_all(self):
        return [self.wrap(n) for n in self.tools]


async def arun(plan, registry=None, confirm=None, log=print):
    """Execute a plan safely. Returns {step_id: result}; raises PlanFailed after rolling back."""
    steps = order(plan, registry)
    tx = Transaction(registry, confirm, log)
    tools, ids, results = tx.tools, {s["id"] for s in steps}, tx.results
    perm = lambda s: not tools[s["tool"]].reversible
    args = lambda s: _resolve(s.get("args", {}), results)
    # point of no return = first permanent step; everything before it can be fully undone
    ponr = next((i for i, s in enumerate(steps) if perm(s)), len(steps))
    tail, checked, current = steps[ponr:], set(), None
    try:
        for s in steps[:ponr]:
            current = s["id"]
            await tx._run_undoable(current, s["tool"], args(s))
        for s in tail:  # dry-run every permanent step whose inputs are already known
            check = tools[s["tool"]].check
            if perm(s) and check and all(r in results for r in _refs(s.get("args", {}), ids)):
                current = s["id"]
                await _call(check, **args(s))
                checked.add(current)
        permanent = [{**s, "args": args(s)} for s in tail if perm(s)]
        current = "confirm"
        if permanent and not await _call(tx.confirm, permanent):
            raise RuntimeError("commit phase not confirmed")
        for s in tail:
            current, a = s["id"], args(s)
            if perm(s):
                # NOTE: a permanent step that needs another permanent step's output can only be
                # checked here, after the point of no return; split the plan if that matters.
                check = tools[s["tool"]].check
                if check and current not in checked:
                    await _call(check, **a)
                await tx._run_permanent(current, s["tool"], a)
            else:
                await tx._run_undoable(current, s["tool"], a)
    except Exception as e:
        raise await tx._fail(current, e) from e
    except BaseException:  # Ctrl+C, task cancellation: still unwind, then propagate
        await tx.arollback()
        raise
    return dict(results)


def run(plan, registry=None, confirm=None, log=print):
    """Sync version of arun(). Inside an event loop, use `await arun(...)` instead."""
    return _sync(arun(plan, registry, confirm, log))


if __name__ == "__main__":
    quiet = lambda m: None

    def expect(exc, fn, *a, **k):
        try:
            fn(*a, **k)
        except exc as e:
            return e
        raise AssertionError(f"expected {exc.__name__}")

    # --- rebooking scenario: the classic agent mistake is cancelling first ---
    db = {"bookings": {"OLD1"}, "holds": [], "seats": {"AI101": 1}, "outbox": []}

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
        db["outbox"].append((to, body))

    def plan(flight="AI101", to="me@x.com"):
        return [
            {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}},
            {"id": "find", "tool": "search", "args": {"route": "BLR-DEL"}},
            {"id": "hold", "tool": "book_flight", "args": {"flight": flight}},
            {"id": "mail", "tool": "send_email", "args": {"to": to, "body": "held $hold"}},
        ]

    def clean():
        return db["bookings"] == {"OLD1"} and db["seats"]["AI101"] == 1 and not db["holds"]

    assert [s["id"] for s in order(plan())] == ["find", "hold", "cancel", "mail"]

    # references: inside text, nested, and "$50" left alone
    assert _resolve({"b": "held $hold for $50", "w": "$hold", "n": [{"x": "$hold"}]},
                    {"hold": {"f": 1}}) == {"b": "held {'f': 1} for $50", "w": {"f": 1}, "n": [{"x": {"f": 1}}]}
    # paths into results: dict keys, list indexes, end-of-sentence dots, missing keys
    res = {"hold": {"flight": "AI101", "legs": [{"no": 7}]}}
    assert _resolve({"a": "$hold.flight", "b": "$hold.legs.0.no", "c": "on $hold.flight.",
                     "d": "$hold.legs.-1", "e": "Booked $hold.Thanks"}, res) == \
        {"a": "AI101", "b": 7, "c": "on AI101.", "d": {"no": 7},
         "e": "Booked {'flight': 'AI101', 'legs': [{'no': 7}]}.Thanks"}
    expect(ValueError, _resolve, {"a": "$hold.nope"}, res)
    assert [s["id"] for s in order([
        {"id": "m", "tool": "send_email", "args": {"to": "a@b", "body": "$h.flight"}},
        {"id": "h", "tool": "book_flight", "args": {"flight": "AI101"}}])] == ["h", "m"]
    # tool() edge cases: objects without __name__, signatures that can't be inspected
    expect(ValueError, tool, functools.partial(print), registry={})
    odd = {}
    tool(functools.partial(print), name="say", registry=odd)
    tool(len, name="length", undo=NOOP, registry=odd)
    assert [t["name"] for t in tools_for_llm(odd)] == ["say", "length"]
    assert [s["id"] for s in order([
        {"id": "m", "tool": "send_email", "args": {"to": "a@b", "body": ["ref", "$h", "$50"]}},
        {"id": "h", "tool": "book_flight", "args": {"flight": "AI101"}}])] == ["h", "m"]

    # malformed plans are rejected before anything runs
    good = {"id": "a", "tool": "search", "args": {"route": "x"}}
    for bad in ["nope", [1], [{**good, "id": "a b"}], [{**good, "id": 3}], [good, good],
                [{**good, "tool": "nope"}], [{**good, "tool": ["x"]}], [{**good, "args": [1]}],
                [{**good, "after": "b"}], [{**good, "after": ["ghost"]}], [{**good, "after": ["a"]}]]:
        expect(ValueError, order, bad)

    # 1. no seats -> old booking survives, nothing committed
    e = expect(PlanFailed, run, plan(flight="XX9"), log=quiet)
    assert e.step == "hold" and e.committed == [] and clean()

    # 2. dry-run blocks -> seat released, nothing permanent happened
    e = expect(PlanFailed, run, plan(to="not-an-email"), log=quiet)
    assert e.step == "mail" and e.undone == ["hold", "find"] and clean()

    # 3. human declines; confirm sees only permanent steps, with references resolved
    seen = []
    e = expect(PlanFailed, run, plan(), confirm=lambda st: seen.extend(st) or False, log=quiet)
    assert e.step == "confirm" and [s["id"] for s in seen] == ["cancel", "mail"] and clean()
    assert seen[1]["args"]["body"] == "held {'flight': 'AI101'}"

    # 4. Ctrl+C mid-run still unwinds, then propagates
    @tool(undo=NOOP)
    def interrupt():
        raise KeyboardInterrupt
    expect(KeyboardInterrupt, run, plan()[1:3] + [{"id": "x", "tool": "interrupt", "after": ["hold"]}], log=quiet)
    assert clean()

    # 5. an undo that fails is reported, and unwinding continues
    @tool(undo=lambda r: 1 / 0)
    def fragile():
        return "f"
    e = expect(PlanFailed, run, [{"id": "f", "tool": "fragile"}] + plan(to="bad")[1:], log=quiet)
    assert e.undo_failed == ["f"] and "hold" in e.undone and clean()

    # 6. happy path
    r = run(plan(), log=quiet)
    assert db["bookings"] == set() and db["outbox"] == [("me@x.com", "held {'flight': 'AI101'}")]
    assert r["hold"] == {"flight": "AI101"}

    # --- async tools, async undo/check/confirm, isolated registry ---
    reg, log2 = {}, []

    async def aundo(r):
        log2.append(("undo", r))

    async def acheck(x):
        if x < 0:
            raise ValueError("negative")

    @tool(undo=aundo, registry=reg)
    async def reserve(x: int) -> int:
        """Async undoable tool."""
        await asyncio.sleep(0)
        return x * 2

    @tool(check=acheck, registry=reg)
    async def charge(x: int):
        """Async permanent tool."""
        log2.append(("charge", x))

    async def aconfirm(steps):
        return True

    p = [{"id": "c", "tool": "charge", "args": {"x": "$r"}}, {"id": "r", "tool": "reserve", "args": {"x": 3}}]
    assert asyncio.run(arun(p, registry=reg, confirm=aconfirm, log=quiet)) == {"r": 6, "c": None}
    assert log2 == [("charge", 6)] and "reserve" not in TOOLS  # registries don't leak
    log2.clear()
    p[1]["args"]["x"] = -1  # charge(-2) fails its async check just-in-time -> async undo runs
    expect(PlanFailed, lambda: asyncio.run(arun(p, registry=reg, log=quiet)))
    assert log2 == [("undo", -2)]

    async def inside_loop():
        return expect(RuntimeError, run, p, registry=reg)
    assert "await arun" in str(asyncio.run(inside_loop()))

    # --- Transaction: guard an agent loop that calls tools one by one ---
    db.update(bookings={"OLD1"}, holds=[], seats={"AI101": 1}, outbox=[])
    with Transaction(log=quiet) as tx:
        hold = tx.wrap("book_flight")
        cancel = tx.wrap("cancel_booking")
        assert hold.__name__ == "book_flight" and "flight" in str(inspect.signature(hold))
        assert hold("AI101") == {"flight": "AI101"}  # positional args work; runs now
        assert "Scheduled" in cancel(ref="OLD1") and db["bookings"] == {"OLD1"}  # deferred
        expect(RuntimeError, cancel, ref="NOPE")  # dry run fails now, agent can react
    assert db["bookings"] == set() and db["holds"] == ["AI101"]  # committed on clean exit

    db.update(bookings={"OLD1"}, holds=[], seats={"AI101": 1})
    try:
        with Transaction(log=quiet) as tx:
            tx.call("book_flight", flight="AI101")
            tx.call("cancel_booking", ref="OLD1")
            raise RuntimeError("agent crashed")
    except RuntimeError:
        pass
    assert clean()  # rolled back, deferred cancel never ran

    tx = Transaction(confirm=lambda st: False, log=quiet)
    tx.call("book_flight", flight="AI101")
    tx.call("cancel_booking", ref="OLD1")
    e = expect(PlanFailed, tx.commit)
    assert e.step == "confirm" and clean()

    async def async_agent():  # async framework calling a sync tool on the loop thread
        async with Transaction(registry=reg, log=quiet) as atx:
            assert await atx.wrap("reserve")(5) == 10
            assert "Scheduled" in await atx.acall("charge", x=1)
        sync_tx = Transaction(log=quiet)
        assert sync_tx.wrap("search")("BLR-DEL") == ["AI101"]
    log2.clear()
    asyncio.run(async_agent())
    assert log2 == [("charge", 1)]

    # --- tool descriptions for models ---
    spec = {t["name"]: t for t in tools_for_llm()}
    assert spec["cancel_booking"]["reversible"] is False and spec["search"]["reversible"] is True
    assert spec["book_flight"]["parameters"] == {"type": "object", "required": ["flight"],
        "properties": {"flight": {"type": "string"}, "note": {"type": "string"}}}
    assert spec["book_flight"]["description"] == "Hold a seat (can be released)."
    print("all checks passed")
