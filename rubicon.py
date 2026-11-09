"""Rubicon: reversibility-aware execution for AI agent plans and tool calls.

Every tool declares once whether it can be undone. Rubicon then guarantees:
  * undoable steps run first and permanent steps last (dependencies respected),
  * every permanent step is dry-run (`check`) before the point of no return,
  * one confirmation covers all permanent steps,
  * a failure, refusal or interruption before anything permanent happens undoes every completed
    step in reverse order. Once a permanent step has committed, the undoable steps before it are
    part of that committed state and are kept; only steps after it are undone.

Two ways to use it:
  run(plan) / await arun(plan)   execute a whole plan, e.g. one written by an LLM
  Transaction()                  guard any agent loop: undoable calls run immediately, permanent
                                 calls are dry-run immediately and deferred until commit()

Tools, undo, check and confirm functions may be sync or async.

Plan format (JSON-friendly, so a model can write it):
  [{"id": "hold", "tool": "hold_seat", "args": {"flight": "AI101"}},
   {"id": "mail", "tool": "send_email", "args": {"body": "Held $hold.flight"}, "after": ["hold"]}]
Step ids start with a letter or _ and may contain letters, digits, _ and -.
"$<id>" as a whole argument value passes that step's raw result; inside text it inserts it as
text. "$<id>.key.0" reaches into dict keys and list indexes. References work in nested
lists/objects and add a dependency automatically.
"""
from __future__ import annotations

import asyncio
import contextlib
import contextvars
import enum
import inspect
import json
import logging
import os
import re
import threading
import time
import types
import typing
import uuid
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import Callable

__all__ = ["FileJournal", "JournalBusy", "NOOP", "PlanFailed", "SQLiteJournal", "TOOLS", "Tool",
           "Transaction", "UNSET", "acknowledge", "arecover", "arun", "compact", "dead_letters", "order",
           "recover", "run", "tool", "tools_for_llm"]

_logger = logging.getLogger("rubicon")

RETRY_DELAY = 0.2  # seconds before the first retry; doubles each attempt


def NOOP(result):
    """Undo for read-only tools: there is nothing to reverse."""


class _Unset:
    __slots__ = ()

    def __repr__(self):
        return "UNSET"

    def __bool__(self):
        return False


UNSET = _Unset()
"""Default for an optional parameter that should be left out of the call entirely (not passed,
not shown at confirmation, not journaled), e.g. so a remote API applies its own default."""


@dataclass
class Tool:
    fn: Callable
    undo: Callable | None = None  # undo(result) or undo(result, args); None means permanent
    check: Callable | None = None  # dry run for permanent tools; raise to block, return a preview
    retries: int = 0  # extra attempts for fn; only for tools that are safe to repeat
    undo_retries: int = 2  # extra attempts for undo, so a network blip doesn't strand a step
    timeout: float | None = None  # seconds per attempt, applies to async functions

    @property
    def reversible(self) -> bool:
        return self.undo is not None


TOOLS: dict[str, Tool] = {}  # default registry; pass registry={} anywhere for an isolated one


def tool(fn=None, *, undo=None, check=None, name=None, registry=None, retries=0, undo_retries=2,
         timeout=None):
    """Register a tool.

    @tool                      permanent
    @tool(check=validate)      permanent, dry-run before the point of no return; if validate
                               returns a value it is shown to whoever confirms
    @tool(undo=reverse)        undoable; reverse(result) or reverse(result, args) on rollback
    @tool(undo=NOOP)           read-only
    """
    def register(f):
        key = name or getattr(f, "__name__", None)
        if not key:
            raise ValueError(f"{f!r} has no __name__; pass tool(..., name='...')")
        (TOOLS if registry is None else registry)[key] = Tool(f, undo, check, retries, undo_retries, timeout)
        return f
    return register(fn) if fn is not None else register


# ---------------------------------------------------------------------------- tool descriptions

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean",
               list: "array", tuple: "array", set: "array", frozenset: "array", dict: "object"}


def _json_schema(h) -> dict:
    """JSON Schema for one type hint."""
    if h is type(None):
        return {"type": "null"}
    origin, args = typing.get_origin(h), typing.get_args(h)
    if origin is typing.Literal or (isinstance(h, type) and issubclass(h, enum.Enum)):
        values = list(args) if origin is typing.Literal else [m.value for m in h]
        kinds = {_JSON_TYPES.get(type(v)) for v in values}
        return {"type": kinds.pop(), "enum": values} if len(kinds) == 1 and None not in kinds else {"enum": values}
    if origin in (typing.Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        return _json_schema(options[0]) if len(options) == 1 else {"anyOf": [_json_schema(a) for a in options]}
    base = origin or h
    if base in (list, tuple, set, frozenset):
        if base is tuple:  # only tuple[X, ...] has one item type
            item = args[0] if len(args) == 2 and args[1] is Ellipsis else None
        else:
            item = args[0] if args else None
        return {"type": "array", "items": _json_schema(item)} if item is not None else {"type": "array"}
    t = _JSON_TYPES.get(base) if isinstance(base, type) else None
    return {"type": t} if t else {}


def _schema(fn) -> dict:
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
        props[p.name] = _json_schema(hints[p.name]) if p.name in hints else {}
        if p.default is p.empty:
            required.append(p.name)
    return {"type": "object", "properties": props, "required": required}


def tools_for_llm(registry=None) -> list[dict]:
    """Registered tools with description, JSON Schema parameters and reversibility."""
    reg = TOOLS if registry is None else registry
    return [{"name": n, "description": inspect.getdoc(t.fn) or "", "reversible": t.reversible,
             "parameters": _schema(t.fn)} for n, t in reg.items()]


# ---------------------------------------------------------------------------- errors

class PlanFailed(Exception):
    """A step failed, a check blocked, or the commit was declined.

    undone       steps that were reversed
    committed    permanent steps that already happened (cannot be undone)
    not_run      permanent steps that never ran
    undo_failed  steps whose undo failed even after retries: these need attention
    results      results of every step that ran
    """

    def __init__(self, step, cause, undone, committed, undo_failed, results, not_run=()):
        self.step, self.cause, self.undone, self.committed = step, cause, undone, committed
        self.undo_failed, self.results, self.not_run = undo_failed, results, list(not_run)
        msg = f"step {step!r} failed: {cause!r}; undone={undone}; committed (cannot undo)={committed}"
        if self.not_run:
            msg += f"; not run={self.not_run}"
        if undo_failed:
            msg += f"; UNDO FAILED for {undo_failed}"
        super().__init__(msg)


class _StepFailed(Exception):
    def __init__(self, step, cause):
        super().__init__(step, cause)
        self.step, self.cause = step, cause


# ---------------------------------------------------------------------------- references

_ID = re.compile(r"[A-Za-z_][\w-]*")
_REF = re.compile(r"\$([A-Za-z_][\w-]*)((?:\.[\w-]+)*)")  # $id, optionally followed by .key.0 path


def _walk(v, f):
    if isinstance(v, str):
        return f(v)
    if isinstance(v, dict):
        return {k: _walk(x, f) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_walk(x, f) for x in v]
    return v


def _refs(args, ids) -> list[str]:
    """Step ids referenced as "$id" anywhere in args ("$50" and unknown names are not references)."""
    found = []
    _walk(args, lambda s: found.extend(m for m, _ in _REF.findall(s) if m in ids) or s)
    return found


def _lookup(results, sid, path):
    """Follow a ".key.0" path. Returns (value, leftover text).

    Stops at a plain value and returns the rest as text ("$file.name.txt" -> "report", ".txt").
    A key or index missing from a dict or list is almost always a typo, so it raises LookupError.
    """
    value, parts = results[sid], path.split(".")[1:]
    for i, part in enumerate(parts):
        if isinstance(value, dict):
            if part not in value:
                raise LookupError(f"${sid}{path}: no key {part!r} (keys: {list(value)[:10]})")
            value = value[part]
        elif isinstance(value, (list, tuple)):
            if not (part.lstrip("-").isdigit() and -len(value) <= int(part) < len(value)):
                raise LookupError(f"${sid}{path}: no index {part!r} in a list of {len(value)}")
            value = value[int(part)]
        else:
            return value, "." + ".".join(parts[i:])
    return value, ""


def _resolve(args, results):
    def text(m):
        if m[1] not in results:
            return m[0]
        value, rest = _lookup(results, m[1], m[2])
        return str(value) + rest

    def sub(s):
        m = _REF.fullmatch(s)
        if m and m[1] in results:
            value, rest = _lookup(results, m[1], m[2])
            return value if not rest else str(value) + rest
        return _REF.sub(text, s)

    try:
        return _walk(args, sub)
    except LookupError as e:
        raise ValueError(f"bad reference {e}") from None


# ---------------------------------------------------------------------------- calling functions

def _prepare(fn, args=(), kwargs=None):
    """Bind a call against fn's signature with defaults applied.

    Returns (positional, keyword, view) where view maps parameter names to values (used for
    checks, undo, confirmation and the journal). Plans pass arguments by name, so positional-only
    parameters given by name are moved to positional. Raises TypeError if the call doesn't fit.
    """
    kwargs = dict(kwargs or {})
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return tuple(args), kwargs, dict(kwargs)
    params = sig.parameters.values()
    if not args:
        args = tuple(kwargs.pop(p.name) for p in params if p.kind is p.POSITIONAL_ONLY and p.name in kwargs)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    for name in [n for n, v in bound.arguments.items() if v is UNSET]:
        del bound.arguments[name]  # left out entirely: the function's own default applies
    view = {}
    for name, value in bound.arguments.items():
        if sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
            view.update(value)
        else:
            view[name] = value
    return bound.args, bound.kwargs, view


def _fit(check, args, kwargs, view):
    """Arguments for a check: the tool's arguments, limited to the parameters the check accepts,
    so a check can ignore optional parameters it doesn't care about."""
    try:
        params = inspect.signature(check).parameters.values()
    except (TypeError, ValueError):
        return args, kwargs
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return args, kwargs
    if not any(p.kind is p.VAR_KEYWORD for p in params):
        names = {p.name for p in params}
        view = {n: v for n, v in view.items() if n in names}
    a, k, _ = _prepare(check, (), view)
    return a, k


async def _call(fn, *args, **kwargs):
    r = fn(*args, **kwargs)
    return await r if inspect.isawaitable(r) else r


_DRIVING = contextvars.ContextVar("rubicon_driving", default=False)


async def _attempt(fn, args, kwargs, retries, timeout):
    """Call with optional timeout (async only) and retries with exponential backoff."""
    for n in range(retries + 1):
        is_async = False
        try:
            r = fn(*args, **kwargs)
            if inspect.isawaitable(r):
                is_async = True
                r = await (asyncio.wait_for(r, timeout) if timeout else r)
            return r
        except Exception:
            if n == retries:
                raise
            if _DRIVING.get() and not is_async:
                # sync code driving this directly inside someone else's event loop (tx.call):
                # there is no task to suspend, so wait synchronously
                time.sleep(RETRY_DELAY * 2 ** n)
            else:  # everywhere else, never block the event loop
                await asyncio.sleep(RETRY_DELAY * 2 ** n)


def _takes_args(undo) -> bool:
    """True if undo accepts (result, args) rather than just (result)."""
    try:
        params = inspect.signature(undo).parameters.values()
    except (TypeError, ValueError):
        return False
    positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    return any(p.kind is p.VAR_POSITIONAL for p in params) or len(positional) >= 2


def _sync(coro, drive=False):
    """Run a coroutine from sync code.

    Inside a running event loop, drive=True runs it to completion directly, which works as long
    as nothing in it waits on real I/O (sync tools and checks); otherwise it raises.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if drive:
        token = _DRIVING.set(True)
        try:
            coro.send(None)
        except StopIteration as finished:
            return finished.value
        finally:
            _DRIVING.reset(token)
    coro.close()
    raise RuntimeError("an async tool, check, undo or confirm can't run from sync code inside a "
                       "running event loop; use the async API (await arun(...), await tx.acall(...), "
                       "async with Transaction())")


_MARK = "__unserializable__"


def _complain(message):
    try:
        _logger.warning(message)
    except Exception:
        pass


def _jsonable(value):
    """A JSON-safe copy: only the parts that can't be serialized are replaced by a marker."""
    if isinstance(value, dict):
        return {k if isinstance(k, str) else str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return {_MARK: repr(value)}


def _has_mark(value) -> bool:
    if isinstance(value, dict):
        return _MARK in value or any(_has_mark(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_mark(v) for v in value)
    return False


# ---------------------------------------------------------------------------- planning

def _deps(step, ids) -> list[str]:
    return [*step.get("after", []), *_refs(step.get("args", {}), ids)]


def order(plan, registry=None) -> list[dict]:
    """Validate a plan and return its execution order: undoable steps first, then plan order.

    Rejects malformed steps, unknown tools, arguments that don't match the tool (or its check),
    references to missing steps and dependency cycles, all before anything runs.
    """
    tools = TOOLS if registry is None else registry
    if not isinstance(plan, list):
        raise ValueError("plan must be a list of steps")
    by_id = {}
    for s in plan:
        if not isinstance(s, dict):
            raise ValueError(f"step is not an object: {s!r}")
        sid, name, after, args = s.get("id"), s.get("tool"), s.get("after", []), s.get("args", {})
        if not isinstance(sid, str) or not _ID.fullmatch(sid):
            raise ValueError(f"invalid step id {sid!r} (start with a letter or _, then letters, digits, _ or -)")
        if sid in by_id:
            raise ValueError(f"duplicate step id {sid!r}")
        if not isinstance(name, str) or name not in tools:
            raise ValueError(f"step {sid!r}: unknown tool {name!r}")
        if not isinstance(args, dict):
            raise ValueError(f"step {sid!r}: args must be an object")
        if not isinstance(after, list) or not all(isinstance(a, str) for a in after):
            raise ValueError(f"step {sid!r}: after must be a list of step ids")
        t = tools[name]
        try:
            a, k, view = _prepare(t.fn, (), args)
            if t.check:
                _fit(t.check, a, k, view)
        except TypeError as e:
            raise ValueError(f"step {sid!r}: arguments {sorted(args)} don't fit {name}: {e}") from None
        by_id[sid] = s
    deps = {sid: _deps(s, by_id) for sid, s in by_id.items()}
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


# ---------------------------------------------------------------------------- execution

class Transaction:
    """Guard tool calls from any agent loop.

    Undoable tools run immediately. Permanent tools are dry-run immediately (so the agent sees a
    check failure and can react) and deferred: the agent gets a "scheduled" message, and they run
    on commit() after one confirmation. An exception inside the `with` block or a declined
    confirmation rolls back every undoable call in reverse order. If a permanent step fails after
    another one committed, only undoable calls made after the last commit are rolled back.

        with Transaction(confirm=ask_human) as tx:        # or: async with
            tools = tx.wrap_all()                         # hand these to your agent framework
            ...                                           # commit on clean exit, rollback on error

    After a rollback, tx.undone and tx.undo_failed say what happened.

    on_event(dict) receives structured events; journal="path.jsonl" writes a crash-safe log that
    recover() can use to undo an interrupted transaction after a restart.
    """

    def __init__(self, registry=None, confirm=None, log=None, *, on_event=None, journal=None):
        self.tools = TOOLS if registry is None else registry
        self.confirm, self.log, self.on_event = confirm or _approve, log or _logger.info, on_event
        self.id = uuid.uuid4().hex
        self.done = []  # undoable calls since the last commit, in completion order
        self.pending = []  # permanent steps waiting for commit
        self.committed, self.results = [], {}
        self.undone, self.undo_failed = [], []
        self._journal, self._lock, self._counter = _as_journal(journal), threading.Lock(), 0
        self._began = False

    # -- bookkeeping

    def _emit(self, event, **data):
        if self.on_event:
            try:
                self.on_event({"event": event, "tx": self.id, "time": time.time(), **data})
            except Exception as e:  # observability must never change what runs or gets undone
                _complain(f"on_event callback failed: {e!r}")

    def _say(self, message):
        try:
            self.log(message)
        except Exception as e:
            _complain(f"log callback failed: {e!r}")

    def _write(self, event, **data):
        if not self._journal:
            return
        now = time.time()
        records = [{"tx": self.id, "event": event, "time": now, **data}]
        with self._lock:
            if not self._began:
                self._began = True
                records.insert(0, {"tx": self.id, "event": "begin", "time": now})
            self._journal.append(records)

    def _settle(self, status):
        """Journal that everything so far is committed or undone. While an earlier undo is still
        owed (tx.undo_failed), write "settled" instead of "finished" so recover() retries it."""
        self._write("finished" if not self.undo_failed else "settled", status=status)

    def _label(self, name):
        with self._lock:
            self._counter += 1
            return f"{name}-{self._counter}"

    def _tool(self, name) -> Tool:
        if name not in self.tools:
            raise ValueError(f"unknown tool {name!r}")
        return self.tools[name]

    # -- running steps

    async def _run(self, label, name, a, k, view, permanent):
        t = self._tool(name)
        # write-ahead: if the process dies mid-step, recover() knows this step may have happened
        self._write("started", step=label, tool=name, permanent=permanent)
        self._emit("step_started", step=label, tool=name, permanent=permanent)
        start = time.perf_counter()
        try:
            r = await _attempt(t.fn, a, k, t.retries, t.timeout)
        except BaseException as e:
            self._write("step_failed", step=label, tool=name, error=repr(e))
            raise
        ms = round((time.perf_counter() - start) * 1000, 1)
        self.results[label] = r
        if permanent:
            self.committed.append(label)
            self.done.clear()  # everything before a committed step is now part of the result
            self._write("committed", step=label, tool=name)
            self._emit("step_committed", step=label, tool=name, duration_ms=ms)
            self._say(f"COMMITTED {label}")
        else:
            self.done.append({"id": label, "tool": name, "args": view, "result": r})
            self._write("done", step=label, tool=name, args=_jsonable(view), result=_jsonable(r))
            self._emit("step_done", step=label, tool=name, duration_ms=ms)
            self._say(f"ran {label}")
        return r

    async def _check(self, label, name, a, k, view):
        t = self._tool(name)
        if not t.check:
            return None
        ca, ck = _fit(t.check, a, k, view)
        preview = await _attempt(t.check, ca, ck, 0, t.timeout)
        self._emit("check_passed", step=label, tool=name, preview=preview)
        return preview

    async def _undo(self, rec):
        t = self.tools[rec["tool"]]
        args = (rec["result"], rec["args"]) if _takes_args(t.undo) else (rec["result"],)
        await _attempt(t.undo, args, {}, t.undo_retries, t.timeout)

    # -- public API

    async def acall(self, name, /, *args, **kwargs):
        """Call a tool through the transaction (async)."""
        t = self._tool(name)
        a, k, view = _prepare(t.fn, args, kwargs)
        label = self._label(name)
        if t.reversible:
            return await self._run(label, name, a, k, view, permanent=False)
        preview = await self._check(label, name, a, k, view)
        self.pending.append({"id": label, "tool": name, "args": view, "preview": preview, "call": (a, k)})
        note = f" Preview: {preview}" if preview is not None else ""
        return (f"Scheduled {name} ({label}): it is a permanent action and will run after "
                f"confirmation when the transaction commits.{note}")

    async def arollback(self):
        """Undo every undoable call since the last commit, in reverse order, and drop pending
        permanent steps. Keeps going if an undo fails or is interrupted; an interruption is
        re-raised once everything has been tried."""
        undone, failed, interrupted = [], [], None
        while self.done:
            rec = self.done.pop()
            try:
                await self._undo(rec)
                undone.append(rec["id"])
                self._write("undone", step=rec["id"])
                self._emit("step_undone", step=rec["id"], tool=rec["tool"])
            except BaseException as e:  # noqa: B036 - an interruption must not strand the rest
                failed.append(rec["id"])
                self._write("undo_failed", step=rec["id"], error=repr(e))
                self._emit("undo_failed", step=rec["id"], tool=rec["tool"], error=repr(e))
                self._say(f"undo of {rec['id']} failed: {e!r}")
                if not isinstance(e, Exception) and interrupted is None:
                    interrupted = e
        self.pending.clear()
        self.undone += undone
        self.undo_failed += failed
        self._settle("rolled_back")
        if interrupted is not None:
            raise interrupted
        return undone, failed

    async def _fail(self, step, cause, not_run=()):
        undone, failed = await self.arollback()
        return PlanFailed(step, cause, undone, list(self.committed), failed, dict(self.results), not_run)

    async def acommit(self):
        """Confirm and run pending permanent steps. Returns all results."""
        steps, current = list(self.pending), "confirm"
        shown = [{key: s[key] for key in ("id", "tool", "args", "preview")} for s in steps]
        try:
            if steps:
                if not await _call(self.confirm, shown):
                    self._emit("gate_declined", steps=[s["id"] for s in steps])
                    raise RuntimeError("commit not confirmed")
                self._emit("gate_confirmed", steps=[s["id"] for s in steps])
            for s in steps:
                current = s["id"]
                a, k = s["call"]
                await self._run(s["id"], s["tool"], a, k, s["args"], permanent=True)
        except Exception as e:
            not_run = [s["id"] for s in steps if s["id"] not in self.committed]
            raise await self._fail(current, e, not_run) from e
        except BaseException:
            await self.arollback()
            raise
        self.pending.clear()
        self.done.clear()  # committed: earlier undoable calls are now final
        self._settle("committed")
        return dict(self.results)

    def call(self, name, /, *args, **kwargs):
        """Call a tool through the transaction. Inside a running event loop this works for sync
        tools and checks; use acall for async ones."""
        return _sync(self.acall(name, *args, **kwargs), drive=True)

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
        """The tool as a guarded function with the original name, docstring and signature.

        The wrapper is async if the tool or its check is async, so agent frameworks await it.
        """
        t = self._tool(name)
        is_async = inspect.iscoroutinefunction(t.fn) or inspect.iscoroutinefunction(t.check)
        if is_async:
            async def guarded(*a, **k):
                return await self.acall(name, *a, **k)
        else:
            def guarded(*a, **k):
                return self.call(name, *a, **k)
        guarded.__name__ = guarded.__qualname__ = name
        guarded.__doc__ = inspect.getdoc(t.fn)
        guarded.__wrapped__ = t.fn  # so inspect.signature() shows the tool's real parameters
        for attr in ("__module__", "__annotations__", "input_schema"):
            if hasattr(t.fn, attr):
                setattr(guarded, attr, getattr(t.fn, attr))
        return guarded

    def wrap_all(self):
        return [self.wrap(n) for n in self.tools]


async def _waves(tx, steps, deps, execute):
    """Run undoable steps in parallel waves of mutually independent steps."""
    remaining, finished = {s["id"]: s for s in steps}, set()
    while remaining:
        wave = [s for sid, s in remaining.items() if set(deps[sid]) & remaining.keys() <= finished]
        tasks = {asyncio.ensure_future(execute(s)): s["id"] for s in wave}
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        except BaseException:
            # we were cancelled or interrupted: stop the wave before anyone rolls back, so no
            # step can finish after the rollback
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if task.cancelled():  # cancelled from inside (e.g. a library's cancel scope): not finished
                raise _StepFailed(tasks[task], RuntimeError(f"step {tasks[task]!r} was cancelled before it finished"))
            if task.exception() is not None:
                raise _StepFailed(tasks[task], task.exception())
        for sid in tasks.values():
            finished.add(sid)
            del remaining[sid]


async def arun(plan, registry=None, confirm=None, log=None, *, parallel=False, on_event=None,
               journal=None) -> dict:
    """Execute a plan safely. Returns {step_id: result}; raises PlanFailed after rolling back.

    parallel=True runs independent undoable steps concurrently. on_event and journal are passed
    to the underlying Transaction.
    """
    steps = order(plan, registry)
    tx = Transaction(registry, confirm, log, on_event=on_event, journal=journal)
    tools, results = tx.tools, tx.results
    ids = {s["id"] for s in steps}
    deps = {s["id"]: _deps(s, ids) for s in steps}
    perm = lambda s: not tools[s["tool"]].reversible
    # point of no return = first permanent step; everything before it can be fully undone
    ponr = next((i for i, s in enumerate(steps) if perm(s)), len(steps))
    pre, tail, previews, current = steps[:ponr], steps[ponr:], {}, None

    def call_for(s):
        return _prepare(tools[s["tool"]].fn, (), _resolve(s.get("args", {}), results))

    async def execute(s):
        a, k, view = call_for(s)
        if perm(s):
            if s["id"] not in previews:  # inputs weren't known before the gate: check now
                await tx._check(s["id"], s["tool"], a, k, view)
            return await tx._run(s["id"], s["tool"], a, k, view, permanent=True)
        return await tx._run(s["id"], s["tool"], a, k, view, permanent=False)

    try:
        if parallel:
            try:
                await _waves(tx, pre, deps, execute)
            except _StepFailed as e:
                current = e.step
                raise e.cause
        else:
            for s in pre:
                current = s["id"]
                await execute(s)
        permanent = []
        for s in tail:
            if not perm(s):
                continue
            current = s["id"]
            # resolve what's known already; references to steps that haven't run stay as written
            a, k, view = call_for(s)
            if all(d in results for d in deps[s["id"]]):  # includes `after`, not just $refs
                previews[s["id"]] = await tx._check(s["id"], s["tool"], a, k, view)
            permanent.append({"id": s["id"], "tool": s["tool"], "args": view, "preview": previews.get(s["id"])})
        current = "confirm"
        if permanent:
            if not await _call(tx.confirm, permanent):
                tx._emit("gate_declined", steps=[s["id"] for s in permanent])
                raise RuntimeError("commit phase not confirmed")
            tx._emit("gate_confirmed", steps=[s["id"] for s in permanent])
        for s in tail:
            # NOTE: a permanent step that needs another permanent step's output can only be
            # checked here, after the point of no return; split the plan if that matters.
            current = s["id"]
            await execute(s)
    except Exception as e:
        not_run = [s["id"] for s in tail if perm(s) and s["id"] not in tx.committed]
        raise await tx._fail(current, e, not_run) from e
    except BaseException:  # Ctrl+C, task cancellation: still unwind, then propagate
        await tx.arollback()
        raise
    tx._settle("committed")
    return dict(results)


def run(plan, registry=None, confirm=None, log=None, **kwargs) -> dict:
    """Sync version of arun(). Inside an event loop, use `await arun(...)` instead."""
    return _sync(arun(plan, registry, confirm, log, **kwargs))


# ---------------------------------------------------------------------------- journals

class JournalBusy(RuntimeError):
    """Another process holds the journal's lock (for example, recover() is already running)."""


class _FileLock:
    """An OS-level exclusive lock on a side file; the OS releases it if the process dies."""

    def __init__(self, path, blocking, timeout=30.0):
        self.path, self.blocking, self.timeout = path, blocking, timeout
        self._file = None

    def __enter__(self):
        f = open(self.path, "a+b")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                if os.name == "nt":
                    import msvcrt
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._file = f
                return self
            except OSError:
                if not self.blocking or time.monotonic() > deadline:
                    f.close()
                    raise JournalBusy(f"{self.path} is locked by another process") from None
                time.sleep(0.01)

    def __exit__(self, *exc):
        f, self._file = self._file, None
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()
        return False


class FileJournal:
    """Append-only JSON-lines file. Every write is flushed and fsynced.

    Several processes on one machine can write to it; writes and compaction are serialized with
    an OS file lock, and recover() takes a separate exclusive lock so only one recovery runs.
    """

    def __init__(self, path):
        self.path = os.fspath(path)

    def __repr__(self):
        return f"FileJournal({self.path!r})"

    def append(self, records):
        data = "".join(json.dumps(r, default=repr) + "\n" for r in records)
        with _FileLock(self.path + ".write.lock", blocking=True):
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

    def records(self):
        if not os.path.exists(self.path):
            return []
        out = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except ValueError:  # a torn last line from a crash
                        continue
        return out

    def lock(self):
        return _FileLock(self.path + ".lock", blocking=False)

    def drop(self, tx_ids):
        """Remove every record of these transactions (atomic rename; writers wait meanwhile)."""
        tx_ids = set(tx_ids)
        with _FileLock(self.path + ".write.lock", blocking=True):
            keep = [r for r in self.records() if r.get("tx") not in tx_ids]
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(r, default=repr) + "\n" for r in keep))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)


class SQLiteJournal:
    """Journal in a SQLite database (WAL mode, synchronous=FULL), standard library only.

    Safe for many threads and processes on one machine. For several machines, implement the same
    small interface on a shared store: append(records) and records(), plus optionally lock()
    (a context manager held during recovery) and drop(tx_ids) (for compaction).
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS rubicon_journal ("
                       "id INTEGER PRIMARY KEY AUTOINCREMENT, tx TEXT NOT NULL, record TEXT NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS rubicon_journal_tx ON rubicon_journal (tx)")

    def __repr__(self):
        return f"SQLiteJournal({self.path!r})"

    def _connect(self):
        import sqlite3
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        return contextlib.closing(db)

    def append(self, records):
        rows = [(r.get("tx", ""), json.dumps(r, default=repr)) for r in records]
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                db.executemany("INSERT INTO rubicon_journal (tx, record) VALUES (?, ?)", rows)
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    def records(self):
        with self._connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT record FROM rubicon_journal ORDER BY id")]

    def lock(self):
        return _FileLock(self.path + ".lock", blocking=False)

    def drop(self, tx_ids):
        tx_ids = list(tx_ids)
        with self._connect() as db:
            for i in range(0, len(tx_ids), 500):
                chunk = tx_ids[i:i + 500]
                db.execute(f"DELETE FROM rubicon_journal WHERE tx IN ({','.join('?' * len(chunk))})", chunk)


def _as_journal(journal):
    if journal is None:
        return None
    if isinstance(journal, (str, os.PathLike)):
        return FileJournal(journal)
    if not (callable(getattr(journal, "append", None)) and callable(getattr(journal, "records", None))):
        raise TypeError("journal must be a path, FileJournal, SQLiteJournal, or an object with "
                        "append(records) and records() methods")
    return journal


def _lock_of(journal):
    return journal.lock() if callable(getattr(journal, "lock", None)) else contextlib.nullcontext()


# ---------------------------------------------------------------------------- crash recovery

def _replay(records):
    """Rebuild each transaction's state from its journal records.

    "done" = undoable steps since the last commit; "owed" = steps whose undo failed during a
    rollback (no later commit makes those final); "open" = started steps with no outcome yet;
    letters = dead letters nobody has resolved.
    """
    txs: dict[str, dict] = {}
    letters: dict[tuple, dict] = {}
    for position, rec in enumerate(records):
        tx_id, event, step = rec.get("tx"), rec.get("event"), rec.get("step")
        if not tx_id or not event:
            continue
        rec = {**rec, "_pos": position}
        tx = txs.setdefault(tx_id, {"done": {}, "owed": {}, "committed": [], "open": {}, "finished": False})
        if event in ("started", "done", "committed"):
            tx["finished"] = False  # new activity reopens a transaction that committed in stages
        if event == "started":
            tx["open"][step] = rec
        elif event == "done":
            tx["open"].pop(step, None)
            tx["done"][step] = rec
        elif event == "committed":
            tx["open"].pop(step, None)
            tx["committed"].append(step)
            tx["done"].clear()
        elif event == "step_failed":
            tx["open"].pop(step, None)
        elif event == "undo_failed":
            if step in tx["done"]:
                tx["owed"][step] = tx["done"].pop(step)
            letters[(tx_id, step)] = {"tx": tx_id, "step": step, "kind": "undo_failed",
                                      "reason": rec.get("error"), "time": rec.get("time")}
        elif event == "dead_letter":
            letters[(tx_id, step)] = {"tx": tx_id, "step": step, "kind": rec.get("kind"),
                                      "reason": rec.get("reason"), "time": rec.get("time")}
        elif event == "undone":
            tx["done"].pop(step, None)
            tx["owed"].pop(step, None)
            letters.pop((tx_id, step), None)
        elif event == "resolved":
            if step:
                for part in ("open", "done", "owed"):
                    tx[part].pop(step, None)
                letters.pop((tx_id, step), None)
            else:
                for part in ("open", "done", "owed"):
                    tx[part].clear()
                tx["finished"] = True
                for key in [k for k in letters if k[0] == tx_id]:
                    del letters[key]
        elif event in ("settled", "finished"):  # everything before this is committed or undone
            tx["done"].clear()
            tx["open"].clear()
            tx["finished"] = event == "finished"
    return txs, letters


def _record(event, tx, **data):
    return {"tx": tx, "event": event, "time": time.time(), **data}


async def arecover(journal, registry=None, log=None, *, on_dead_letter=None) -> list[dict]:
    """Undo transactions that a crash left unfinished, using their journal.

    For each unfinished transaction, undoable steps completed after its last commit are undone in
    reverse order (steps before a commit are part of the committed result and are kept). Each undo
    is journaled as it happens, so recovery itself can crash and be run again without undoing
    anything twice. A transaction is marked finished only when every undo succeeded; run
    recover() again later to retry the ones in "undo_failed".

    If the crash happened while a permanent step was running, nobody knows whether it went
    through, so nothing is undone (undoing what it relied on could leave things worse) and the
    step is listed in "needs_attention" for a person to check. An undoable step that started but
    never reported a result is listed in "uncertain".

    Everything that needs a person becomes a dead letter: it stays in the journal, is listed by
    dead_letters() until acknowledge() resolves it, and is passed once to on_dead_letter(dict).
    Only one recover() runs per journal at a time; another one raises JournalBusy.

    Returns one report per transaction:
    {"tx", "undone", "undo_failed", "committed", "uncertain", "needs_attention", "kept"}.
    """
    tools = TOOLS if registry is None else registry
    j = _as_journal(journal)
    say = log or _logger.warning
    with _lock_of(j):
        txs, letters = _replay(j.records())
        reports = []
        for tx_id, tx in txs.items():
            if tx["finished"] and not tx["owed"]:
                continue

            def dead_letter(step, kind, reason, write=True):
                if (tx_id, step) in letters:
                    return  # already recorded and reported on an earlier run
                letter = {"tx": tx_id, "step": step, "kind": kind, "reason": reason, "time": time.time()}
                if write:
                    j.append([_record("dead_letter", tx_id, step=step, kind=kind, reason=reason)])
                letters[(tx_id, step)] = letter
                if on_dead_letter:
                    try:
                        on_dead_letter(dict(letter))
                    except Exception as e:
                        _complain(f"on_dead_letter callback failed: {e!r}")

            in_flight = [s for s, r in tx["open"].items() if r.get("permanent")]
            uncertain = [s for s, r in tx["open"].items() if not r.get("permanent")]
            to_undo = sorted([*tx["done"].values(), *tx["owed"].values()], key=lambda r: r["_pos"], reverse=True)
            if not (in_flight or uncertain or to_undo):  # nothing left to do: close it quietly
                j.append([_record("finished", tx_id, status="recovered")])
                continue
            report = {"tx": tx_id, "undone": [], "undo_failed": [], "committed": tx["committed"],
                      "uncertain": uncertain, "needs_attention": in_flight, "kept": []}
            if in_flight:
                report["kept"] = [r["step"] for r in reversed(to_undo)]
                say(f"recover: transaction {tx_id} stopped during permanent step(s) {in_flight}; it may "
                    f"or may not have happened, so {report['kept']} were kept. Check by hand, then "
                    f"call acknowledge().")
                for s in in_flight:
                    dead_letter(s, "in_flight_permanent", "the process stopped while this permanent step "
                                "was running; it may or may not have happened")
                reports.append(report)
                continue
            for s in uncertain:
                dead_letter(s, "uncertain", "the process stopped while this undoable step was running; "
                            "it may have happened and can't be undone automatically")
            if uncertain:
                say(f"recover: step(s) {uncertain} of {tx_id} started but never reported a result; "
                    f"they may have happened and can't be undone automatically.")
            for rec in to_undo:
                t = tools.get(rec["tool"])
                result, args = rec.get("result"), rec.get("args", {})
                try:
                    if t is None or not t.reversible:
                        raise LookupError(f"tool {rec['tool']!r} is not registered as undoable")
                    with_args = _takes_args(t.undo)
                    if _has_mark(result) or (with_args and _has_mark(args)):
                        raise ValueError("the step's result or arguments weren't JSON-serializable, so "
                                         "they can't be replayed; undo it by hand")
                    await _attempt(t.undo, (result, args) if with_args else (result,), {}, t.undo_retries, t.timeout)
                    report["undone"].append(rec["step"])
                    j.append([_record("undone", tx_id, step=rec["step"], by="recover")])
                    letters.pop((tx_id, rec["step"]), None)
                except Exception as e:
                    report["undo_failed"].append(rec["step"])
                    j.append([_record("undo_failed", tx_id, step=rec["step"], error=repr(e))])
                    dead_letter(rec["step"], "undo_failed", repr(e), write=False)
                    say(f"recover: undo of {rec['step']} failed: {e!r}")
            if not report["undo_failed"]:
                j.append([_record("finished", tx_id, status="recovered")])
            reports.append(report)
        return reports


def recover(journal, registry=None, log=None, *, on_dead_letter=None) -> list[dict]:
    """Sync version of arecover()."""
    return _sync(arecover(journal, registry, log, on_dead_letter=on_dead_letter))


def dead_letters(journal) -> list[dict]:
    """Steps that need a person, oldest first, each {"tx", "step", "kind", "reason", "time"}.

    kind is "undo_failed" (an undo kept failing), "uncertain" (a crash during an undoable step)
    or "in_flight_permanent" (a crash during a permanent step). Listed until a later successful
    undo or acknowledge() resolves them. Run recover() first to record crash-related ones.
    """
    _, letters = _replay(_as_journal(journal).records())
    return sorted(letters.values(), key=lambda letter: letter.get("time") or 0)


def acknowledge(journal, tx, step=None, note=""):
    """Record that a person handled a dead letter, so recover() and dead_letters() stop reporting it.

    With a step: that step is settled as it is (Rubicon won't undo it). Without a step: the whole
    transaction is settled, including any steps recover() was keeping.
    """
    _as_journal(journal).append([_record("resolved", tx, step=step, note=note)])


def compact(journal) -> int:
    """Remove settled transactions from the journal so it doesn't grow forever.

    Keeps every transaction that is unfinished, owes an undo or has an unresolved dead letter.
    Returns how many transactions were removed. Needs a journal with drop(tx_ids); both built-in
    journals have one.
    """
    j = _as_journal(journal)
    if not callable(getattr(j, "drop", None)):
        raise TypeError(f"{j!r} has no drop(tx_ids) method, so it can't be compacted")
    with _lock_of(j):
        txs, letters = _replay(j.records())
        busy = {key[0] for key in letters}
        settled = [t for t, tx in txs.items() if tx["finished"] and not tx["owed"] and t not in busy]
        if settled:
            j.drop(settled)
        return len(settled)


# ---------------------------------------------------------------------------- self-check

if __name__ == "__main__":
    import functools
    import sys
    import tempfile

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

    # --- rebooking scenario: the classic agent mistake is cancelling first ---
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

    def plan(flight="AI101", to="me@x.com"):
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

    assert [s["id"] for s in order(plan())] == ["find", "hold", "cancel", "mail"]

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
               log=quiet) == {"h": {"flight": "AI101"}}
    reset()

    # 1. no seats -> old booking survives, nothing committed
    e = expect(PlanFailed, run, plan(flight="XX9"), log=quiet)
    assert e.step == "hold" and e.committed == [] and e.not_run == ["cancel", "mail"] and clean()

    # 2. dry-run blocks -> seat released, nothing permanent happened
    e = expect(PlanFailed, run, plan(to="not-an-email"), log=quiet)
    assert e.step == "mail" and e.undone == ["hold", "find"] and clean()

    # 3. human declines; confirm sees only permanent steps, resolved args and previews
    seen = []
    e = expect(PlanFailed, run, plan(), confirm=lambda st: seen.extend(st) or False, log=quiet)
    assert e.step == "confirm" and [s["id"] for s in seen] == ["cancel", "mail"] and clean()
    assert seen[1]["args"]["body"] == "held AI101" and seen[0]["preview"] == "will cancel OLD1"

    # 4. CRITICAL: a permanent step fails after another committed -> the hold it relies on stays
    db["smtp_down"] = True
    e = expect(PlanFailed, run, plan(), log=quiet)
    assert e.step == "mail" and e.committed == ["cancel"] and e.undone == [] and e.not_run == ["mail"]
    assert db["bookings"] == set() and db["holds"] == ["AI101"]  # user still has the new flight
    reset()

    # 5. Ctrl+C mid-run still unwinds, then propagates
    @tool(undo=NOOP)
    def interrupt():
        raise KeyboardInterrupt
    expect(KeyboardInterrupt, run, plan()[1:3] + [{"id": "x", "tool": "interrupt", "after": ["hold"]}], log=quiet)
    assert clean()

    # 6. an undo that keeps failing is retried, reported, and unwinding continues
    calls = []
    @tool(undo=lambda r: calls.append(1) or 1 / 0, undo_retries=1)
    def fragile():
        return "f"
    e = expect(PlanFailed, run, [{"id": "f", "tool": "fragile"}] + plan(to="bad")[1:], log=quiet)
    assert e.undo_failed == ["f"] and "hold" in e.undone and len(calls) == 2 and clean()

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
         {"id": "p", "tool": "publish", "args": {"name": "doc"}, "after": ["c"]}], log=quiet)

    # 8. a bad reference in a permanent step is blamed on that step
    e = expect(PlanFailed, run, [plan()[2], {"id": "mail", "tool": "send_email",
                                             "args": {"to": "a@b", "body": "$hold.nope"}}], log=quiet)
    assert e.step == "mail" and clean()

    # 9. happy path, with structured events
    events = []
    r = run(plan(), log=quiet, on_event=events.append)
    assert db["bookings"] == set() and db["outbox"] == [("me@x.com", "held AI101")]
    assert r["hold"] == {"flight": "AI101"}
    kinds = [ev["event"] for ev in events]
    assert kinds[:2] == ["step_started", "step_done"] and "gate_confirmed" in kinds and kinds[-1] == "step_committed"
    assert all("duration_ms" in ev for ev in events if ev["event"] in ("step_done", "step_committed"))
    reset()

    # --- undo receives the call's arguments when it asks for them ---
    files = {}

    @tool(undo=lambda result, args: files.pop(args["path"]))
    def put_file(path: str, content: str) -> str:
        files[path] = content
        return "ok"
    e = expect(PlanFailed, run, [{"id": "p", "tool": "put_file", "args": {"path": "/a", "content": "x"}},
                                 {"id": "m", "tool": "send_email", "args": {"to": "bad", "body": "x"}}], log=quiet)
    assert files == {} and e.undone == ["p"]

    # --- retries and timeouts ---
    attempts = []

    @tool(undo=NOOP, retries=2)
    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("blip")
        return "ok"
    assert run([{"id": "f", "tool": "flaky"}], log=quiet) == {"f": "ok"} and len(attempts) == 3

    @tool(undo=NOOP, timeout=0.05)
    async def slow():
        await asyncio.sleep(1)
    expect(PlanFailed, run, [{"id": "s", "tool": "slow"}], log=quiet)

    # --- async tools, async undo/check/confirm, isolated registry, parallel waves ---
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

    async def aconfirm(steps):
        return True

    p = [{"id": "c", "tool": "charge", "args": {"x": "$r"}}, {"id": "r", "tool": "reserve", "args": {"x": 3}}]
    assert asyncio.run(arun(p, registry=reg, confirm=aconfirm, log=quiet)) == {"r": 6, "c": None}
    assert log2 == [("charge", 6)] and "reserve" not in TOOLS  # registries don't leak
    log2.clear()
    p[1]["args"]["x"] = -1  # charge(-2) fails its async check -> async undo runs
    expect(PlanFailed, lambda: asyncio.run(arun(p, registry=reg, log=quiet)))
    assert log2 == [("undo", -2)]

    wide = [{"id": f"r{i}", "tool": "reserve", "args": {"x": i}} for i in range(4)]
    live["peak"] = 0
    asyncio.run(arun(wide, registry=reg, log=quiet))
    assert live["peak"] == 1  # sequential by default
    assert asyncio.run(arun(wide, registry=reg, log=quiet, parallel=True)) == {f"r{i}": i * 2 for i in range(4)}
    assert live["peak"] == 4  # all four ran at the same time (counted, not timed)
    log2.clear()
    e = expect(PlanFailed, lambda: asyncio.run(arun(wide + [{"id": "bad", "tool": "reserve", "args": {"x": 99}}],
                                                    registry=reg, log=quiet, parallel=True)))
    assert e.step == "bad" and sorted(e.undone) == ["r0", "r1", "r2", "r3"] and len(log2) == 4

    async def inside_loop():
        return expect(RuntimeError, run, p, registry=reg)
    assert "await arun" in str(asyncio.run(inside_loop()))

    # --- Transaction: guard an agent loop that calls tools one by one ---
    with Transaction(log=quiet) as tx:
        hold = tx.wrap("book_flight")
        cancel = tx.wrap("cancel_booking")
        assert hold.__name__ == "book_flight" and "flight" in str(inspect.signature(hold))
        assert hold("AI101") == {"flight": "AI101"}  # positional args work; runs now
        assert "will cancel OLD1" in cancel(ref="OLD1") and db["bookings"] == {"OLD1"}  # deferred, previewed
        expect(RuntimeError, cancel, ref="NOPE")  # dry run fails now, agent can react
    assert db["bookings"] == set() and db["holds"] == ["AI101"]  # committed on clean exit
    reset()

    tx = Transaction(log=quiet)
    try:
        with tx:
            tx.call("book_flight", flight="AI101")
            tx.call("cancel_booking", ref="OLD1")
            raise RuntimeError("agent crashed")
    except RuntimeError:
        pass
    assert clean() and tx.undone == ["book_flight-1"] and tx.undo_failed == []

    tx = Transaction(confirm=lambda st: False, log=quiet)
    tx.call("book_flight", flight="AI101")
    tx.call("cancel_booking", ref="OLD1")
    e = expect(PlanFailed, tx.commit)
    assert e.step == "confirm" and clean()

    # Transaction: a failing permanent step after a commit keeps the earlier state
    db["smtp_down"] = True
    tx = Transaction(log=quiet)
    tx.call("book_flight", flight="AI101")
    tx.call("cancel_booking", ref="OLD1")
    tx.call("send_email", to="a@b", body="x")
    e = expect(PlanFailed, tx.commit)
    assert e.committed == ["cancel_booking-2"] and e.undone == [] and db["holds"] == ["AI101"]
    reset()

    # wrap(): *args, **kwargs, positional-only, defaults reaching the check
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
    with Transaction(registry=odd, log=quiet) as tx:
        tx.wrap("varargs")(1, 2, 3, y=4)
        assert tx.wrap("length")([1, 2]) == 2
        tx.wrap("mail")("a@b.com")
    assert got[:2] == [(1, (2, 3), {"y": 4}), ("check", "a@b.com", "none")]
    assert run([{"id": "n", "tool": "length", "args": {"obj": [1, 2, 3]}}], registry=odd, log=quiet) == {"n": 3}

    # async framework: sync tools run on its loop thread; async checks get an async wrapper
    async def async_agent():
        async with Transaction(registry=reg, log=quiet) as atx:
            assert inspect.iscoroutinefunction(atx.wrap("charge"))
            assert await atx.wrap("reserve")(5) == 10
            assert "Scheduled" in await atx.acall("charge", x=1)
        sync_tx = Transaction(log=quiet)
        assert sync_tx.wrap("search")("BLR-DEL") == ["AI101"]
        assert sync_tx.call("book_flight", "AI101") == {"flight": "AI101"}
        await sync_tx.arollback()
    log2.clear()
    asyncio.run(async_agent())
    assert log2 == [("charge", 1)] and clean()

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
    tx = Transaction(registry=odd, log=quiet)
    tx.call("step_a", 1)
    tx.call("step_b")
    tx.call("step_a", 2)
    expect(KeyboardInterrupt, tx.rollback)
    assert order_undone == [2, 1] and tx.undo_failed == ["step_b-2"]

    # --- journal and crash recovery ---
    path = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(journal=path, log=quiet)
    tx.call("book_flight", flight="AI101")
    tx.call("search", route="x")
    # ...the process dies here: no rollback, no commit
    assert db["holds"] == ["AI101"]
    report = recover(path, log=quiet)
    assert report[0]["undone"] == ["search-2", "book_flight-1"] and clean()
    assert recover(path, log=quiet) == []  # already finished
    run(plan(), journal=path, log=quiet)  # a finished run leaves nothing to recover
    assert recover(path, log=quiet) == []
    reset()

    # cancelling a parallel run stops the wave first, so no step finishes after the rollback
    trail = []

    @tool(undo=lambda r: trail.append(("undo", r)), registry=reg)
    async def wave_step(n: int):
        await asyncio.sleep(0 if n == 0 else 0.3)
        trail.append(("done", n))
        return n

    async def cancel_midway():
        task = asyncio.ensure_future(arun([{"id": f"w{i}", "tool": "wave_step", "args": {"n": i}} for i in range(3)],
                                          registry=reg, log=quiet, parallel=True))
        await asyncio.sleep(0.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        await asyncio.sleep(0.4)  # an orphaned step would finish in this time
    asyncio.run(cancel_midway())
    assert trail == [("done", 0), ("undo", 0)]

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
    tx = Transaction(registry=odd, log=quiet)
    tx.call("remote", "y")
    assert tx.done[-1]["args"] == {"a": "y"}  # not shown, not journaled

    # journal: only the values that can't be serialized are marked
    assert _jsonable({"a": 1, "o": object, 2: [object]}) == {
        "a": 1, "o": {_MARK: repr(object)}, "2": [{_MARK: repr(object)}]}

    # recovery journals each undo, so it can be re-run without undoing anything twice
    path2 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
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
    tx = Transaction(registry=odd, journal=path2, log=quiet)
    tx.call("part_a")
    tx.call("part_b")
    # ...crash
    first = recover(path2, registry=odd, log=quiet)
    assert first[0]["undone"] == ["part_a-1"] and first[0]["undo_failed"] == ["part_b-2"]
    outage["down"] = False
    second = recover(path2, registry=odd, log=quiet)
    assert second[0]["undone"] == ["part_b-2"] and second[0]["undo_failed"] == []
    assert outage["calls"] == ["b", "a", "b"]  # part_a undone exactly once
    assert recover(path2, registry=odd, log=quiet) == []

    # an in-process rollback whose undo fails leaves the journal open for recover() to retry
    outage.update(down=True, calls=[])
    tx = Transaction(registry=odd, journal=path2, log=quiet)
    tx.call("part_b")
    tx.rollback()
    assert tx.undo_failed == ["part_b-1"]
    outage["down"] = False
    assert recover(path2, registry=odd, log=quiet)[0]["undone"] == ["part_b-1"]

    # recover() also covers a Transaction that committed in stages and crashed in a later stage
    path3 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    outage.update(down=False, calls=[])
    tx = Transaction(registry=odd, journal=path3, log=quiet)
    tx.call("part_a")
    tx.commit()  # stage 1 done: part_a is final
    tx.call("part_b")  # stage 2... crash
    staged = recover(path3, registry=odd, log=quiet)
    assert staged[0]["undone"] == ["part_b-2"] and outage["calls"] == ["b"]

    # a step that cancels itself inside a parallel wave is a failure, not a finished step
    @tool(undo=NOOP, registry=reg)
    async def self_cancel():
        raise asyncio.CancelledError()
    e = expect(PlanFailed, lambda: asyncio.run(arun([{"id": "sc", "tool": "self_cancel"},
                                                    {"id": "r", "tool": "reserve", "args": {"x": 1}}],
                                                   registry=reg, log=quiet, parallel=True)))
    assert e.step == "sc" and "cancelled" in str(e.cause)

    # a sync tool's retries also work through tx.call() inside a running event loop
    tries = []

    @tool(undo=NOOP, retries=2, registry=odd)
    def blip():
        tries.append(1)
        if len(tries) < 2:
            raise ConnectionError("blip")
        return "ok"

    async def in_framework():
        return Transaction(registry=odd, log=quiet).call("blip")
    assert asyncio.run(in_framework()) == "ok" and len(tries) == 2

    # a broken on_event or log callback never changes what runs or gets undone
    def broken(*a):
        raise RuntimeError("logging backend down")
    with captured_warnings() as warned:
        assert run(plan(), on_event=broken, log=broken) == {"find": ["AI101"], "hold": {"flight": "AI101"},
                                                            "cancel": None, "mail": None}
    assert any("on_event callback failed" in w for w in warned) and any("log callback failed" in w for w in warned)
    reset()

    # a real crash (the process dies) in the middle of a step
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))

    def crash_run(dies_in, journal_path):
        child = (
            "import os, sys\n"
            f"sys.path.insert(0, {here!r})\n"
            "from rubicon import tool, run, NOOP\n"
            "reg = {}\n"
            "tool(lambda flight: {'flight': flight}, name='hold', undo=NOOP, registry=reg)\n"
            f"tool(lambda: os._exit(3) if {dies_in!r} == 'probe' else 'x', name='probe', undo=NOOP, registry=reg)\n"
            f"tool(lambda ref: os._exit(3) if {dies_in!r} == 'cancel' else 'ok', name='cancel', registry=reg)\n"
            "run([{'id': 'h', 'tool': 'hold', 'args': {'flight': 'AI101'}},\n"
            "     {'id': 'p', 'tool': 'probe', 'after': ['h']},\n"
            "     {'id': 'c', 'tool': 'cancel', 'args': {'ref': 'OLD1'}}],\n"
            f"    registry=reg, journal={journal_path!r}, log=lambda m: None)\n")
        assert subprocess.run([sys.executable, "-c", child]).returncode == 3

    released = []
    after_crash = {}
    tool(lambda flight: None, name="hold", undo=lambda r: released.append(r), registry=after_crash)
    tool(lambda: None, name="probe", undo=NOOP, registry=after_crash)
    tool(lambda ref: None, name="cancel", registry=after_crash)

    # dies during the permanent cancel: it may have gone through, so the hold must be kept
    path4 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("cancel", path4)
    rep = recover(path4, registry=after_crash, log=quiet)[0]
    assert rep["needs_attention"] == ["c"] and rep["kept"] == ["h", "p"] and released == []
    assert recover(path4, registry=after_crash, log=quiet)[0]["needs_attention"] == ["c"]  # still flagged

    # dies during an undoable step: completed steps are undone, the half-run one is reported
    path5 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("probe", path5)
    rep = recover(path5, registry=after_crash, log=quiet)[0]
    assert rep["undone"] == ["h"] and rep["uncertain"] == ["p"] and released == [{"flight": "AI101"}]
    assert recover(path5, registry=after_crash, log=quiet) == []

    # a failed undo stays owed to recover(): through a second rollback and through later commits
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

    path6 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    try:
        with Transaction(registry=odd, journal=path6, log=quiet) as tx:
            tx.call("take")
            tx.call("break_commit")
            tx.commit()  # fails, its rollback can't undo "take", then __exit__ rolls back again
    except PlanFailed:
        pass
    owed["down"] = False
    assert recover(path6, registry=odd, log=quiet)[0]["undone"] == ["take-1"] and owed["undone"] == ["seat"]

    owed.update(down=True, undone=[])
    path7 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=odd, journal=path7, log=quiet)
    tx.call("take")
    tx.rollback()  # stage 1: undo fails
    tx.call("part_a")
    tx.commit()  # stage 2 commits; part_a is final, "take" is still owed
    owed["down"] = False
    outage.update(calls=[])
    rep = recover(path7, registry=odd, log=quiet)[0]
    assert rep["undone"] == ["take-1"] and owed["undone"] == ["seat"] and outage["calls"] == []
    assert recover(path7, registry=odd, log=quiet) == []

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
                     registry=odd, log=quiet, parallel=True))
    assert loop_state["seen"][1] - loop_state["seen"][0] >= 3  # the ticker kept running during backoff

    # the confirmation resolves what it can, even for a step that waits on another permanent step
    shown2 = []

    @tool(undo=NOOP, registry=odd)
    def make_hold():
        return {"flight": "AI101"}

    @tool(registry=odd)
    def do_cancel():
        return "ok"

    @tool(registry=odd)
    def notify(flight: str, note: str = "", channel: str = "email"):
        return "sent"
    run([{"id": "h", "tool": "make_hold"}, {"id": "c", "tool": "do_cancel"},
         {"id": "n", "tool": "notify", "args": {"flight": "$h.flight", "note": "after $c"}, "after": ["c"]}],
        registry=odd, confirm=lambda st: shown2.extend(st) or True, log=quiet)
    assert shown2[1]["args"] == {"flight": "AI101", "note": "after $c", "channel": "email"}

    # --- journals: SQLite, custom stores, concurrent writers, recovery lock ---
    def crash_with(journal):
        tx = Transaction(registry=odd, journal=journal, log=quiet)
        tx.call("part_a")
        tx.call("make_hold")
        return tx.id  # ...the process dies here

    outage.update(down=False, calls=[])
    db_path = os.path.join(tempfile.mkdtemp(), "rubicon.db")
    tx_id = crash_with(SQLiteJournal(db_path))
    rep = recover(SQLiteJournal(db_path), registry=odd, log=quiet)
    assert rep[0]["tx"] == tx_id and rep[0]["undone"] == ["make_hold-2", "part_a-1"] and outage["calls"] == ["a"]
    assert recover(SQLiteJournal(db_path), registry=odd, log=quiet) == []

    class MemoryJournal:  # what a Redis/Postgres-backed journal needs to provide
        def __init__(self):
            self.rows = []

        def append(self, records):
            self.rows.extend(json.loads(json.dumps(r, default=repr)) for r in records)

        def records(self):
            return list(self.rows)

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

    # --- dead letters ---
    owed.update(down=True, undone=[])
    path8 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=odd, journal=path8, log=quiet)
    tx.call("take")
    tx.rollback()  # undo fails: a dead letter, owed to recover()
    letters = dead_letters(path8)
    assert [(d["step"], d["kind"]) for d in letters] == [("take-1", "undo_failed")]
    notified = []
    recover(path8, registry=odd, log=quiet, on_dead_letter=notified.append)  # still down
    assert notified == []  # already a dead letter: reported once, not on every run
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

    # crash during a permanent step; a person says "it didn't happen, undo the rest"
    path11 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    crash_run("cancel", path11)
    released.clear()
    tx_id = recover(path11, registry=after_crash, log=quiet)[0]["tx"]
    acknowledge(path11, tx_id, "c")
    assert recover(path11, registry=after_crash, log=quiet)[0]["undone"] == ["p", "h"] and len(released) == 1

    # --- compact ---
    for journal in (FileJournal(os.path.join(tempfile.mkdtemp(), "j.jsonl")),
                    SQLiteJournal(os.path.join(tempfile.mkdtemp(), "j.db"))):
        for _ in range(3):
            run([{"id": "a", "tool": "part_a"}], registry=odd, journal=journal, log=quiet)  # settled
        owed.update(down=True)
        stuck = Transaction(registry=odd, journal=journal, log=quiet)
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
    print("all checks passed")
