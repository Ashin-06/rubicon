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
import functools
import heapq
import inspect
import json
import logging
import os
import re
import socket
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
        if timeout and not inspect.iscoroutinefunction(f):
            _logger.warning("tool %r: timeout= applies only to async functions; this one is sync", key)
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
    target = fn
    while isinstance(target, functools.partial):
        target = target.func
    if not inspect.isroutine(target) and callable(getattr(target, "__call__", None)):
        target = target.__call__  # an object with __call__: its hints live on the method
    try:
        hints = typing.get_type_hints(target)
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
        hint = hints.get(p.name, p.annotation if not isinstance(p.annotation, str) else p.empty)
        props[p.name] = _json_schema(hint) if hint is not p.empty else {}
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


_MAX_DEPTH = 100  # arguments nested deeper than this are refused, never a RecursionError


def _walk(v, f, depth=0):
    if isinstance(v, str):
        return f(v)
    if isinstance(v, (dict, list, tuple)) and depth >= _MAX_DEPTH:
        raise ValueError(f"arguments are nested more than {_MAX_DEPTH} levels deep")
    if isinstance(v, dict):
        return {k: _walk(x, f, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_walk(x, f, depth + 1) for x in v]
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
            digits = part[1:] if part.startswith("-") else part
            if not (digits.isascii() and digits.isdigit() and -len(value) <= int(part) < len(value)):
                raise LookupError(f"${sid}{path}: no index {part!r} in a list of {len(value)}")
            value = value[int(part)]
        elif value is None or isinstance(value, (str, int, float, bool)):
            return value, "." + ".".join(parts[i:])  # a plain value: the rest is ordinary text
        else:  # an object: dumping it into the text is never what was meant
            raise LookupError(f"${sid}{path}: {type(value).__name__} is not a dict or list, so it has "
                              f"no {part!r}")
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
    if not args:  # in order: an omitted one takes its default, and a missing required one stops here
        positional, defaults = [], []
        for p in params:
            if p.kind is not p.POSITIONAL_ONLY:
                break
            if p.name in kwargs:
                positional += defaults + [kwargs.pop(p.name)]
                defaults = []
            elif p.default is not p.empty and p.default is not UNSET:
                defaults.append(p.default)
            else:  # required, or UNSET (which can't be passed by position and then left out)
                break  # bind() reports it, along with any later one still given by name
        args = tuple(positional)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    gap = False  # an UNSET positional parameter was left out, so later ones can't go by position
    for name in list(bound.arguments):
        p, value = sig.parameters[name], bound.arguments[name]
        if value is UNSET:  # left out entirely: the function's own default applies
            gap = gap or p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            del bound.arguments[name]
        elif gap and p.kind is p.POSITIONAL_ONLY:  # a default (a given one is rejected above)
            del bound.arguments[name]
        elif gap and p.kind is p.VAR_POSITIONAL:
            if value:
                raise TypeError(f"extra positional arguments can't follow the left-out argument "
                                f"before *{name}")
            del bound.arguments[name]
    view = {}
    for name, value in bound.arguments.items():
        if sig.parameters[name].kind is inspect.Parameter.VAR_KEYWORD:
            view.update(value)
        else:
            view[name] = value
    validate = getattr(fn, "validate_args", None)  # for tools whose names can't be parameters
    if callable(validate):
        validate(bound.kwargs)
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


def _not_driven(r):
    """Inside a running loop, sync code (tx.call) can run only sync functions. Refuse an
    awaitable before any of it runs, so nothing half-happens."""
    if _DRIVING.get():
        if inspect.iscoroutine(r):
            r.close()  # never started: nothing of it ran
        raise RuntimeError(_ASYNC_IN_SYNC)


async def _call(fn, *args, **kwargs):
    r = fn(*args, **kwargs)
    if inspect.isawaitable(r):
        _not_driven(r)
        return await r
    return r


_DRIVING = contextvars.ContextVar("rubicon_driving", default=False)

_HOST = socket.gethostname()
_LIVE: set = set()  # ids of journaled transactions in this process that have steps not yet settled


def _pid_alive(pid) -> bool:
    """Whether a process with this id exists on this machine. Errs towards "alive" when unsure,
    because treating a live transaction as crashed would undo work that is still in use."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # declare the real signatures: a HANDLE is 64-bit and would be truncated as a default int
        kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        # SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION: a process that exited with code 259
        # (STILL_ACTIVE's value) would look alive by its exit code, so wait on it instead
        handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)
        if not handle:
            return ctypes.get_last_error() == 5  # access denied: it exists
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # Linux: a zombie (exited, not yet reaped by its parent) still answers kill(pid, 0)
        with open(f"/proc/{pid}/stat", "rb") as f:
            state = f.read().rsplit(b")", 1)[-1].split()[:1]
        return state not in ([b"Z"], [b"X"])
    except OSError:
        return True


async def _attempt(fn, args, kwargs, retries, timeout):
    """Call with optional timeout (async only) and retries with exponential backoff."""
    for n in range(retries + 1):
        is_async = False
        try:
            r = fn(*args, **kwargs)
            if inspect.isawaitable(r):
                _not_driven(r)
                is_async = True
                r = await (asyncio.wait_for(r, timeout) if timeout else r)
            return r
        except Exception as e:
            if n == retries or (isinstance(e, RuntimeError) and e.args == (_ASYNC_IN_SYNC,)):
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
    if any(p.kind is p.VAR_POSITIONAL for p in params):
        return True
    # undo(result, args); a second parameter with a default (undo(result, force=False)) is an
    # option of its own, unless it is named args
    return len(positional) >= 2 and (positional[1].default is positional[1].empty
                                     or positional[1].name == "args")


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
    raise RuntimeError(_ASYNC_IN_SYNC)


_ASYNC_IN_SYNC = ("an async tool, check, undo or confirm can't run from sync code inside a running "
                  "event loop; use the async API (await arun(...), await tx.acall(...), async with "
                  "Transaction())")


_MARK = "__unserializable__"


def _complain(message):
    try:
        _logger.warning(message)
    except Exception:
        pass  # a broken logging setup must never break execution; there is nowhere left to report it


def _mark(value):
    try:
        text = repr(value)
    except Exception:  # a broken or self-referencing __repr__
        text = f"<{type(value).__name__}>"
    return {_MARK: text[:1000]}


def _jsonable(value, depth=0):
    """A JSON-safe copy: only the parts that can't be serialized are replaced by a marker.
    Cycles and very deep nesting are marked too, never a crash."""
    if depth == 0:
        try:
            json.dumps(value)
            return value  # the common case: already JSON, one pass
        except (TypeError, ValueError, RecursionError):
            pass  # something inside isn't: mark just those parts below
    if depth > 100:
        return _mark(value)
    if isinstance(value, dict):
        return {k if isinstance(k, str) else str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v, depth + 1) for v in value]
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError, RecursionError):
        return _mark(value)


def _has_mark(value) -> bool:
    stack = [value]  # a loop, not recursion: a journaled value may be nested thousands deep
    while stack:
        v = stack.pop()
        if isinstance(v, dict):
            if _MARK in v:
                return True
            stack.extend(v.values())
        elif isinstance(v, list):
            stack.extend(v)
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
        if not all(isinstance(key, str) for key in args):
            raise ValueError(f"step {sid!r}: argument names must be strings, got {list(args)[:10]!r}")
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
    try:
        deps = {sid: _deps(s, by_id) for sid, s in by_id.items()}
    except ValueError as e:
        raise ValueError(f"invalid arguments: {e}") from None
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
        for i in ts.get_ready():
            heapq.heappush(ready, (not tools[by_id[i]["tool"]].reversible, pos[i], i))
        i = heapq.heappop(ready)[2]
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
        self._began = self._dirty = False
        self._running = set()  # labels of steps currently executing
        self._rollbacks = 0  # a step that finishes after a rollback started undoes itself

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
        # every record says who runs this transaction, so recover() elsewhere can tell it is
        # still alive even after compact() removed its earlier records
        who = {"pid": os.getpid(), "host": _HOST}
        records = [{"tx": self.id, "event": event, "time": now, **data, **who}]
        opens = event in ("started", "done", "committed")
        with self._lock:
            if not self._began:
                records.insert(0, {"tx": self.id, "event": "begin", "time": now, **who})
            was_live = self.id in _LIVE
            if opens:  # live before the record is durable: a recover() in this process that
                _LIVE.add(self.id)  # reads it must already see the transaction as running
            try:
                self._journal.append(records)
            except BaseException:
                if opens and not was_live:
                    _LIVE.discard(self.id)
                raise
            self._began = True  # state changes only once the write succeeded
            if opens:
                self._dirty = True

    def _settle(self, status):
        """Journal that everything so far is committed or undone. While an earlier undo is still
        owed (tx.undo_failed), write "settled" instead of "finished" so recover() retries it.
        Writes nothing if no step ran since the last settle (no empty or duplicate records)."""
        if self._dirty:
            self._write("finished" if not self.undo_failed else "settled", status=status)
            self._dirty = False
            # a later stage starts afresh with its own "begin" (pid and host): once settled, its
            # records can be compacted away (after a person acknowledges what was owed, too)
            self._began = False
        _LIVE.discard(self.id)  # only now: until the record above is durable, it is still running

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
        self._running.add(label)  # before "started": a settle elsewhere must see it as running
        try:
            # write-ahead: if the process dies mid-step, recover() knows this step may have happened
            self._write("started", step=label, tool=name, permanent=permanent)
        except BaseException:
            self._running.discard(label)
            raise
        self._emit("step_started", step=label, tool=name, permanent=permanent)
        rollbacks, start = self._rollbacks, time.perf_counter()
        try:
            r = await _attempt(t.fn, a, k, t.retries, t.timeout)
        except BaseException as e:
            self._running.discard(label)
            try:
                self._write("step_failed", step=label, tool=name, error=repr(e))
            except Exception as journal_error:  # the step's own error is what the caller must see
                _complain(f"could not journal the failure of {label}: {journal_error!r}")
            self._emit("step_failed", step=label, tool=name, error=repr(e))
            raise
        self._running.discard(label)
        ms = round((time.perf_counter() - start) * 1000, 1)
        self.results[label] = r
        if not permanent and self._rollbacks != rollbacks:
            # the transaction was rolled back while this step ran (another thread): undo it too
            self.done.append({"id": label, "tool": name, "args": view, "result": r})
            if self._journal:
                self._write("done", step=label, tool=name, args=_jsonable(view), result=_jsonable(r))
            await self.arollback()
            raise RuntimeError(f"the transaction was rolled back while {label} ran; it was undone")
        if permanent:
            self.committed.append(label)
            self.done.clear()  # everything before a committed step is now part of the result
            self._write("committed", step=label, tool=name)
            self._emit("step_committed", step=label, tool=name, duration_ms=ms)
            self._say(f"COMMITTED {label}")
        else:
            self.done.append({"id": label, "tool": name, "args": view, "result": r})
            if self._journal:  # no copy to make without one
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
            try:
                return await self._run(label, name, a, k, view, permanent=False)
            except BaseException:
                # an agent call is the one path with no rollback after a failure: if nothing is
                # outstanding, settle, so the journal doesn't show the transaction as running
                if not (self.done or self.pending or self._running):
                    try:
                        self._settle("step_failed")
                    except BaseException:
                        _LIVE.discard(self.id)
                raise
        preview = await self._check(label, name, a, k, view)
        self.pending.append({"id": label, "tool": name, "args": view, "preview": preview, "call": (a, k)})
        note = f" Preview: {preview}" if preview is not None else ""
        return (f"Scheduled {name} ({label}): it is a permanent action and will run after "
                f"confirmation when the transaction commits.{note}")

    async def arollback(self):
        """Undo every undoable call since the last commit, in reverse order, and drop pending
        permanent steps. Keeps going if an undo fails or is interrupted; an interruption is
        re-raised once everything has been tried."""
        self._rollbacks += 1
        undone, failed, interrupted = [], [], None

        def journal(event, **data):
            try:
                self._write(event, **data)
            except Exception as e:  # keep unwinding: the undos matter more than their record
                _complain(f"could not journal {event} of {data.get('step')}: {e!r}; recover() may "
                          f"repeat that undo (undos must be idempotent)")

        while self.done:
            rec = self.done.pop()
            try:
                await self._undo(rec)
            except BaseException as e:  # noqa: B036 - an interruption must not strand the rest
                failed.append(rec["id"])
                journal("undo_failed", step=rec["id"], error=repr(e))
                self._emit("undo_failed", step=rec["id"], tool=rec["tool"], error=repr(e))
                self._say(f"undo of {rec['id']} failed: {e!r}")
                if not isinstance(e, Exception) and interrupted is None:
                    interrupted = e
                continue
            undone.append(rec["id"])
            journal("undone", step=rec["id"])
            self._emit("step_undone", step=rec["id"], tool=rec["tool"])
        self.pending.clear()
        self.undone += undone
        self.undo_failed += failed
        try:
            self._settle("rolled_back")
        except Exception as e:  # still running as far as the journal knows: recover() skips it here
            _complain(f"could not journal the end of the rollback of {self.id}: {e!r}")
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
        return _sync(self.acommit(), drive=True)

    def rollback(self):
        return _sync(self.arollback(), drive=True)

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
        failures = []
        for task in sorted(done, key=lambda t: tasks[t]):  # collect every outcome, so none is left unread
            if task.cancelled():  # cancelled from inside (e.g. a library's cancel scope): not finished
                failures.append((tasks[task], RuntimeError(f"step {tasks[task]!r} was cancelled before it finished")))
            elif task.exception() is not None:
                failures.append((tasks[task], task.exception()))
        if failures:
            raise _StepFailed(*failures[0])
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


def run(plan, registry=None, confirm=None, log=None, *, parallel=False, on_event=None,
        journal=None) -> dict:
    """Sync version of arun(). Inside an event loop, use `await arun(...)` instead."""
    return _sync(arun(plan, registry, confirm, log, parallel=parallel, on_event=on_event, journal=journal))


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
            try:  # a line torn by a power cut or full disk: end it, or this record joins it
                with open(self.path, "rb") as f:
                    f.seek(-1, os.SEEK_END)
                    if f.read(1) != b"\n":
                        data = "\n" + data
            except OSError:
                pass  # no file yet, or empty: nothing to end
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())

    def records(self):
        # read under the write lock: never half of a line being written, and on Windows the file
        # is never open here while drop() replaces it
        with _FileLock(self.path + ".write.lock", blocking=True):
            return self._read()

    def _read(self):
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

    def drop(self, tx_ids, upto=None):
        """Remove the records of these transactions (atomic rename; writers wait meanwhile).
        With upto, only among the first `upto` records: ones written since compact() read the
        journal belong to a new stage and stay."""
        tx_ids = set(tx_ids)
        with _FileLock(self.path + ".write.lock", blocking=True):
            records = self._read()
            upto = len(records) if upto is None else upto
            keep = [r for i, r in enumerate(records) if i >= upto or r.get("tx") not in tx_ids]
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

    def drop(self, tx_ids, upto=None):
        """Remove the records of these transactions; with upto, only among the first `upto`
        records (ones written since compact() read the journal stay)."""
        tx_ids = list(tx_ids)
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")  # all chunks or none
            try:
                last = None
                if upto is not None:  # the id of the last record compact() saw
                    row = db.execute("SELECT id FROM rubicon_journal ORDER BY id LIMIT 1 OFFSET ?",
                                     (max(upto - 1, 0),)).fetchone()
                    last = row[0] if row and upto > 0 else -1
                for i in range(0, len(tx_ids), 500):
                    chunk = tx_ids[i:i + 500]
                    # only "?" placeholders are formatted into the SQL; the ids are bound parameters
                    where = f"tx IN ({','.join('?' * len(chunk))})" + ("" if last is None else " AND id <= ?")
                    db.execute(f"DELETE FROM rubicon_journal WHERE {where}",
                               chunk + ([] if last is None else [last]))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise


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
        tx = txs.setdefault(tx_id, {"id": tx_id, "done": {}, "owed": {}, "committed": [], "open": {},
                                    "finished": False, "settled": False, "begin": None})
        if event in ("started", "done", "committed"):
            # new activity reopens a transaction that committed in stages
            tx["finished"] = tx["settled"] = False
        if "pid" in rec:  # written by the Transaction itself: who runs it, most recent wins
            tx["begin"] = rec
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
            # not "settled": a live Transaction may hold permanent steps that are only journaled
            # at commit. The Transaction settles itself when nothing is outstanding.
            tx["open"].pop(step, None)
        elif event == "undo_failed":
            tx["open"].pop(step, None)
            if step in tx["done"]:
                tx["owed"][step] = tx["done"].pop(step)
            old = letters.get((tx_id, step), {})  # a retry that failed again is the same letter
            letters[(tx_id, step)] = {"tx": tx_id, "step": step, "kind": "undo_failed",
                                      "reason": rec.get("error"), "time": old.get("time", rec.get("time")),
                                      **({"_notified": True} if old.get("_notified") else {})}
        elif event == "notified":  # passed to an on_dead_letter callback
            if (tx_id, step) in letters:
                letters[(tx_id, step)]["_notified"] = True
        elif event == "dead_letter":
            letters[(tx_id, step)] = {"tx": tx_id, "step": step, "kind": rec.get("kind"),
                                      "reason": rec.get("reason"), "time": rec.get("time")}
            # only recover() writes these, and only for a transaction it found not running
            tx["settled"] = True
            if rec.get("kind") == "uncertain":  # reported; nothing more recover() can do about it
                tx["open"].pop(step, None)
        elif event == "undone":
            tx["open"].pop(step, None)
            tx["done"].pop(step, None)
            tx["owed"].pop(step, None)
            letters.pop((tx_id, step), None)
        elif event == "resolved":
            # never marks the transaction as not running: a person resolving an old dead letter
            # says nothing about a later stage that may be running right now
            if step:
                for part in ("open", "done", "owed"):
                    tx[part].pop(step, None)
                letters.pop((tx_id, step), None)
            elif "steps" in rec:
                # the steps a person handled as they were: settled. Undos owed from an earlier
                # rollback are separate debts, and a later stage was not part of what they saw
                for s in rec["steps"]:
                    if s not in tx["owed"]:
                        tx["open"].pop(s, None)
                        tx["done"].pop(s, None)
                        letters.pop((tx_id, s), None)
                if not (tx["open"] or tx["done"]):
                    tx["settled"] = True
            else:  # written before 0.8.6: everything kept or in flight at that point
                tx["open"].clear()
                tx["done"].clear()
                tx["settled"] = True
                for key in [k for k in letters if k[0] == tx_id and k[1] not in tx["owed"]]:
                    del letters[key]
        elif event in ("settled", "finished"):  # everything before this is committed or undone
            tx["done"].clear()
            tx["open"].clear()
            tx["finished"] = event == "finished"
            tx["settled"] = True
    for tx in txs.values():  # settled, and a person resolved whatever was still owed
        if tx["settled"] and not (tx["open"] or tx["done"] or tx["owed"]):
            tx["finished"] = True
    return txs, letters


def _maybe_running(tx) -> bool:
    """Whether a transaction may still be running, so recover() must leave it alone."""
    if tx["id"] in _LIVE:
        return True  # running in this process, whatever the journal says
    if tx["settled"]:
        return False  # settled and not reopened: only an owed undo can be left, and nothing runs
    begin = tx["begin"]
    if not begin or "pid" not in begin:
        return False  # written before liveness was recorded
    if begin.get("host") != _HOST:
        return True  # another machine: can't check, so don't assume it died
    if begin["pid"] == os.getpid():
        return begin["tx"] in _LIVE
    return _pid_alive(begin["pid"])


def _record(event, tx, **data):
    return {"tx": tx, "event": event, "time": time.time(), **data}


async def arecover(journal, registry=None, log=None, *, on_dead_letter=None, include_live=False) -> list[dict]:
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

    Transactions that may still be running are left alone: their process is alive on this
    machine, or they belong to another machine (which can't be checked from here). So recover()
    is safe to run while other workers use the same journal. Run it on each machine, or pass
    include_live=True when you know those processes are gone.

    Returns one report per transaction:
    {"tx", "undone", "undo_failed", "committed", "uncertain", "needs_attention", "kept"}.
    """
    tools = TOOLS if registry is None else registry
    j = _as_journal(journal)
    say = log or _logger.warning
    def notify(letter):
        # Each dead letter reaches the callback once, whoever recorded it (a rollback in a
        # Transaction or run() records its own). Called before "notified" is written, so a crash
        # in between means a second call, never a missed one.
        if not on_dead_letter or letter.get("_notified"):
            return
        letter["_notified"] = True
        try:
            on_dead_letter({k: v for k, v in letter.items() if k != "_notified"})
        except Exception as e:
            _complain(f"on_dead_letter callback failed: {e!r}")
        j.append([_record("notified", letter["tx"], step=letter["step"])])

    with _lock_of(j):
        txs, letters = _replay(j.records())
        if not include_live:
            # A transaction of this process is judged by _LIVE, which may have changed since the
            # read: one that settled meanwhile left _LIVE only after its closing record was
            # durable, so a second read now shows it. Judge those by the second read.
            mine = [t for t, tx in txs.items() if not (tx["finished"] and not tx["owed"])
                    and (tx["begin"] or {}).get("pid") == os.getpid()
                    and (tx["begin"] or {}).get("host") == _HOST and not _maybe_running(tx)]
            if mine:
                fresh, fresh_letters = _replay(j.records())
                for t in mine:
                    if t in fresh:
                        txs[t] = fresh[t]
                    else:
                        del txs[t]  # compacted away meanwhile: settled
                letters.update({key: v for key, v in fresh_letters.items() if key[0] in mine})
        for letter in sorted(letters.values(), key=lambda letter: letter.get("time") or 0):
            notify(letter)
        reports = []
        for tx_id, tx in txs.items():
            if tx["finished"] and not tx["owed"]:
                continue
            if not include_live and _maybe_running(tx):
                _logger.info("recover: skipping transaction %s; it may still be running (pid %s on %s)",
                             tx_id, tx["begin"].get("pid"), tx["begin"].get("host"))
                continue

            def dead_letter(step, kind, reason, write=True):
                if (tx_id, step) in letters:
                    return  # already recorded and reported on an earlier run
                letter = {"tx": tx_id, "step": step, "kind": kind, "reason": reason, "time": time.time()}
                if write:
                    j.append([_record("dead_letter", tx_id, step=step, kind=kind, reason=reason)])
                letters[(tx_id, step)] = letter
                notify(letter)

            in_flight = [s for s, r in tx["open"].items() if r.get("permanent")]
            uncertain = [s for s, r in tx["open"].items() if not r.get("permanent")]
            newest_first = lambda recs: sorted(recs, key=lambda r: r["_pos"], reverse=True)
            if not (in_flight or uncertain or tx["done"] or tx["owed"]):  # nothing to do: close quietly
                j.append([_record("finished", tx_id, status="recovered")])
                continue
            report = {"tx": tx_id, "undone": [], "undo_failed": [], "committed": tx["committed"],
                      "uncertain": uncertain, "needs_attention": in_flight, "kept": []}

            async def undo_all(recs):
                for rec in recs:
                    t = tools.get(rec["tool"])
                    result, args = rec.get("result"), rec.get("args", {})
                    try:
                        if t is None or not t.reversible:
                            raise LookupError(f"tool {rec['tool']!r} is not registered as undoable")
                        with_args = _takes_args(t.undo)
                        if t.undo is not NOOP and (_has_mark(result) or (with_args and _has_mark(args))):
                            raise ValueError("the step's result or arguments weren't JSON-serializable, "
                                             "so they can't be replayed; undo it by hand")
                        await _attempt(t.undo, (result, args) if with_args else (result,), {},
                                       t.undo_retries, t.timeout)
                    except Exception as e:
                        report["undo_failed"].append(rec["step"])
                        j.append([_record("undo_failed", tx_id, step=rec["step"], error=repr(e))])
                        dead_letter(rec["step"], "undo_failed", repr(e), write=False)
                        say(f"recover: undo of {rec['step']} failed: {e!r}")
                        continue
                    report["undone"].append(rec["step"])
                    # if this write fails, the error propagates and the next recover() repeats
                    # the undo: never reported as failed when it happened
                    j.append([_record("undone", tx_id, step=rec["step"], by="recover")])
                    letters.pop((tx_id, rec["step"]), None)

            if in_flight:
                # keep what the in-flight permanent step may rely on: the steps since the last
                # commit. Undos owed from an earlier rollback don't depend on it: retry them.
                report["kept"] = [r["step"] for r in sorted(tx["done"].values(), key=lambda r: r["_pos"])]
                say(f"recover: transaction {tx_id} stopped during permanent step(s) {in_flight}; it may "
                    f"or may not have happened, so {report['kept']} were kept. Check by hand, then "
                    f"call acknowledge().")
                for s in in_flight:
                    dead_letter(s, "in_flight_permanent", "the process stopped while this permanent step "
                                "was running; it may or may not have happened")
                await undo_all(newest_first(tx["owed"].values()))
                reports.append(report)
                continue
            for s in uncertain:
                dead_letter(s, "uncertain", "the process stopped while this undoable step was running; "
                            "it may have happened and can't be undone automatically")
            if uncertain:
                say(f"recover: step(s) {uncertain} of {tx_id} started but never reported a result; "
                    f"they may have happened and can't be undone automatically.")
            await undo_all(newest_first([*tx["done"].values(), *tx["owed"].values()]))
            # A transaction that was already settled only had owed undos to retry; its process may be
            # alive and may have started a new stage meanwhile, which a closing record written now
            # would wipe. Each "undone" above is enough: it is finished once nothing is owed.
            if not tx["settled"]:
                # "settled" when an undo is still owed: recovery handled it, nothing runs, retry later
                j.append([_record("settled" if report["undo_failed"] else "finished", tx_id,
                                  status="recovered")])
            reports.append(report)
        return reports


def recover(journal, registry=None, log=None, *, on_dead_letter=None, include_live=False) -> list[dict]:
    """Sync version of arecover()."""
    return _sync(arecover(journal, registry, log, on_dead_letter=on_dead_letter, include_live=include_live))


def dead_letters(journal) -> list[dict]:
    """Steps that need a person, oldest first, each {"tx", "step", "kind", "reason", "time"}.

    kind is "undo_failed" (an undo kept failing), "uncertain" (a crash during an undoable step)
    or "in_flight_permanent" (a crash during a permanent step). Listed until a later successful
    undo or acknowledge() resolves them. Run recover() first to record crash-related ones.
    """
    _, letters = _replay(_as_journal(journal).records())
    public = [{k: v for k, v in letter.items() if k != "_notified"} for letter in letters.values()]
    return sorted(public, key=lambda letter: letter.get("time") or 0)


def acknowledge(journal, tx, step=None, note=""):
    """Record that a person handled a dead letter, so recover() and dead_letters() stop reporting it.

    With a step: that step is settled as it is (Rubicon won't undo it). Without a step: the whole
    transaction is settled, including any steps recover() was keeping. Raises ValueError for a
    transaction or step the journal doesn't contain, so a typo can't leave a dead letter open.
    """
    j = _as_journal(journal)
    records = [r for r in j.records() if r.get("tx") == tx]
    if not records:
        raise ValueError(f"no transaction {tx!r} in {j!r}")
    if step is not None and not any(r.get("step") == step for r in records):
        raise ValueError(f"transaction {tx!r} has no step {step!r}")
    if step is not None:
        j.append([_record("resolved", tx, step=step, note=note)])
        return
    # What recover() kept or found in flight when it reported the crash: the steps before its
    # last crash report, plus the reported steps. Anything after that is a stage the worker
    # started later, which the person never checked, so it stays (named, never implied).
    state, letters = _replay(records)
    state = state[tx]
    crash = max((i for i, r in enumerate(records) if r.get("event") == "dead_letter"
                 and r.get("kind") in ("in_flight_permanent", "uncertain")), default=-1)
    steps = sorted({*(s for s, r in {**state["open"], **state["done"]}.items() if r["_pos"] < crash),
                    *(s for (t, s) in letters if t == tx and s not in state["owed"])})
    j.append([_record("resolved", tx, step=None, steps=steps, note=note)])


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
        records = j.records()
        txs, letters = _replay(records)
        busy = {key[0] for key in letters}
        settled = [t for t, tx in txs.items() if tx["finished"] and not tx["owed"] and t not in busy]
        if settled:
            # a settled Transaction may start its next stage while this runs: drop only the
            # records read above, so that stage's new records stay
            try:
                takes_upto = "upto" in inspect.signature(j.drop).parameters
            except (TypeError, ValueError):
                takes_upto = False
            if takes_upto:
                j.drop(settled, upto=len(records))
            else:
                j.drop(settled)  # a custom journal: see the README on concurrent compaction
        return len(settled)


# ---------------------------------------------------------------------------- self-check

if __name__ == "__main__":
    import datetime
    import sys
    import tempfile

    import atexit

    previous_tempdir = tempfile.tempdir
    scratch = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    atexit.register(scratch.cleanup)  # removed at exit, also if a check fails
    tempfile.tempdir = scratch.name  # every mkdtemp() below lands inside it

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
    assert db["bookings"] == set() and db["outbox"] == [("me@example.com", "held AI101")]
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
        pass  # expected: the simulated agent crash; the rollback is checked below
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
        tx.wrap("mail")("a@example.com")
    assert got[:2] == [(1, (2, 3), {"y": 4}), ("check", "a@example.com", "none")]
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
    crashed(tx)  # the process dies here: no rollback, no commit
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
    crashed(tx)
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
    tx.call("part_b")  # stage 2...
    crashed(tx)
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
        pass  # expected: what matters is that the owed undo survives, checked below
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
        crashed(tx)
        return tx.id

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

    # --- recover() never touches a transaction that may still be running ---
    path12 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    outage.update(down=False, calls=[])
    live_tx = Transaction(registry=odd, journal=path12, log=quiet)
    live_tx.call("part_a")  # still running in this process
    assert recover(path12, registry=odd, log=quiet) == [] and outage["calls"] == []
    crashed(live_tx)
    assert recover(path12, registry=odd, log=quiet)[0]["undone"] == ["part_a-1"]

    # ...in another process that is alive, then killed
    path13 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    child = (f"import sys, time; sys.path.insert(0, {here!r})\n"
             "from rubicon import Transaction, tool, NOOP\n"
             "reg = {}\n"
             "tool(lambda: 'a', name='part_a', undo=NOOP, registry=reg)\n"
             f"tx = Transaction(registry=reg, journal={path13!r})\n"
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

    # ...on another machine (can't be checked from here) unless told otherwise
    elsewhere = MemoryJournal()
    elsewhere.append([{"tx": "t9", "event": "begin", "pid": 1, "host": "another-machine"},
                      {"tx": "t9", "event": "started", "step": "part_a-1", "tool": "part_a", "permanent": False},
                      {"tx": "t9", "event": "done", "step": "part_a-1", "tool": "part_a", "args": {}, "result": "a"}])
    assert recover(elsewhere, registry=odd, log=quiet) == []
    assert recover(elsewhere, registry=odd, log=quiet, include_live=True)[0]["undone"] == ["part_a-1"]

    # ...but a settled transaction that owes an undo is retried even while its worker lives on
    path18 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    child = (f"import sys, time; sys.path.insert(0, {here!r})\n"
             "from rubicon import Transaction, tool\n"
             "reg = {}\n"
             "def release(r):\n"
             "    raise ConnectionError('service down')\n"
             "tool(lambda: 'seat', name='take', undo=release, undo_retries=0, registry=reg)\n"
             f"tx = Transaction(registry=reg, journal={path18!r}, log=lambda m: None)\n"
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

    # a standalone call whose first step fails leaves nothing marked live
    lone = Transaction(journal=os.path.join(tempfile.mkdtemp(), "j.jsonl"), log=quiet)
    expect(RuntimeError, lone.call, "book_flight", flight="XX9")
    assert lone.id not in _LIVE

    # liveness: running, killed, and (Linux) zombie processes
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    assert _pid_alive(sleeper.pid)
    sleeper.kill()
    if sys.platform.startswith("linux"):
        time.sleep(0.5)  # exited but not reaped yet: a zombie
        assert not _pid_alive(sleeper.pid)
    sleeper.wait()
    assert not _pid_alive(sleeper.pid) and _pid_alive(os.getpid()) and not _pid_alive(0)

    # --- maintenance while a staged transaction runs never touches the running stage ---
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

    # compact() between stages, then a live stage 2
    path20 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path20, log=quiet)
    tx.call("sa")
    tx.commit()
    assert compact(path20) == 1  # stage 1 removed, "begin" included
    tx.call("sb")
    assert [r["event"] for r in FileJournal(path20).records()][0] == "begin"  # stage 2 has its own
    with from_other_process(path20) as other:
        assert recover(other, registry=stage, log=quiet) == [] and stage_undone == []
    assert recover(path20, registry=stage, log=quiet) == []  # and in this process
    tx.rollback()

    # acknowledging stage 1's failed undo while stage 2 runs
    path21 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path21, log=quiet)
    tx.call("sc")
    tx.rollback()  # undo fails: owed
    tx.call("sb")  # stage 2, running
    acknowledge(path21, tx.id, "sc-1")
    stage_undone.clear()
    with from_other_process(path21) as other:
        assert recover(other, registry=stage, log=quiet) == [] and stage_undone == []
    tx.rollback()

    # a stage whose rollback owed an undo, acknowledged and compacted by another worker,
    # followed by a new stage (found by the randomised tests, seed 3079722)
    path24 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path24, log=quiet)
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

    # a lone failed first step settles itself, so it can be compacted
    path22 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    lone = Transaction(registry=stage, journal=path22, log=quiet)
    expect(ZeroDivisionError, lone.call, "sbroken")
    assert compact(path22) == 1

    # ...but not while a permanent step is pending: the journal doesn't know about it yet
    path23 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path23, log=quiet)
    tx.call("sp")  # scheduled, not journaled until commit
    expect(ZeroDivisionError, tx.call, "sbroken")
    with from_other_process(path23) as other:
        compact(other)  # must not remove the running transaction's "begin"
    tx.call("sb")
    stage_undone.clear()
    with from_other_process(path23) as other:
        assert recover(other, registry=stage, log=quiet) == [] and stage_undone == []
    tx.rollback()

    # ...nor when an earlier stage's owed undo is acknowledged and compacted: the records left
    # must still say who runs the transaction
    path25 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=stage, journal=path25, log=quiet)
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

    # a failed permanent step after an earlier commit leaves one settle, from the plan's end
    path28 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    seq = {}
    tool(lambda: "1", name="q1", registry=seq)
    tool(lambda: 1 / 0, name="q2", registry=seq)
    expect(PlanFailed, run, [{"id": "a", "tool": "q1"}, {"id": "b", "tool": "q2", "after": ["a"]}],
           registry=seq, journal=path28, log=quiet)
    settles = [r.get("status") for r in FileJournal(path28).records() if r["event"] in ("finished", "settled")]
    assert "step_failed" not in settles, settles

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
    tx = Transaction(registry=racing, journal=path29, log=quiet)
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

    # an undo journaled for a step whose "done" was never written closes that step too
    for outcome in ("undone", "undo_failed"):
        torn = _replay([{"tx": "T", "event": "started", "step": "s", "tool": "sa", "permanent": False},
                        {"tx": "T", "event": outcome, "step": "s"}])[0]["T"]
        assert not torn["open"], torn

    # a read-only step whose result can't be stored needs nothing undone after a crash
    dated = {}
    tool(lambda: datetime.date(2026, 1, 1), name="today", undo=NOOP, registry=dated)
    path30 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=dated, journal=path30, log=quiet)
    tx.call("today")
    crashed(tx)
    rep = recover(path30, registry=dated, log=quiet)
    assert rep[0]["undone"] == ["today-1"] and not rep[0]["undo_failed"] and not dead_letters(path30), rep

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
    assert run([{"id": "s", "tool": "unset_gap", "args": {"a": "1"}}], registry=gap, log=quiet) == {"s": "1"}

    # list indexes: anything but plain ASCII digits is a bad reference, not a crash
    for part in ("--0", chr(0xB2), "-"):  # chr(0xB2) is a superscript two
        expect(LookupError, _lookup, {"s": [1, 2]}, "s", "." + part)
    assert _lookup({"s": [1, 2]}, "s", ".-1") == (2, "")

    # --- settle records: one per settle, none for an empty transaction ---
    path14 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    with Transaction(registry=odd, journal=path14, log=quiet) as tx:
        tx.call("part_a")
        tx.commit()  # explicit commit inside the block; leaving it must not write again
    events = [r["event"] for r in FileJournal(path14).records()]
    assert events.count("finished") == 1 and events[0] == "begin"
    with Transaction(registry=odd, journal=path14, log=quiet):
        pass
    assert len(FileJournal(path14).records()) == len(events)  # nothing written

    # --- acknowledge() rejects ids the journal doesn't have ---
    expect(ValueError, acknowledge, path14, "no-such-tx")
    expect(ValueError, acknowledge, path14, FileJournal(path14).records()[0]["tx"], "no-such-step")

    # --- compact() removes a transaction once its last dead letter is acknowledged ---
    path15 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    owed.update(down=True)
    tx = Transaction(registry=odd, journal=path15, log=quiet)
    tx.call("take")
    tx.rollback()  # undo fails: owed, a dead letter
    acknowledge(path15, tx.id, "take-1", note="released by hand")
    assert compact(path15) == 1 and FileJournal(path15).records() == []
    owed.update(down=False)

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

    # ---------------------------------------------------------------- independent audits, 0.8.6
    world6 = set()
    reg6 = {}
    tool(lambda k: world6.add(k) or k, name="put", undo=lambda r: world6.discard(r), registry=reg6)
    tool(lambda: "p", name="perm", registry=reg6)

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
        tx = Transaction(registry=reg6, journal=racy, log=quiet)
        tx.call("put", "a")
        tx.commit()
        racy.hook = lambda: tx.call("put", "b")
        assert compact(racy) == 1
        assert [r["event"] for r in racy.records() if r["tx"] == tx.id] == ["begin", "started", "done"]
        crashed(tx)
        recover(racy, registry=reg6, log=quiet)
        assert world6 == {"a"}, (kind, world6)
        world6.clear()

    # a line torn by a power cut is ended, so the next record is not lost with it
    path31 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    FileJournal(path31).append([{"tx": "T", "event": "begin"}])
    with open(path31, "a", encoding="utf-8") as f:
        f.write('{"tx": "other", "event": "done", "resu')
    FileJournal(path31).append([{"tx": "T", "event": "started", "step": "p-1", "tool": "perm", "permanent": True}])
    assert [r["event"] for r in FileJournal(path31).records()] == ["begin", "started"]

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
        tx = Transaction(registry=reg6, journal=watched, log=quiet)
        watched.when = moment if moment == "started" else None
        tx.call("put", "seat")
        watched.when = moment if moment == "finished" else None
        tx.commit()
        assert watched.found == [] and world6 == {"seat"} and not dead_letters(watched), (moment, watched.found)
        world6.clear()

    # acknowledging a whole transaction never settles a stage it started afterwards
    flaky_undo_down = {"down": True}

    def flaky_undo(r):
        if flaky_undo_down["down"]:
            raise RuntimeError("down")
        world6.discard(r)

    reg7 = {}
    tool(lambda k: world6.add(k) or k, name="put", undo=flaky_undo, undo_retries=0, registry=reg7)
    path32 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    tx = Transaction(registry=reg7, journal=path32, log=quiet)
    tx.call("put", "a")
    tx.rollback()  # owed
    tx.call("put", "b")  # the next stage
    acknowledge(path32, tx.id, note="looked at it")
    crashed(tx)
    flaky_undo_down["down"] = False
    recover(path32, registry=reg7, log=quiet)
    assert world6 == set(), world6
    world6.clear()

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
                   registry=reg6, journal=failing, log=quiet)
    assert isinstance(e.cause, ZeroDivisionError) and e.undone == ["a"] and e.undo_failed == [], e
    assert world6 == set()

    # results that can't be journaled are marked, never a failure; without a journal, not even copied
    def cyclic():
        node = {"children": []}
        node["children"].append({"parent": node})
        return node

    tool(cyclic, name="cyclic", undo=NOOP, registry=reg6)
    tool(lambda: functools.reduce(lambda acc, _: [acc], range(2000), []), name="deep", undo=NOOP, registry=reg6)
    assert set(run([{"id": "c", "tool": "cyclic"}, {"id": "d", "tool": "deep"}], registry=reg6, log=quiet)) == {"c", "d"}
    path33 = os.path.join(tempfile.mkdtemp(), "rubicon.jsonl")
    run([{"id": "c", "tool": "cyclic"}, {"id": "d", "tool": "deep"}], registry=reg6, journal=path33, log=quiet)
    done33 = {r["step"]: r["result"] for r in FileJournal(path33).records() if r["event"] == "done"}
    assert _has_mark(done33["c"]) and not _has_mark(done33["d"])  # the cycle is marked; deep JSON is kept

    # sync Transaction code inside a running event loop: commit and rollback work like call()
    async def sync_code_in_loop(fail):
        with Transaction(registry=reg6, log=quiet) as tx:
            tx.call("put", "inside")
            if fail:
                raise KeyError("agent error")

    expect(KeyError, lambda: asyncio.run(sync_code_in_loop(True)))
    assert world6 == set()
    asyncio.run(sync_code_in_loop(False))
    assert world6 == {"inside"}
    world6.clear()

    # an async tool called from sync code inside a loop is refused before any of it runs
    async def put_async(k):
        world6.add(k)
        await asyncio.sleep(0)

    tool(put_async, name="put_async", undo=NOOP, registry=reg6)

    async def drive_async():
        Transaction(registry=reg6, log=quiet).call("put_async", "half")

    expect(RuntimeError, lambda: asyncio.run(drive_async()))
    assert world6 == set()

    # a step is marked running before "started" is written, so no settle can miss it
    class Peeking(FileJournal):
        seen = None

        def append(self, records):
            if records[-1]["event"] == "started":
                self.seen = set(peek_tx._running)
            super().append(records)

    peeking = Peeking(os.path.join(tempfile.mkdtemp(), "rubicon.jsonl"))
    peek_tx = Transaction(registry=reg6, journal=peeking, log=quiet)
    peek_tx.call("put", "x")
    assert peeking.seen == {"put-1"}, peeking.seen
    peek_tx.rollback()

    # a step that finishes on another thread after a rollback undoes itself
    gate = threading.Event()

    def slow_put(k):
        gate.wait(10)
        world6.add(k)
        return k

    tool(slow_put, name="slow_put", undo=lambda r: world6.discard(r), registry=reg6)
    tx = Transaction(registry=reg6, log=quiet)
    outcome = []
    worker = threading.Thread(target=lambda: outcome.append(expect(RuntimeError, tx.call, "slow_put", "late")))
    worker.start()
    time.sleep(0.05)
    tx.rollback()
    gate.set()
    worker.join()
    assert world6 == set() and "slow_put-1" in tx.undone and outcome, (world6, tx.undone)

    # a process that exited with code 259 (STILL_ACTIVE's value on Windows) is not alive
    child = subprocess.Popen([sys.executable, "-c", "import os; os._exit(259)"])
    child.wait()
    assert not _pid_alive(child.pid)  # the Popen object still holds its handle

    # plan inputs: names must be strings, *args after a left-out argument, big plans stay fast
    expect(ValueError, order, [{"id": "s", "tool": "put", "args": {1: 2, "k": 3}}], reg6)

    @tool(undo=NOOP, registry=reg6)
    def star(a, b=UNSET, *rest):
        return a

    assert run([{"id": "s", "tool": "star", "args": {"a": 1}}], registry=reg6, log=quiet) == {"s": 1}
    started = time.perf_counter()
    order([{"id": f"s{i}", "tool": "put", "args": {"k": i}} for i in range(5000)], reg6)
    assert time.perf_counter() - started < 3, "order() must not be quadratic"

    # a path into an object that isn't a dict or list is a bad reference, not dumped text
    @dataclass
    class Seat:
        seat: str

    tool(lambda: Seat("12A"), name="seat_obj", undo=NOOP, registry=reg6)
    tool(lambda body: body, name="say", undo=NOOP, registry=reg6)
    e = expect(PlanFailed, run, [{"id": "h", "tool": "seat_obj"},
                                 {"id": "m", "tool": "say", "args": {"body": "Seat $h.seat"}}], registry=reg6, log=quiet)
    assert "not a dict or list" in str(e.cause)

    # an undo with an option of its own gets just the result
    got = []
    tool(lambda k: k, name="opt", undo=lambda r, force=False: got.append((r, force)), registry=reg6)
    t = Transaction(registry=reg6, log=quiet)
    t.call("opt", "z")
    t.rollback()
    assert got == [("z", False)]

    # tool descriptions see the hints of partials and callable objects
    class Typed:
        def __call__(self, x: int):
            return x

    assert _schema(Typed())["properties"] == {"x": {"type": "integer"}}

    def two_args(a: int, b: str = "b"):
        return a

    assert _schema(functools.partial(two_args, b="c"))["properties"] == {"a": {"type": "integer"}, "b": {"type": "string"}}

    tempfile.tempdir = previous_tempdir
    scratch.cleanup()
    print("all checks passed")
