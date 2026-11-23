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
import atexit
import contextlib
import contextvars
import difflib
import enum
import functools
import hashlib
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

__all__ = ["APPROVE_ALL", "BudgetExceeded", "adry_run", "dry_run", "FileJournal", "JournalBusy", "NOOP", "PlanFailed", "PostgresJournal",
           "SQLiteJournal", "TOOLS", "Tool", "Transaction", "UNSET", "acknowledge", "arecover", "arun", "compact", "dead_letters", "order",
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


class BudgetExceeded(RuntimeError):
    """A transaction made more tool calls, or ran longer, than its max_calls / max_seconds allow."""


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

    def scan(s):
        for m, _ in _REF.findall(s):
            if m in ids:
                found.append(m)
            else:  # a near miss is almost always a typo the model made; it would be sent as plain text
                near = difflib.get_close_matches(m, list(ids), n=1, cutoff=0.75)
                if near:
                    _logger.warning("$%s is not a step id (did you mean $%s?); it is sent as plain text", m, near[0])
        return s

    _walk(args, scan)
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
    by_name = [p.name for p in params if p.kind is p.POSITIONAL_ONLY and p.name in kwargs]
    if by_name and not any(p.kind is p.VAR_KEYWORD for p in params):
        # newer Pythons' bind() lets this through and the call then fails: refuse it here
        raise TypeError(f"positional-only arguments {by_name} can't be given by name here "
                        f"(an earlier one is missing or left out)")
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


class _Runner:
    """One event loop per thread for sync calls.

    A new loop per call opens a loopback socket pair each time; on Windows, thousands of them in a
    row can leave a new loop stuck in accept(). Reusing the loop also makes each call cheaper.
    The loop is closed when its thread ends (this object is dropped with the thread's locals) or,
    for the main thread, at exit, before the interpreter starts tearing things down.
    """

    def __init__(self):
        self.runner = asyncio.Runner()

    def close(self):
        runner, self.runner = self.runner, None
        if runner is not None:
            with contextlib.suppress(Exception):
                runner.close()

    __del__ = close


_THREAD_LOCAL = threading.local()
atexit.register(lambda: getattr(_THREAD_LOCAL, "runner", None) and _THREAD_LOCAL.runner.close())


def _run(coro):
    if not hasattr(asyncio, "Runner"):  # Python 3.10
        return asyncio.run(coro)
    holder = getattr(_THREAD_LOCAL, "runner", None)
    if holder is None or holder.runner is None:
        holder = _THREAD_LOCAL.runner = _Runner()
    runner = holder.runner
    try:
        # the caller's context vars, as asyncio.run() would see them; Ctrl+C cancels the task
        return runner.run(coro, context=contextvars.copy_context())
    finally:
        # like asyncio.run(): nothing started by this call outlives it
        loop = runner.get_loop()
        leftover = asyncio.all_tasks(loop)
        for task in leftover:
            task.cancel()
        if leftover:
            loop.run_until_complete(asyncio.gather(*leftover, return_exceptions=True))


def _sync(coro, drive=False):
    """Run a coroutine from sync code.

    Inside a running event loop, drive=True runs it to completion directly, which works as long
    as nothing in it waits on real I/O (sync tools and checks); otherwise it raises.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run(coro)
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


def APPROVE_ALL(steps):
    """confirm=APPROVE_ALL runs permanent steps with nobody asked. Without a confirm they are refused."""
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

    def __init__(self, registry=None, confirm=None, log=None, *, on_event=None, journal=None,
                 max_calls=None, max_seconds=None):
        self.tools = TOOLS if registry is None else registry
        # budgets for a looping agent: going over rolls back (calls count undoable and scheduled ones)
        self.max_calls, self.max_seconds = max_calls, max_seconds
        self._calls, self._started = 0, time.monotonic()
        self.confirm, self.log, self.on_event = confirm, log or _logger.info, on_event
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

    def _over_time(self):
        return self.max_seconds is not None and time.monotonic() - self._started > self.max_seconds

    def _spend(self):
        """Count one tool call against the budget; raise BudgetExceeded when it is used up."""
        self._calls += 1
        if self.max_calls is not None and self._calls > self.max_calls:
            raise BudgetExceeded(f"more than {self.max_calls} tool calls in one transaction")
        if self._over_time():
            raise BudgetExceeded(f"transaction ran longer than {self.max_seconds} seconds")

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
        self._emit("tx_settled", status=status, undo_failed=list(self.undo_failed))
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
        try:
            self._spend()
        except BudgetExceeded:
            self._say(f"budget used up at {name}; rolling back")
            await self.arollback()
            raise
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

    async def _gate(self, shown):
        """Fail closed: only a confirm that returns exactly True lets permanent steps run."""
        ids = [s["id"] for s in shown]
        if self.confirm is None:
            self._emit("gate_declined", steps=ids)
            raise RuntimeError("permanent steps need a confirm= callable; pass confirm=APPROVE_ALL "
                               "to run them unattended")
        try:
            answer = await _call(self.confirm, shown)
        except Exception as e:
            self._emit("gate_declined", steps=ids)
            raise RuntimeError(f"confirm raised {e!r}, so the permanent steps were declined") from e
        if answer is not True:
            self._emit("gate_declined", steps=ids)
            raise RuntimeError(f"permanent steps not confirmed (confirm returned {answer!r}, not True)")
        self._emit("gate_confirmed", steps=ids)

    async def _fail(self, step, cause, not_run=()):
        self._emit("tx_failed", step=step, error=repr(cause), not_run=list(not_run))
        undone, failed = await self.arollback()
        return PlanFailed(step, cause, undone, list(self.committed), failed, dict(self.results), not_run)

    async def acommit(self):
        """Confirm and run pending permanent steps. Returns all results."""
        steps, current = list(self.pending), "confirm"
        shown = [{key: s[key] for key in ("id", "tool", "args", "preview")} for s in steps]
        try:
            if steps:
                await self._gate(shown)
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
               journal=None, max_calls=None, max_seconds=None) -> dict:
    """Execute a plan safely. Returns {step_id: result}; raises PlanFailed after rolling back.

    parallel=True runs independent undoable steps concurrently. on_event and journal are passed
    to the underlying Transaction. max_calls rejects a longer plan before anything runs; past
    max_seconds, the run stops before the point of no return and rolls back.
    """
    steps = order(plan, registry)
    if max_calls is not None and len(steps) > max_calls:
        raise BudgetExceeded(f"plan has {len(steps)} steps, more than max_calls={max_calls}; nothing ran")
    tx = Transaction(registry, confirm, log, on_event=on_event, journal=journal,
                     max_calls=max_calls, max_seconds=max_seconds)
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
        if not perm(s):
            tx._spend()  # never after the point of no return: a commit, once started, finishes
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
        if tx._over_time():
            raise BudgetExceeded(f"transaction ran longer than {max_seconds} seconds")
        if permanent:
            await tx._gate(permanent)
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
        journal=None, max_calls=None, max_seconds=None) -> dict:
    """Sync version of arun(). Inside an event loop, use `await arun(...)` instead."""
    return _sync(arun(plan, registry, confirm, log, parallel=parallel, on_event=on_event, journal=journal,
                      max_calls=max_calls, max_seconds=max_seconds))


async def adry_run(plan, registry=None) -> dict:
    """What run() would do, with nothing executed: validates the plan, orders it, marks the point of
    no return, and runs the checks (dry runs by contract) of permanent steps whose inputs are known
    now. Raises ValueError for an invalid plan, as run() would before running anything."""
    steps = order(plan, registry)
    tools = TOOLS if registry is None else registry
    ids = {s["id"] for s in steps}
    out, ponr = [], None
    for s in steps:
        t, deps = tools[s["tool"]], _deps(s, ids)
        row = {"id": s["id"], "tool": s["tool"], "permanent": not t.reversible, "after": deps}
        if not t.reversible:
            ponr = ponr or s["id"]
            if not t.check:
                row["check"] = None
            elif deps:
                row["check"] = {"waits_for": deps}  # checked at run time, once those steps ran
            else:
                a, k, view = _prepare(t.fn, (), s.get("args", {}))
                ca, ck = _fit(t.check, a, k, view)
                try:
                    row["check"] = {"preview": await _attempt(t.check, ca, ck, 0, t.timeout)}
                except Exception as e:
                    row["check"] = {"error": f"{type(e).__name__}: {e}"}
        out.append(row)
    return {"steps": out, "point_of_no_return": ponr,
            "undoable": [r["id"] for r in out if not r["permanent"]],
            "permanent": [r["id"] for r in out if r["permanent"]],
            "would_fail": [r["id"] for r in out if isinstance(r.get("check"), dict) and "error" in r["check"]]}


def dry_run(plan, registry=None) -> dict:
    """Sync version of adry_run()."""
    return _sync(adry_run(plan, registry))


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


class PostgresJournal:
    """Journal in a PostgreSQL table, for workers on several machines. Needs psycopg 3
    (`pip install "rubicon-agents[postgres]"`).

        journal = PostgresJournal("postgresql://user@db/app")   # or any psycopg conninfo string

    Writes are serialized with a transaction-level advisory lock, so record ids follow commit
    order and compact() never drops a record written after it read the journal. lock() is a
    session-level advisory lock held during recover() and compact(), released by the server if
    the holder's connection dies. Each call opens its own connection, so put a pooler such as
    PgBouncer in front for many busy workers.
    """

    def __init__(self, conninfo, table="rubicon_journal"):
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", table):
            raise ValueError(f"table must be a plain SQL identifier, got {table!r}")
        self.conninfo, self.table = conninfo, table
        # two advisory-lock keys per table: one orders writes, one is the recovery lock
        digest = hashlib.sha256(f"rubicon:{table}".encode()).digest()
        self._key = int.from_bytes(digest[:4], "big", signed=True)
        with self._connect() as db:
            db.execute(f"CREATE TABLE IF NOT EXISTS {table} ("
                       "id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
                       "tx TEXT NOT NULL, record TEXT NOT NULL)")
            db.execute(f"CREATE INDEX IF NOT EXISTS {table}_tx ON {table} (tx)")

    def __repr__(self):
        return f"PostgresJournal(table={self.table!r})"  # the conninfo may hold a password

    def _connect(self):
        import psycopg
        return psycopg.connect(self.conninfo, autocommit=False)

    def append(self, records):
        rows = [(r.get("tx", ""), json.dumps(r, default=repr)) for r in records]
        with self._connect() as db:  # commits on success, rolls back on error
            db.execute("SELECT pg_advisory_xact_lock(%s, 1)", (self._key,))
            with db.cursor() as cur:
                cur.executemany(f"INSERT INTO {self.table} (tx, record) VALUES (%s, %s)", rows)

    def records(self):
        with self._connect() as db:
            return [json.loads(row[0]) for row in
                    db.execute(f"SELECT record FROM {self.table} ORDER BY id").fetchall()]

    @contextlib.contextmanager
    def lock(self):
        with self._connect() as db:
            db.autocommit = True
            if not db.execute("SELECT pg_try_advisory_lock(%s, 2)", (self._key,)).fetchone()[0]:
                raise JournalBusy(f"{self!r} is locked by another process")
            try:
                yield self
            finally:
                db.execute("SELECT pg_advisory_unlock(%s, 2)", (self._key,))

    def drop(self, tx_ids, upto=None):
        """Remove the records of these transactions; with upto, only among the first `upto`
        records (ones written since compact() read the journal stay)."""
        tx_ids = list(tx_ids)
        with self._connect() as db:  # one transaction: all or nothing
            db.execute("SELECT pg_advisory_xact_lock(%s, 1)", (self._key,))
            last = None
            if upto is not None:  # the id of the last record compact() saw
                row = db.execute(f"SELECT id FROM {self.table} ORDER BY id LIMIT 1 OFFSET %s",
                                 (max(upto - 1, 0),)).fetchone()
                last = row[0] if row and upto > 0 else -1
            if last is None:
                db.execute(f"DELETE FROM {self.table} WHERE tx = ANY(%s)", (tx_ids,))
            else:
                db.execute(f"DELETE FROM {self.table} WHERE tx = ANY(%s) AND id <= %s", (tx_ids, last))


def _as_journal(journal):
    if journal is None:
        return None
    if isinstance(journal, (str, os.PathLike)):
        return FileJournal(journal)
    if not (callable(getattr(journal, "append", None)) and callable(getattr(journal, "records", None))):
        raise TypeError("journal must be a path, FileJournal, SQLiteJournal, PostgresJournal, or an "
                        "object with append(records) and records() methods")
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
