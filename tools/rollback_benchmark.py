"""How often does a failure leave the world half-changed? A rebooking agent against a real local
HTTP service and a real SQLite database, with injected timeouts, HTTP 500s, connection resets,
slow responses, a locked or full database, no seats, and process kills between steps or in the
middle of one.

    python tools/rollback_benchmark.py                 # 300 trials per strategy
    python tools/rollback_benchmark.py --trials 1000 --seed 7
    python tools/rollback_benchmark.py --write docs/BENCHMARK.md

Strategies, all with the same tools and the same faults:

  plain      run the model's tool calls in its order; stop at the first error
  naive      the same inside try/except, undoing every step that returned a result, newest first
  toolnode   a plain LangGraph ToolNode running the same calls (skipped if langgraph is missing)
  rubicon    rubicon.arun(): undoable steps first, checks, one confirmation, rollback; after a
             crash, arecover() from the journal

A trial ends consistent when the world is either rebooked (old booking cancelled, exactly one
seat held, the note saved) or untouched (old booking active, nothing held, saved or sent).
Anything else is inconsistent: a traveller with a cancelled booking and no seat, a seat or note
left behind with the old booking still active, or an email about a change that never happened.
A confirmation email that fails after the cancel leaves the traveller rebooked; it is counted
under "rebooked, email failed". Inconsistent states that a dead letter flags for a person are
counted separately.

The mutating tools send an idempotency key and, when the connection fails, ask the service
whether the request was applied, as a production client should. All strategies get that.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import json
import math
import os
import platform
import random
import socket
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rubicon import (APPROVE_ALL, NOOP, PlanFailed, SQLiteJournal, arecover, arun,  # noqa: E402
                     dead_letters, order, tool)

CLIENT_TIMEOUT = 0.3  # seconds; a "timeout" fault sleeps longer than this
MODES = ["timeout", "timeout_after_apply", "500", "reset", "reset_after_apply", "slow"]
ENDPOINTS = ["flights", "hold", "release", "cancel", "email"]
PHANTOM = [0]  # undos that found nothing to undo: the step they undo never happened
FILLER = "~"  # rows that fill the database in a db_full trial; not notes


def _sync(coro):
    # a fresh event loop per trial keeps trials independent of each other
    return asyncio.run(coro)


# ---------------------------------------------------------------------------- the service

class Service:
    """A booking API on 127.0.0.1 with faults that fire once per endpoint."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset_state()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self.server.daemon_threads = True
        self.server.handle_error = lambda request, address: None  # resets and timeouts are the point
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reset_state(self, seats=1):
        with self.lock:
            self.bookings = {"OLD1": "active"}
            self.seats = {"AI101": seats}
            self.holds = {}
            self.outbox = []
            self.applied = {}  # idempotency key -> response
            self.faults = {}   # endpoint -> mode, consumed when it fires

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def state(self):
        with self.lock:
            return {"booking": self.bookings["OLD1"], "holds": len(self.holds), "emails": len(self.outbox)}

    def _apply(self, endpoint, path, body):
        """The business logic; returns (status, response)."""
        with self.lock:
            if endpoint == "flights":
                return 200, [f for f, n in self.seats.items() if n > 0]
            if endpoint == "hold":
                if self.seats.get(body["flight"], 0) < 1:
                    return 409, {"error": f"no seats on {body['flight']}"}
                self.seats[body["flight"]] -= 1
                hold_id = uuid.uuid4().hex[:8]
                self.holds[hold_id] = body["flight"]
                return 200, {"hold_id": hold_id, "flight": body["flight"]}
            if endpoint == "release":
                flight = self.holds.pop(path.rsplit("/", 1)[1], None)
                if flight:
                    self.seats[flight] += 1
                return 200, {"released": bool(flight)}
            if endpoint == "cancel":
                ref = path.split("/")[2]
                if self.bookings.get(ref) != "active":
                    return 409, {"error": f"booking {ref} is not active"}
                self.bookings[ref] = "cancelled"
                return 200, {"cancelled": ref}
            if endpoint == "email":
                self.outbox.append(body)
                return 200, {"sent": len(self.outbox)}
        raise AssertionError(endpoint)

    def _handler(self):
        service = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _endpoint(self):
                p = self.path
                if self.command == "GET" and p.startswith("/flights"):
                    return "flights"
                if self.command == "GET" and p.startswith("/bookings/"):
                    return "booking"
                if self.command == "GET" and p.startswith("/requests/"):
                    return "lookup"
                if self.command == "POST" and p == "/holds":
                    return "hold"
                if self.command == "DELETE" and p.startswith("/holds/"):
                    return "release"
                if self.command == "POST" and p.endswith("/cancel"):
                    return "cancel"
                if self.command == "POST" and p == "/emails":
                    return "email"
                return None

            def _send(self, status, payload):
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _reset(self):
                self.close_connection = True
                try:  # an abrupt RST instead of a clean close
                    self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
                    self.connection.close()
                except OSError:
                    pass

            def _handle(self):
                endpoint = self._endpoint()
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"null")
                if endpoint is None:
                    return self._send(404, {"error": "not found"})
                if endpoint == "booking":
                    with service.lock:
                        return self._send(200, {"status": service.bookings.get(self.path.split("/")[2])})
                if endpoint == "lookup":
                    with service.lock:
                        found = service.applied.get(self.path.split("/")[2])
                    return self._send(200 if found else 404, found or {"error": "unknown request"})
                with service.lock:
                    mode = service.faults.pop(endpoint, None)
                if mode == "slow":
                    time.sleep(CLIENT_TIMEOUT / 3)
                if mode in ("timeout", "500", "reset"):  # fails before anything is applied
                    if mode == "timeout":
                        time.sleep(CLIENT_TIMEOUT * 3)
                    return self._reset() if mode == "reset" else self._send(500, {"error": "internal"})
                key = self.headers.get("Idempotency-Key")
                with service.lock:
                    done = service.applied.get(key) if key else None
                if done is None:
                    status, payload = service._apply(endpoint, self.path, body)
                    done = {"status": status, "body": payload}
                    if key and status == 200:
                        with service.lock:
                            service.applied[key] = done
                if mode == "timeout_after_apply":
                    time.sleep(CLIENT_TIMEOUT * 3)
                if mode == "reset_after_apply":
                    return self._reset()
                self._send(done["status"], done["body"])

            def do_GET(self):
                self._handle()

            do_POST = do_DELETE = do_GET

        return Handler


# ---------------------------------------------------------------------------- the tools

class ServiceError(RuntimeError):
    pass


def _request(url, method, path, body=None, key=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", **({"Idempotency-Key": key} if key else {})})
    try:
        with urllib.request.urlopen(req, timeout=CLIENT_TIMEOUT) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        raise ServiceError(f"{method} {path}: HTTP {e.code} {e.read()[:200]!r}") from None


def _mutate(url, method, path, body=None):
    """A mutating request with an idempotency key. If the connection fails, ask whether the
    service applied it: the answer decides between success and a real failure."""
    key = uuid.uuid4().hex
    try:
        return _request(url, method, path, body, key)
    except ServiceError:
        raise
    except (OSError, urllib.error.URLError, http_client_errors()) as e:
        for _ in range(3):
            try:
                found = _request(url, "GET", f"/requests/{key}")
                return found["body"]
            except ServiceError:  # 404: never applied
                raise ConnectionError(f"{method} {path} failed and was not applied: {e!r}") from None
            except (OSError, urllib.error.URLError, http_client_errors()):
                time.sleep(0.05)
        raise ConnectionError(f"{method} {path}: outcome unknown after {e!r}") from None


def http_client_errors():
    import http.client
    return http.client.HTTPException


def make_tools(url, db_path, registry, crash_at=None, db_full=False):
    """The rebooking tools against the service and a SQLite notes table.

    crash_at names a tool whose call ends the process right after the service applied it, or
    "before:<tool>" to end it as that tool is called, between two steps (for crash trials).
    db_full caps the database at its current size, so the note can't be written."""

    def maybe_crash(name):
        if crash_at == name:
            os._exit(3)

    def between(name):
        maybe_crash("before:" + name)

    def search_flights(route: str) -> list:
        """Flights on a route with free seats."""
        between("search_flights")
        r = _request(url, "GET", f"/flights?route={route}")
        maybe_crash("search_flights")
        return r

    def release(hold):
        if not _mutate(url, "DELETE", f"/holds/{hold['hold_id']}")["released"]:
            PHANTOM[0] += 1

    def hold_seat(flight: str) -> dict:
        """Hold a seat; the hold can be released."""
        between("hold_seat")
        r = _mutate(url, "POST", "/holds", {"flight": flight})
        maybe_crash("hold_seat")
        return r

    def delete_note(row_id):
        with sqlite3.connect(db_path, timeout=CLIENT_TIMEOUT) as db:
            if db.execute("DELETE FROM notes WHERE id = ?", (row_id,)).rowcount == 0:
                PHANTOM[0] += 1
        db.close()

    def save_note(text: str) -> int:
        """Save a note in the local database; it can be deleted."""
        between("save_note")
        with sqlite3.connect(db_path, timeout=CLIENT_TIMEOUT) as db:
            if db_full:  # "database or disk is full" once the insert needs a new page
                db.execute(f"PRAGMA max_page_count = {db.execute('PRAGMA page_count').fetchone()[0]}")
            row = db.execute("INSERT INTO notes (text) VALUES (?)", (text,)).lastrowid
        db.close()
        maybe_crash("save_note")
        return row

    def booking_active(ref):
        status = _request(url, "GET", f"/bookings/{ref}")["status"]
        if status != "active":
            raise ServiceError(f"booking {ref} is {status}")
        return f"cancel {ref}"

    def cancel_booking(ref: str) -> dict:
        """Cancel a booking. PERMANENT."""
        between("cancel_booking")
        r = _mutate(url, "POST", f"/bookings/{ref}/cancel")
        maybe_crash("cancel_booking")
        return r

    def send_email(to: str, body: str) -> dict:
        """Send an email. PERMANENT."""
        between("send_email")
        r = _mutate(url, "POST", "/emails", {"to": to, "body": body})
        maybe_crash("send_email")
        return r

    tool(search_flights, undo=NOOP, registry=registry)
    tool(hold_seat, undo=release, registry=registry)
    tool(save_note, undo=delete_note, registry=registry)
    tool(cancel_booking, check=booking_active, registry=registry)
    tool(send_email, registry=registry)
    return registry


PLAN = [  # the order a careless model writes: the permanent cancel first
    {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}},
    {"id": "find", "tool": "search_flights", "args": {"route": "BLR-DEL"}},
    {"id": "hold", "tool": "hold_seat", "args": {"flight": "AI101"}},
    {"id": "note", "tool": "save_note", "args": {"text": "rebooked OLD1 to AI101"}},
    {"id": "mail", "tool": "send_email", "args": {"to": "me@example.com", "body": "Moved to AI101"}},
]
TOOL_ENDPOINT = {"search_flights": "flights", "hold_seat": "hold", "cancel_booking": "cancel",
                 "send_email": "email"}


def new_db(path, full=False):
    """A fresh notes table. full=True fills the database until a capped insert fails, so the
    next insert under the same cap fails with SQLITE_FULL."""
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, text TEXT)")
        db.execute("DELETE FROM notes")
    db.close()
    if full:
        db = sqlite3.connect(path, isolation_level=None)
        db.execute(f"PRAGMA max_page_count = {db.execute('PRAGMA page_count').fetchone()[0]}")
        try:
            while True:
                db.execute("INSERT INTO notes (text) VALUES (?)", (FILLER * 8,))
        except sqlite3.OperationalError:
            pass
        finally:
            db.close()


def notes(path):
    with sqlite3.connect(path) as db:
        n = db.execute("SELECT COUNT(*) FROM notes WHERE text NOT LIKE ?", (FILLER + "%",)).fetchone()[0]
    db.close()
    return n


# ---------------------------------------------------------------------------- strategies

def run_plain(registry):
    for step in PLAN:
        registry[step["tool"]].fn(**step["args"])  # stop at the first error


def run_naive(registry):
    """try/except with manual cleanup: undo, newest first, every step that returned a result.
    The cancel can't be undone, so a failure after it leaves the traveller without a booking."""
    done = []
    try:
        for step in PLAN:
            t = registry[step["tool"]]
            done.append((t, t.fn(**step["args"])))
    except Exception:
        for t, result in reversed(done):
            if t.undo not in (None, NOOP):
                try:
                    t.undo(result)
                except Exception:
                    pass
        raise


def run_toolnode(registry):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import ToolNode

    from langgraph.graph import END, START, MessagesState, StateGraph

    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([StructuredTool.from_function(t.fn, name=n) for n, t in registry.items()],
                                       handle_tool_errors=True))  # errors go back to the model
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    graph = builder.compile()
    for i, step in enumerate(PLAN):  # one call per turn, as an agent loop issues them
        graph.invoke({"messages": [AIMessage("", tool_calls=[{"name": step["tool"], "args": step["args"], "id": f"c{i}"}])]})


def run_rubicon(registry, journal=None, on_event=None):
    try:
        _sync(arun(PLAN, registry=registry, confirm=APPROVE_ALL, log=lambda m: None, journal=journal,
                   on_event=on_event))
    except PlanFailed:
        pass


STRATEGIES = {"plain": run_plain, "naive": run_naive, "toolnode": run_toolnode, "rubicon": run_rubicon}
CLEANS_UP = {"naive", "rubicon"}  # strategies that undo anything after a failure


def langgraph_available():
    try:
        import langgraph.prebuilt  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------- trials

def classify(service, db_path):
    s = service.state()
    n = notes(db_path)
    if s["booking"] == "cancelled" and s["holds"] == 1 and n == 1:
        return "complete" if s["emails"] == 1 else "email_failed"
    if s == {"booking": "active", "holds": 0, "emails": 0} and n == 0:
        return "untouched"
    return "inconsistent"


def leftovers(service, db_path):
    """Undoable effects (held seats, notes) left while the old booking is still active: parts of
    a change that did not happen."""
    s = service.state()
    return s["holds"] + notes(db_path) if s["booking"] == "active" else 0


def make_fault(rng):
    """One fault per trial: (kind, target, mode)."""
    roll = rng.random()
    if roll < 0.1:
        return ("none", None, None)
    if roll < 0.2:
        return ("no_seats", None, None)
    if roll < 0.3:
        return ("db_locked", None, None)
    if roll < 0.35:
        return ("db_full", None, None)
    if roll < 0.45:  # killed mid-step, after the service applied the call
        return ("crash", rng.choice([s["tool"] for s in PLAN]), None)
    if roll < 0.55:  # killed between two steps
        return ("crash", "before:" + rng.choice([s["tool"] for s in PLAN]), None)
    return ("http", rng.choice(ENDPOINTS), rng.choice(MODES))


def _child(args):
    """Run one strategy in this process until the injected crash."""
    between = args.crash_at.startswith("before:") and args.strategy == "rubicon"
    registry = make_tools(args.url, args.db, {}, crash_at=None if between else args.crash_at)
    if between:
        # between two steps as Rubicon runs them: right after the previous step's record is
        # durable, before the next one starts (inside the tool would be mid-step for the journal)
        ids = [s["id"] for s in order(PLAN, registry)]
        i = ids.index(next(s["id"] for s in PLAN if s["tool"] == args.crash_at[len("before:"):]))
        if i == 0:
            os._exit(3)

        def on_event(e):
            if e["event"] in ("step_done", "step_committed") and e["step"] == ids[i - 1]:
                os._exit(3)
        run_rubicon(registry, journal=SQLiteJournal(args.journal), on_event=on_event)
    elif args.strategy == "rubicon":
        run_rubicon(registry, journal=SQLiteJournal(args.journal))
    else:
        try:
            STRATEGIES[args.strategy](registry)
        except Exception:
            pass


def _timed(fn, failed_at):
    @functools.wraps(fn)
    def timed(*a, **k):
        try:
            return fn(*a, **k)
        except Exception:
            failed_at.append(time.perf_counter())
            raise
    return timed


def trial(strategy, fault, service, workdir, stats=None):
    """Run one trial; returns (outcome, flagged). stats, if given, collects "phantom" and
    "leftover", "recovery_ms" when something was cleaned up, and for a crash "before_recover"
    (the outcome right after the crash)."""
    stats = {} if stats is None else stats
    kind, target, mode = fault
    db_path = os.path.join(workdir, "notes.db")
    new_db(db_path, full=kind == "db_full")
    service.reset_state(seats=0 if kind == "no_seats" else 1)
    PHANTOM[0], flagged = 0, False
    if kind == "http":
        service.faults[target] = mode
    if kind == "crash":
        journal = os.path.join(workdir, f"journal-{uuid.uuid4().hex[:6]}.db")
        subprocess.run([sys.executable, __file__, "--child", strategy, "--url", service.url, "--db", db_path,
                        "--journal", journal, "--crash-at", target], check=False)
        stats["before_recover"] = classify(service, db_path)
        if strategy == "rubicon":  # the restarted worker
            registry = make_tools(service.url, db_path, {})
            started = time.perf_counter()
            _sync(arecover(SQLiteJournal(journal), registry=registry, log=lambda m: None))
            stats["recovery_ms"] = (time.perf_counter() - started) * 1000
            flagged = bool(dead_letters(SQLiteJournal(journal)))
    else:
        registry = make_tools(service.url, db_path, {}, db_full=kind == "db_full")
        failed_at = []
        if strategy in CLEANS_UP:  # time from the failing step to the end of the cleanup
            for t in registry.values():
                t.fn = _timed(t.fn, failed_at)
        blocker = None
        if kind == "db_locked":  # another writer holds the database for the whole trial
            blocker = sqlite3.connect(db_path, isolation_level=None)
            blocker.execute("BEGIN EXCLUSIVE")
        try:
            STRATEGIES[strategy](registry)
        except Exception:
            pass
        finally:
            ended = time.perf_counter()
            if blocker is not None:
                blocker.execute("ROLLBACK")
                blocker.close()
        if failed_at:
            stats["recovery_ms"] = (ended - failed_at[0]) * 1000
    stats["phantom"], stats["leftover"] = PHANTOM[0], leftovers(service, db_path)
    return classify(service, db_path), flagged


def benchmark(trials=300, seed=1, strategies=None, say=print):
    strategies = strategies or [s for s in STRATEGIES if s != "toolnode" or langgraph_available()]
    rng = random.Random(seed)
    faults = [make_fault(rng) for _ in range(trials)]
    service = Service()
    results = {}
    try:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as workdir:
            for strategy in strategies:
                c = {"complete": 0, "email_failed": 0, "untouched": 0, "inconsistent": 0, "flagged": 0,
                     "phantom": 0, "leftover": 0, "leftover_flagged": 0, "crashes": 0,
                     "crash_inconsistent_before": 0, "crash_inconsistent_after": 0}
                times = []
                started = time.monotonic()
                for fault in faults:
                    stats = {}
                    outcome, flagged = trial(strategy, fault, service, workdir, stats)
                    c[outcome] += 1
                    c["flagged"] += outcome == "inconsistent" and flagged
                    c["phantom"] += stats["phantom"]
                    c["leftover"] += stats["leftover"]
                    c["leftover_flagged"] += stats["leftover"] if flagged else 0
                    if "before_recover" in stats:
                        c["crashes"] += 1
                        c["crash_inconsistent_before"] += stats["before_recover"] == "inconsistent"
                        c["crash_inconsistent_after"] += outcome == "inconsistent"
                    if "recovery_ms" in stats:
                        times.append(stats["recovery_ms"])
                times.sort()
                c["median_ms"] = statistics.median(times) if times else None
                c["p95_ms"] = times[math.ceil(0.95 * len(times)) - 1] if times else None  # nearest rank
                c["seconds"] = round(time.monotonic() - started, 1)
                results[strategy] = c
                say(f"{strategy}: {c}")
    finally:
        service.close()
    return results


def table(results, trials):
    pct = lambda n: f"{100 * n / trials:.1f}%"  # noqa: E731
    ms = lambda v: "-" if v is None else f"{v:.0f}"  # noqa: E731
    lines = ["| Strategy | Consistent | Left inconsistent | Of those, flagged for a person | Rebooked | "
             "Rebooked, email failed | Untouched | Undos for steps that never ran | Leftover partial effects "
             "(flagged) | Recovery ms, median / p95 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in results.items():
        lines.append(f"| {name} | {pct(trials - c['inconsistent'])} | {pct(c['inconsistent'])} | {c['flagged']} | "
                     f"{pct(c['complete'])} | {pct(c['email_failed'])} | {pct(c['untouched'])} | {c['phantom']} | "
                     f"{c['leftover']} ({c['leftover_flagged']}) | {ms(c['median_ms'])} / {ms(c['p95_ms'])} |")
    return "\n".join(lines)


def crash_table(results):
    lines = ["| Strategy | Crash trials | Inconsistent right after the crash | Inconsistent after recover() | "
             "Of those, flagged for a person |", "|---|---|---|---|---|"]
    for name, c in results.items():
        has = name == "rubicon"
        lines.append(f"| {name} | {c['crashes']} | {c['crash_inconsistent_before']} | "
                     f"{c['crash_inconsistent_after'] if has else 'no recover()'} | {c['flagged'] if has else '-'} |")
    return "\n".join(lines)


DOC = """# Rollback benchmark

Generated by `python tools/rollback_benchmark.py --trials {trials} --seed {seed} --write docs/BENCHMARK.md`
on Python {py} ({system}). Every number below comes from that one run.

## Setup

A rebooking agent runs the same five-step plan, written in the careless order a model might
produce: cancel booking OLD1 (permanent), search flights (read-only), hold a seat (undoable:
release), save a note in SQLite (undoable: delete), send an email (permanent).

The tools call a real local HTTP service (stdlib `http.server`) and a real SQLite file. Each
trial injects one failure:

- HTTP, at one endpoint: a timeout or a connection reset, before or after the service applied
  the request; an HTTP 500; or a slow reply that still succeeds. Mutating requests carry an
  idempotency key, and when the connection fails the tool asks the service whether the request
  was applied, as a production client should. Every strategy gets those same tools.
- SQLite: another connection holds an exclusive lock ("database is locked"), or the database is
  capped at its size ("database or disk is full").
- No seats left on the flight.
- The process is killed (`os._exit`), either between two steps or in the middle of a step, right
  after the service applied the call and before the tool returned.
- No failure.

Strategies, on identical faults:

- `plain`: the calls in plan order, stopping at the first error.
- `naive`: the same inside `try/except`, undoing every step that returned a result, newest first.
- `toolnode`: a LangGraph `ToolNode` running the same calls (only when `langgraph` is installed).
- `rubicon`: `rubicon.arun()` with a `SQLiteJournal`; after a kill, `arecover()` in a new process.

A trial is consistent when it ends rebooked (cancelled, one seat held, note saved) or untouched
(booking active, nothing held, saved or sent). A failed confirmation email after the cancel counts
as rebooked. "Undos for steps that never ran" counts undo calls that found nothing to undo.
"Leftover partial effects" counts seats and notes left behind while the old booking is still
active. Recovery time runs from the failing step to the end of the cleanup, or covers the whole
`recover()` run after a kill.

## Results

{table}

### Process kills: end state before and after recover()

{crash_table}

## What rubicon's inconsistent trials are

Every inconsistent end state Rubicon leaves comes from a process killed in the middle of an
undoable step, after the service had applied it but before the step's result reached the
journal. Nobody can know from the journal alone whether that step happened, so `recover()` does
not guess. It undoes everything else, then records the step as a dead letter (kind `uncertain`)
for a person to check. These trials are counted in "flagged for a person", and none of them is
silent: `dead_letters(journal)` lists each one until `acknowledge()` resolves it. A kill between
steps, or a kill during a permanent step, ends consistent after `recover()`.

The other strategies have nothing to recover from: whatever a kill or a failure left behind
stays, and nothing records it. Cleanup alone does not fix the order: `naive` scores below `plain`
because, once the cancel has run, undoing the held seat after a later failure leaves the
traveller with no booking at all.

## Limits

- One failure per trial. Undos that keep failing are covered by `tests/test_fuzz.py`, not here.
- A service that applies a write after the client gave up, and gives no way to ask about it
  afterwards, is beyond what any of these strategies can clean up.
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trials", type=int, default=300)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--write", metavar="PATH", help="also write the results as Markdown to PATH")
    p.add_argument("--child", help=argparse.SUPPRESS)
    p.add_argument("--url", help=argparse.SUPPRESS)
    p.add_argument("--db", help=argparse.SUPPRESS)
    p.add_argument("--journal", help=argparse.SUPPRESS)
    p.add_argument("--crash-at", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.child:
        args.strategy = args.child
        return _child(args)
    results = benchmark(args.trials, args.seed)
    print()
    print(f"{args.trials} trials per strategy, seed {args.seed}:")
    print(table(results, args.trials))
    print()
    print(crash_table(results))
    if args.write:
        Path(args.write).write_text(DOC.format(trials=args.trials, seed=args.seed, py=platform.python_version(),
                                               system=platform.system(), table=table(results, args.trials),
                                               crash_table=crash_table(results)), encoding="utf-8")
    return results


if __name__ == "__main__":
    main()
