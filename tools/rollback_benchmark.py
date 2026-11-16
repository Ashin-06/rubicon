"""Rollback correctness under real failures: Rubicon against two common ways of running steps.

    python tools/rollback_benchmark.py                 # 200 transactions, seed 1, prints tables
    python tools/rollback_benchmark.py -n 500 --seed 7
    python tools/rollback_benchmark.py --write docs/BENCHMARK.md

Each transaction is 3 to 6 undoable steps (HTTP PUTs to a local server, SQLite inserts) plus one
permanent "send" at a random place in the plan. One real failure is injected per transaction:

    http_500                the server answers 500 and applies nothing
    http_timeout            the server hangs past the client timeout and applies nothing
    http_drop               the server reads the request, then sends half a status line and closes
    http_drop_after_apply   the server applies the write, then drops the connection mid-response
    sqlite_locked           another connection holds an exclusive lock ("database is locked")
    sqlite_full             the database hits its page limit ("database or disk is full")
    kill                    the process is killed (TerminateProcess / SIGKILL) between two steps
    none                    no failure

The same plans and failures run under three strategies:

    rubicon   rubicon.run() with a FileJournal; after a kill, recover() in a new process
    naive     try/except that undoes every step it attempted, including the one that failed
    plain     the steps one after another, no cleanup

Every tool uses client-chosen keys, so an ambiguous HTTP failure (timeout, dropped connection) is
settled by asking the server whether the key exists. That is the same for all three strategies.
"""
from __future__ import annotations

import argparse
import asyncio
import http.client
import http.server
import json
import math
import random
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout
import rubicon  # noqa: E402

CLIENT_TIMEOUT = 0.25  # seconds; the server hangs for HANG on an http_timeout fault
HANG = 0.5
FAULTS = ["http_500", "http_timeout", "http_drop", "http_drop_after_apply", "sqlite_locked",
          "sqlite_full", "kill", "none"]
STRATEGIES = ["rubicon", "naive", "plain"]

BASE = ("127.0.0.1", 0)  # the server's address, set by serve() or the child's arguments
DB = ""  # SQLite file holding the inserted rows
DB_FAULT = None  # (kind, key) for the next SQLite insert that should fail
PHANTOM = [0]  # undos that found nothing to undo: the step never happened
FAILED_AT = [None]  # perf_counter() when an injected failure surfaced in a tool
LOOP = None  # one event loop for every arun()/arecover(), closed at the end


# ---------------------------------------------------------------------------- the HTTP server

class Store:
    keys: set = set()
    sent: set = set()
    fault = None  # (kind, path) for the next PUT/POST to that path
    lock = threading.Lock()


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, status):
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _take_fault(self):
        with Store.lock:
            if Store.fault and Store.fault[1] == self.path:
                kind, Store.fault = Store.fault[0], None
                return kind
        return None

    def _write(self, collection):
        kind = self._take_fault()
        if kind == "http_500":
            return self._reply(500)
        if kind == "http_timeout":
            time.sleep(HANG)  # the client gave up already; close without a word
            return
        if kind == "http_drop":
            self.wfile.write(b"HTTP/1.1 2")  # half a status line, then the connection closes
            return
        with Store.lock:
            collection.add(self.path.split("/", 2)[2])
        if kind == "http_drop_after_apply":
            self.wfile.write(b"HTTP/1.1 2")
            return
        self._reply(201)

    def do_PUT(self):
        self._write(Store.keys)

    def do_POST(self):
        self._write(Store.sent)

    def do_GET(self):
        self._reply(200 if self.path.split("/", 2)[2] in Store.keys else 404)

    def do_DELETE(self):
        key = self.path.split("/", 2)[2]
        with Store.lock:
            found = key in Store.keys
            Store.keys.discard(key)
        self._reply(204 if found else 404)


def serve():
    global BASE
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    BASE = server.server_address[:2]
    return server


def _http(method, path):
    c = http.client.HTTPConnection(*BASE, timeout=CLIENT_TIMEOUT)
    try:
        c.request(method, path)
        r = c.getresponse()
        r.read()
        return r.status
    finally:
        c.close()


# ---------------------------------------------------------------------------- the tools

def _failed(e):
    FAILED_AT[0] = time.perf_counter()
    raise e


def http_put(key):
    try:
        status = _http("PUT", f"/k/{key}")
    except (OSError, http.client.HTTPException) as e:
        # the write may or may not have landed; the key is ours, so ask
        if _http("GET", f"/k/{key}") == 200:
            return {"key": key}
        _failed(e)
    if status != 201:
        _failed(RuntimeError(f"PUT {key}: HTTP {status}"))
    return {"key": key}


def http_undo(result):
    status = _http("DELETE", f"/k/{result['key']}")
    if status == 404:
        PHANTOM[0] += 1
    elif status != 204:
        raise RuntimeError(f"DELETE {result['key']}: HTTP {status}")


def db_insert(key):
    global DB_FAULT
    fault = None
    if DB_FAULT and DB_FAULT[1] == key:
        fault, DB_FAULT = DB_FAULT[0], None
    con = sqlite3.connect(DB, timeout=0.05)
    locker = None
    try:
        if fault == "sqlite_locked":
            locker = sqlite3.connect(DB, isolation_level=None)
            locker.execute("BEGIN EXCLUSIVE")
        if fault == "sqlite_full":  # no room for one more page; auto_vacuum keeps no free pages
            pages = con.execute("PRAGMA page_count").fetchone()[0]
            con.execute(f"PRAGMA max_page_count = {pages}")
        con.execute("INSERT INTO rows (key, data) VALUES (?, ?)", (key, b"x" * 8192))
        con.commit()
    except sqlite3.OperationalError as e:
        _failed(e)
    finally:
        con.close()
        if locker:
            locker.execute("ROLLBACK")
            locker.close()
    return {"key": key}


def db_undo(result):
    with sqlite3.connect(DB) as con:
        if con.execute("DELETE FROM rows WHERE key = ?", (result["key"],)).rowcount == 0:
            PHANTOM[0] += 1
    con.close()


def send(key):
    status = _http("POST", f"/send/{key}")
    if status != 201:
        raise RuntimeError(f"send {key}: HTTP {status}")
    return "sent"


FN = {"http_put": http_put, "db_insert": db_insert, "send": send}
UNDO = {"http_put": http_undo, "db_insert": db_undo}
REG = {}
rubicon.tool(http_put, undo=http_undo, registry=REG)
rubicon.tool(db_insert, undo=db_undo, registry=REG)
rubicon.tool(send, registry=REG)


# ---------------------------------------------------------------------------- the strategies

def run_rubicon(plan, journal, after_step=None):
    rplan = [{"id": s["id"], "tool": s["tool"], "args": {"key": s["key"]}} for s in plan]
    on_event = (lambda e: e["event"] == "step_done" and after_step()) if after_step else None
    try:
        LOOP.run_until_complete(rubicon.arun(rplan, registry=REG, confirm=rubicon.APPROVE_ALL,
                                             journal=journal, on_event=on_event))
        return True
    except rubicon.PlanFailed:
        return False


def run_naive(plan, journal=None, after_step=None):
    attempted = []
    try:
        for s in plan:
            attempted.append(s)
            FN[s["tool"]](s["key"])
            if after_step:
                after_step()
        return True
    except Exception:
        for s in reversed(attempted):  # undo all we tried: the failed one may have half-happened
            if s["tool"] in UNDO:
                try:
                    UNDO[s["tool"]]({"key": s["key"]})
                except Exception:
                    pass
        return False


def run_plain(plan, journal=None, after_step=None):
    try:
        for s in plan:
            FN[s["tool"]](s["key"])
            if after_step:
                after_step()
        return True
    except Exception:
        return False


RUN = {"rubicon": run_rubicon, "naive": run_naive, "plain": run_plain}


def child(spec):
    """Run one plan and wait to be killed after spec["kill_after"] completed steps."""
    global BASE, DB, LOOP
    BASE, DB, LOOP = tuple(spec["base"]), spec["db"], asyncio.new_event_loop()
    count = [0]

    def after_step():
        count[0] += 1
        if count[0] == spec["kill_after"]:
            print("KILL", flush=True)
            time.sleep(60)
    RUN[spec["strategy"]](spec["plan"], spec["journal"], after_step)
    print("FINISHED", flush=True)


# ---------------------------------------------------------------------------- the benchmark

def make_case(rng, t):
    n = rng.randint(3, 6)
    undoable = [rng.choice(["http_put", "db_insert"]) for _ in range(n)]
    fault = rng.choice(FAULTS)
    if fault.startswith("http") and "http_put" not in undoable:
        undoable[0] = "http_put"
    if fault.startswith("sqlite") and "db_insert" not in undoable:
        undoable[0] = "db_insert"
    want = {"http": "http_put", "sqlite": "db_insert"}.get(fault.split("_")[0])
    target = rng.choice([i for i, k in enumerate(undoable) if want in (None, k)])
    tools = undoable[:]
    tools.insert(rng.randrange(n + 1), "send")
    return {"t": t, "tools": tools, "fault": fault, "target": target,
            "kill_after": rng.randint(1, n)}


def run_case(case, strategy, tmp):
    prefix = f"{strategy}-{case['t']}-"
    plan, u = [], 0
    for i, name in enumerate(case["tools"]):
        plan.append({"id": f"s{i}", "tool": name, "key": f"{prefix}{i}"})
        if name != "send":
            if u == case["target"]:
                target_key = f"{prefix}{i}"
            u += 1
    send_key = next(s["key"] for s in plan if s["tool"] == "send")
    journal_path = str(Path(tmp, f"{prefix}journal.jsonl"))
    journal = rubicon.FileJournal(journal_path)
    PHANTOM[0], FAILED_AT[0] = 0, None
    fault, recovery_ms = case["fault"], None

    if fault == "kill":
        spec = {"strategy": strategy, "plan": plan, "kill_after": case["kill_after"],
                "base": list(BASE), "db": DB, "journal": journal_path}
        p = subprocess.Popen([sys.executable, __file__, "--child", json.dumps(spec)],
                             stdout=subprocess.PIPE, text=True)
        line = p.stdout.readline().strip()
        p.kill()
        p.wait()
        p.stdout.close()
        if line != "KILL":
            raise RuntimeError(f"child for {prefix} did not reach its kill point: {line!r}")
        ok = False
        if strategy == "rubicon":
            start = time.perf_counter()
            LOOP.run_until_complete(rubicon.arecover(journal, registry=REG, log=lambda m: None,
                                                     include_live=True))
            recovery_ms = (time.perf_counter() - start) * 1000
    else:
        global DB_FAULT
        if fault.startswith("http"):
            Store.fault = (fault, f"/k/{target_key}")
        elif fault.startswith("sqlite"):
            DB_FAULT = (fault, target_key)
        ok = RUN[strategy](plan, journal)
        end = time.perf_counter()
        Store.fault, DB_FAULT = None, None
        if not ok and FAILED_AT[0] is not None and strategy != "plain":
            recovery_ms = (end - FAILED_AT[0]) * 1000

    undoable_keys = {s["key"] for s in plan if s["tool"] != "send"}
    with sqlite3.connect(DB) as con:
        rows = {k for (k,) in con.execute("SELECT key FROM rows WHERE key LIKE ?", (prefix + "%",))}
    con.close()
    with Store.lock:
        present = (Store.keys & undoable_keys) | rows
        sent = send_key in Store.sent
    if ok:
        consistent = present == undoable_keys and sent
    else:
        consistent = not present and not sent
    return {"strategy": strategy, "fault": fault, "ok": ok, "consistent": consistent,
            "phantom": PHANTOM[0], "leftover": 0 if ok else len(present),
            "sent_on_failure": (not ok) and sent, "recovery_ms": recovery_ms}


def benchmark(n=200, seed=1):
    global DB, LOOP
    server, LOOP = serve(), asyncio.new_event_loop()
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        DB = str(Path(tmp, "rows.db"))
        with sqlite3.connect(DB) as con:
            con.execute("PRAGMA auto_vacuum = FULL")
            con.execute("CREATE TABLE rows (key TEXT PRIMARY KEY, data BLOB)")
        con.close()
        rng = random.Random(seed)
        cases = [make_case(rng, t) for t in range(n)]
        try:
            return [run_case(c, s, tmp) for c in cases for s in STRATEGIES]
        finally:
            server.shutdown()
            server.server_close()
            LOOP.close()


def summarize(rows):
    out = {}
    for s in STRATEGIES:
        mine = [r for r in rows if r["strategy"] == s]
        times = sorted(r["recovery_ms"] for r in mine if r["recovery_ms"] is not None)
        out[s] = {
            "n": len(mine),
            "consistent": sum(r["consistent"] for r in mine),
            "phantom": sum(r["phantom"] for r in mine),
            "leftover": sum(r["leftover"] for r in mine),
            "sent_on_failure": sum(r["sent_on_failure"] for r in mine),
            "median_ms": statistics.median(times) if times else None,
            "p95_ms": times[math.ceil(0.95 * len(times)) - 1] if times else None,  # nearest rank
            "by_fault": {f: (sum(r["consistent"] for r in mine if r["fault"] == f),
                             sum(1 for r in mine if r["fault"] == f)) for f in FAULTS},
        }
    return out


def table(summary):
    ms = lambda v: "-" if v is None else f"{v:.1f}"
    lines = ["| Strategy | Consistent end states | Undos for steps that never ran | Leftover partial "
             "effects | Sends for failed transactions | Recovery time median (ms) | p95 (ms) |",
             "|---|---|---|---|---|---|---|"]
    for s, v in summary.items():
        lines.append(f"| {s} | {v['consistent']}/{v['n']} ({100 * v['consistent'] / v['n']:.1f}%) | "
                     f"{v['phantom']} | {v['leftover']} | {v['sent_on_failure']} | "
                     f"{ms(v['median_ms'])} | {ms(v['p95_ms'])} |")
    lines += ["", "Consistent end states by injected failure:", "",
              "| Failure | " + " | ".join(summary) + " |", "|---" * (len(summary) + 1) + "|"]
    for f in FAULTS:
        cells = [f"{v['by_fault'][f][0]}/{v['by_fault'][f][1]}" for v in summary.values()]
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


DOC = """# Rollback benchmark

Generated by `python tools/rollback_benchmark.py -n {n} --seed {seed} --write docs/BENCHMARK.md`
on Python {py} ({platform}).

{n} randomised transactions, each run under three strategies. A transaction is 3 to 6 undoable
steps (HTTP PUTs to a local `http.server`, SQLite inserts) plus one permanent `send` placed at a
random point in the plan, with one real failure injected:

- `http_500`: the server answers 500 and applies nothing.
- `http_timeout`: the server hangs past the client timeout and applies nothing.
- `http_drop`: the server reads the request, sends half a status line and closes.
- `http_drop_after_apply`: the server applies the write, then drops the connection mid-response.
  The tool asks the server whether its key exists, finds it, and carries on, so these
  transactions should succeed.
- `sqlite_locked`: another connection holds an exclusive lock, so the insert fails with
  "database is locked".
- `sqlite_full`: the database is capped at its current page count, so the insert fails with
  "database or disk is full".
- `kill`: the process is killed between two steps. Rubicon then runs `recover()` on the journal
  from a new process; the other strategies have nothing to recover with.
- `none`: no failure.

Strategies:

- `rubicon`: `rubicon.run()` with a `FileJournal`. Permanent steps run last, after every
  undoable step succeeded.
- `naive`: steps in plan order inside `try/except`; on an error it undoes every step it
  attempted, including the one that failed, since it can't tell whether that one half-happened.
- `plain`: steps in plan order, no cleanup.

A transaction's end state is consistent when it either succeeded with every effect in place, or
failed with no effect left and nothing sent. "Undos for steps that never ran" counts undo calls
that found nothing to undo. "Leftover partial effects" counts rows and keys left behind by failed
transactions. Recovery time runs from the moment the failure surfaced (or, after a kill, from the
start of `recover()`) until the strategy is done cleaning up.

{table}

Limits: each transaction has one failure, and the undos themselves are not failed here (the
fuzz tests in `tests/test_fuzz.py` cover failing undos). A server that applies a write after
the client timed out, with no way to ask it afterwards, is outside what any of these strategies
can clean up.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-n", type=int, default=200, help="transactions per strategy")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--write", metavar="PATH", help="write the results as Markdown to PATH")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.child:
        return child(json.loads(a.child))
    summary = summarize(benchmark(a.n, a.seed))
    print(table(summary))
    if a.write:
        import platform
        Path(a.write).write_text(DOC.format(n=a.n, seed=a.seed, table=table(summary),
                                            py=platform.python_version(),
                                            platform=platform.system()), encoding="utf-8")
    return summary


if __name__ == "__main__":
    main()
