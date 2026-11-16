"""PostgresJournal: the same contract as SQLiteJournal, plus the crash and maintenance fuzz
scenarios with their rows kept in PostgreSQL.

Runs when RUBICON_PG_DSN points at a database the tests may create tables in, for example
RUBICON_PG_DSN=postgresql://postgres:postgres@localhost:5432/postgres
"""
import os
import threading

import pytest

pytest.importorskip("psycopg")
DSN = os.environ.get("RUBICON_PG_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="set RUBICON_PG_DSN to run the PostgreSQL tests")

import psycopg  # noqa: E402

import test_fuzz  # noqa: E402
from rubicon import (APPROVE_ALL, NOOP, JournalBusy, PlanFailed, PostgresJournal,  # noqa: E402
                     Transaction, _LIVE, compact, dead_letters, recover, run, tool)

FUZZ_COUNT = int(os.environ.get("RUBICON_FUZZ_PG", "100"))


@pytest.fixture
def journal(request):
    table = "rubicon_test_" + request.node.name.lower().replace("[", "_").replace("]", "")[:40]
    with psycopg.connect(DSN, autocommit=True) as db:
        db.execute(f"DROP TABLE IF EXISTS {table}")
    yield PostgresJournal(DSN, table=table)
    with psycopg.connect(DSN, autocommit=True) as db:
        db.execute(f"DROP TABLE IF EXISTS {table}")


def test_append_and_records_keep_order(journal):
    journal.append([{"tx": "a", "n": 1}, {"tx": "b", "n": 2}])
    journal.append([{"tx": "a", "n": 3, "odd": object()}])
    rows = journal.records()
    assert [r["n"] for r in rows] == [1, 2, 3] and rows[2]["odd"].startswith("<object")


def test_table_name_must_be_an_identifier():
    with pytest.raises(ValueError):
        PostgresJournal(DSN, table="x; DROP TABLE users")


def test_repr_hides_the_conninfo(journal):
    assert "postgres" not in repr(journal).replace("PostgresJournal", "")


def test_lock_is_exclusive_and_released(journal):
    other = PostgresJournal(DSN, table=journal.table)
    with journal.lock():
        with pytest.raises(JournalBusy):
            with other.lock():
                pass
        with pytest.raises(JournalBusy):
            recover(other, log=lambda m: None)  # one recovery at a time, across workers
    with other.lock():
        pass


def test_drop_with_upto_keeps_later_records(journal):
    journal.append([{"tx": "a", "n": 1}, {"tx": "b", "n": 2}])
    seen = len(journal.records())
    journal.append([{"tx": "a", "n": 3}])  # written after compact() read the journal
    journal.drop(["a"], upto=seen)
    assert [r["n"] for r in journal.records()] == [2, 3]
    journal.drop(["a", "b"])
    assert journal.records() == []


def test_concurrent_writers_lose_nothing(journal):
    def write(w):
        for i in range(25):
            PostgresJournal(DSN, table=journal.table).append([{"tx": f"w{w}", "i": i}])

    threads = [threading.Thread(target=write, args=(w,)) for w in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = journal.records()
    assert len(rows) == 100
    for w in range(4):  # each writer's records stay in its own order
        assert [r["i"] for r in rows if r["tx"] == f"w{w}"] == list(range(25))


def test_crash_recover_and_compact(journal):
    state = {"held": [], "sent": []}
    reg = {}
    tool(lambda f: state["held"].append(f) or f, name="hold", undo=lambda r: state["held"].remove(r),
         registry=reg)
    tool(lambda to: state["sent"].append(to), name="send", registry=reg)
    tool(lambda: "ok", name="look", undo=NOOP, registry=reg)

    run([{"id": "h", "tool": "hold", "args": {"f": "AI101"}}, {"id": "s", "tool": "send", "args": {"to": "x"}}],
        registry=reg, journal=journal, confirm=APPROVE_ALL, log=lambda m: None)
    assert recover(journal, registry=reg, log=lambda m: None) == []

    tx = Transaction(registry=reg, journal=journal, log=lambda m: None)
    tx.call("hold", "BA1")
    _LIVE.discard(tx.id)  # the worker died here
    report = recover(journal, registry=reg, log=lambda m: None)
    assert report[0]["undone"] == ["hold-1"] and state["held"] == ["AI101"]
    assert not dead_letters(journal)
    assert compact(journal) == 2 and journal.records() == []

    with pytest.raises(PlanFailed):
        run([{"id": "s", "tool": "send", "args": {"to": "y"}}], registry=reg, journal=journal,
            log=lambda m: None)  # no confirm: refused before anything ran
    assert recover(journal, registry=reg, log=lambda m: None) == [] and state["sent"] == ["x"]


@pytest.fixture
def backed_fuzz(journal):
    def fresh():
        with psycopg.connect(DSN, autocommit=True) as db:
            db.execute(f"TRUNCATE {journal.table}")
        return journal

    test_fuzz.CrashJournal.backing = staticmethod(fresh)
    yield
    test_fuzz.CrashJournal.backing = None


@pytest.mark.timeout(900)  # a sweep of scenarios, not one test
def test_crash_and_recovery_invariants_on_postgres(backed_fuzz):
    test_fuzz._sweep(test_fuzz.scenario_crash, 5_000_000, FUZZ_COUNT)


@pytest.mark.timeout(900)  # a sweep of scenarios, not one test
def test_concurrent_maintenance_invariants_on_postgres(backed_fuzz):
    test_fuzz._sweep(test_fuzz.scenario_maintenance, 6_000_000, FUZZ_COUNT)
