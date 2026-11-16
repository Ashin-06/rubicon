"""Real failures against real services: a local stdlib HTTP booking API and a SQLite database.

Timeouts, HTTP 500s, connection resets (before and after the service applied the request), slow
responses, a locked or full database, no seats, and a process killed between steps or in the
middle of one. After rollback, and after recover() for kills, the world must be rebooked or
untouched, never half changed; anything else must be flagged in a dead letter for a person.
"""
import importlib.util
import sqlite3
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "rollback_benchmark.py"
if not SCRIPT.exists():  # the wheel job copies only tests/
    pytest.skip("tools/rollback_benchmark.py is not next to the tests", allow_module_level=True)
_spec = importlib.util.spec_from_file_location("rollback_benchmark", SCRIPT)
bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench)

pytestmark = pytest.mark.timeout(120)


@pytest.fixture(scope="module")
def service():
    svc = bench.Service()
    yield svc
    svc.close()


def outcome(service, tmp_path, fault, strategy="rubicon", stats=None):
    return bench.trial(strategy, fault, service, str(tmp_path), stats)


BEFORE_APPLY = ["timeout", "500", "reset"]
AFTER_APPLY = ["timeout_after_apply", "reset_after_apply"]

EXPECTED = {
    # the search fails: nothing permanent was tried, the plan is rolled back
    **{("flights", m): "untouched" for m in BEFORE_APPLY + AFTER_APPLY},
    # the hold fails before it is applied: rolled back; applied but the reply was lost: the
    # client finds it by its idempotency key and carries on
    **{("hold", m): "untouched" for m in BEFORE_APPLY},
    **{("hold", m): "complete" for m in AFTER_APPLY},
    # the release only runs on rollback, so a fault there is never reached on the happy path
    **{("release", m): "complete" for m in BEFORE_APPLY + AFTER_APPLY},
    **{("cancel", m): "untouched" for m in BEFORE_APPLY},
    **{("cancel", m): "complete" for m in AFTER_APPLY},
    # the email runs after the cancel committed: the traveller stays rebooked
    **{("email", m): "email_failed" for m in BEFORE_APPLY},
    **{("email", m): "complete" for m in AFTER_APPLY},
    **{(e, "slow"): "complete" for e in bench.ENDPOINTS},
}


@pytest.mark.parametrize("endpoint,mode", sorted(EXPECTED), ids=lambda v: str(v))
def test_http_faults(service, tmp_path, endpoint, mode):
    stats = {}
    assert outcome(service, tmp_path, ("http", endpoint, mode), stats=stats) == (EXPECTED[endpoint, mode], False)
    assert stats["phantom"] == 0 and stats["leftover"] == 0


def test_no_seats_leaves_the_booking_alone(service, tmp_path):
    assert outcome(service, tmp_path, ("no_seats", None, None)) == ("untouched", False)
    # the same failure strands an agent that cancels first, with or without cleanup
    assert outcome(service, tmp_path, ("no_seats", None, None), "plain") == ("inconsistent", False)
    assert outcome(service, tmp_path, ("no_seats", None, None), "naive") == ("inconsistent", False)


def test_a_locked_database_rolls_back_the_held_seat(service, tmp_path):
    assert outcome(service, tmp_path, ("db_locked", None, None)) == ("untouched", False)


def test_a_full_database_is_a_real_sqlite_error_and_rolls_back(service, tmp_path):
    db = str(tmp_path / "full.db")
    bench.new_db(db, full=True)
    tools = bench.make_tools(service.url, db, {}, db_full=True)
    with pytest.raises(sqlite3.OperationalError, match="full"):
        tools["save_note"].fn(text="x")
    stats = {}
    assert outcome(service, tmp_path, ("db_full", None, None), stats=stats) == ("untouched", False)
    assert stats["leftover"] == 0 and stats["recovery_ms"] > 0
    assert outcome(service, tmp_path, ("db_full", None, None), "plain")[0] == "inconsistent"


def test_an_undo_that_fails_once_is_retried(service, tmp_path):
    db = str(tmp_path / "notes.db")
    bench.new_db(db)
    service.reset_state()
    service.faults["release"] = "500"  # the rollback's first release attempt fails
    blocker = sqlite3.connect(db, isolation_level=None)
    blocker.execute("BEGIN EXCLUSIVE")  # so saving the note fails after the seat was held
    try:
        bench.run_rubicon(bench.make_tools(service.url, db, {}))
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()
    assert bench.classify(service, db) == "untouched" and "release" not in service.faults


@pytest.mark.parametrize("tool_name", [s["tool"] for s in bench.PLAN])
def test_a_crash_mid_step_is_recovered_or_flagged(service, tmp_path, tool_name):
    result, flagged = outcome(service, tmp_path, ("crash", tool_name, None))
    assert result != "inconsistent" or flagged, (tool_name, result)
    # only an undoable step whose outcome the journal never saw can stay half-done
    assert result != "inconsistent" or tool_name in ("hold_seat", "save_note"), (tool_name, result)


@pytest.mark.parametrize("tool_name", [s["tool"] for s in bench.PLAN])
def test_a_kill_between_steps_is_recovered_completely(service, tmp_path, tool_name):
    stats = {}
    result, _ = outcome(service, tmp_path, ("crash", "before:" + tool_name, None), stats=stats)
    assert result in ("untouched", "email_failed"), (tool_name, result)
    assert stats["phantom"] == 0 and stats["leftover"] == 0


def test_the_benchmark_runs_and_rubicon_never_leaves_silent_damage():
    results = bench.benchmark(trials=40, seed=11, strategies=["plain", "naive", "rubicon"], say=lambda m: None)
    rubicon, plain = results["rubicon"], results["plain"]
    assert rubicon["inconsistent"] == rubicon["flagged"]
    assert rubicon["phantom"] == 0 and rubicon["leftover"] == rubicon["leftover_flagged"]
    assert plain["inconsistent"] > rubicon["inconsistent"]
    assert "| rubicon |" in bench.table(results, 40) and "| rubicon |" in bench.crash_table(results)
