"""A short run of tools/rollback_benchmark.py: real HTTP, SQLite and process-kill failures."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "rollback_benchmark.py"


@pytest.mark.timeout(300)
def test_rollback_benchmark_short_run():
    if not SCRIPT.exists():  # the wheel test copies only tests/
        pytest.skip("tools/rollback_benchmark.py is not next to this test")
    spec = importlib.util.spec_from_file_location("rollback_benchmark", SCRIPT)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    # every failure kind at least once, kills included
    summary = bench.main(["-n", "12", "--seed", "7"])
    r = summary["rubicon"]
    assert r["consistent"] == r["n"] == 12
    assert r["phantom"] == 0 and r["leftover"] == 0 and r["sent_on_failure"] == 0
    assert all(total for _, total in r["by_fault"].values()), r["by_fault"]
    # the baselines really were hit by the failures, so the numbers above mean something
    assert summary["plain"]["leftover"] > 0
    assert summary["naive"]["phantom"] > 0
