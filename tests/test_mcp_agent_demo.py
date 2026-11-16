"""The MCP agent demo end to end: a real stdio server, a failure mid-plan, then a clean run."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")
DEMO = Path(__file__).resolve().parents[1] / "examples" / "mcp_agent_demo" / "run_demo.py"


@pytest.mark.timeout(180)
def test_mcp_agent_demo(tmp_path):
    if not DEMO.exists():  # the wheel test copies only tests/
        pytest.skip("examples/mcp_agent_demo is not next to this test")
    work = tmp_path / "work"
    out = subprocess.run([sys.executable, str(DEMO), "--workdir", str(work)],
                         capture_output=True, text=True, encoding="utf-8", timeout=150)
    assert out.returncode == 0, out.stderr
    run1, run2 = out.stdout.split("== Run 2")
    assert "no space left on device while writing changelog.md" in run1
    assert "rolled back: ['write_file-3', 'write_file-2', 'read_file-1']" in run1
    assert "Scheduled send_message" in run1 and "HELD for confirmation" not in run1
    assert "HELD for confirmation" in run2 and "COMMITTED send_message-4" in run2
    after1 = run1.split("workspace after run 1")[1]
    assert "summary.md:" not in after1 and "outbox.jsonl:" not in after1  # run 1 left nothing
    # run 2 sent exactly one message
    assert (work / "index.md").read_text(encoding="utf-8") == "- notes.md\n- summary.md\n"
    sent = [json.loads(line) for line in (work / "outbox.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sent == [{"to": "team@example.com", "text": "The summary is ready."}]
