"""`python -m rubicon`: inspect a journal, run recover() with your tools, acknowledge dead letters."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from rubicon import main

ROOT = Path(__file__).resolve().parent.parent

TOOLS = '''
from rubicon import NOOP, tool
@tool(undo=lambda r: open({log!r}, "a").write(f"undo {{r}}\\n"))
def hold(flight: str):
    return flight
@tool(undo=NOOP)
def probe():
    return "x"
@tool
def cancel(ref: str):
    return "ok"
'''

CRASH = '''
import os, sys
sys.path.insert(0, {root!r}); sys.path.insert(0, {here!r})
import mytools
from rubicon import APPROVE_ALL, run, tool
tool(lambda ref: os._exit(3), name="cancel", registry=None)  # dies inside the permanent step
run([{{"id": "h", "tool": "hold", "args": {{"flight": "AI101"}}}},
     {{"id": "c", "tool": "cancel", "args": {{"ref": "OLD1"}}}}],
    journal={journal!r}, confirm=APPROVE_ALL, log=lambda m: None)
'''


def cli(*args, cwd):
    done = subprocess.run([sys.executable, "-m", "rubicon", *args], cwd=cwd, capture_output=True, text=True,
                          env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    return done.returncode, done.stdout, done.stderr


@pytest.fixture
def crashed(tmp_path):
    log = tmp_path / "undo.log"
    (tmp_path / "mytools.py").write_text(TOOLS.format(log=str(log)))
    journal = tmp_path / "j.jsonl"
    code = CRASH.format(root=str(ROOT), here=str(tmp_path), journal=str(journal))
    assert subprocess.run([sys.executable, "-c", code]).returncode == 3
    return tmp_path, journal, log


def test_inspect_recover_and_ack(crashed):
    here, journal, log = crashed

    code, out, _ = cli("inspect", str(journal), cwd=here)
    assert code == 0 and "1 unfinished transaction(s)" in out and "not running" in out
    assert "in progress: c" in out and "undoable: h" in out and "0 dead letter(s)" in out

    code, out, err = cli("recover", str(journal), "--module", "mytools", cwd=here)
    assert code == 0, err
    (report,) = json.loads(out)
    assert report["needs_attention"] == ["c"] and report["kept"] == ["h"]
    assert not log.exists()  # the hold is kept: the cancel may have gone through

    code, out, _ = cli("inspect", str(journal), "--json", cwd=here)
    data = json.loads(out)
    (letter,) = data["dead_letters"]
    assert letter["kind"] == "in_flight_permanent" and letter["step"] == "c"

    code, out, _ = cli("ack", str(journal), letter["tx"], "--note", "checked: cancelled", cwd=here)
    assert code == 0 and out.startswith("acknowledged")
    code, out, _ = cli("inspect", str(journal), cwd=here)
    assert "0 unfinished transaction(s)" in out and "0 dead letter(s)" in out


def test_recover_undoes_with_a_named_registry(tmp_path):
    log = tmp_path / "undo.log"
    (tmp_path / "regtools.py").write_text(
        "from rubicon import NOOP, tool\nREG = {}\n"
        f"tool(lambda f: f, name='hold', undo=lambda r: open({str(log)!r}, 'a').write(r), registry=REG)\n"
        "tool(lambda: __import__('os')._exit(3), name='die', undo=NOOP, registry=REG)\n")
    journal = tmp_path / "j.db"
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); sys.path.insert(0, {str(tmp_path)!r})\n"
            "import regtools\nfrom rubicon import run, SQLiteJournal\n"
            "run([{'id': 'h', 'tool': 'hold', 'args': {'f': 'AI101'}}, {'id': 'd', 'tool': 'die', 'after': ['h']}],"
            f" registry=regtools.REG, journal=SQLiteJournal({str(journal)!r}), log=lambda m: None)\n")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 3
    code, out, err = cli("recover", str(journal), "--module", "regtools:REG", cwd=tmp_path)
    assert code == 0, err
    assert json.loads(out)[0]["undone"] == ["h"] and log.read_text() == "AI101"


def test_errors_are_short_messages(tmp_path, capsys):
    journal = tmp_path / "j.jsonl"
    journal.write_text("")
    assert main(["ack", str(journal), "no-such-tx"]) == 1
    assert "no transaction" in capsys.readouterr().err
    assert main(["recover", str(journal), "--module", "no_such_module_here"]) == 1
    assert main(["inspect", str(journal)]) == 0
    assert "0 unfinished transaction(s)" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([])
