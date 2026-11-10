"""Run each module's built-in self-check under pytest (and coverage).

The checks live next to the code they test, so `python rubicon.py` still works on its own.
"""
import importlib.util
import runpy
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _missing(module):
    try:
        return importlib.util.find_spec(module) is None
    except ModuleNotFoundError:  # parent package (e.g. "google") not installed
        return True


def _run(module):
    previous = tempfile.tempdir  # the checks point it at their own scratch folder
    try:
        runpy.run_path(str(ROOT / f"{module}.py"), run_name="__main__")
    finally:
        tempfile.tempdir = previous


def test_core():
    _run("rubicon")


@pytest.mark.skipif(_missing("mcp"), reason="mcp SDK not installed")
def test_mcp():
    _run("rubicon_mcp")


@pytest.mark.skipif(_missing("google.genai"), reason="google-genai not installed")
def test_planners():
    _run("rubicon_llm")
