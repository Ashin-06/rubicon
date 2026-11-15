"""Planners and agent(): plan parsing, repair, retries and confirmation, with stub clients."""
import asyncio
import builtins
import json
import logging
import math
import os
import tempfile
import threading
import time as _time
import warnings as _warnings
from functools import partial
from rubicon import NOOP, PlanFailed, tool
from types import SimpleNamespace as NS

import pytest

from rubicon_llm import (CLAUDE_MODEL, GEMINI_MODEL, PlanRejected, _logger, aagent, agent, ask_human, parse_plan, plan_with_claude, plan_with_gemini, plan_with_openai, planning_request)

@pytest.fixture(autouse=True, scope="module")
def _scratch(tmp_path_factory):
    """Every mkdtemp() in this module lands in one scratch folder that pytest cleans up."""
    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("scratch"))
    yield
    tempfile.tempdir = previous


# Offline checks with stub clients: no API keys, no network.


state = {"held": 0, "cancelled": []}
reg = {}

@tool(undo=lambda r: state.__setitem__("held", state["held"] - 1), registry=reg)
def hold_seat(flight: str):
    """Hold a seat; can be released."""
    state["held"] += 1
    return {"flight": flight}

@tool(check=lambda ref: None, registry=reg)
def cancel_booking(ref: str):
    """Cancel a booking. PERMANENT."""
    state["cancelled"].append(ref)

@tool(undo=NOOP, registry=reg)
def search(route: str):
    """Find flights."""
    return ["AI101"]

naive = [  # what a model might emit: permanent step first
    {"id": "c", "tool": "cancel_booking", "args_json": '{"ref": "OLD1"}', "after": []},
    {"id": "s", "tool": "search", "args_json": '{"route": "BLR-DEL"}', "after": []},
    {"id": "h", "tool": "hold_seat", "args_json": '{"flight": "AI101"}', "after": ["s"]},
]
text = json.dumps({"steps": naive})

class Gemini:
    def __init__(self, text):
        self.text, self.kw = text, None
        self.models = NS(generate_content=self.gen)

    def gen(self, **kw):
        self.kw = kw
        return NS(text=self.text, candidates=[NS(finish_reason="SAFETY")], prompt_feedback=None)

class Claude:
    def __init__(self, text, stop="end_turn"):
        self.text, self.stop, self.kw = text, stop, None
        self.beta = NS(messages=NS(create=self.create))

    def create(self, **kw):
        self.kw = kw
        return NS(stop_reason=self.stop, stop_details=None, content=[NS(type="text", text=self.text)])

class OpenAI:
    def __init__(self, text, finish="stop"):
        self.text, self.finish, self.kw = text, finish, None
        self.chat = NS(completions=NS(create=self.create))

    def create(self, **kw):
        self.kw = kw
        return NS(choices=[NS(finish_reason=self.finish, message=NS(content=self.text, refusal=None))])

quiet = lambda m: None

def expect(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"expected {exc.__name__}")


def test_gemini_default_planner():
    global g
    # Gemini (default planner): reordered; only the permanent step needs confirmation.
    g, seen = Gemini(text), []
    agent("move", client=g, registry=reg, confirm=lambda st: seen.extend(s["id"] for s in st) or True, log=quiet)
    assert seen == ["c"] and state == {"held": 1, "cancelled": ["OLD1"]}
    cfg = g.kw["config"]  # a real GenerateContentConfig, so field names are validated by the SDK
    assert g.kw["model"] == GEMINI_MODEL and cfg.response_mime_type == "application/json"
    assert cfg.max_output_tokens == 16000  # output (and its cost) is bounded
    assert cfg.response_json_schema["properties"]["steps"]["items"]["properties"]["tool"]["enum"] == \
        ["cancel_booking", "hold_seat", "search"]
    assert '"reversible": false' in g.kw["contents"] and '"type": "string"' in g.kw["contents"]
    assert "SAFETY" in str(expect(RuntimeError, plan_with_gemini, "x", Gemini(""), reg))


def test_model_env_vars_set_after_import_e_g_by_load_dotenv_still():
    global c, g
    # model env vars set after import (e.g. by load_dotenv) still apply; model= wins over both
    os.environ["GEMINI_MODEL"], os.environ["CLAUDE_MODEL"] = "late-gemini", "late-claude"
    try:
        g = Gemini(text)
        plan_with_gemini("x", g, reg)
        assert g.kw["model"] == "late-gemini"
        plan_with_gemini("x", g, reg, model="explicit")
        assert g.kw["model"] == "explicit"
        c = Claude(text)
        plan_with_claude("x", c, reg)
        assert c.kw["model"] == "late-claude"
    finally:
        del os.environ["GEMINI_MODEL"], os.environ["CLAUDE_MODEL"]


def test_claude():
    global c
    # Claude.
    c = Claude(text)
    agent("move", planner=plan_with_claude, client=c, registry=reg, confirm=lambda st: True, log=quiet)
    assert c.kw["model"] == CLAUDE_MODEL and c.kw["extra_body"] == {"fallbacks": "default"}
    assert "refusal" in str(expect(RuntimeError, plan_with_claude, "x", Claude(text, "refusal"), reg))


def test_openai_compatible():
    # OpenAI-compatible: strict schema, JSON mode, and no-JSON-mode servers.
    o = OpenAI(text)
    agent("move", planner=partial(plan_with_openai, model="m1"), client=o, registry=reg,
          confirm=lambda st: True, log=quiet)
    assert o.kw["model"] == "m1" and o.kw["response_format"]["json_schema"]["strict"] is True
    o = OpenAI("```json\n" + text + "\n```")  # fenced output from a local model
    plan_with_openai("x", o, reg, model="m2", json_mode="object")
    assert o.kw["response_format"] == {"type": "json_object"} and "matching this schema" in o.kw["messages"][1]["content"]
    o = OpenAI("Sure! Here is the plan:\n" + text + "\nGood luck.")  # prose around the JSON
    plan_with_openai("x", o, reg, model="m3", json_mode="none")
    assert "response_format" not in o.kw
    expect(RuntimeError, plan_with_openai, "x", OpenAI(text, finish="length"), reg, model="m")
    expect(RuntimeError, plan_with_openai, "x", OpenAI("", finish="stop"), reg, model="m")
    assert plan_with_openai("x", OpenAI(text, finish="eos"), reg, model="m")  # other normal finishes
    assert state == {"held": 3, "cancelled": ["OLD1"] * 3}


def test_bring_your_own_model():
    # Bring-your-own model: request + parse, with args as an object instead of args_json.
    req = planning_request("move", reg)
    assert set(req) == {"system", "prompt", "schema"}
    assert parse_plan('{"steps": [{"id": "s", "tool": "search", "args": {"route": "x"}}]}', reg)[0]["args"] == {"route": "x"}
    assert parse_plan('{"steps": [{"id": "s", "tool": "search", "args_json": {"route": "y"}}]}', reg)[0]["args"] == {"route": "y"}


def test_human_declines():
    # Human declines -> held seat released, nothing cancelled.
    expect(PlanFailed, agent, "move", client=Gemini(text), registry=reg, confirm=lambda st: False, log=quiet)
    assert state == {"held": 3, "cancelled": ["OLD1"] * 3}


def test_untrusted_output_is_rejected_before_anything_runs():
    # Untrusted output is rejected before anything runs.
    for bad in ["not json", '{"plan": []}', '{"steps": [1]}',
                json.dumps({"steps": [{**naive[2], "after": ["ghost"]}]}),
                json.dumps({"steps": [{**naive[0], "args_json": "[1]"}]}),
                json.dumps({"steps": [{**naive[0], "id": "has space"}]}),
                json.dumps({"steps": [{**naive[0], "id": "1"}]}),
                json.dumps({"steps": [{**naive[0], "args_json": '{"booking_id": "OLD1"}'}]})]:
        expect(ValueError, parse_plan, bad, reg)
    expect(ValueError, planning_request, "x", {})  # nothing to plan with
    bare = {}
    tool(lambda: 1, name="ping", undo=NOOP, registry=bare)
    # args_json written as the text "null" means no args_json: "args" stands, or there are none
    assert parse_plan(json.dumps([{"id": "s", "tool": "search", "args": {"route": "R"},
                                   "args_json": "null"}]), reg)[0]["args"] == {"route": "R"}
    assert parse_plan(json.dumps([{"id": "p", "tool": "ping", "args_json": " null "}]), bare)[0]["args"] == {}
    # a cut-off args_json is refused whole, not read as the inner object it happens to contain
    expect(ValueError, parse_plan, json.dumps([{"id": "a", "tool": "search",
                                                "args_json": '{"broken": {"route": "BLR"}'}]), reg)
    # a list of objects in the prose before the plan is not taken for the plan
    prose = ('Context: [{"flight": "AI101"}]\nPlan: {"steps": [{"id": "s", "tool": "search", '
             '"args": {"route": "x"}}]}')
    assert [s["id"] for s in parse_plan(prose, reg)] == ["s"]
    for not_args in ([1, 2], 3, True):  # args_json that is not an object is refused, never read as {}
        expect(ValueError, parse_plan, json.dumps([{"id": "a", "tool": "ping", "args_json": not_args}]), bare)


def test_lenient_parsing():
    # Lenient parsing: braces in trailing prose, bare lists, empty args_json, null after.
    fenced = "Here is the plan:\n```json\n" + text + "\n```\nNote: replaces {OLD1}."
    assert [s["id"] for s in parse_plan(fenced, reg)] == ["c", "s", "h"]
    assert len(parse_plan(json.dumps(naive), reg)) == 3
    loose = parse_plan('{"steps": [{"id": "s", "tool": "search", "args": {"route": "x"}, '
                       '"args_json": "", "after": null}]}', reg)
    assert loose == [{"id": "s", "tool": "search", "args": {"route": "x"}, "after": []}]
    assert parse_plan('[{"id": "s", "tool": "search", "args": {"route": "x"}, "args_json": null}]', reg)


def test_local_repair():
    global calls, fixes, rep
    # Local repair: small, unambiguous mistakes are fixed without asking the model again.
    rep = {}

    @tool(undo=NOOP, registry=rep)
    def book(flight_number: str, seats: int, refundable: bool = False, price: float = 0.0):
        """Typed tool for repair tests."""
        return flight_number

    @tool(undo=NOOP, registry=rep)
    def count_seats() -> dict:
        return {"n": 2}

    fixes = []
    fixed = parse_plan(json.dumps([{"id": "b", "tool": "Book", "args": {
        "flightNumber": 101, "Seats": "2", "refundable": "TRUE", "price": "9.5"}}]), rep, repairs=fixes)
    assert fixed[0]["tool"] == "book" and fixed[0]["args"] == {
        "flight_number": "101", "seats": 2, "refundable": True, "price": 9.5}
    assert len(fixes) == 7  # tool name, two argument names, four values
    python_style = "[{'id': 'b', 'tool': 'book', 'args': {'flight_number': 'AI1', 'seats': 1, 'refundable': True}}]"
    assert parse_plan(python_style, rep)[0]["args"]["refundable"] is True
    twice = json.dumps([{"id": "b", "tool": "book", "args_json": json.dumps(json.dumps({"flight_number": "X", "seats": 1}))}])
    assert parse_plan(twice, rep)[0]["args"] == {"flight_number": "X", "seats": 1}
    # never touched: references, NaN/inf, values that don't plainly convert
    kept = parse_plan(json.dumps([{"id": "c", "tool": "count_seats"},
                                  {"id": "b", "tool": "book", "args": {"flight_number": "X", "seats": "$c.n", "price": "NaN"}}]), rep)
    assert kept[1]["args"] == {"flight_number": "X", "seats": "$c.n", "price": "NaN"}
    assert parse_plan(json.dumps([{"id": "b", "tool": "book", "args": {"flight_number": "X", "seats": "two"}}]), rep)
    # ambiguous names are not guessed: the plan goes back to the model
    amb = {}
    tool(lambda: 1, name="hold_seat", undo=NOOP, registry=amb)
    tool(lambda: 1, name="holdseat", undo=NOOP, registry=amb)
    expect(PlanRejected, parse_plan, json.dumps([{"id": "h", "tool": "Hold-Seat"}]), amb)
    # repair=False is strict
    expect(PlanRejected, parse_plan, python_style, rep, repair=False)
    expect(PlanRejected, parse_plan, json.dumps([{"id": "b", "tool": "Book", "args": {"flight_number": "X", "seats": 1}}]),
           rep, repair=False)
    # agent() repairs locally instead of spending a second model call
    calls = []

    def sloppy(goal, client, registry):
        calls.append(goal)
        return parse_plan(json.dumps([{"id": "b", "tool": "BOOK", "args": {"FlightNumber": 7, "seats": "1"}}]), registry)
    assert agent("book", planner=sloppy, registry=rep, confirm=lambda st: True) == {"b": "7"} and len(calls) == 1


def test_bracketed_prose_before_the_plan_doesn_t_hijack_parsing():
    # Bracketed prose before the plan doesn't hijack parsing (JSON or Python-style).
    good = [{"id": "b", "tool": "book", "args": {"flight_number": "X", "seats": 1}}]
    assert parse_plan("See note [1] below:\n```json\n" + json.dumps(good) + "\n```", rep)[0]["id"] == "b"
    assert parse_plan("Option {2} is best: " + json.dumps({"steps": good}), rep)[0]["id"] == "b"
    assert parse_plan("Step [1]:\n" + repr(good).replace('"', "'"), rep)[0]["id"] == "b"


def test_two_keys_aiming_at_one_parameter_are_ambiguous():
    global fixes
    # Two keys aiming at one parameter are ambiguous: neither is renamed.
    @tool(undo=NOOP, registry=rep)
    def flex(flight_number: str = "", **extra):
        return flight_number
    fixes = []
    both = parse_plan(json.dumps([{"id": "f", "tool": "flex", "args": {"flightNumber": "A", "flight-number": "B"}}]),
                      rep, repairs=fixes)
    assert both[0]["args"] == {"flightNumber": "A", "flight-number": "B"} and fixes == []


def test_nan_never_produces_a_bogus_repair_and_isn_t_turned_into():
    global fixes
    # NaN never produces a bogus repair, and isn't turned into the string "nan".
    fixes = []
    nan_plan = parse_plan('[{"id": "b", "tool": "book", "args": {"flight_number": NaN, "seats": 1, "price": NaN}}]',
                          rep, repairs=fixes)
    assert fixes == [] and math.isnan(nan_plan[0]["args"]["flight_number"])


def test_after_must_be_a_list():
    # "after" must be a list; only null means "none", even in strict mode.
    for bad_after in ('""', "0", "{}"):
        expect(PlanRejected, parse_plan, f'[{{"id": "b", "tool": "book", "args": {{"flight_number": "X", "seats": 1}}, '
               f'"after": {bad_after}}}]', rep, repair=False)
    assert parse_plan('[{"id": "b", "tool": "book", "args": {"flight_number": "X", "seats": 1}, "after": null}]',
                      rep, repair=False)[0]["after"] == []


def test_repairs_are_logged_only_for_plans_that_are_accepted():
    # Repairs are logged only for plans that are accepted.
    logged = []

    class Grab(logging.Handler):
        def emit(self, record):
            logged.append(record.getMessage())
    grab = Grab(logging.INFO)
    _logger.addHandler(grab)
    old_level = _logger.level
    _logger.setLevel(logging.INFO)  # setLevel, not .level =, so the logger's cache is cleared
    try:
        expect(PlanRejected, parse_plan, json.dumps([{"id": "b", "tool": "Book", "args": {"seats": "2"}}]), rep)
        assert logged == []  # partly repaired, still rejected (flight_number missing): nothing logged
        parse_plan(json.dumps([{"id": "b", "tool": "Book", "args": {"flight_number": "X", "seats": "2"}}]), rep)
        assert len(logged) == 2 and all("repaired plan" in m for m in logged)
    finally:
        _logger.removeHandler(grab)
        _logger.setLevel(old_level)


def test_empty_responses_are_clear_errors_not_stopiteration():
    # Empty responses are clear errors, not StopIteration / IndexError.
    no_text = NS(stop_reason="end_turn", content=[NS(type="thinking")])
    expect(RuntimeError, plan_with_claude, "x", NS(beta=NS(messages=NS(create=lambda **kw: no_text))), reg)
    expect(RuntimeError, plan_with_openai, "x", NS(chat=NS(completions=NS(create=lambda **kw: NS(choices=[])))),
           reg, model="m")


def test_a_rejected_plan_is_sent_back_to_the_model_once_with_the():
    # A rejected plan is sent back to the model once, with the error, for a fix.
    goals = []

    def picky(goal, client, registry):
        goals.append(goal)
        args = '{"booking_id": "OLD1"}' if len(goals) == 1 else '{"ref": "OLD1"}'
        return parse_plan(json.dumps([{"id": "c", "tool": "cancel_booking", "args_json": args}]), registry)
    agent("cancel OLD1", planner=picky, registry=reg, confirm=lambda st: True, log=quiet)
    assert len(goals) == 2 and "rejected" in goals[1] and "booking_id" in goals[1]
    assert '"args_json": "{\\"booking_id\\"' in goals[1]  # the model sees the plan it wrote
    goals.clear()
    e = expect(PlanRejected, agent, "x", planner=picky, registry=reg, attempts=1, log=quiet)
    assert isinstance(e, ValueError) and "booking_id" in e.text
    for bad_attempts in (0, -1, 1.5, True):
        expect(ValueError, agent, "x", planner=picky, registry=reg, attempts=bad_attempts, log=quiet)
    expect(ValueError, lambda: asyncio.run(aagent("x", planner=picky, registry=reg, attempts=0, log=quiet)))
    assert state["cancelled"] == ["OLD1"] * 4


def test_async_agent():
    global async_planner
    # Async agent: sync confirm runs off the event loop thread; async planners are awaited.
    threads = []

    def confirm_in_thread(steps):
        threads.append(threading.current_thread() is threading.main_thread())
        return True

    async def async_planner(goal, client, registry):
        return parse_plan(text, registry)

    async def run_async():
        await aagent("move", client=Gemini(text), registry=reg, confirm=confirm_in_thread, log=quiet)
        return await aagent("move", planner=async_planner, registry=reg, confirm=lambda st: True, log=quiet)
    assert asyncio.run(run_async())["c"] is None and threads == [False]

    real_input, builtins.input = builtins.input, lambda *a: (_ for _ in ()).throw(EOFError)
    try:
        assert ask_human([{"id": "c", "tool": "cancel_booking", "args": {}, "preview": "p"}]) is False
    finally:
        builtins.input = real_input
    assert state == {"held": 5, "cancelled": ["OLD1"] * 6}


def test_an_async_confirm_reached_through_a_lambda_or_an_object():
    # An async confirm reached through a lambda or an object is awaited: a "no" stays a no.
    async def says_no(steps):
        return False

    class AsyncNo:
        async def __call__(self, steps):
            return False

    for no in (lambda st: says_no(st), AsyncNo()):
        expect(PlanFailed, lambda: asyncio.run(aagent("move", client=Gemini(text), registry=reg,
                                                      confirm=no, log=quiet)))
    # ...and so is a planner reached through a lambda
    plan = asyncio.run(aagent("move", planner=lambda g, c, r: async_planner(g, c, r), registry=reg,
                              confirm=lambda st: True, log=quiet))
    assert plan["c"] is None
    assert state["cancelled"] == ["OLD1"] * 7


def test_only_a_rejected_plan_goes_back_to_the_model():
    global calls
    # Only a rejected plan goes back to the model: a configuration error is raised at once.
    calls = []

    def misconfigured(goal, client, registry):
        calls.append(goal)
        raise ValueError("Invalid API key format: sk-...")
    expect(ValueError, agent, "x", planner=misconfigured, registry=reg, attempts=3, log=quiet)
    assert len(calls) == 1
    expect(ValueError, plan_with_openai, "x", NS(), reg, model="m", json_mode="bogus")


def test_two_different_plans_in_one_answer_are_ambiguous():
    # Two different plans in one answer are ambiguous: refused, never the first one guessed.
    two = ('I considered [{"id": "c", "tool": "cancel_booking", "args": {"ref": "ALL"}}] but use this: '
           '{"steps": [{"id": "s", "tool": "search", "args": {"route": "x"}}]}')
    expect(PlanRejected, parse_plan, two, reg)
    one = json.dumps({"steps": naive})
    assert len(parse_plan(f"```json {one} ``` Again: {one}", reg)) == 3  # the same plan twice is fine


def test_hostile_shapes():
    # Hostile shapes: deep nesting is refused cleanly, and long garbage stays fast.
    for deep in ("[" * 100000 + "]" * 100000, "[" * 100000,
                 '[{"id": "s", "tool": "search", "args": {"route": ' + "[" * 1500 + "]" * 1500 + "}}]"):
        expect(PlanRejected, parse_plan, deep, reg)
    started = _time.perf_counter()
    expect(PlanRejected, parse_plan, "[" * 64 + "a," * 4000 + "]" * 64, reg)
    assert _time.perf_counter() - started < 3, "the literal fallback must stay linear"
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # the library never prints, not even a SyntaxWarning
        parse_plan("[{'id': 's', 'tool': 'search', 'args': {'route': '" + chr(92) + "d'}}]", reg)


def test_a_misnamed_step_field_is_read_as_what_it_means_or():
    # A misnamed step field is read as what it means, or refused; never silently dropped.
    named = parse_plan(json.dumps([{"id": "s", "tool": "search", "arguments": {"route": "x"}},
                                   {"id": "h", "tool": "hold_seat", "args": {"flight": "f"},
                                    "depends_on": ["s"]}]), reg)
    assert named[0]["args"] == {"route": "x"} and named[1]["after"] == ["s"]
    expect(PlanRejected, parse_plan, json.dumps([{"id": "s", "tool": "search", "arguments": {"route": "x"}}]),
           reg, repair=False)
    expect(PlanRejected, parse_plan, json.dumps([{"id": "s", "tool": "search", "args": {"route": "x"},
                                                  "arguments": {"route": "y"}}]), reg)
    assert parse_plan(json.dumps([{"id": "s", "tool": "search", "args": {"route": "x"},
                                   "reason": "find flights first"}]), reg)  # notes are fine
