"""Let any LLM write the plan; Rubicon executes it safely.

    agent("Move booking OLD1 to AI101 and email me@x.com")                       # Gemini
    agent("...", planner=plan_with_claude)                                       # Claude
    agent("...", planner=partial(plan_with_openai, model="gpt-5.1"))             # OpenAI
    agent("...", planner=partial(plan_with_openai, model="llama3.3",             # any OpenAI-
                                 base_url="http://localhost:11434/v1"))          # compatible API

Any other model or framework: build the request yourself and parse the answer.

    req = planning_request(goal)          # {"system", "prompt", "schema"}
    text = my_model(req["system"], req["prompt"], json_schema=req["schema"])
    run(parse_plan(text))

The model sees every registered tool (local or MCP) with its JSON Schema and reversibility label.
Its output is untrusted: parse_plan() rejects malformed JSON, unknown tools, bad ids, arguments
that don't fit the tool, dangling references and cycles before anything runs, and permanent steps
still need confirmation. agent() sends a rejected plan's error back to the model for a fix.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os

from rubicon import TOOLS, arun, order, run, tools_for_llm

__all__ = ["CLAUDE_MODEL", "GEMINI_MODEL", "PlanRejected", "SYSTEM", "aagent", "agent", "ask_human",
           "parse_plan", "plan_schema", "plan_with_claude", "plan_with_gemini", "plan_with_openai",
           "planning_request"]

# Defaults can be overridden with environment variables (or a .env loaded before import).
# gemini-3.5-flash measured 2026-11-08 at ~10s per plan; flash-latest and 3.8 timed out then.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL") or "gemini-3.5-flash"
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL") or "claude-opus-5-5"


class PlanRejected(ValueError):
    """A model's plan failed parsing or validation. `text` is what the model wrote."""

    def __init__(self, message, text):
        super().__init__(message)
        self.text = text

SYSTEM = """You turn a goal into a plan of tool calls for an agent.
Each tool is labelled reversible or not. The executor runs reversible steps first and permanent \
steps last, dry-runs permanent steps before committing, and rolls back on failure, so you don't \
need to order steps for safety. Focus on correctness: include every step the goal needs and no \
others, and use "after" only for real ordering needs. Step ids must start with a letter and may \
contain letters, digits, _ and - (for example "hold_seat" or "step1", never "1"). To use an \
earlier step's result, write "$<step id>": as a whole argument value it passes the raw result, \
inside text it inserts it as text, and "$<step id>.<key>" picks one field of a result that is an \
object. args_json must be a JSON object, encoded as a string, using exactly the parameter names \
in the tool's schema and every required one. Respond with JSON only."""


def _registry(registry):
    reg = TOOLS if registry is None else registry
    if not reg:
        raise ValueError("no tools registered")
    return reg


def plan_schema(registry=None):
    """JSON Schema of a plan (strict-mode compatible: every field required, no extras)."""
    return {
        "type": "object",
        "properties": {"steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "tool": {"type": "string", "enum": sorted(_registry(registry))},
                "args_json": {"type": "string"},
                "after": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "tool", "args_json", "after"],
            "additionalProperties": False,
        }}},
        "required": ["steps"],
        "additionalProperties": False,
    }


def planning_request(goal, registry=None):
    """Everything a model needs to write a plan: {"system", "prompt", "schema"}."""
    tools = json.dumps(tools_for_llm(_registry(registry)), indent=1, default=str)
    return {"system": SYSTEM, "prompt": f"Tools:\n{tools}\n\nGoal: {goal}", "schema": plan_schema(registry)}


def _loads(text):
    """Parse the first JSON object or array in the text, ignoring fences and prose around it."""
    t = text.strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    decoder = json.JSONDecoder()
    for i, ch in enumerate(t):
        if ch in "{[":
            try:
                return decoder.raw_decode(t, i)[0]
            except ValueError:
                continue
    raise ValueError(f"model output is not JSON: {text[:200]!r}")


def parse_plan(text, registry=None):
    """Model output -> validated plan.

    Accepts {"steps": [...]} or a bare [...] list, with "args_json" (a JSON string) or "args".
    Raises PlanRejected (a ValueError) carrying the model's text.
    """
    try:
        return _parse_plan(text, registry)
    except ValueError as e:  # includes json.JSONDecodeError
        raise PlanRejected(str(e), text) from e


def _parse_plan(text, registry):
    data = _loads(text)
    steps = data if isinstance(data, list) else data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        raise ValueError('model output has no "steps" list')
    plan = []
    for s in steps:
        if not isinstance(s, dict):
            raise ValueError(f"step is not an object: {s!r}")
        args, raw = s.get("args"), s.get("args_json")
        if isinstance(raw, str) and raw.strip():
            args = json.loads(raw)
        elif isinstance(raw, dict):
            args = raw
        plan.append({"id": s.get("id"), "tool": s.get("tool"),
                     "args": {} if args is None else args, "after": s.get("after") or []})
    order(plan, registry)  # full validation before anything can run
    return plan


def plan_with_gemini(goal, client=None, registry=None, model=GEMINI_MODEL):
    """Gemini via google-genai. Reads GEMINI_API_KEY (or GOOGLE_API_KEY)."""
    from google import genai
    from google.genai import types

    req = planning_request(goal, registry)
    # timeout is in ms so a stalled model can't hang forever; retry rate limits and overloads
    # (429/5xx) like the Anthropic and OpenAI SDKs do by default
    client = client or genai.Client(http_options=types.HttpOptions(
        timeout=120_000,
        retry_options=types.HttpRetryOptions(attempts=4, http_status_codes=[408, 429, 500, 502, 503, 504]),
    ))
    resp = client.models.generate_content(
        model=model,
        contents=req["prompt"],
        config=types.GenerateContentConfig(
            system_instruction=req["system"],
            response_mime_type="application/json",
            response_json_schema=req["schema"],
        ),
    )
    if not resp.text:  # blocked or empty
        why = resp.candidates[0].finish_reason if resp.candidates else resp.prompt_feedback
        raise RuntimeError(f"planner returned no plan: {why}")
    return parse_plan(resp.text, registry)


def plan_with_claude(goal, client=None, registry=None, model=CLAUDE_MODEL):
    """Claude via anthropic. Reads ANTHROPIC_API_KEY."""
    req = planning_request(goal, registry)
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},  # retry a refusal on another model; SDK 0.86 has no kwarg
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": req["schema"]}},
        system=req["system"],
        messages=[{"role": "user", "content": req["prompt"]}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"planner stopped: {resp.stop_reason} {getattr(resp, 'stop_details', '')}")
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        raise RuntimeError("planner returned no text block")
    return parse_plan(text, registry)


def plan_with_openai(goal, client=None, registry=None, *, model, base_url=None, api_key=None,
                     json_mode="schema"):
    """Any OpenAI-compatible Chat Completions API: OpenAI, Azure, Ollama, LM Studio, vLLM, Groq,
    OpenRouter, Together, DeepSeek, Mistral, xAI, Gemini's OpenAI endpoint, ...

    json_mode="schema" sends a strict JSON Schema; use "object" for servers that only support
    JSON mode, or "none" for servers with neither (the schema is then put in the prompt).
    Reads OPENAI_API_KEY unless api_key is given; local servers need no key.
    """
    req = planning_request(goal, registry)
    if client is None:
        import openai
        key = api_key or os.environ.get("OPENAI_API_KEY") or ("not-needed" if base_url else None)
        client = openai.OpenAI(base_url=base_url, api_key=key, timeout=120)
    prompt, extra = req["prompt"], {}
    if json_mode == "schema":
        extra["response_format"] = {"type": "json_schema",
                                    "json_schema": {"name": "plan", "strict": True, "schema": req["schema"]}}
    else:
        prompt += f"\n\nRespond with JSON matching this schema:\n{json.dumps(req['schema'])}"
        if json_mode == "object":
            extra["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": req["system"]}, {"role": "user", "content": prompt}],
        **extra,
    )
    if not resp.choices:
        raise RuntimeError("planner returned no choices")
    choice = resp.choices[0]
    # servers disagree on the normal finish value ("stop", "eos", "end_turn", None), so only
    # reject the ones that mean the answer is cut off or withheld
    if not choice.message.content or choice.finish_reason in ("length", "content_filter", "tool_calls"):
        raise RuntimeError(f"planner returned no plan: finish_reason={choice.finish_reason} "
                           f"refusal={getattr(choice.message, 'refusal', None)}")
    return parse_plan(choice.message.content, registry)


def ask_human(steps):
    """Default confirmation: list the permanent actions and ask on the terminal."""
    print("Permanent actions about to run:")
    for s in steps:
        print(f"  {s['id']}: {s['tool']}({s.get('args', {})})")
        if s.get("preview") is not None:
            print(f"      {s['preview']}")
    try:
        return input("Commit? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:  # no terminal attached: never approve by default
        return False


def _retry_goal(goal, error):
    """The goal again, with the rejected plan and the reason, since planners are single-turn."""
    previous = getattr(error, "text", None)
    shown = f"\n\nYour previous plan was:\n{previous[:4000]}" if previous else ""
    return f"{goal}{shown}\n\nIt was rejected: {error}\nWrite a corrected plan."


def _check_attempts(attempts):
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError(f"attempts must be a whole number of at least 1, got {attempts!r}")


def agent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=print,
          *, attempts=2, **run_kwargs):
    """Plan with a model, then run the plan safely.

    If the plan is rejected (bad JSON, unknown tool, wrong argument names, ...), the error is
    sent back to the model for a corrected plan, up to `attempts` plans in total. run_kwargs go
    to run(): parallel=, on_event=, journal=.
    """
    _check_attempts(attempts)
    error = None
    for n in range(attempts):
        try:
            plan = planner(goal if error is None else _retry_goal(goal, error), client, registry)
            break
        except ValueError as e:
            error = e
            if n == attempts - 1:
                raise
    return run(plan, registry, confirm, log, **run_kwargs)


async def aagent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=print,
                 *, attempts=2, **run_kwargs):
    """Async agent(): use inside event loops and with async (e.g. MCP) tools.

    The planner may be sync (run in a thread) or async. A sync confirm, such as the default
    terminal prompt, runs in a thread so it doesn't block the event loop.
    """
    _check_attempts(attempts)

    async def make_plan(g):
        if inspect.iscoroutinefunction(planner):
            return await planner(g, client, registry)
        return await asyncio.to_thread(planner, g, client, registry)

    error = None
    for n in range(attempts):
        try:
            plan = await make_plan(goal if error is None else _retry_goal(goal, error))
            break
        except ValueError as e:
            error = e
            if n == attempts - 1:
                raise
    if not inspect.iscoroutinefunction(confirm):
        sync_confirm = confirm

        async def confirm(steps):
            return await asyncio.to_thread(sync_confirm, steps)
    return await arun(plan, registry, confirm, log, **run_kwargs)


if __name__ == "__main__":
    # Offline checks with stub clients: no API keys, no network.
    from functools import partial
    from types import SimpleNamespace as NS

    from rubicon import NOOP, PlanFailed, tool

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

    # Gemini (default planner): reordered; only the permanent step needs confirmation.
    g, seen = Gemini(text), []
    agent("move", client=g, registry=reg, confirm=lambda st: seen.extend(s["id"] for s in st) or True, log=quiet)
    assert seen == ["c"] and state == {"held": 1, "cancelled": ["OLD1"]}
    cfg = g.kw["config"]  # a real GenerateContentConfig, so field names are validated by the SDK
    assert g.kw["model"] == GEMINI_MODEL and cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema["properties"]["steps"]["items"]["properties"]["tool"]["enum"] == \
        ["cancel_booking", "hold_seat", "search"]
    assert '"reversible": false' in g.kw["contents"] and '"type": "string"' in g.kw["contents"]
    assert "SAFETY" in str(expect(RuntimeError, plan_with_gemini, "x", Gemini(""), reg))

    # Claude.
    c = Claude(text)
    agent("move", planner=plan_with_claude, client=c, registry=reg, confirm=lambda st: True, log=quiet)
    assert c.kw["model"] == CLAUDE_MODEL and c.kw["extra_body"] == {"fallbacks": "default"}
    assert "refusal" in str(expect(RuntimeError, plan_with_claude, "x", Claude(text, "refusal"), reg))

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

    # Bring-your-own model: request + parse, with args as an object instead of args_json.
    req = planning_request("move", reg)
    assert set(req) == {"system", "prompt", "schema"}
    assert parse_plan('{"steps": [{"id": "s", "tool": "search", "args": {"route": "x"}}]}', reg)[0]["args"] == {"route": "x"}
    assert parse_plan('{"steps": [{"id": "s", "tool": "search", "args_json": {"route": "y"}}]}', reg)[0]["args"] == {"route": "y"}

    # Human declines -> held seat released, nothing cancelled.
    expect(PlanFailed, agent, "move", client=Gemini(text), registry=reg, confirm=lambda st: False, log=quiet)
    assert state == {"held": 3, "cancelled": ["OLD1"] * 3}

    # Untrusted output is rejected before anything runs.
    for bad in ["not json", '{"plan": []}', '{"steps": [1]}',
                json.dumps({"steps": [{**naive[2], "after": ["ghost"]}]}),
                json.dumps({"steps": [{**naive[0], "args_json": "[1]"}]}),
                json.dumps({"steps": [{**naive[0], "id": "has space"}]}),
                json.dumps({"steps": [{**naive[0], "id": "1"}]}),
                json.dumps({"steps": [{**naive[0], "args_json": '{"booking_id": "OLD1"}'}]})]:
        expect(ValueError, parse_plan, bad, reg)
    expect(ValueError, planning_request, "x", {})  # nothing to plan with

    # Lenient parsing: braces in trailing prose, bare lists, empty args_json, null after.
    fenced = "Here is the plan:\n```json\n" + text + "\n```\nNote: replaces {OLD1}."
    assert [s["id"] for s in parse_plan(fenced, reg)] == ["c", "s", "h"]
    assert len(parse_plan(json.dumps(naive), reg)) == 3
    loose = parse_plan('{"steps": [{"id": "s", "tool": "search", "args": {"route": "x"}, '
                       '"args_json": "", "after": null}]}', reg)
    assert loose == [{"id": "s", "tool": "search", "args": {"route": "x"}, "after": []}]
    assert parse_plan('[{"id": "s", "tool": "search", "args": {"route": "x"}, "args_json": null}]', reg)

    # Empty responses are clear errors, not StopIteration / IndexError.
    no_text = NS(stop_reason="end_turn", content=[NS(type="thinking")])
    expect(RuntimeError, plan_with_claude, "x", NS(beta=NS(messages=NS(create=lambda **kw: no_text))), reg)
    expect(RuntimeError, plan_with_openai, "x", NS(chat=NS(completions=NS(create=lambda **kw: NS(choices=[])))),
           reg, model="m")

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
    for bad_attempts in (0, -1, 1.5):
        expect(ValueError, agent, "x", planner=picky, registry=reg, attempts=bad_attempts, log=quiet)
    expect(ValueError, lambda: asyncio.run(aagent("x", planner=picky, registry=reg, attempts=0, log=quiet)))
    assert state["cancelled"] == ["OLD1"] * 4

    # Async agent: sync confirm runs off the event loop thread; async planners are awaited.
    import threading
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

    import builtins
    real_input, builtins.input = builtins.input, lambda *a: (_ for _ in ()).throw(EOFError)
    try:
        assert ask_human([{"id": "c", "tool": "cancel_booking", "args": {}, "preview": "p"}]) is False
    finally:
        builtins.input = real_input
    assert state == {"held": 5, "cancelled": ["OLD1"] * 6}
    print("all planner checks passed")
