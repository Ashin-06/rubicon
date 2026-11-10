"""Let any LLM write the plan; Rubicon executes it safely.

    agent("Move booking OLD1 to AI101 and email me@example.com")                       # Gemini
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

import ast
import asyncio
import inspect
import json
import logging
import math
import os
import re
import warnings

from rubicon import TOOLS, _schema, arun, order, run, tools_for_llm

_logger = logging.getLogger("rubicon")

__all__ = ["CLAUDE_MODEL", "GEMINI_MODEL", "PlanRejected", "SYSTEM", "aagent", "agent", "ask_human",
           "parse_plan", "plan_schema", "plan_with_claude", "plan_with_gemini", "plan_with_openai",
           "planning_request"]

# Default models. The GEMINI_MODEL / CLAUDE_MODEL environment variables override them, read
# each time a planner runs, so loading a .env after importing this module still works.
# gemini-3.5-flash measured 2026-11-08 at ~10s per plan; flash-latest and 3.8 timed out then.
GEMINI_MODEL = "gemini-3.5-flash"
CLAUDE_MODEL = "claude-opus-5-5"


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


_MAX_STARTS = 64  # bracket positions tried when looking for JSON inside prose
_NOT_READABLE = (ValueError, SyntaxError, MemoryError, RecursionError, TypeError)


def _plan_like(value):
    """A {"steps": [...]} object, or a bare list of steps (objects that name a tool)."""
    return (isinstance(value, dict) and isinstance(value.get("steps"), list)) or (
        isinstance(value, list) and bool(value) and all(isinstance(x, dict) and "tool" in x for x in value))


def _close(text, start):
    """End (exclusive) of the bracketed value opening at start, skipping quoted strings; None if
    it never closes. One pass over the text, so trying many starts stays cheap."""
    depth, quote, i, n = 0, None, start, len(text)
    while i < n:
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _values(text, read):
    """Each top-level bracketed value in the text that `read` accepts, in order. A value found
    inside one already read (the steps list of a plan) is part of it, not another candidate."""
    found, taken_until, tried = [], -1, 0
    for start, ch in enumerate(text):
        if ch not in "{[" or start < taken_until:
            continue
        tried += 1
        if tried > _MAX_STARTS:
            break
        end = _close(text, start)
        if end is None:
            continue
        try:
            value = read(text[start:end])
        except _NOT_READABLE:
            continue
        if isinstance(value, (dict, list)):
            found.append(value)
            taken_until = end
    return found


def _pick(values):
    """The plan among the values, else the first value at all, so a stray "[1]" in the prose
    before the plan doesn't win. Two different plans are ambiguous: neither is guessed."""
    plans = [v for v in values if _plan_like(v)]
    if any(p != plans[0] for p in plans[1:]):
        raise ValueError(f"model output contains {len(plans)} different plans; write exactly one")
    if plans:
        return plans[0]
    return values[0] if values else None


def _quiet_literal(text):
    with warnings.catch_warnings():  # e.g. SyntaxWarning for "\d": the library never prints
        warnings.simplefilter("ignore")
        return ast.literal_eval(text)


def _literal(text):
    """Python-style dict/list output (single quotes, True/None) from models without a JSON mode.
    ast.literal_eval only reads literals; it never runs code."""
    return _pick(_values(text, _quiet_literal))


def _whole_literal(text):
    try:
        value = _quiet_literal(text.strip())
    except _NOT_READABLE:
        return None
    return value if isinstance(value, (dict, list)) else None


def _loads(text, repairs=None):
    """Parse the JSON object or array in the text, ignoring fences and prose around it.
    With a repairs list, Python-style literals are accepted too (and noted in the list)."""
    t = text.strip()
    try:
        return json.loads(t)
    except RecursionError:
        raise ValueError("model output is nested too deeply") from None
    except ValueError:
        pass  # not bare JSON: look for JSON inside prose or fences below
    value = _pick(_values(t, json.loads))
    if value is not None and (_plan_like(value) or repairs is None):
        return value
    if repairs is not None:
        literal = _literal(t)
        if literal is not None and (_plan_like(literal) or value is None):
            repairs.append("read Python-style literal output as JSON")
            return literal
    if value is not None:
        return value
    raise ValueError(f"model output is not JSON: {text[:200]!r}")


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower()) if isinstance(name, str) else None


def _unique(name, options):
    """The single option `name` can mean, ignoring case and _ - spaces; None if zero or several."""
    key = _norm(name)
    matches = [o for o in options if _norm(o) == key]
    return matches[0] if key and len(matches) == 1 else None


_INT = re.compile(r"[+-]?\d+")


def _coerce(value, prop):
    """Fix a value whose JSON type plainly differs from the schema's. References are untouched."""
    kind = prop.get("type") if isinstance(prop, dict) else None
    if isinstance(value, str) and value.lstrip().startswith("$"):
        return value
    if kind == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value) if isinstance(value, int) or math.isfinite(value) else value
    if kind == "integer":
        if isinstance(value, str) and _INT.fullmatch(value.strip()):
            return int(value.strip())
        if isinstance(value, float) and value.is_integer():
            return int(value)
    if kind == "number" and isinstance(value, str):
        text = value.strip()
        if _INT.fullmatch(text):
            return int(text)
        try:
            number = float(text)
        except ValueError:
            return value
        return number if number == number and abs(number) != float("inf") else value  # not NaN/inf
    if kind == "boolean" and isinstance(value, str) and value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    return value


def _repair(plan, tools, repairs):
    """Deterministic fixes for small, unambiguous mistakes, so the model isn't asked again."""
    for step in plan:
        name = step["tool"]
        if isinstance(name, str) and name not in tools:
            fixed = _unique(name, tools)
            if fixed:
                repairs.append(f"step {step['id']!r}: tool {name!r} -> {fixed!r}")
                step["tool"] = name = fixed
        if not isinstance(name, str) or name not in tools or not isinstance(step["args"], dict):
            continue
        props = (_schema(tools[name].fn) or {}).get("properties") or {}
        if not props:
            continue
        args = step["args"]
        free = [p for p in props if p not in args]
        targets = {k: _unique(k, free) for k in args if k not in props}
        for key, fixed in targets.items():
            # two keys aiming at one parameter ("flightNumber" and "flight-number") is ambiguous
            if fixed and list(targets.values()).count(fixed) == 1:
                repairs.append(f"step {step['id']!r}: argument {key!r} -> {fixed!r}")
                args[fixed] = args.pop(key)
        for key, value in list(args.items()):
            new = _coerce(value, props.get(key))
            if new is not value:  # _coerce returns the same object when it changes nothing
                repairs.append(f"step {step['id']!r}: {key}={value!r} -> {new!r}")
                args[key] = new


def parse_plan(text, registry=None, *, repair=True, repairs=None):
    """Model output -> validated plan.

    Accepts {"steps": [...]} or a bare [...] list, with "args_json" (a JSON string) or "args".
    With repair=True, small unambiguous mistakes are fixed locally before validation: a tool or
    argument name that differs only in case or separators, a number sent as a string (or the
    reverse) where the schema says otherwise, "true"/"false" for booleans, doubly-encoded
    args_json and Python-style literals. Each change is logged, and appended to `repairs` if you
    pass a list. Raises PlanRejected (a ValueError) carrying the model's text.
    """
    changes = [] if repair else None
    try:
        plan = _parse_plan(text, registry, changes)
    except ValueError as e:  # includes json.JSONDecodeError
        raise PlanRejected(str(e), text) from e
    finally:
        if repairs is not None:
            repairs.extend(changes or [])
    for change in changes or []:  # logged only for plans that were accepted
        _logger.info("repaired plan: %s", change)
    return plan


# What models write for a step's fields, normalised (see _norm). A misnamed field must never be
# dropped silently: the tool would run without its arguments, or before what it depends on.
_FIELD_NAMES = {
    "id": "id", "stepid": "id",
    "tool": "tool", "toolname": "tool",
    "args": "args", "arguments": "args", "params": "args", "parameters": "args", "kwargs": "args",
    "input": "args", "inputs": "args",
    "argsjson": "args_json", "argumentsjson": "args_json",
    "after": "after", "dependson": "after", "dependencies": "after", "deps": "after",
    "requires": "after", "needs": "after",
}


def _fields(s, repairs):
    """A step with its fields under their canonical names; other keys (notes, reasons) are ignored."""
    out = {}
    for key, value in s.items():
        name = _FIELD_NAMES.get(_norm(key))
        if name is None:
            continue
        if name in out:
            raise ValueError(f"step {s.get('id')!r}: both {out[name][0]!r} and {key!r} given")
        if key != name:
            if repairs is None:
                raise ValueError(f"step {s.get('id')!r}: unknown field {key!r}; did you mean {name!r}?")
            repairs.append(f"step {s.get('id')!r}: read field {key!r} as {name!r}")
        out[name] = (key, value)
    return {name: value for name, (_, value) in out.items()}


def _parse_plan(text, registry, repairs):
    data = _loads(text, repairs)
    steps = data if isinstance(data, list) else data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        raise ValueError('model output has no "steps" list')
    plan = []
    for s in steps:
        if not isinstance(s, dict):
            raise ValueError(f"step is not an object: {s!r}")
        s = _fields(s, repairs)
        args, raw = s.get("args"), s.get("args_json")
        if isinstance(raw, str) and raw.strip() == "null":
            raw = None  # JSON null written as text: no args_json, so "args" (if any) stands
        if isinstance(raw, str) and raw.strip():
            try:
                args = json.loads(raw)
            except RecursionError:
                raise ValueError(f"step {s.get('id')!r}: args_json is nested too deeply") from None
            except ValueError:
                # the field holds just the arguments, so read it whole: a fragment of broken JSON
                # (an inner object of a cut-off one) must not pass for the arguments
                args = _whole_literal(raw) if repairs is not None else None
                if args is None:
                    raise
                repairs.append(f"step {s.get('id')!r}: read Python-style args_json")
            if isinstance(args, str) and repairs is not None:  # JSON encoded twice
                try:
                    args = json.loads(args)
                    repairs.append(f"step {s.get('id')!r}: decoded doubly-encoded args_json")
                except ValueError:
                    pass  # a plain string, not encoded twice: order() reports it as invalid args
        elif raw is not None and not isinstance(raw, str):  # an object, or a list/number order() rejects
            args = raw
        after = s.get("after")
        plan.append({"id": s.get("id"), "tool": s.get("tool"),
                     "args": {} if args is None else args, "after": [] if after is None else after})
    if repairs is not None:
        _repair(plan, _registry(registry), repairs)
    order(plan, registry)  # full validation before anything can run
    return plan


def plan_with_gemini(goal, client=None, registry=None, model=None):
    """Gemini via google-genai. Reads GEMINI_API_KEY (or GOOGLE_API_KEY)."""
    from google import genai
    from google.genai import types

    model = model or (os.environ.get("GEMINI_MODEL") or "").strip() or GEMINI_MODEL
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
            max_output_tokens=16000,  # bounds cost; a plan is far smaller, and a cut-off one is rejected
            system_instruction=req["system"],
            response_mime_type="application/json",
            response_json_schema=req["schema"],
        ),
    )
    if not resp.text:  # blocked or empty
        why = resp.candidates[0].finish_reason if resp.candidates else resp.prompt_feedback
        raise RuntimeError(f"planner returned no plan: {why}")
    return parse_plan(resp.text, registry)


def plan_with_claude(goal, client=None, registry=None, model=None):
    """Claude via anthropic. Reads ANTHROPIC_API_KEY."""
    model = model or (os.environ.get("CLAUDE_MODEL") or "").strip() or CLAUDE_MODEL
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
    if json_mode not in ("schema", "object", "none"):
        raise ValueError(f'json_mode must be "schema", "object" or "none", got {json_mode!r}')
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
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError(f"attempts must be a whole number of at least 1, got {attempts!r}")


def agent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=None,
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
        except PlanRejected as e:  # only a rejected plan goes back to the model, not a config error
            error = e
            if n == attempts - 1:
                raise
    return run(plan, registry, confirm, log, **run_kwargs)


async def aagent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=None,
                 *, attempts=2, **run_kwargs):
    """Async agent(): use inside event loops and with async (e.g. MCP) tools.

    The planner may be sync (run in a thread) or async. A sync confirm, such as the default
    terminal prompt, runs in a thread so it doesn't block the event loop.
    """
    _check_attempts(attempts)

    async def make_plan(g):
        if inspect.iscoroutinefunction(planner):
            return await planner(g, client, registry)
        plan = await asyncio.to_thread(planner, g, client, registry)
        return await plan if inspect.isawaitable(plan) else plan  # a lambda around an async planner

    error = None
    for n in range(attempts):
        try:
            plan = await make_plan(goal if error is None else _retry_goal(goal, error))
            break
        except PlanRejected as e:
            error = e
            if n == attempts - 1:
                raise
    if not inspect.iscoroutinefunction(confirm):
        sync_confirm = confirm

        async def confirm(steps):
            answer = await asyncio.to_thread(sync_confirm, steps)
            # a lambda around an async function, or an object with async __call__: its answer
            # is a coroutine, which must be awaited, never taken as a yes
            return await answer if inspect.isawaitable(answer) else answer
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
    assert cfg.max_output_tokens == 16000  # output (and its cost) is bounded
    assert cfg.response_json_schema["properties"]["steps"]["items"]["properties"]["tool"]["enum"] == \
        ["cancel_booking", "hold_seat", "search"]
    assert '"reversible": false' in g.kw["contents"] and '"type": "string"' in g.kw["contents"]
    assert "SAFETY" in str(expect(RuntimeError, plan_with_gemini, "x", Gemini(""), reg))

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

    # Lenient parsing: braces in trailing prose, bare lists, empty args_json, null after.
    fenced = "Here is the plan:\n```json\n" + text + "\n```\nNote: replaces {OLD1}."
    assert [s["id"] for s in parse_plan(fenced, reg)] == ["c", "s", "h"]
    assert len(parse_plan(json.dumps(naive), reg)) == 3
    loose = parse_plan('{"steps": [{"id": "s", "tool": "search", "args": {"route": "x"}, '
                       '"args_json": "", "after": null}]}', reg)
    assert loose == [{"id": "s", "tool": "search", "args": {"route": "x"}, "after": []}]
    assert parse_plan('[{"id": "s", "tool": "search", "args": {"route": "x"}, "args_json": null}]', reg)

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

    # Bracketed prose before the plan doesn't hijack parsing (JSON or Python-style).
    good = [{"id": "b", "tool": "book", "args": {"flight_number": "X", "seats": 1}}]
    assert parse_plan("See note [1] below:\n```json\n" + json.dumps(good) + "\n```", rep)[0]["id"] == "b"
    assert parse_plan("Option {2} is best: " + json.dumps({"steps": good}), rep)[0]["id"] == "b"
    assert parse_plan("Step [1]:\n" + repr(good).replace('"', "'"), rep)[0]["id"] == "b"

    # Two keys aiming at one parameter are ambiguous: neither is renamed.
    @tool(undo=NOOP, registry=rep)
    def flex(flight_number: str = "", **extra):
        return flight_number
    fixes = []
    both = parse_plan(json.dumps([{"id": "f", "tool": "flex", "args": {"flightNumber": "A", "flight-number": "B"}}]),
                      rep, repairs=fixes)
    assert both[0]["args"] == {"flightNumber": "A", "flight-number": "B"} and fixes == []

    # NaN never produces a bogus repair, and isn't turned into the string "nan".
    fixes = []
    nan_plan = parse_plan('[{"id": "b", "tool": "book", "args": {"flight_number": NaN, "seats": 1, "price": NaN}}]',
                          rep, repairs=fixes)
    assert fixes == [] and math.isnan(nan_plan[0]["args"]["flight_number"])

    # "after" must be a list; only null means "none", even in strict mode.
    for bad_after in ('""', "0", "{}"):
        expect(PlanRejected, parse_plan, f'[{{"id": "b", "tool": "book", "args": {{"flight_number": "X", "seats": 1}}, '
               f'"after": {bad_after}}}]', rep, repair=False)
    assert parse_plan('[{"id": "b", "tool": "book", "args": {"flight_number": "X", "seats": 1}, "after": null}]',
                      rep, repair=False)[0]["after"] == []

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
    for bad_attempts in (0, -1, 1.5, True):
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

    # Only a rejected plan goes back to the model: a configuration error is raised at once.
    calls = []

    def misconfigured(goal, client, registry):
        calls.append(goal)
        raise ValueError("Invalid API key format: sk-...")
    expect(ValueError, agent, "x", planner=misconfigured, registry=reg, attempts=3, log=quiet)
    assert len(calls) == 1
    expect(ValueError, plan_with_openai, "x", NS(), reg, model="m", json_mode="bogus")

    # Two different plans in one answer are ambiguous: refused, never the first one guessed.
    two = ('I considered [{"id": "c", "tool": "cancel_booking", "args": {"ref": "ALL"}}] but use this: '
           '{"steps": [{"id": "s", "tool": "search", "args": {"route": "x"}}]}')
    expect(PlanRejected, parse_plan, two, reg)
    one = json.dumps({"steps": naive})
    assert len(parse_plan(f"```json {one} ``` Again: {one}", reg)) == 3  # the same plan twice is fine

    # Hostile shapes: deep nesting is refused cleanly, and long garbage stays fast.
    for deep in ("[" * 100000 + "]" * 100000, "[" * 100000,
                 '[{"id": "s", "tool": "search", "args": {"route": ' + "[" * 1500 + "]" * 1500 + "}}]"):
        expect(PlanRejected, parse_plan, deep, reg)
    import time as _time
    started = _time.perf_counter()
    expect(PlanRejected, parse_plan, "[" * 64 + "a," * 4000 + "]" * 64, reg)
    assert _time.perf_counter() - started < 3, "the literal fallback must stay linear"
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("error")  # the library never prints, not even a SyntaxWarning
        parse_plan("[{'id': 's', 'tool': 'search', 'args': {'route': '" + chr(92) + "d'}}]", reg)

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
    print("all planner checks passed")
