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


def plan_with_claude(goal, client=None, registry=None, model=None, effort="high", fallback=True):
    """Claude via anthropic. Reads ANTHROPIC_API_KEY. Pass effort=None for models or gateways
    without the effort setting, and fallback=False where the server-side fallback beta is not
    available (for example Bedrock or Vertex)."""
    model = model or (os.environ.get("CLAUDE_MODEL") or "").strip() or CLAUDE_MODEL
    req = planning_request(goal, registry)
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    output_config = {"format": {"type": "json_schema", "schema": req["schema"]}}
    if effort:
        output_config["effort"] = effort
    extra = {}
    if fallback:  # retry a refusal on another model; SDK 0.86 has no kwarg for it
        extra = {"betas": ["server-side-fallback-2026-07-01"], "extra_body": {"fallbacks": "default"}}
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        output_config=output_config,
        system=req["system"],
        messages=[{"role": "user", "content": req["prompt"]}],
        **extra,
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
    if confirm is not None and not inspect.iscoroutinefunction(confirm):
        sync_confirm = confirm

        async def confirm(steps):
            answer = await asyncio.to_thread(sync_confirm, steps)
            # a lambda around an async function, or an object with async __call__: its answer
            # is a coroutine, which must be awaited, never taken as a yes
            return await answer if inspect.isawaitable(answer) else answer
    return await arun(plan, registry, confirm, log, **run_kwargs)
