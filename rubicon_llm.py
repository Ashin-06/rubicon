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
Its output is untrusted: parse_plan() rejects malformed JSON, unknown tools, bad ids, dangling
references and cycles before anything runs, and permanent steps still need confirmation.
"""
import asyncio
import json
import os

from rubicon import TOOLS, arun, order, run, tools_for_llm

GEMINI_MODEL = "gemini-3.5-flash"  # measured 2026-11-08: ~10s per plan; flash-latest / 3.8 timed out
CLAUDE_MODEL = "claude-opus-5-5"

SYSTEM = """You turn a goal into a plan of tool calls for an agent.
Each tool is labelled reversible or not. The executor runs reversible steps first and permanent \
steps last, dry-runs permanent steps before committing, and rolls back on failure, so you don't \
need to order steps for safety. Focus on correctness: include every step the goal needs and no \
others, and use "after" only for real ordering needs. Step ids may use only letters, digits, _ \
and -. To use an earlier step's result, write "$<step id>": as a whole argument value it passes \
the raw result, inside text it inserts it as text. args_json must be a JSON object, encoded as a \
string, matching the tool's parameters. Respond with JSON only."""


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
    t = text.strip()
    if t.startswith("```"):  # ```json ... ``` fences from models without a JSON mode
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except ValueError:
        i, j = t.find("{"), t.rfind("}")
        if i == -1 or j < i:
            raise ValueError(f"model output is not JSON: {text[:200]!r}") from None
        return json.loads(t[i:j + 1])


def parse_plan(text, registry=None):
    """Model output -> validated plan. Accepts {"steps": [...]} with args_json or args."""
    data = _loads(text)
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        raise ValueError('model output has no "steps" list')
    plan = []
    for s in steps:
        if not isinstance(s, dict):
            raise ValueError(f"step is not an object: {s!r}")
        args = s.get("args", {})
        if "args_json" in s:
            raw = s["args_json"]
            args = (json.loads(raw) if raw.strip() else {}) if isinstance(raw, str) else raw
        plan.append({"id": s.get("id"), "tool": s.get("tool"), "args": args, "after": s.get("after", [])})
    order(plan, registry)  # full validation before anything can run
    return plan


def plan_with_gemini(goal, client=None, registry=None, model=GEMINI_MODEL):
    """Gemini via google-genai. Reads GEMINI_API_KEY (or GOOGLE_API_KEY)."""
    from google import genai
    from google.genai import types

    req = planning_request(goal, registry)
    # timeout is in ms so a stalled model can't hang forever
    client = client or genai.Client(http_options=types.HttpOptions(timeout=120_000))
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
    import anthropic

    req = planning_request(goal, registry)
    client = client or anthropic.Anthropic()
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
    return parse_plan(next(b.text for b in resp.content if b.type == "text"), registry)


def plan_with_openai(goal, client=None, registry=None, *, model, base_url=None, api_key=None,
                     json_mode="schema"):
    """Any OpenAI-compatible Chat Completions API: OpenAI, Azure, Ollama, LM Studio, vLLM, Groq,
    OpenRouter, Together, DeepSeek, Mistral, xAI, Gemini's OpenAI endpoint, ...

    json_mode="schema" sends a strict JSON Schema; use "object" for servers that only support
    JSON mode, or "none" for servers with neither (the schema is then put in the prompt).
    Reads OPENAI_API_KEY unless api_key is given; local servers need no key.
    """
    import openai

    req = planning_request(goal, registry)
    key = api_key or os.environ.get("OPENAI_API_KEY") or ("not-needed" if base_url else None)
    client = client or openai.OpenAI(base_url=base_url, api_key=key, timeout=120)
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
    choice = resp.choices[0]
    if not choice.message.content or choice.finish_reason not in ("stop", None):
        raise RuntimeError(f"planner returned no plan: finish_reason={choice.finish_reason} "
                           f"refusal={getattr(choice.message, 'refusal', None)}")
    return parse_plan(choice.message.content, registry)


def ask_human(steps):
    """Default confirmation: list the permanent actions and ask on the terminal."""
    print("Permanent actions about to run:")
    for s in steps:
        print(f"  {s['id']}: {s['tool']}({s.get('args', {})})")
    try:
        return input("Commit? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:  # no terminal attached: never approve by default
        return False


def agent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=print):
    """Plan with a model, then run the plan safely."""
    return run(planner(goal, client, registry), registry, confirm, log)


async def aagent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=print):
    """Async agent(): use inside event loops and with async (e.g. MCP) tools."""
    plan = await asyncio.to_thread(planner, goal, client, registry)
    return await arun(plan, registry, confirm, log)


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
                json.dumps({"steps": [{**naive[0], "id": "has space"}]})]:
        expect(ValueError, parse_plan, bad, reg)
    expect(ValueError, planning_request, "x", {})  # nothing to plan with

    # Async agent, and the default confirmation never approves without a terminal.
    async def run_async():
        return await aagent("move", client=Gemini(text), registry=reg, confirm=lambda st: True, log=quiet)
    assert asyncio.run(run_async())["c"] is None
    import builtins
    real_input, builtins.input = builtins.input, lambda *a: (_ for _ in ()).throw(EOFError)
    try:
        assert ask_human([]) is False
    finally:
        builtins.input = real_input
    assert state == {"held": 4, "cancelled": ["OLD1"] * 4}
    print("all planner checks passed")
