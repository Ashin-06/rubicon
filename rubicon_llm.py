"""An LLM writes the plan from a goal; rubicon executes it safely.

    results = agent("Move my BLR-DEL booking OLD1 to flight AI101 and email me@x.com")      # Gemini
    results = agent("...", planner=plan_with_claude)                                       # Claude

Planners: plan_with_gemini (GEMINI_API_KEY, google-genai) and plan_with_claude (ANTHROPIC_API_KEY,
anthropic). The model sees every registered tool (local or MCP) with its reversibility label and
returns the plan as schema-constrained JSON. The plan is untrusted model output: rubicon.order()
rejects unknown tools, duplicate ids, dangling dependencies and cycles before anything runs, and a
human confirms the permanent steps (default `ask_human`).
"""
import json

from rubicon import TOOLS, run, tools_for_llm

# Measured 2026-11-08 on the planning request: 3.5-flash ~9.5s; gemini-3.8-flash and
# gemini-flash-latest timed out (504) or took minutes for this key.
GEMINI_MODEL = "gemini-3.5-flash"
CLAUDE_MODEL = "claude-opus-5-5"

SYSTEM = """You turn a goal into a plan of tool calls for an agent.
Each tool is labelled reversible or not. The executor runs reversible steps first and permanent \
steps last, dry-runs permanent steps before committing, and rolls back on failure, so you don't \
need to order steps for safety. Focus on correctness: include every step the goal needs and no \
others, and use "after" only for real ordering needs. To use an earlier step's result, write \
"$<step id>": as a whole argument value it passes the raw result, inside text it inserts it as \
text. args_json must be a JSON object matching the tool's args."""


def _schema():
    return {
        "type": "object",
        "properties": {"steps": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "tool": {"type": "string", "enum": sorted(TOOLS)},
                "args_json": {"type": "string"},
                "after": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["id", "tool", "args_json", "after"],
            "additionalProperties": False,
        }}},
        "required": ["steps"],
        "additionalProperties": False,
    }


def _prompt(goal):
    return f"Tools:\n{json.dumps(tools_for_llm(), indent=1)}\n\nGoal: {goal}"


def _parse(text):
    plan = []
    for s in json.loads(text)["steps"]:
        args = json.loads(s["args_json"])
        if not isinstance(args, dict):
            raise ValueError(f"step {s['id']}: args_json is not an object")
        plan.append({"id": s["id"], "tool": s["tool"], "args": args, "after": s["after"]})
    return plan


def plan_with_gemini(goal, client=None, model=GEMINI_MODEL):
    from google import genai
    from google.genai import types

    # reads GEMINI_API_KEY (or GOOGLE_API_KEY); timeout in ms so a stalled model can't hang forever
    client = client or genai.Client(http_options=types.HttpOptions(timeout=120_000))
    resp = client.models.generate_content(
        model=model,
        contents=_prompt(goal),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_json_schema=_schema(),
        ),
    )
    if not resp.text:  # blocked or empty
        why = resp.candidates[0].finish_reason if resp.candidates else resp.prompt_feedback
        raise RuntimeError(f"planner returned no plan: {why}")
    return _parse(resp.text)


def plan_with_claude(goal, client=None, model=CLAUDE_MODEL):
    import anthropic

    client = client or anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        extra_body={"fallbacks": "default"},  # SDK 0.86 has no `fallbacks` kwarg yet
        output_config={"effort": "high", "format": {"type": "json_schema", "schema": _schema()}},
        system=SYSTEM,
        messages=[{"role": "user", "content": _prompt(goal)}],
    )
    if resp.stop_reason != "end_turn":
        raise RuntimeError(f"planner stopped: {resp.stop_reason} {getattr(resp, 'stop_details', '')}")
    return _parse(next(b.text for b in resp.content if b.type == "text"))


def ask_human(steps):
    print("Permanent actions about to run:")
    for s in steps:
        print(f"  {s['id']}: {s['tool']}({s.get('args', {})})")
    return input("Commit? [y/N] ").strip().lower() == "y"


def agent(goal, planner=plan_with_gemini, client=None, confirm=ask_human, **kw):
    return run(planner(goal, client), confirm=confirm, **kw)


if __name__ == "__main__":
    # Offline checks with stub clients (no API key needed).
    from types import SimpleNamespace as NS

    from rubicon import NOOP, PlanFailed, tool

    state = {"held": 0, "cancelled": []}

    @tool(undo=lambda r: state.__setitem__("held", state["held"] - 1))
    def hold_seat(flight: str):
        """Hold a seat; can be released."""
        state["held"] += 1
        return {"flight": flight}

    @tool(check=lambda ref: None)
    def cancel_booking(ref: str):
        """Cancel a booking. PERMANENT."""
        state["cancelled"].append(ref)

    @tool(undo=NOOP)
    def search(route: str):
        """Find flights."""
        return ["AI101"]

    class Gemini:
        def __init__(self, steps=None, text=None):
            self.text = text if text is not None else json.dumps({"steps": steps or []})
            self.kw = None
            self.models = NS(generate_content=self.gen)

        def gen(self, **kw):
            self.kw = kw
            return NS(text=self.text, candidates=[NS(finish_reason="SAFETY")], prompt_feedback=None)

    class Claude:
        def __init__(self, steps=None, stop="end_turn"):
            self.text, self.stop, self.kw = json.dumps({"steps": steps or []}), stop, None
            self.beta = NS(messages=NS(create=self.create))

        def create(self, **kw):
            self.kw = kw
            return NS(stop_reason=self.stop, stop_details=None, content=[NS(type="text", text=self.text)])

    naive = [  # what a model might emit: permanent step first
        {"id": "c", "tool": "cancel_booking", "args_json": '{"ref": "OLD1"}', "after": []},
        {"id": "s", "tool": "search", "args_json": '{"route": "BLR-DEL"}', "after": []},
        {"id": "h", "tool": "hold_seat", "args_json": '{"flight": "AI101"}', "after": ["s"]},
    ]
    quiet = lambda m: None

    # Gemini (default planner): reordered, only the permanent step needs confirmation.
    g = Gemini(naive)
    seen = []
    agent("move my booking", client=g, confirm=lambda st: seen.extend(s["id"] for s in st) or True, log=quiet)
    assert seen == ["c"] and state == {"held": 1, "cancelled": ["OLD1"]}
    cfg = g.kw["config"]  # a real GenerateContentConfig, so field names are validated by the SDK
    assert g.kw["model"] == GEMINI_MODEL and cfg.response_mime_type == "application/json"
    assert cfg.response_json_schema["properties"]["steps"]["items"]["properties"]["tool"]["enum"] \
        == ["cancel_booking", "hold_seat", "search"]
    assert '"reversible": false' in g.kw["contents"]  # labels reach the model

    # Blocked / empty Gemini response -> nothing runs.
    try:
        agent("x", client=Gemini(text=""), confirm=lambda st: True, log=quiet)
        raise AssertionError("should fail")
    except RuntimeError as e:
        assert "SAFETY" in str(e)

    # Human declines -> held seat released, nothing cancelled.
    try:
        agent("move my booking", client=Gemini(naive), confirm=lambda st: False, log=quiet)
        raise AssertionError("should fail")
    except PlanFailed:
        pass
    assert state == {"held": 1, "cancelled": ["OLD1"]}

    # Untrusted output is rejected before anything runs.
    for bad in ([{**naive[2], "after": ["ghost"]}], [{**naive[0], "args_json": "[1]"}]):
        try:
            agent("x", client=Gemini(bad), confirm=lambda st: True, log=quiet)
            raise AssertionError("should fail")
        except ValueError:
            pass

    # Claude planner still works the same way.
    c = Claude(naive)
    agent("move my booking", planner=plan_with_claude, client=c, confirm=lambda st: True, log=quiet)
    assert c.kw["model"] == CLAUDE_MODEL and state == {"held": 2, "cancelled": ["OLD1", "OLD1"]}
    try:
        plan_with_claude("x", client=Claude(stop="refusal"))
        raise AssertionError("should fail")
    except RuntimeError as e:
        assert "refusal" in str(e)
    print("all planner checks passed")
