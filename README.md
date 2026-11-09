# Rubicon

Reversibility-aware execution for AI agents.

An agent should do everything it can take back first, check everything it can't take back, and
only then cross the point of no return. Rubicon enforces that. If anything fails, gets refused or
is interrupted before that point, every step that already ran is undone.

```
plan from the model           what Rubicon runs
-------------------           ------------------------------------------------
1. cancel old booking   ->    1. search flights        (undoable)
2. search flights             2. hold new seat         (undoable)
3. hold new seat              -- gate: dry-run every permanent step, ask once --
4. email customer             3. cancel old booking    (PERMANENT)
                              4. email customer        (PERMANENT)
```

It works with any model (Gemini, Claude, OpenAI and anything OpenAI-compatible, or your own
call), any tool source (plain Python functions, sync or async, and MCP servers on SDK 1.x or 2.x)
and any agent loop (whole plans, or step-by-step agents through `Transaction`).

## Contents

- [Why](#why)
- [Features](#features)
- [Install](#install)
- [Quick start](#quick-start)
- [Two ways to use it](#two-ways-to-use-it)
- [Plan format](#plan-format)
- [MCP servers](#mcp-servers)
- [Let a model write the plan](#let-a-model-write-the-plan)
- [Async](#async)
- [Multiple agents in one process](#multiple-agents-in-one-process)
- [Live demo](#live-demo)
- [API reference](#api-reference)
- [Security and limitations](#security-and-limitations)
- [Development](#development)

## Why

Agent frameworks treat every tool call the same. A model that writes "cancel the booking, then
look for a new flight" will cancel first, find no seats, and leave the user with nothing.

The pieces that would prevent this already exist on their own: MCP's `destructiveHint`
annotations, saga-style compensations, preview-then-confirm flows. Nothing puts them into the
execution loop. Rubicon does.

| Without Rubicon | With Rubicon |
|---|---|
| Steps run in the order the model wrote them | Undoable steps run first, permanent steps last, dependencies respected |
| A failure halfway leaves half-done work | Steps that already ran are undone in reverse order |
| Ctrl+C or a cancelled task leaves a mess | Interruptions roll back too, then propagate |
| Validation happens when the permanent call fails | Every permanent step is dry-run before anything permanent happens |
| Approval per call, or none | One approval for all permanent actions |
| The model can't see which tools are dangerous | Reversibility is part of the tool list the model plans against |

## Features

- Zero-dependency core: `rubicon.py` uses only the Python 3.10+ standard library.
- Safe ordering: a topological sort that prefers undoable steps and never breaks a dependency.
- Dry runs before commit: per-tool `check` functions run before the point of no return.
- Rollback on anything: errors, a declined confirmation, `KeyboardInterrupt`, task cancellation.
  If an undo fails, it is reported and unwinding carries on.
- `Transaction` for agents that call tools one at a time: undoable calls run, permanent calls are
  dry-run and held until commit.
- Sync and async: tools, undo, check and confirm functions can be either.
- MCP adapter: loads every tool from an MCP server (SDK 1.x and 2.x, all pages) and labels it
  conservatively from its annotations.
- Model-agnostic planning: Gemini, Claude, any OpenAI-compatible API, or your own model through
  `planning_request()` and `parse_plan()`. Model output is validated before anything runs.
- Separate registries, so several agents with different tools can share one process.

## Install

```bash
git clone https://github.com/Ashin-06/rubicon.git
cd rubicon
pip install -e ".[all]"
```

Extras: `mcp`, `gemini`, `claude`, `openai`, or `all`. The core needs none of them.

## Quick start

```python
from rubicon import tool, run, NOOP

@tool(undo=NOOP)                                  # read-only, nothing to undo
def search_flights(route: str) -> list: ...

@tool(undo=lambda hold: release_seat(hold))       # undoable
def hold_seat(flight: str) -> dict: ...

@tool(check=lambda ref: assert_exists(ref))       # permanent, dry-run first
def cancel_booking(ref: str) -> str: ...

plan = [
    {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}},
    {"id": "find",   "tool": "search_flights", "args": {"route": "BLR-DEL"}},
    {"id": "hold",   "tool": "hold_seat",      "args": {"flight": "AI101"}},
]

run(plan, confirm=lambda permanent_steps: input("Commit? ") == "y")
# runs: find, hold, [dry-run cancel, ask], cancel
```

If `hold_seat` raises because there are no seats, the booking is never cancelled. If the dry run
fails or the answer is no, the held seat is released.

`@tool` with no arguments registers a permanent tool. Without `confirm`, commits are approved
automatically, so pass one whenever a person should approve.

## Two ways to use it

### 1. Whole plans: `run` and `arun`

Hand Rubicon a complete plan, written by you or by a model, and it runs it safely as shown above.

### 2. Any agent loop: `Transaction`

Most agent frameworks call tools one at a time and decide the next call from the last result.
`Transaction` guards that loop:

- Undoable tools run immediately.
- Permanent tools are dry-run immediately, so the agent sees a failed check and can react. Then
  they are held, and the agent gets a `"Scheduled ..."` message instead of a result.
- On a clean exit, the held actions are confirmed once and run.
- On any exception, every undoable call is rolled back and the held actions are dropped.

```python
from rubicon import Transaction

with Transaction(confirm=ask_human) as tx:
    tools = tx.wrap_all()            # guarded functions with the same names, docs and signatures
    my_agent.run(goal, tools=tools)  # your agent loop
# commits here; an exception inside the block rolls everything back
```

Wrapped tools keep their name, docstring and signature. Pass them wherever your framework accepts
plain Python functions, or wrap them with the framework's own tool decorator. You can also call
tools directly with `tx.call("hold_seat", flight="AI101")` and finish with `tx.commit()` or
`tx.rollback()`.

> A permanent tool's real result only exists after commit. Design the agent so it doesn't need
> that result to choose its next step, or commit in stages.

## Plan format

Plans are plain JSON so a model can write them:

```json
[
  {"id": "hold", "tool": "hold_seat", "args": {"flight": "AI101"}},
  {"id": "mail", "tool": "send_email",
   "args": {"to": "me@x.com", "body": "Moved to $hold.flight"}, "after": ["hold"]}
]
```

| Field | Meaning |
|---|---|
| `id` | Unique step name: letters, digits, `_` or `-` |
| `tool` | A registered tool name |
| `args` | Keyword arguments for the tool (optional) |
| `after` | Step ids that must run first (optional) |

References:

- `"$hold"` as a whole argument passes that step's raw result.
- `"$hold"` inside text inserts the result as text.
- `"$hold.flight"` or `"$hold.legs.0"` picks a field or list item out of the result. As a whole
  argument a missing field is an error; inside text it is left as written.
- References work inside nested lists and objects and add a dependency automatically. Only real
  step ids count, so `"$50"` stays as it is.

A plan is rejected before anything runs if it has unknown tools, invalid or duplicate ids,
arguments that aren't objects, references to missing steps, or dependency cycles.

## MCP servers

```python
from rubicon import arun, Transaction
from rubicon_mcp import register_mcp

async def main(session):                      # mcp ClientSession (1.x or 2.x) or mcp.Client (2.x)
    call = await register_mcp(
        session,
        trust_readonly=True,                   # see "Security and limitations"
        undo={"create_issue": lambda r: call("delete_issue", id=r["id"])},
        checks={"send_email": validate_address},
    )
    await arun(plan)                           # or: async with Transaction() as tx
```

| Condition | Label |
|---|---|
| Listed in `undo` | Undoable with your undo function (sync or async, it can call another MCP tool) |
| `readOnlyHint: true` and `trust_readonly=True` | Undoable, nothing to undo |
| Anything else | Permanent, matching MCP's own default of assuming a tool is destructive |

Names in `undo` and `checks` must exist on the server. A typo raises an error instead of quietly
leaving a tool unprotected. Tool lists are fetched across all pages, and `prefix=` namespaces
tools when you register several servers.

## Let a model write the plan

```python
from functools import partial
from rubicon_llm import agent, plan_with_claude, plan_with_openai

agent("Move my booking OLD1 to AI101 and email me@x.com")                 # Gemini (default)
agent("...", planner=plan_with_claude)                                    # Claude
agent("...", planner=partial(plan_with_openai, model="gpt-5.1"))          # OpenAI
agent("...", planner=partial(plan_with_openai, model="llama3.3",          # Ollama, LM Studio,
                             base_url="http://localhost:11434/v1"))       # vLLM, Groq, OpenRouter, ...
```

| Planner | Extra | Key | Default model |
|---|---|---|---|
| `plan_with_gemini` | `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-3.5-flash` |
| `plan_with_claude` | `claude` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| `plan_with_openai` | `openai` | `OPENAI_API_KEY` (not needed for local servers) | none, pass `model=` |

`plan_with_openai` sends a strict JSON Schema by default. Use `json_mode="object"` for servers that
only have a JSON mode, and `json_mode="none"` for servers with neither. In that case the schema goes
into the prompt, and code fences or extra text around the answer are tolerated.

To use any other model or framework:

```python
from rubicon import run
from rubicon_llm import planning_request, parse_plan

req = planning_request(goal)                   # {"system", "prompt", "schema"}
text = my_model(req["system"], req["prompt"], json_schema=req["schema"])
run(parse_plan(text))
```

The model sees every registered tool with its JSON Schema parameters (from type hints, or from
MCP) and whether it can be undone. The tool name in the schema is limited to registered tools.

## Async

Everything has an async form, and tools, undo, check and confirm functions can be `async def`.

| Sync | Async |
|---|---|
| `run(plan)` | `await arun(plan)` |
| `agent(goal)` | `await aagent(goal)` |
| `with Transaction() as tx` | `async with Transaction() as tx` |
| `tx.call`, `tx.commit`, `tx.rollback` | `await tx.acall`, `tx.acommit`, `tx.arollback` |

Calling a sync entry point from inside a running event loop raises an error that points you to the
async form. Wrapped sync tools still work when an async framework calls them on its loop thread.

## Multiple agents in one process

Every function takes `registry=`. Pass your own dict to keep tool sets apart:

```python
billing = {}

@tool(check=validate, registry=billing)
def charge(amount: int): ...

run(plan, registry=billing)
```

## Live demo

The demo uses fake in-memory travel tools, so nothing real is booked or sent.

```bash
cp .env.example .env       # add your key
python demo.py             # Gemini
python demo.py claude      # Claude
OPENAI_MODEL=llama3.3 OPENAI_BASE_URL=http://localhost:11434/v1 python demo.py openai
```

## API reference

### `rubicon`

| Name | Description |
|---|---|
| `tool(fn=None, *, undo=None, check=None, name=None, registry=None)` | Register a tool. `undo(result)` reverses it (`None` means permanent). `check(**args)` dry-runs a permanent tool; raise to block. |
| `NOOP` | Undo function for read-only tools. |
| `run(plan, registry=None, confirm=None, log=print)`, `arun(...)` | Run a plan and return `{step_id: result}`. `confirm(steps)` gets the permanent steps with references resolved. |
| `Transaction(registry=None, confirm=None, log=print)` | Guard an agent loop. Methods: `call`/`acall`, `commit`/`acommit`, `rollback`/`arollback`, `wrap(name)`, `wrap_all()`. Works as a sync or async context manager. |
| `order(plan, registry=None)` | Validate a plan and return the order it would run in. |
| `tools_for_llm(registry=None)` | Tools as `{name, description, reversible, parameters}`, with JSON Schema parameters. |
| `PlanFailed` | Raised after rollback. Attributes: `step`, `cause`, `undone`, `committed`, `undo_failed`, `results`. |
| `TOOLS` | The default registry. |

### `rubicon_mcp`

| Name | Description |
|---|---|
| `await register_mcp(session, *, prefix="", trust_readonly=False, undo=None, checks=None, registry=None)` | Register every tool of an MCP server. Returns an async `call(name, **args)`. |
| `await list_all_tools(session)` | All tools, following pagination. |

### `rubicon_llm`

| Name | Description |
|---|---|
| `agent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=print)`, `aagent(...)` | Plan with a model, then run the plan safely. |
| `plan_with_gemini`, `plan_with_claude`, `plan_with_openai` | Turn a goal into a validated plan. Signature `(goal, client=None, registry=None, ...)`; pass `client=` to reuse or customise an SDK client. |
| `planning_request(goal, registry=None)` | `{"system", "prompt", "schema"}` for any model. |
| `parse_plan(text, registry=None)` | Turn model output into a validated plan. |
| `plan_schema(registry=None)` | The plan's JSON Schema. |
| `ask_human(steps)` | Terminal confirmation. Returns `False` when no terminal is attached. |

## Security and limitations

- MCP annotations are claims, not guarantees. A malicious server could mark a destructive tool as
  read-only, so read-only tools only run before the safety gate with `trust_readonly=True`. Use it
  for servers you trust.
- Model output is untrusted. Plans are checked for structure, but argument values are up to your
  tools. Validate at the tool boundary and give permanent tools a `check`.
- An undo is a compensation, not a time machine: it is whatever your function does, such as
  releasing a hold or deleting a draft. If an undo fails, Rubicon keeps unwinding and lists it in
  `PlanFailed.undo_failed`.
- If a permanent step needs another permanent step's output, its dry run can only happen right
  before it runs, after the point of no return. Split the plan if that matters.
- Rollback happens in-process. A hard kill (power loss, `SIGKILL`) between steps can't be unwound,
  so keep undoable actions cheap to clean up, for example holds that expire.
- Keep keys out of the repo. `.env` is git-ignored; `.env.example` is the template.

## Development

Each module has its own self-check. No test framework, API keys or network are needed.

```bash
python rubicon.py        # core: ordering, validation, references, dry runs, rollback, async, Transaction
python rubicon_mcp.py    # MCP adapter against an in-memory MCP server, plus pagination
python rubicon_llm.py    # planners with stub clients, parsing, validation
```

CI runs all three on Python 3.10 to 3.13, on Linux, Windows and macOS, against MCP SDK 1.x and 2.x.

```
rubicon.py        core: registry, ordering, gate, rollback, Transaction
rubicon_mcp.py    MCP adapter
rubicon_llm.py    Gemini, Claude and OpenAI-compatible planners, plus bring-your-own-model helpers
demo.py           live demo with fake travel tools
```

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
