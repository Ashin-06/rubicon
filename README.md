# Rubicon

**Reversibility-aware execution for AI agents.**

Rubicon makes an agent do everything it can take back *first*, check everything it can't take
back, and only then cross the point of no return. If anything fails, is refused, or is interrupted
before that point, every completed step is undone.

```
plan from the model          what Rubicon runs
───────────────────          ─────────────────────────────────────────────
1. cancel old booking  ──►   1. search flights        (undoable)
2. search flights            2. hold new seat         (undoable)
3. hold new seat             ── gate: dry-run every permanent step, ask once ──
4. email customer            3. cancel old booking    (PERMANENT)
                             4. email customer        (PERMANENT)
```

Works with **any model** (Gemini, Claude, OpenAI and every OpenAI-compatible API, or your own
call), **any tool source** (Python functions, sync or async, and MCP servers on SDK 1.x or 2.x),
and **any agent loop** (whole plans, or step-by-step agents through `Transaction`).

---

## Contents

- [Why](#why) · [Features](#features) · [Install](#install) · [Quick start](#quick-start)
- [Two ways to use it](#two-ways-to-use-it): [whole plans](#1-whole-plans-run--arun) ·
  [any agent loop](#2-any-agent-loop-transaction)
- [Plan format](#plan-format) · [MCP servers](#mcp-servers) · [Models](#let-a-model-write-the-plan)
- [Async](#async) · [Multiple agents](#multiple-agents-in-one-process) · [Live demo](#live-demo)
- [API reference](#api-reference) · [Security and limitations](#security-and-limitations) ·
  [Development](#development)

## Why

Agent frameworks treat every tool call the same. A model that writes "cancel the booking, then
look for a new flight" will happily cancel first, find no seats, and leave the user with nothing.
The pieces to prevent this exist in isolation — MCP's `destructiveHint` annotations, saga-style
compensations, preview-then-confirm flows — but nothing puts them in the execution loop.

| Without Rubicon | With Rubicon |
|---|---|
| Steps run in the order the model wrote them | Undoable steps run first, permanent steps last (dependencies respected) |
| A failure halfway leaves half-done work | Completed undoable steps are rolled back in reverse order |
| Ctrl+C or a cancelled task leaves a mess | Interruptions roll back too, then propagate |
| Validation happens when the permanent call fails | Every permanent step is dry-run *before* anything permanent happens |
| Approval per call, or none | One approval for all permanent actions |
| The model can't see which tools are dangerous | Reversibility is part of the tool list the model plans against |

## Features

- **Zero-dependency core** — `rubicon.py` is Python 3.10+ standard library only.
- **Safe ordering** — topological sort that prefers undoable steps and never breaks a dependency.
- **Pre-commit dry runs** — per-tool `check` functions run before the point of no return.
- **Rollback on anything** — errors, declined confirmation, `KeyboardInterrupt`, task
  cancellation. Failed undos are reported, and unwinding continues.
- **`Transaction`** — guards agents that call tools one at a time: undoable calls run, permanent
  calls are dry-run and deferred to commit.
- **Sync and async** — tools, undo, check and confirm functions can be either.
- **MCP adapter** — loads every tool from an MCP server (SDK 1.x and 2.x, with pagination);
  conservative labelling from annotations.
- **Model-agnostic planning** — Gemini, Claude, any OpenAI-compatible API, or your own model via
  `planning_request()` + `parse_plan()`. Model output is validated before anything runs.
- **Isolated registries** — run several agents with different tool sets in one process.

## Install

```bash
git clone https://github.com/Ashin-06/rubicon.git
cd rubicon
pip install -e ".[all]"     # or pick: .[mcp] .[gemini] .[claude] .[openai] — or nothing for the core
```

## Quick start

```python
from rubicon import tool, run, NOOP

@tool(undo=NOOP)                                  # read-only: nothing to undo
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
# runs: find → hold → [dry-run cancel, ask] → cancel
```

If `hold_seat` raises (no seats), the booking is never cancelled. If the dry run fails or the
answer is no, the held seat is released.

`@tool` on its own registers a permanent tool. Without `confirm`, commits are approved
automatically — pass one whenever a human should approve.

## Two ways to use it

### 1. Whole plans: `run` / `arun`

Give Rubicon a complete plan (written by you or by a model) and it executes it safely, as above.

### 2. Any agent loop: `Transaction`

Most agent frameworks call tools one at a time, deciding the next call from the last result.
`Transaction` guards that loop:

- **Undoable tools** run immediately.
- **Permanent tools** are dry-run immediately — so the agent sees a failed check and can react —
  then deferred. The agent receives a `"Scheduled ..."` message instead of a result.
- **On a clean exit**, pending permanent actions are confirmed once and run.
- **On any exception**, every undoable call is rolled back and pending actions are dropped.

```python
from rubicon import Transaction

with Transaction(confirm=ask_human) as tx:
    tools = tx.wrap_all()            # guarded functions: same names, docstrings and signatures
    my_agent.run(goal, tools=tools)  # your agent loop
# commits here; any exception inside the block rolls everything back
```

Wrapped tools keep their name, docstring and signature, so they can be passed wherever your
framework accepts plain Python functions, or wrapped with your framework's own tool decorator.
You can also call tools directly: `tx.call("hold_seat", flight="AI101")`, then `tx.commit()` or
`tx.rollback()`.

> A permanent tool's real result is only available after commit. Design agents so they don't need
> the output of a permanent action to decide the next step, or commit in stages.

## Plan format

Plans are JSON-friendly so a model can write them:

```json
[
  {"id": "hold", "tool": "hold_seat", "args": {"flight": "AI101"}},
  {"id": "mail", "tool": "send_email",
   "args": {"to": "me@x.com", "body": "Moved to $hold"}, "after": ["hold"]}
]
```

| Field | Meaning |
|---|---|
| `id` | Unique step name: letters, digits, `_` or `-` |
| `tool` | A registered tool name |
| `args` | Keyword arguments for the tool (optional) |
| `after` | Step ids that must run first (optional) |

**References:** `"$hold"` as a whole argument passes that step's raw result; `"$hold"` inside text
inserts it as text. References work inside nested lists and objects and add a dependency
automatically. Only real step ids count — `"$50"` is left alone.

Before anything runs, a plan is rejected if it has unknown tools, invalid or duplicate ids,
non-object arguments, references to missing steps, or dependency cycles.

## MCP servers

```python
from rubicon import arun, Transaction
from rubicon_mcp import register_mcp

async def main(session):                      # mcp ClientSession (1.x / 2.x) or mcp.Client (2.x)
    call = await register_mcp(
        session,
        trust_readonly=True,                   # see "Security" below
        undo={"create_issue": lambda r: call("delete_issue", id=r["id"])},
        checks={"send_email": validate_address},
    )
    await arun(plan)                           # or: async with Transaction() as tx
```

| Condition | Label |
|---|---|
| Listed in `undo` | Undoable, with your undo function (sync or async; it can call another MCP tool) |
| `readOnlyHint: true` **and** `trust_readonly=True` | Undoable (nothing to undo) |
| Anything else | Permanent — MCP's own default is to assume a tool is destructive |

Names in `undo` and `checks` must exist on the server — a typo raises instead of silently leaving
a tool unprotected. Tool lists are fetched across all pages. Use `prefix=` to namespace tools when
registering several servers.

## Let a model write the plan

```python
from functools import partial
from rubicon_llm import agent, plan_with_claude, plan_with_openai

agent("Move my booking OLD1 to AI101 and email me@x.com")                 # Gemini (default)
agent("...", planner=plan_with_claude)                                    # Claude
agent("...", planner=partial(plan_with_openai, model="gpt-5.1"))          # OpenAI
agent("...", planner=partial(plan_with_openai, model="llama3.3",          # Ollama, LM Studio,
                             base_url="http://localhost:11434/v1"))       # vLLM, Groq, OpenRouter…
```

| Planner | SDK extra | Key | Default model |
|---|---|---|---|
| `plan_with_gemini` | `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-3.5-flash` |
| `plan_with_claude` | `claude` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| `plan_with_openai` | `openai` | `OPENAI_API_KEY` (none for local servers) | required: `model=` |

`plan_with_openai` sends a strict JSON Schema by default. For servers that only support JSON mode
use `json_mode="object"`; for servers with neither use `json_mode="none"` (the schema goes in the
prompt, and code fences or surrounding prose in the answer are tolerated).

**Any other model or framework:**

```python
from rubicon import run
from rubicon_llm import planning_request, parse_plan

req = planning_request(goal)                   # {"system", "prompt", "schema"}
text = my_model(req["system"], req["prompt"], json_schema=req["schema"])
run(parse_plan(text))
```

The model sees every registered tool with its JSON Schema parameters (taken from type hints, or
from MCP) and its reversibility label. The tool name is constrained to an enum of registered
tools.

## Async

Everything has an async form; tools, undo, check and confirm functions may be `async def`.

| Sync | Async |
|---|---|
| `run(plan)` | `await arun(plan)` |
| `agent(goal)` | `await aagent(goal)` |
| `with Transaction() as tx` | `async with Transaction() as tx` |
| `tx.call / tx.commit / tx.rollback` | `await tx.acall / tx.acommit / tx.arollback` |

Calling a sync entry point from inside a running event loop raises a clear error pointing to the
async form. Wrapped sync tools still work when an async framework calls them on its loop thread.

## Multiple agents in one process

Every function accepts `registry=`. Pass your own dict to keep tool sets apart:

```python
billing = {}

@tool(check=validate, registry=billing)
def charge(amount: int): ...

run(plan, registry=billing)
```

## Live demo

Fake in-memory travel tools — nothing real is booked or sent.

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
| `tool(fn=None, *, undo=None, check=None, name=None, registry=None)` | Register a tool. `undo(result)` reverses it (`None` = permanent); `check(**args)` dry-runs a permanent tool — raise to block. |
| `NOOP` | Undo function for read-only tools. |
| `run(plan, registry=None, confirm=None, log=print)` / `arun(...)` | Execute a plan; returns `{step_id: result}`. `confirm(steps)` receives the permanent steps with references resolved. |
| `Transaction(registry=None, confirm=None, log=print)` | Guard an agent loop. Methods: `call`/`acall`, `commit`/`acommit`, `rollback`/`arollback`, `wrap(name)`, `wrap_all()`; sync and async context manager. |
| `order(plan, registry=None)` | Validate a plan and return the execution order. |
| `tools_for_llm(registry=None)` | Tools as `{name, description, reversible, parameters}` with JSON Schema parameters. |
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
| `agent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=print)` / `aagent(...)` | Plan with a model, then run safely. |
| `plan_with_gemini / plan_with_claude / plan_with_openai` | Goal → validated plan. Signature `(goal, client=None, registry=None, ...)`; pass `client=` to reuse or customise an SDK client. |
| `planning_request(goal, registry=None)` | `{"system", "prompt", "schema"}` for any model. |
| `parse_plan(text, registry=None)` | Model output → validated plan. |
| `plan_schema(registry=None)` | The plan's JSON Schema. |
| `ask_human(steps)` | Terminal confirmation; returns `False` when no terminal is attached. |

## Security and limitations

- **MCP annotations are claims, not guarantees.** A malicious server could mark a destructive tool
  as read-only. Read-only tools run before the safety gate only with `trust_readonly=True` — use
  it for servers you trust.
- **Model output is untrusted.** Plans are validated structurally, but argument *values* are the
  tools' responsibility. Validate at your tool boundary, and use `check` for permanent tools.
- **Undo is compensation, not time travel.** An undo is whatever your function does (release a
  hold, delete a draft). If an undo fails, Rubicon keeps unwinding and reports it in
  `PlanFailed.undo_failed`.
- **Permanent → permanent dependencies.** If a permanent step needs another permanent step's
  output, its dry run can only happen just before it runs, after the point of no return. Split the
  plan if that matters.
- **Process crashes.** Rollback runs in-process. A hard kill (power loss, `SIGKILL`) between steps
  can't be unwound; keep undoable actions cheap to clean up, such as holds with expiry.
- **Keep keys out of the repo.** `.env` is git-ignored; use `.env.example` as the template.

## Development

Each module carries its own self-check — no test framework, no API keys, no network.

```bash
python rubicon.py        # core: ordering, validation, dry runs, rollback, async, Transaction
python rubicon_mcp.py    # MCP adapter against an in-memory MCP server, plus pagination
python rubicon_llm.py    # planners with stub clients, parsing, validation
```

CI runs all three on Python 3.10–3.13, on Linux, Windows and macOS, against MCP SDK 1.x and 2.x.

```
rubicon.py        core: registry, ordering, gate, rollback, Transaction
rubicon_mcp.py    MCP adapter
rubicon_llm.py    Gemini / Claude / OpenAI-compatible planners, bring-your-own-model helpers
demo.py           live demo with fake travel tools
```

See [CHANGELOG.md](CHANGELOG.md) for release notes.

## License

[MIT](LICENSE)
