# Rubicon

**Reversibility-aware execution for AI agent plans.**

Rubicon makes an agent do everything it can take back *first*, check everything it can't take back,
and only then cross the point of no return. If anything fails before that point, every completed
step is undone.

```
plan from the model          what Rubicon runs
───────────────────          ─────────────────────────────────────────────
1. cancel old booking  ──►   1. search flights        (undoable)
2. search flights            2. hold new seat         (undoable)
3. hold new seat             ── gate: dry-run every permanent step, ask a human ──
4. email customer            3. cancel old booking    (PERMANENT)
                             4. email customer        (PERMANENT)
```

---

## Why

Agent frameworks treat every tool call the same. A model that writes "cancel the booking, then
look for a new flight" will happily cancel first, find no seats, and leave the user with nothing.
The pieces to prevent this already exist in isolation — MCP's `destructiveHint` annotations, saga
style compensations, preview-then-confirm flows — but nothing puts them in the execution loop.

Rubicon does. Each tool declares once whether it can be undone; the executor handles the rest:

| Without Rubicon | With Rubicon |
|---|---|
| Steps run in the order the model wrote them | Undoable steps run first, permanent steps last (dependencies respected) |
| A failure halfway leaves half-done work | Completed undoable steps are rolled back in reverse order |
| Validation happens when the permanent call fails | Every permanent step is dry-run *before* anything permanent happens |
| Human approval per call, or none | One approval for the whole commit phase |
| The model can't see which tools are dangerous | Reversibility is part of the tool list the model plans against |

## Features

- **Zero-dependency core** — `rubicon.py` is plain Python 3.10+ standard library.
- **Safe reordering** — topological sort that prefers undoable steps, never breaks a dependency.
- **Pre-commit dry runs** — per-tool `check` functions run before the point of no return.
- **Automatic rollback** — reverse-order undo, with a precise report of what was undone and what
  was already committed.
- **MCP adapter** — load every tool from any MCP server; labels come from tool annotations, with a
  conservative default.
- **LLM planners** — Gemini (default) or Claude turns a plain-language goal into a plan, using
  schema-constrained JSON. The plan is treated as untrusted input and validated before it runs.

## Install

```bash
git clone https://github.com/Ashin-06/rubicon.git
cd rubicon
pip install -e ".[all]"        # or: .[mcp]  .[gemini]  .[claude]  — or nothing, for the core only
```

## Quick start

```python
from rubicon import tool, run, NOOP

@tool(undo=NOOP)                                   # read-only: nothing to undo
def search_flights(route): ...

@tool(undo=lambda hold: release_seat(hold))        # undoable
def hold_seat(flight): ...

@tool(check=lambda ref: assert_booking_exists(ref))  # permanent, with a dry run
def cancel_booking(ref): ...

plan = [
    {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}},
    {"id": "find",   "tool": "search_flights", "args": {"route": "BLR-DEL"}},
    {"id": "hold",   "tool": "hold_seat",      "args": {"flight": "AI101"}},
]

run(plan, confirm=lambda permanent_steps: input("Commit? ") == "y")
# runs: find → hold → [dry-run cancel, ask] → cancel
```

If `hold_seat` raises (no seats), the booking is never cancelled. If the dry run or the human says
no, the held seat is released.

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
| `id` | Unique step name |
| `tool` | A registered tool name |
| `args` | Keyword arguments for the tool |
| `after` | Optional list of step ids that must run first |

**References:** `"$hold"` as a whole argument passes that step's raw result; `"$hold"` inside text
inserts it as text. A reference also adds a dependency automatically. Only real step ids count —
`"$50"` is left alone.

## Use with MCP servers

```python
import asyncio
from rubicon import run
from rubicon_mcp import register_mcp

async def main(session):                       # an initialized mcp.ClientSession
    call = await register_mcp(
        session,
        trust_readonly=True,                    # see "Security" below
        undo={"create_issue": lambda r: call("delete_issue", id=r["id"])},
        checks={"send_email": validate_address},
    )
    results = await asyncio.to_thread(run, plan)  # run() is sync; MCP calls hop back to this loop
```

How each MCP tool is labelled:

| Condition | Label |
|---|---|
| Listed in `undo` | Undoable, with your undo function (which can call another MCP tool) |
| `readOnlyHint: true` **and** `trust_readonly=True` | Undoable (nothing to undo) |
| Anything else | Permanent — MCP's own default is to assume a tool is destructive |

## Let a model write the plan

```python
from rubicon_llm import agent, plan_with_claude

agent("Move my BLR-DEL booking OLD1 to flight AI101 and email me@x.com")      # Gemini
agent("...", planner=plan_with_claude)                                        # Claude
```

- The model sees every registered tool (local or MCP) with its arguments and reversibility label.
- Output is constrained to a JSON schema whose `tool` field is an enum of registered tools.
- Before anything runs, the plan is rejected if it has unknown tools, duplicate ids, references to
  missing steps, cycles, or non-object arguments. A blocked or refused response stops everything.
- `confirm` defaults to an interactive prompt listing the permanent actions.

Keys are read from the environment: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for Gemini,
`ANTHROPIC_API_KEY` for Claude. Default models are `gemini-3.5-flash` and `claude-opus-5-5`;
pass `model=` to change them.

## Live demo

Fake in-memory travel tools — nothing real is booked or sent.

```bash
cp .env.example .env     # add your key
python demo.py           # Gemini
python demo.py claude    # Claude
```

## API reference

### `rubicon`

| Name | Description |
|---|---|
| `tool(undo=None, check=None)` | Decorator registering a tool. `undo(result)` reverses it; `undo=None` means permanent. `check(**args)` is a dry run for permanent tools — raise to block. |
| `NOOP` | Undo function for read-only tools. |
| `run(plan, confirm=..., log=print)` | Execute a plan safely; returns `{step_id: result}`. `confirm(steps)` receives the permanent steps. |
| `order(plan)` | The execution order `run` would use. Validates the plan. |
| `tools_for_llm()` | Registered tools with docs, argument schemas and reversibility. |
| `PlanFailed` | Raised on failure. Attributes: `step`, `cause`, `undone`, `committed`. |
| `TOOLS` | The tool registry. |

### `rubicon_mcp`

| Name | Description |
|---|---|
| `await register_mcp(session, *, prefix="", trust_readonly=False, undo=None, checks=None)` | Register every tool of an MCP session. Returns a sync `call(name, **args)` for use inside undo/check functions. |

### `rubicon_llm`

| Name | Description |
|---|---|
| `agent(goal, planner=plan_with_gemini, client=None, confirm=ask_human, **kw)` | Plan with a model, then `run` it. |
| `plan_with_gemini(goal, client=None, model=...)` | Goal → validated plan via Gemini. |
| `plan_with_claude(goal, client=None, model=...)` | Goal → validated plan via Claude. |
| `ask_human(steps)` | Default terminal confirmation. |

## Security and limitations

- **MCP annotations are claims, not guarantees.** A malicious server could mark a destructive tool
  as read-only. Read-only tools run before the safety gate only if you opt in with
  `trust_readonly=True` — use it for servers you trust.
- **Model output is untrusted.** Plans are validated structurally, but argument *values* are the
  tools' responsibility. Validate at your tool boundary.
- **Undo is compensation, not time travel.** An undo is whatever your function does (release a
  hold, delete a draft). If an undo itself fails, Rubicon keeps unwinding and logs the failure.
- **Permanent → permanent dependencies.** If a permanent step needs another permanent step's
  output, its dry run can only happen just before it runs, after the point of no return. Split the
  plan if that matters.
- **Keep keys out of the repo.** `.env` is git-ignored; use `.env.example` as the template.

## Running the checks

Each module carries its own self-check — no test framework needed, no API keys used.

```bash
python rubicon.py        # core: ordering, dry runs, rollback, references
python rubicon_mcp.py    # MCP adapter against an in-memory MCP server
python rubicon_llm.py    # planners, with stub clients
```

CI runs all three on Python 3.10, 3.12 and 3.13.

## Project layout

```
rubicon.py        core: tool registry, ordering, gate, rollback
rubicon_mcp.py    MCP adapter
rubicon_llm.py    Gemini / Claude planners
demo.py           live demo with fake travel tools
```

## License

[MIT](LICENSE)
