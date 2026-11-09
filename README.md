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
- [What happens when something fails](#what-happens-when-something-fails)
- [Two ways to use it](#two-ways-to-use-it)
- [Plan format](#plan-format)
- [Tools: undo, checks, previews, retries](#tools-undo-checks-previews-retries)
- [MCP servers](#mcp-servers)
- [Let a model write the plan](#let-a-model-write-the-plan)
- [Parallel steps, events and crash recovery](#parallel-steps-events-and-crash-recovery)
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
| A wrong argument name is found when the call fails | Every step's arguments are checked against the tool before anything runs |
| Validation happens when the permanent call fails | Every permanent step is dry-run before anything permanent happens |
| Approval per call, or none | One approval for all permanent actions, with a preview of each |
| The model can't see which tools are dangerous | Reversibility is part of the tool list the model plans against |

## Features

- Zero-dependency core: `rubicon.py` uses only the Python 3.10+ standard library.
- Safe ordering: a topological sort that prefers undoable steps and never breaks a dependency.
- Plans are checked before anything runs: unknown tools, bad ids, arguments that don't fit the
  tool, missing references, cycles.
- Dry runs before commit, with an optional human-readable preview for the confirmation.
- Rollback on anything: errors, a declined confirmation, `KeyboardInterrupt`, task cancellation.
  Failed undos are retried, then reported, and unwinding carries on.
- Partial commits handled correctly: undoable steps that a committed permanent step relies on are
  never undone.
- `Transaction` for agents that call tools one at a time: undoable calls run, permanent calls are
  dry-run and held until commit.
- Parallel execution of independent undoable steps.
- Crash-safe journal with `recover()`: a JSON-lines file, SQLite (several processes), or your own
  store such as Postgres or Redis. One recovery at a time per journal, enforced with an OS lock.
- Dead letters: every step Rubicon can't fix by itself stays recorded until a person resolves it,
  with a callback to push it to your alerting.
- Structured events and standard `logging` (logger `"rubicon"`); the library never prints.
- Per-tool retries and timeouts.
- Sync and async: tools, undo, check and confirm functions can be either.
- MCP adapter: loads every tool from an MCP server (SDK 1.x and 2.x, all pages), gives each a real
  Python signature, and labels it conservatively from its annotations.
- Model-agnostic planning: Gemini, Claude, any OpenAI-compatible API, or your own model through
  `planning_request()` and `parse_plan()`. Small mistakes in a plan are repaired locally; a plan
  that is still wrong goes back to the model for a fix.
- Separate registries, so several agents with different tools can share one process.

## Install

```bash
git clone https://github.com/Ashin-06/rubicon.git
cd rubicon
pip install -e ".[all]"
```

Extras: `mcp`, `gemini`, `claude`, `openai`, `all`, and `dev` (pytest). The core needs none of
them.

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

## What happens when something fails

| When it fails | What Rubicon does |
|---|---|
| A step before any permanent action | Undoes every step that ran, newest first |
| A dry run, or the confirmation is declined | Same: nothing permanent has happened, so everything is undone |
| A permanent step, after another permanent step already committed | Keeps the steps the committed one relied on (undoing them would leave things worse), undoes only steps that ran after the last commit, and lists the permanent steps that never ran |
| Ctrl+C or task cancellation | Undoes what it can, then lets the interruption through |
| An undo itself | Retries it, then reports it in `PlanFailed.undo_failed` and keeps unwinding |

Every failure raises `PlanFailed` with `step`, `cause`, `undone`, `committed`, `not_run`,
`undo_failed` and `results`.

For example: hold seat AI101, cancel booking OLD1, then the email fails. OLD1 is already gone, so
releasing the AI101 seat would leave the user with no flight at all. Rubicon keeps the seat,
reports `committed=["cancel"]` and `not_run=["mail"]`, and lets you retry the email.

## Two ways to use it

### 1. Whole plans: `run` and `arun`

Hand Rubicon a complete plan, written by you or by a model, and it runs it safely as shown above.

### 2. Any agent loop: `Transaction`

Most agent frameworks call tools one at a time and decide the next call from the last result.
`Transaction` guards that loop:

- Undoable tools run immediately.
- Permanent tools are dry-run immediately, so the agent sees a failed check and can react. Then
  they are held, and the agent gets a `"Scheduled ..."` message (with the preview, if any)
  instead of a result.
- On a clean exit, the held actions are confirmed once and run.
- On any exception, every undoable call is rolled back and the held actions are dropped.
  `tx.undone` and `tx.undo_failed` say what happened.

```python
from rubicon import Transaction

with Transaction(confirm=ask_human) as tx:
    tools = tx.wrap_all()            # guarded functions with the same names, docs and signatures
    my_agent.run(goal, tools=tools)  # your agent loop
# commits here; an exception inside the block rolls everything back
```

Wrapped tools keep their name, docstring and signature, take positional and keyword arguments
like the original, and are async when the tool or its check is async. Pass them wherever your
framework accepts plain Python functions, or wrap them with the framework's own tool decorator.
You can also call tools directly with `tx.call("hold_seat", "AI101")` and finish with
`tx.commit()` or `tx.rollback()`.

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
| `id` | Unique step name. Starts with a letter or `_`, then letters, digits, `_` or `-` |
| `tool` | A registered tool name |
| `args` | Keyword arguments for the tool (optional). Checked against the tool's signature |
| `after` | Step ids that must run first (optional) |

References:

- `"$hold"` as a whole argument passes that step's raw result.
- `"$hold"` inside text inserts the result as text.
- `"$hold.flight"` or `"$hold.legs.0"` picks a field or list item out of the result.
- When the path reaches a plain value, the rest is ordinary text: with `file = {"name": "report"}`,
  `"$file.name.txt"` becomes `"report.txt"`.
- A key or index missing from a dict or list is treated as a typo and fails the step, so a
  message never goes out with a dumped dictionary in it.
- References work inside nested lists and objects and add a dependency automatically. Only real
  step ids count, and ids can't start with a digit, so `"$50"` or `"$1"` stay as money.

## Tools: undo, checks, previews, retries

```python
@tool(
    undo=lambda result, args: delete_file(args["path"]),  # undo(result) or undo(result, args)
    retries=0,          # extra attempts for the tool itself; only for tools safe to repeat
    undo_retries=2,     # extra attempts for undo, with backoff (default 2)
    timeout=30,         # seconds per attempt, for async functions
)  # a sync tool waits between retries synchronously; an async one with asyncio.sleep
def put_file(path: str, content: str) -> str: ...

def preview_cancel(ref):
    booking = find(ref)                      # raise to block the commit
    return f"Cancel {ref} ({booking.route}), fee {booking.fee}"

@tool(check=preview_cancel)                  # a check's return value is shown when confirming
def cancel_booking(ref: str): ...
```

- An undo that takes two arguments gets the call's arguments as well as its result, for tools
  that only return `"ok"`.
- A check receives the tool's arguments (with defaults filled in), limited to the parameters it
  declares, so `def must_exist(ref)` works for `def cancel(ref, reason="user")`. Raise to block;
  return a string to show the person confirming what will happen. `confirm(steps)` receives each
  step's `id`, `tool`, resolved `args` and `preview`.
- Use `UNSET` as the default of an optional parameter that should be left out of the call
  entirely (not passed, not shown at confirmation, not journaled), for example so a remote API
  applies its own default. MCP tools use it for every optional parameter.

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

- Each tool gets a real Python signature from its input schema (keyword-only, with type
  annotations), so frameworks that inspect signatures see the actual parameters. Optional
  parameters you leave out are not sent, so the server's own defaults apply.
- Arguments the schema doesn't declare are rejected before anything runs, unless the schema sets
  `additionalProperties` to `true` (or to a schema) or declares no properties at all. Many servers
  silently ignore unknown arguments, so a hallucinated one would otherwise go unnoticed.
- Names in `undo` and `checks` must exist on the server; a typo raises an error.
- Tool lists are fetched across all pages. `prefix=` namespaces tools (including their function
  names) when you register several servers.
- Text results that are JSON are parsed; `"NaN"` and `"Infinity"` stay text. Images and other
  non-text content are returned instead of dropped.

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

- Override the default models with `GEMINI_MODEL` and `CLAUDE_MODEL` (environment or `.env`), or
  pass `model=`. The variables are read each time a planner runs, so loading a `.env` after
  importing works too.
- Before rejecting a plan, `parse_plan()` fixes small mistakes locally when there is exactly one
  right answer, so they cost no extra model call:
  - a tool or argument name that differs only in case or separators (`Hold_Seat`, `flightNumber`
    for `flight_number`), when exactly one tool or parameter matches,
  - a value whose type plainly differs from the schema (`"2"` for an integer, `101` for a string,
    `"true"` for a boolean),
  - Python-style output (single quotes, `True`, `None`) and doubly-encoded `args_json`.

  References (`"$step..."`) are never changed, and nothing ambiguous is guessed. Every repair is
  logged; pass `repairs=[]` to collect them, or `repair=False` for strict parsing.
- If a plan is still rejected (bad JSON, unknown tool, wrong argument names, bad ids), `agent()`
  sends the model its previous plan and the error, and asks for a corrected plan. `attempts=2`
  by default. A final rejection raises `PlanRejected`, a `ValueError` whose `text` is what the
  model wrote.
- `plan_with_openai` sends a strict JSON Schema by default. Use `json_mode="object"` for servers
  that only have a JSON mode, and `json_mode="none"` for servers with neither.
- Answers wrapped in code fences or surrounded by text are handled, as are bare lists of steps
  and `args` given as an object instead of `args_json`.
- `agent()` passes `parallel=`, `on_event=` and `journal=` through to `run()`.

To use any other model or framework:

```python
from rubicon import run
from rubicon_llm import planning_request, parse_plan

req = planning_request(goal)                   # {"system", "prompt", "schema"}
text = my_model(req["system"], req["prompt"], json_schema=req["schema"])
run(parse_plan(text))
```

The model sees every registered tool with its JSON Schema parameters (from type hints, including
`Optional`, `Literal`, enums and typed lists, or from MCP) and whether it can be undone.

## Parallel steps, events and crash recovery

```python
run(plan, parallel=True)            # independent undoable steps run at the same time
run(plan, on_event=print)           # structured events
run(plan, journal="rubicon.jsonl")  # crash-safe log

from rubicon import recover
recover("rubicon.jsonl")            # after a restart: undo what a crash left half-done
```

- `parallel=True` runs undoable steps in waves of steps that don't depend on each other. If one
  fails, or the whole run is cancelled or interrupted, the rest of the wave is stopped first and
  then everything completed is undone, so no step can finish after the rollback. Permanent steps
  always run one at a time, in order.
- `on_event` receives dicts such as `{"event": "step_done", "step": "hold", "tool": "hold_seat",
  "duration_ms": 12.3, "tx": "...", "time": ...}`. Events: `step_started`, `step_done`,
  `check_passed`, `gate_confirmed`, `gate_declined`, `step_committed`, `step_undone`,
  `undo_failed`. Forward them to your logger or tracing system. If your `on_event` or `log`
  callback raises, the error is logged as a warning and execution carries on unchanged.
- `journal=` records every step durably with the arguments and result needed to undo it. If the
  process is killed, `recover()` undoes the steps of every unfinished transaction since its last
  commit, using the same tools.
- Recovery journals each undo as it happens, so if recovery itself is interrupted, running it
  again never undoes anything twice. A transaction is marked finished only when every undo
  succeeded; anything in a report's `undo_failed` (for example because a service was down) is
  retried the next time you run `recover()`. The same applies to an in-process rollback whose
  undo failed.
- Values that aren't JSON-serializable are marked individually in the journal. A step whose
  undo would need such a value is reported in `undo_failed` instead of being replayed wrongly.
- Every step is journaled as "started" before it runs. If the crash happened while a permanent
  step was running, nobody knows whether it went through, so `recover()` undoes nothing for that
  transaction: undoing what it relied on could leave things worse. It reports the step in
  `needs_attention` and the steps it kept in `kept`, for a person to check. An undoable step that
  started but never reported a result is listed in `uncertain`.
- A `Transaction` that commits in stages is covered too: a crash in a later stage undoes only
  that stage.

Each report is `{"tx", "undone", "undo_failed", "committed", "uncertain", "needs_attention",
"kept"}`. Only one `recover()` runs per journal at a time; a second one raises `JournalBusy`
instead of undoing the same step twice.

`recover()` is safe to run while other workers are using the same journal. Each transaction
records which process and machine runs it, and `recover()` leaves alone any transaction whose
process is still alive on this machine, or that belongs to another machine (it can't check
that from here). Run `recover()` on each machine, or pass `include_live=True` when you know the
other processes are gone.

### Where the journal lives

```python
from rubicon import FileJournal, SQLiteJournal

run(plan, journal="rubicon.jsonl")                  # same as FileJournal("rubicon.jsonl")
run(plan, journal=SQLiteJournal("rubicon.db"))      # many processes and threads on one machine
run(plan, journal=MyPostgresJournal(...))           # several machines: bring your own store
```

| Journal | Use it for |
|---|---|
| `FileJournal(path)` | One machine. JSON lines, each write flushed and fsynced; writes from several processes are serialized with an OS file lock. |
| `SQLiteJournal(path)` | One machine, many processes or threads. Standard library only (WAL mode, full sync). |
| Your own object | Several machines (Postgres, Redis, a queue). It needs `append(records)` and `records()`; optionally `lock()`, a context manager held during recovery, and `drop(tx_ids)` for compaction. Records are plain JSON-serializable dicts. |

### Dead letters

Some steps need a person: an undo that kept failing (the service was down), a crash in the middle
of a step, or a permanent step whose outcome is unknown. These are dead letters. They stay in the
journal until someone resolves them, so nothing gets lost between restarts.

```python
from rubicon import dead_letters, acknowledge, recover

recover(journal, on_dead_letter=page_on_call)   # called once per new dead letter
for letter in dead_letters(journal):            # {"tx", "step", "kind", "reason", "time"}
    ...
acknowledge(journal, tx, step)    # that step is handled; Rubicon won't undo it
acknowledge(journal, tx)          # the whole transaction is handled as it is
```

- `kind` is `"undo_failed"`, `"uncertain"` (crash during an undoable step) or
  `"in_flight_permanent"` (crash during a permanent step).
- A failed undo is retried by every `recover()` run until it succeeds or is acknowledged.
- For a crash during a permanent step, acknowledge just that step if it did not happen, and the
  next `recover()` undoes the steps it was keeping; acknowledge the whole transaction if it did.
- `acknowledge()` raises `ValueError` for a transaction or step the journal doesn't contain, so
  a typo can't leave the real dead letter open.

### Keeping the journal small

`compact(journal)` removes settled transactions and keeps everything that is unfinished, owes an
undo or has an unresolved dead letter. Run it on a schedule or at startup. With `FileJournal`,
compaction rewrites the file atomically while writers wait for its lock.

### Logging

The library logs to the standard `logging` module under the name `"rubicon"` and never prints:
progress at INFO, problems at WARNING. Pass `log=print` (or any function) to `run()`,
`Transaction()` or `agent()` to send progress messages somewhere else.

## Async

Everything has an async form, and tools, undo, check and confirm functions can be `async def`.

| Sync | Async |
|---|---|
| `run(plan)` | `await arun(plan)` |
| `agent(goal)` | `await aagent(goal)` |
| `recover(path)` | `await arecover(path)` |
| `with Transaction() as tx` | `async with Transaction() as tx` |
| `tx.call`, `tx.commit`, `tx.rollback` | `await tx.acall`, `tx.acommit`, `tx.arollback` |

- Calling `run()` from inside a running event loop raises an error that points you to `arun()`.
- `tx.call()` and wrapped sync tools still work when an async framework calls them on its loop
  thread, as long as the tool and its check are sync.
- `aagent()` accepts sync or async planners, and runs a sync `confirm` (like the terminal prompt)
  in a thread so it doesn't block the event loop.

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
| `tool(fn=None, *, undo=None, check=None, name=None, registry=None, retries=0, undo_retries=2, timeout=None)` | Register a tool. See [Tools](#tools-undo-checks-previews-retries). |
| `NOOP` | Undo function for read-only tools. |
| `run(plan, registry=None, confirm=None, log=None, *, parallel=False, on_event=None, journal=None)`, `arun(...)` | Run a plan and return `{step_id: result}`. |
| `Transaction(registry=None, confirm=None, log=None, *, on_event=None, journal=None)` | Guard an agent loop. Methods: `call`/`acall`, `commit`/`acommit`, `rollback`/`arollback`, `wrap(name)`, `wrap_all()`. Attributes: `results`, `committed`, `undone`, `undo_failed`. Works as a sync or async context manager. |
| `recover(journal, registry=None, log=None, *, on_dead_letter=None, include_live=False)`, `arecover(...)` | Undo transactions a crash left unfinished, leaving running ones alone. Returns one report per transaction (see [crash recovery](#parallel-steps-events-and-crash-recovery)). Safe to run repeatedly and alongside live workers. |
| `dead_letters(journal)` | Steps that need a person, until resolved. |
| `acknowledge(journal, tx, step=None, note="")` | Mark a dead letter (or a whole transaction) as handled. |
| `compact(journal)` | Remove settled transactions; returns how many. |
| `FileJournal(path)`, `SQLiteJournal(path)` | Built-in journals. `journal=` also accepts a path or your own object. |
| `JournalBusy` | Raised when another process holds the journal's recovery lock. |
| `order(plan, registry=None)` | Validate a plan and return the order it would run in. |
| `tools_for_llm(registry=None)` | Tools as `{name, description, reversible, parameters}`, with JSON Schema parameters. |
| `PlanFailed` | Raised after rollback. Attributes: `step`, `cause`, `undone`, `committed`, `not_run`, `undo_failed`, `results`. |
| `UNSET` | Default for optional parameters that should be left out of the call entirely. |
| `TOOLS` | The default registry. |

### `rubicon_mcp`

| Name | Description |
|---|---|
| `await register_mcp(session, *, prefix="", trust_readonly=False, undo=None, checks=None, registry=None)` | Register every tool of an MCP server. Returns an async `call(name, **args)`. |
| `await list_all_tools(session)` | All tools, following pagination. |

### `rubicon_llm`

| Name | Description |
|---|---|
| `agent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=None, *, attempts=2, **run_kwargs)`, `aagent(...)` | Plan with a model (repairing small mistakes, retrying rejected plans with the error), then run the plan safely. |
| `plan_with_gemini`, `plan_with_claude`, `plan_with_openai` | Turn a goal into a validated plan. Signature `(goal, client=None, registry=None, ...)`; pass `client=` to reuse or customise an SDK client. |
| `planning_request(goal, registry=None)` | `{"system", "prompt", "schema"}` for any model. |
| `parse_plan(text, registry=None, *, repair=True, repairs=None)` | Turn model output into a validated plan, repairing small unambiguous mistakes first. |
| `plan_schema(registry=None)` | The plan's JSON Schema. |
| `PlanRejected` | `ValueError` raised for a plan that fails parsing or validation; `.text` is the model's output. |
| `GEMINI_MODEL`, `CLAUDE_MODEL` | Default models, overridable with environment variables of the same name. |
| `ask_human(steps)` | Terminal confirmation with previews. Returns `False` when no terminal is attached. |

## Security and limitations

- MCP annotations are claims, not guarantees. A malicious server could mark a destructive tool as
  read-only, so read-only tools only run before the safety gate with `trust_readonly=True`. Use it
  for servers you trust.
- Model output is untrusted. Plans are checked for structure and argument names, but argument
  values are up to your tools. Validate at the tool boundary and give permanent tools a `check`.
- An undo is a compensation, not a time machine: it is whatever your function does, such as
  releasing a hold or deleting a draft.
- If a permanent step needs another permanent step's output, its dry run can only happen right
  before it runs, after the point of no return. Split the plan if that matters.
- With `parallel=True`, a step cancelled mid-flight because a sibling failed is not undone (it
  never reported a result). Keep undoable tools atomic.
- Timeouts apply to async functions. A sync function can't be interrupted safely, so give slow
  sync tools their own timeouts.
- Without `journal=`, rollback happens in-process only, and a hard kill between steps can't be
  unwound. With it, run `recover()` on startup and alert on `dead_letters()`.
- The built-in journals cover one machine. Across machines, use a shared store through your own
  journal object, and give it a `lock()` that works across machines if more than one of them may
  run `recover()`. `recover()` only checks process liveness on its own machine, so it skips other
  machines' transactions unless you pass `include_live=True`.
- Liveness is checked by process id. If the operating system reuses a crashed process's id for a
  new process, that transaction is skipped until the new process exits: a delay, never a wrong
  undo.
- A sync undo that hangs can't be interrupted. Give remote calls their own timeouts, or make the
  undo async so `timeout=` applies.
- Keep keys out of the repo. `.env` is git-ignored; `.env.example` is the template.

## Development

Each module has its own self-check, and pytest runs all of them. No API keys or network needed.

```bash
python rubicon.py              # core
python rubicon_mcp.py          # MCP adapter against an in-memory MCP server
python rubicon_llm.py          # planners with stub clients
pip install -e ".[all,dev]"
pytest --cov                   # everything, with coverage
```

CI runs the core with nothing installed, then the full suite with coverage on Python 3.10 to
3.13, on Linux, Windows and macOS, against MCP SDK 1.x and 2.x.

```
rubicon.py        core: registry, ordering, validation, gate, rollback, Transaction, journal
rubicon_mcp.py    MCP adapter
rubicon_llm.py    Gemini, Claude and OpenAI-compatible planners, plus bring-your-own-model helpers
demo.py           live demo with fake travel tools
tests/            pytest entry point for the self-checks
```

Release notes are in [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
