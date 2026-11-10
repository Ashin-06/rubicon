# Rubicon

[![CI](https://github.com/Ashin-06/rubicon/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashin-06/rubicon/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Ashin-06/rubicon/blob/main/LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)

**Reversibility-aware execution for AI agents.** Rubicon runs what can be taken back first,
checks what can't, asks once, and only then crosses the point of no return. If anything fails,
is refused or is interrupted before that point, every step that already ran is undone.

```
plan from the model           what Rubicon runs
-------------------           ------------------------------------------------
1. cancel old booking   ->    1. search flights        (undoable)
2. search flights             2. hold new seat         (undoable)
3. hold new seat              -- check every permanent step, ask once --
4. email customer             3. cancel old booking    (PERMANENT)
                              4. email customer        (PERMANENT)
```

It works with:

- **any model**: Gemini, Claude, OpenAI and anything OpenAI-compatible (Ollama, vLLM, Groq,
  OpenRouter, ...), or your own call;
- **any tools**: plain Python functions (sync or async) and MCP servers (SDK 1.x and 2.x);
- **any agent loop**: whole plans with `run()`, or step-by-step agents through `Transaction`.

The core is one file with no dependencies beyond the Python standard library.

> **Status: beta (0.x).** The behaviour described here is tested heavily (see
> [Testing](#testing)), but names and signatures may still change before 1.0. See the
> [changelog](https://github.com/Ashin-06/rubicon/blob/main/CHANGELOG.md).

## Contents

- [Why](#why)
- [When to use it, and when not](#when-to-use-it-and-when-not)
- [Install](#install)
- [Quick start](#quick-start)
- [Words used here](#words-used-here)
- [What happens when something fails](#what-happens-when-something-fails)
- [Two ways to use it](#two-ways-to-use-it)
- [Plan format](#plan-format)
- [Tools: undo, checks, previews, retries](#tools-undo-checks-previews-retries)
- [MCP servers](#mcp-servers)
- [Let a model write the plan](#let-a-model-write-the-plan)
- [Crash safety: journals and recovery](#crash-safety-journals-and-recovery)
- [Events, logging and parallel steps](#events-logging-and-parallel-steps)
- [Async](#async)
- [Multiple agents in one process](#multiple-agents-in-one-process)
- [Live demo](#live-demo)
- [API reference](#api-reference)
- [How it compares](#how-it-compares)
- [Security and limitations](#security-and-limitations)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Why

Agent frameworks treat every tool call the same. A model that writes "cancel the booking, then
look for a new flight" will cancel first, find no seats, and leave the user with nothing.

The ideas that prevent this are old: compensating actions (undo steps, as in the saga
pattern), dry runs, and one confirmation before anything irreversible. Rubicon puts them into
the place where agent tool calls actually run.

| Without Rubicon | With Rubicon |
|---|---|
| Steps run in the order the model wrote them | Undoable steps run first, permanent steps last, dependencies respected |
| A failure halfway leaves half-done work | Steps that already ran are undone in reverse order |
| Ctrl+C or a cancelled task leaves a mess | Interruptions roll back too, then propagate |
| A wrong argument name is found when the call fails | Every step's arguments are checked against the tool before anything runs |
| Problems with a permanent step show up when it fails | Every permanent step's check runs before anything permanent happens |
| Approval per call, or none | One approval for all permanent actions, with a preview of each |
| The model can't see which tools are dangerous | Reversibility is part of the tool list the model plans against |
| A crash mid-plan is forgotten | A journal lets `recover()` undo it after a restart |

## When to use it, and when not

Use Rubicon when an agent mixes actions you can take back (holds, drafts, reservations,
reads) with actions you can't (payments, emails, deletions, cancellations), and a half-done
plan is worse than none.

It is not the right tool when:

- the agent only reads: there is nothing to undo or protect;
- a workflow must wait for days or resume halfway after a restart: use a durable workflow
  engine. Rubicon undoes interrupted work; it does not resume it;
- you can't write an undo for your actions: Rubicon can then only order, check and confirm them.

## Install

Rubicon is not on PyPI yet. Install it from GitHub:

```bash
pip install "rubicon-agents @ git+https://github.com/Ashin-06/rubicon.git"
```

or from a clone, for development:

```bash
git clone https://github.com/Ashin-06/rubicon.git
cd rubicon
pip install -e .
```

The package is called `rubicon-agents`; you import it as `rubicon` (plus `rubicon_llm` and
`rubicon_mcp`). The core needs nothing else. Optional extras add the integrations:

| Extra | Adds | For |
|---|---|---|
| `gemini` | `google-genai` | `plan_with_gemini` |
| `claude` | `anthropic` | `plan_with_claude` |
| `openai` | `openai` | `plan_with_openai` (OpenAI and compatible servers) |
| `mcp` | `mcp` | `register_mcp` |
| `all` | all of the above | |
| `dev` | `pytest`, `pytest-cov` | running the tests |

For example `pip install -e ".[gemini,mcp]"`.

> The import name `rubicon` is also used by BeeWare's `rubicon-objc` (used by Toga and
> Briefcase on macOS and iOS). Don't install both into the same environment.

Requires Python 3.10 or newer, on Linux, macOS or Windows.

## Quick start

Copy this into a file and run it. Nothing real is booked.

```python
from rubicon import NOOP, PlanFailed, run, tool

seats = {"AI101": 1}
bookings = {"OLD1"}

@tool(undo=NOOP)                       # read-only: nothing to undo
def search_flights(route: str) -> list:
    return [flight for flight, free in seats.items() if free]

def release(hold):                     # the undo gets the step's result
    seats[hold["flight"]] += 1

@tool(undo=release)                    # undoable
def hold_seat(flight: str) -> dict:
    if not seats.get(flight):
        raise RuntimeError(f"no seats on {flight}")
    seats[flight] -= 1
    return {"flight": flight}

def booking_exists(ref):               # the check: raise to block, return text to preview
    if ref not in bookings:
        raise LookupError(f"no booking {ref}")
    return f"Booking {ref} will be cancelled for good"

@tool(check=booking_exists)            # no undo: permanent
def cancel_booking(ref: str) -> str:
    bookings.remove(ref)
    return "cancelled"

plan = [  # the order a careless model might write
    {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}},
    {"id": "find", "tool": "search_flights", "args": {"route": "BLR-DEL"}},
    {"id": "hold", "tool": "hold_seat", "args": {"flight": "AI101"}},
]

def ask(steps):
    for s in steps:
        print(f"about to run {s['tool']}{s['args']}: {s['preview']}")
    return input("Commit? [y/N] ") == "y"

try:
    print(run(plan, confirm=ask))      # runs find, hold, then checks cancel, asks, cancels
except PlanFailed as e:
    print("nothing permanent happened:", e)
print(seats, bookings)
```

Answer `n` and the held seat is released: `{'AI101': 1} {'OLD1'}`. Set `seats = {"AI101": 0}` and
`hold_seat` fails, so the booking is never touched.

> **`run()` without `confirm=` approves permanent steps automatically.** Pass a `confirm`
> whenever a person should decide. (`agent()`, the model-driven version, asks on the terminal by
> default.)

## Words used here

| Word | Meaning |
|---|---|
| Undoable step | A tool registered with `undo=`. Rubicon can reverse it by calling the undo. |
| Read-only step | An undoable step whose undo is `NOOP`, because there is nothing to reverse. |
| Permanent step | A tool registered without `undo=`. Once it ran, it stays. |
| Check | A function you attach with `check=`. Rubicon calls it with the same arguments before anything permanent runs; it never calls the tool itself for this. Raise to block, return text to show a preview. A permanent tool without a check only gets its arguments validated. Elsewhere this is called a dry run. |
| Gate | The single confirmation before the first permanent step. |
| Commit | Running the permanent steps after the gate. Undoable steps before a commit become final with it. |
| Stage | In a `Transaction`, the calls between two commits. |
| Journal | A durable log of every step, so `recover()` can undo what a crash left half-done. |
| Dead letter | Something Rubicon could not fix by itself, recorded until a person resolves it. |
| Idempotent | Safe to run twice: releasing a seat that is already released does nothing. Undos should be. |

## What happens when something fails

| When it fails | What Rubicon does |
|---|---|
| The plan itself is invalid (unknown tool, wrong argument name, cycle, ...) | Raises `ValueError` before anything runs |
| A step before any permanent action | Undoes every step that ran, newest first |
| A check, or the confirmation is declined | The same: nothing permanent has happened, so everything is undone |
| A permanent step, after another permanent step already ran | Keeps the steps the earlier permanent step relied on, undoes only the ones after it, lists the permanent steps that never ran |
| Ctrl+C or task cancellation | Undoes what it can, then lets the interruption through |
| An undo itself | Retries it, reports it in `PlanFailed.undo_failed`, keeps unwinding |

After anything starts, every failure raises `PlanFailed` with:

| Attribute | Meaning |
|---|---|
| `step` | The step that failed, or `"confirm"` when the confirmation was declined |
| `cause` | The original exception |
| `undone` | Steps that were reversed |
| `committed` | Permanent steps that already happened |
| `not_run` | Permanent steps that never ran |
| `undo_failed` | Steps whose undo failed: these need a person (see [dead letters](#dead-letters)) |
| `results` | Results of every step that ran |

For example, with a plan of `hold` (hold seat AI101), `cancel` (cancel booking OLD1) and `mail`
(send the confirmation): if `mail` fails, OLD1 is already gone, so releasing the AI101 seat
would leave the user with no flight at all. Rubicon keeps the seat, reports
`committed=["cancel"]` and `not_run=["mail"]`, and you can retry the email.

## Two ways to use it

### 1. Whole plans: `run` and `arun`

Hand Rubicon a complete plan, written by you or by a model, and it runs it as shown above.

### 2. Any agent loop: `Transaction`

Most agent frameworks call tools one at a time and decide the next call from the last result.
`Transaction` guards that loop:

- Undoable tools run immediately.
- Permanent tools are checked immediately, so the agent sees a failed check and can react. Then
  they are held, and the agent gets a `"Scheduled ..."` message (with the preview, if any)
  instead of a result.
- On a clean exit, the held actions are confirmed once and run.
- On any exception, every undoable call is rolled back and the held actions are dropped.

```python
from rubicon import Transaction
from rubicon_llm import ask_human

with Transaction(confirm=ask_human) as tx:
    tools = tx.wrap_all()            # a list of guarded functions: same names, docs, signatures
    my_agent.run(goal, tools=tools)  # your agent loop
# commits here; an exception inside the block rolls everything back
```

- Wrapped tools take positional and keyword arguments like the originals, and are async when the
  tool or its check is async. Pass them wherever your framework accepts plain Python functions.
- You can also call tools directly with `tx.call("hold_seat", "AI101")` and finish with
  `tx.commit()` or `tx.rollback()`.
- Each call gets a step id of the form `<tool>-<n>` (`hold_seat-1`, `cancel_booking-2`). These ids
  appear in `tx.results`, the confirmation, the journal and dead letters.
- `tx.id` identifies the transaction in the journal (you need it for `acknowledge()`).
  `tx.undone`, `tx.undo_failed`, `tx.committed` and `tx.results` say what happened.
- You can commit in stages: call `tx.commit()` in the middle, and keep going. Everything up to
  that point becomes final; a later failure only rolls back what came after.

> A permanent tool's real result only exists after commit. Design the agent so it doesn't need
> that result to choose its next step, or commit in stages.

## Plan format

Plans are plain JSON so a model can write them:

```json
[
  {"id": "hold", "tool": "hold_seat", "args": {"flight": "AI101"}},
  {"id": "mail", "tool": "send_email",
   "args": {"to": "me@example.com", "body": "Moved to $hold.flight"}, "after": ["hold"]}
]
```

| Field | Meaning |
|---|---|
| `id` | Unique step name. Starts with a letter or `_`, then letters, digits, `_` or `-` |
| `tool` | A registered tool name |
| `args` | Keyword arguments for the tool (optional). Checked against the tool's signature |
| `after` | Step ids that must run first (optional) |

References let a step use an earlier step's result:

- `"$hold"` as a whole argument passes that step's result as it is.
- `"$hold"` inside text inserts the result as text.
- `"$hold.flight"` or `"$hold.legs.0"` picks a key or list item out of the result.
- Rubicon follows the path while each part matches a key or an index. Once it reaches a string
  or a number, the rest is kept as text: with `file = {"name": "report"}`, `"$file.name.txt"`
  becomes `"report.txt"`.
- A key or index that doesn't exist, or a path into an object that isn't a dict or list, fails
  the step, so a message never goes out with a dumped dictionary or object in it.
- References work inside nested lists and objects and add a dependency automatically.
- Only real step ids count, and ids can't start with a digit, so `"$50"` stays money. This also
  means a misspelt reference such as `"$hodl.flight"` is sent as ordinary text. Check the
  confirmation preview, or validate such values in your tool.

## Tools: undo, checks, previews, retries

```python
from rubicon import tool

@tool(
    undo=lambda result, args: delete_file(args["path"]),  # undo(result) or undo(result, args)
    retries=0,          # extra attempts for the tool itself; only for tools safe to repeat
    undo_retries=2,     # extra attempts for the undo, with backoff (the default)
    timeout=30,         # seconds per attempt; async functions only
)
def put_file(path: str, content: str) -> str: ...

def preview_cancel(ref):
    booking = find(ref)                      # raise to block the commit
    return f"Cancel {ref} ({booking.route}), fee {booking.fee}"

@tool(check=preview_cancel)                  # its return value is shown when confirming
def cancel_booking(ref: str): ...
```

- An undo with a second required parameter (or one named `args`) gets the call's arguments as
  well as its result, for tools that only return `"ok"`. A second parameter with a default,
  such as `undo(result, force=False)`, is left alone.
- Make every undo idempotent. If the process dies after an undo but before it is journaled,
  `recover()` runs it again; the same is true of any compensation-based system.
- A check receives the tool's arguments (with defaults filled in), limited to the parameters it
  declares, so `def must_exist(ref)` works for `def cancel(ref, reason="user")`.
- `confirm(steps)` receives a list with each permanent step's `id`, `tool`, resolved `args` and
  `preview`. Return true to go ahead.
- Use `UNSET` as the default of an optional parameter that should be left out of the call
  entirely (not passed, not shown at confirmation, not journaled), for example so a remote API
  applies its own default. MCP tools use it for every optional parameter.
- `timeout=` only applies to async functions; a sync function can't be interrupted safely.
  Registering a sync tool with a timeout logs a warning.
- Registering a name again replaces the earlier tool in that registry.

## MCP servers

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rubicon import arun
from rubicon_mcp import register_mcp

async def main():
    server = StdioServerParameters(command="python", args=["my_server.py"])
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        call = await register_mcp(
            session,
            trust_readonly=True,       # see "Security and limitations"
            undo={"create_issue": lambda r: call("delete_issue", id=r["id"])},
        )
        plan = [{"id": "i", "tool": "create_issue", "args": {"title": "Seat moved"}}]
        print(await arun(plan))

asyncio.run(main())
```

`register_mcp` accepts an SDK 1.x or 2.x `ClientSession`, or an SDK 2.x `mcp.Client`. It
labels each tool:

| Condition | Label |
|---|---|
| Listed in `undo` | Undoable with your undo function (sync or async; it can call another MCP tool) |
| `readOnlyHint: true`, not also `destructiveHint: true`, and `trust_readonly=True` | Read-only |
| Anything else | Permanent, matching MCP's own default of assuming a tool is destructive |

- Each tool gets a real Python signature from its input schema (keyword-only parameters,
  annotated where the schema's type maps to a Python type), so frameworks that inspect
  signatures see the actual parameters. Optional parameters you leave out are not sent, so the
  server's own defaults apply.
- Arguments the schema doesn't declare are rejected before anything runs, unless the schema sets
  `additionalProperties` to `true` (or to a schema) or declares no properties at all. Property
  names that can't be Python parameters (`"from"`, `"user-id"`) are checked the same way.
- Names in `undo` and `checks` must exist on the server, and a server that lists one tool name
  twice is refused, so no tool ends up with the wrong label.
- Tool lists are fetched across all pages. `prefix=` namespaces tools when you register several
  servers.
- Text results that are JSON are parsed; `"NaN"` and `"Infinity"` stay text. Images and other
  non-text content are returned instead of dropped. An error result raises with the server's
  message.

## Let a model write the plan

```python
from functools import partial
from rubicon_llm import agent, plan_with_claude, plan_with_openai

agent("Move my booking OLD1 to AI101 and email me@example.com")      # Gemini (default)
agent("...", planner=plan_with_claude)                               # Claude
agent("...", planner=partial(plan_with_openai, model="gpt-5.1"))     # OpenAI
agent("...", planner=partial(plan_with_openai, model="llama3.3",     # Ollama, LM Studio,
                             base_url="http://localhost:11434/v1"))  # vLLM, Groq, OpenRouter, ...
```

| Planner | Extra | Key | Default model (0.8.x) |
|---|---|---|---|
| `plan_with_gemini` | `gemini` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `gemini-3.5-flash` |
| `plan_with_claude` | `claude` | `ANTHROPIC_API_KEY` | `claude-opus-5-5` |
| `plan_with_openai` | `openai` | `OPENAI_API_KEY` (not needed for local servers) | none, pass `model=` |

- Override the default models with the `GEMINI_MODEL` and `CLAUDE_MODEL` environment variables,
  or pass `model=`. They are read each time a planner runs. (Only `demo.py` reads a `.env` file;
  in your own code set the variables, or load the file yourself, for example with python-dotenv.)
- `agent()` asks for confirmation on the terminal by default (`ask_human`), which answers no when
  no terminal is attached. Pass `confirm=` to decide another way.
- Before rejecting a plan, `parse_plan()` fixes small mistakes locally when there is exactly one
  right answer, so they cost no extra model call:
  - a tool or argument name that differs only in case or separators (`Hold_Seat`, `flightNumber`
    for `flight_number`), when exactly one tool or parameter matches;
  - a value whose type plainly differs from the schema (`"2"` for an integer, `"true"` for a
    boolean);
  - a step field under a common other name (`arguments` or `params` for `args`, `depends_on`
    for `after`);
  - Python-style output (single quotes, `True`, `None`) and doubly-encoded `args_json`.

  References (`"$step..."`) are never changed, and nothing ambiguous is guessed: an answer with
  two different plans in it is rejected. Every repair is logged; pass `repairs=[]` to collect
  them, or `repair=False` for strict parsing.
- If a plan is still rejected (bad JSON, unknown tool, wrong argument names, bad ids), `agent()`
  sends the model its previous plan and the error and asks for a corrected one, up to
  `attempts=2` plans in total. A final rejection raises `PlanRejected`, a `ValueError` whose
  `text` is what the model wrote. Other errors, such as a missing API key, are raised at once.
- `plan_with_openai` sends a strict JSON Schema by default. Use `json_mode="object"` for servers
  that only have a JSON mode, and `json_mode="none"` for servers with neither.
- `agent()` passes `parallel=`, `on_event=` and `journal=` through to `run()`.

To use any other model or framework:

```python
from rubicon import run
from rubicon_llm import parse_plan, planning_request

req = planning_request(goal)                   # {"system", "prompt", "schema"}
text = my_model(req["system"], req["prompt"], json_schema=req["schema"])
run(parse_plan(text))
```

The model sees every registered tool with its JSON Schema parameters (from type hints, including
`Optional`, `Literal`, enums and typed lists, or from MCP) and whether it can be undone.

## Crash safety: journals and recovery

In short: pass `journal=`, call `recover()` when your service starts, and alert on dead
letters.

```python
from rubicon import recover, run

run(plan, journal="rubicon.jsonl")            # every step is recorded durably
recover("rubicon.jsonl", on_dead_letter=page)  # after a restart: undo what a crash left half-done
```

How recovery decides:

- Every step is journaled as "started" before it runs, and with its arguments and result when it
  finishes. If the process is killed, `recover()` undoes every unfinished transaction's steps
  since its last commit, newest first, using the same tools.
- If the crash happened while a permanent step was running, nobody knows whether it went
  through. `recover()` then undoes nothing that step may rely on: it reports the step in
  `needs_attention` and the steps it kept in `kept`, for a person to check. An undoable step that
  started but never reported a result is listed in `uncertain`.
- Undos that failed earlier (a service was down) are retried on every run until they succeed.
- Recovery journals each undo as it happens, so if recovery itself is interrupted, running it
  again never undoes anything twice.
- Values that aren't JSON-serializable are marked in the journal. A step whose undo would need
  such a value is reported in `undo_failed` instead of being replayed wrongly. A read-only step
  needs nothing, so its result can be anything.
- A `Transaction` that commits in stages is covered too: a crash in a later stage undoes only
  that stage.

Each report is `{"tx", "undone", "undo_failed", "committed", "uncertain", "needs_attention",
"kept"}`. Only one `recover()` runs per journal at a time; a second one raises `JournalBusy`.

`recover()` is safe to run while other workers use the same journal. Every record says which
process and machine runs the transaction, and `recover()` leaves alone any transaction whose
process is still alive on this machine, or that belongs to another machine (it can't check from
here). Run `recover()` on each machine, or pass `include_live=True` when you know the other
processes are gone.

### Where the journal lives

```python
from rubicon import FileJournal, SQLiteJournal

run(plan, journal="rubicon.jsonl")                  # same as FileJournal("rubicon.jsonl")
run(plan, journal=SQLiteJournal("rubicon.db"))      # many processes and threads on one machine
run(plan, journal=MyPostgresJournal(...))           # several machines: bring your own store
```

| Journal | Use it for |
|---|---|
| `FileJournal(path)` | One machine. JSON lines, each write flushed and fsynced; writers in several processes take turns through an OS file lock. |
| `SQLiteJournal(path)` | One machine, many processes or threads. Standard library only (WAL mode, full sync). |
| Your own object | Several machines (Postgres, Redis, a queue). See below. |

Your own journal needs `append(records)` (store a list of JSON-serializable dicts, in order,
atomically) and `records()` (return them all, oldest first). Optionally:

- `lock()`: a context manager held during recovery and compaction, exclusive across every
  machine that may run them;
- `drop(tx_ids, upto=None)`: delete those transactions' records, for `compact()`. With `upto`,
  only among the first `upto` records: records added after `compact()` read the journal belong to
  a new stage and must stay. A `drop(tx_ids)` without `upto` works, but then don't run
  `compact()` while those workers are writing.

### Dead letters

Some steps need a person: an undo that kept failing, a crash in the middle of a step, or a
permanent step whose outcome is unknown. These are dead letters. They stay in the journal until
someone resolves them, so nothing gets lost between restarts.

```python
from rubicon import acknowledge, dead_letters, recover

recover(journal, on_dead_letter=page_on_call)   # called once for each dead letter
for letter in dead_letters(journal):            # {"tx", "step", "kind", "reason", "time"}
    ...
acknowledge(journal, letter["tx"], letter["step"], note="fixed by hand")
acknowledge(journal, letter["tx"], note="the payment went through")
```

- `kind` is `"undo_failed"`, `"uncertain"` (a crash during an undoable step) or
  `"in_flight_permanent"` (a crash during a permanent step).
- `on_dead_letter` is called once for each dead letter, including those a rollback in `run()` or
  a `Transaction` recorded itself, the next time `recover()` runs with it.
- For a crash during a permanent step: if the step did not happen, acknowledge just that step,
  and the next `recover()` undoes the steps it was keeping. If it did happen, acknowledge the
  whole transaction, which settles what recovery kept and found in flight.
- Acknowledging a whole transaction never settles undos still owed from an earlier rollback
  (they keep being retried, or acknowledge them by name), nor a stage the worker started after
  the crash report.
- `acknowledge()` raises `ValueError` for a transaction or step the journal doesn't contain, so
  a typo can't leave the real dead letter open.

### Keeping the journal small

`compact(journal)` removes settled transactions and keeps everything that is unfinished, owes an
undo or has an unresolved dead letter. Run it on a schedule or at startup. It is safe to run
while workers write: records written after it read the journal are kept.

## Events, logging and parallel steps

```python
run(plan, on_event=print)      # structured events
run(plan, parallel=True)       # independent undoable steps at the same time
run(plan, log=print)           # progress messages somewhere other than logging
```

- `on_event` receives dicts such as `{"event": "step_done", "step": "hold", "tool": "hold_seat",
  "duration_ms": 12.3, "tx": "...", "time": ...}`. Events: `step_started`, `step_done`,
  `step_failed`, `check_passed`, `gate_confirmed`, `gate_declined`, `step_committed`,
  `step_undone`, `undo_failed`. Forward them to your logger or tracing system.
- If an `on_event` or `log` callback raises, the error is logged as a warning and execution
  carries on unchanged.
- The library logs to the standard `logging` module under the name `"rubicon"`: progress at
  INFO, problems at WARNING. It never prints, apart from `ask_human`, the terminal prompt.
- `parallel=True` runs undoable steps in waves of steps that don't depend on each other. If one
  fails, or the run is cancelled or interrupted, the rest of the wave is stopped first and then
  everything completed is undone, so no step can finish after the rollback. Permanent steps
  always run one at a time, in order.

## Async

Everything has an async form, and tools, undo, check and confirm functions can be `async def`.

| Sync | Async |
|---|---|
| `run(plan)` | `await arun(plan)` |
| `agent(goal)` | `await aagent(goal)` |
| `recover(path)` | `await arecover(path)` |
| `with Transaction() as tx` | `async with Transaction() as tx` |
| `tx.call`, `tx.commit`, `tx.rollback` | `await tx.acall`, `tx.acommit`, `tx.arollback` |

- Calling `run()` from inside a running event loop (for example in Jupyter) raises an error that
  points you to `arun()`.
- Sync `Transaction` code (`tx.call`, `tx.commit`, `tx.rollback`, `with Transaction()`) also
  works on an event loop's thread, as long as the tools, checks, undos and confirm are sync. An
  async one is refused before any of it runs.
- `aagent()` accepts sync or async planners and confirm functions, and runs a sync `confirm`
  (like the terminal prompt) in a thread so it doesn't block the event loop.

## Multiple agents in one process

Every function takes `registry=`. Pass your own dict to keep tool sets apart:

```python
billing = {}

@tool(check=validate, registry=billing)
def charge(amount: int): ...

run(plan, registry=billing)
```

Without `registry=`, tools go into the shared default registry, `rubicon.TOOLS`.

## Live demo

The demo plans a real booking change with a real model, against fake in-memory travel tools, so
nothing is booked or sent.

```bash
pip install -e ".[gemini]"     # or [claude], [openai]
cp .env.example .env           # then put your key in .env (Windows: copy .env.example .env)
python demo.py                 # Gemini
python demo.py claude          # Claude
python demo.py openai          # OpenAI-compatible: set OPENAI_MODEL (and OPENAI_BASE_URL) in .env
```

## API reference

### `rubicon`

| Name | Description |
|---|---|
| `tool(fn=None, *, undo=None, check=None, name=None, registry=None, retries=0, undo_retries=2, timeout=None)` | Register a tool. See [Tools](#tools-undo-checks-previews-retries). |
| `NOOP` | Undo function for read-only tools. |
| `UNSET` | Default for optional parameters that should be left out of the call entirely. |
| `run(plan, registry=None, confirm=None, log=None, *, parallel=False, on_event=None, journal=None)`, `arun(...)` | Run a plan and return `{step_id: result}`. |
| `order(plan, registry=None)` | Validate a plan and return the order it would run in. Raises `ValueError`. |
| `Transaction(registry=None, confirm=None, log=None, *, on_event=None, journal=None)` | Guard an agent loop. Methods: `call`/`acall`, `commit`/`acommit`, `rollback`/`arollback`, `wrap(name)`, `wrap_all()` (a list). Attributes: `id`, `results`, `committed`, `undone`, `undo_failed`. A sync or async context manager. |
| `PlanFailed` | Raised after rollback. Attributes: `step`, `cause`, `undone`, `committed`, `not_run`, `undo_failed`, `results`. |
| `recover(journal, registry=None, log=None, *, on_dead_letter=None, include_live=False)`, `arecover(...)` | Undo what crashes left unfinished, leaving running transactions alone. Returns one report per transaction it worked on. |
| `dead_letters(journal)` | Steps that need a person, oldest first. |
| `acknowledge(journal, tx, step=None, note="")` | Mark a dead letter, or a whole transaction, as handled. |
| `compact(journal)` | Remove settled transactions; returns how many. |
| `FileJournal(path)`, `SQLiteJournal(path)` | Built-in journals. `journal=` also accepts a path or your own object. |
| `JournalBusy` | Raised when another process holds the journal's lock. |
| `tools_for_llm(registry=None)` | Tools as `{name, description, reversible, parameters}`, with JSON Schema parameters. |
| `Tool` | A registered tool (`fn`, `undo`, `check`, `retries`, `undo_retries`, `timeout`, `reversible`). |
| `TOOLS` | The default registry. |

### `rubicon_mcp`

| Name | Description |
|---|---|
| `await register_mcp(session, *, prefix="", trust_readonly=False, undo=None, checks=None, registry=None)` | Register every tool of an MCP server. Returns an async `call(name, **args)` for calling server tools directly, for example from an undo. |
| `await list_all_tools(session)` | All tools, following pagination. |

### `rubicon_llm`

| Name | Description |
|---|---|
| `agent(goal, planner=plan_with_gemini, client=None, registry=None, confirm=ask_human, log=None, *, attempts=2, **run_kwargs)`, `aagent(...)` | Plan with a model (repairing small mistakes, retrying rejected plans with the error), then run the plan. |
| `plan_with_gemini`, `plan_with_claude`, `plan_with_openai` | Turn a goal into a validated plan. Signature `(goal, client=None, registry=None, ...)`; pass `client=` to reuse or customise an SDK client. |
| `planning_request(goal, registry=None)` | `{"system", "prompt", "schema"}` for any model. |
| `parse_plan(text, registry=None, *, repair=True, repairs=None)` | Turn model output into a validated plan, repairing small unambiguous mistakes first. |
| `plan_schema(registry=None)` | The plan's JSON Schema. |
| `SYSTEM` | The system prompt the planners use. |
| `PlanRejected` | `ValueError` raised for a plan that fails parsing or validation; `.text` is the model's output. |
| `GEMINI_MODEL`, `CLAUDE_MODEL` | Default models, overridable with environment variables of the same name. |
| `ask_human(steps)` | Terminal confirmation with previews. Returns `False` when no terminal is attached. |

## How it compares

| | Rubicon | Saga libraries | Durable workflow engines (Temporal, DBOS, ...) | Human-in-the-loop interrupts in agent frameworks |
|---|---|---|---|---|
| Reorders a plan so undoable steps come first | Yes | No | No | No |
| Undoes completed steps on failure | Yes | Yes | You write it | No |
| One confirmation before anything permanent | Yes | No | You write it | Yes, per interrupt |
| Checks arguments against tools before running | Yes | No | No | Partly |
| Survives a crash | Undoes the interrupted work | Varies | Resumes the workflow | Resumes from a checkpoint |
| Infrastructure | None (a file or SQLite) | Varies | A server | Your framework |

Rubicon is a small, in-process executor built for model-written plans. It complements a
durable workflow engine rather than replacing one: use the engine when work must resume, and
Rubicon around the tool calls an agent makes.

## Security and limitations

- **MCP annotations are claims, not guarantees.** A malicious server could mark a destructive
  tool as read-only, so read-only tools only run before the gate with `trust_readonly=True`. Use
  it for servers you trust.
- **Model output is untrusted.** Plans are checked for structure, tools and argument names, but
  argument values are up to your tools. Validate at the tool boundary, and give permanent tools a
  `check`.
- **An undo is a compensation, not a time machine.** It is whatever your function does, such as
  releasing a hold or deleting a draft. Make it idempotent.
- If a permanent step needs another permanent step's output, its check can only run right before
  it, after the point of no return. Split the plan if that matters.
- With `parallel=True`, a step cancelled mid-flight because a sibling failed is not undone (it
  never reported a result). Keep undoable tools atomic.
- `timeout=` covers async functions only. A sync tool or undo that hangs can't be interrupted;
  give remote calls their own timeouts, or make them async.
- Functions without an inspectable signature (some built-ins) can't have their arguments checked
  before the run.
- Without `journal=`, rollback happens in-process only, and a hard kill between steps can't be
  unwound. With it, run `recover()` on startup and alert on dead letters.
- The built-in journals cover one machine. Across machines, use a shared store through your own
  journal object, with a `lock()` that works across machines if more than one of them may run
  `recover()`. `recover()` only checks process liveness on its own machine, so it skips other
  machines' transactions unless you pass `include_live=True`.
- Liveness is checked by process id. If the operating system reuses a crashed process's id for a
  new process, that transaction is skipped until the new process exits: a delay, never a wrong
  undo.
- Keep keys out of your repository. Here `.env` is git-ignored and `.env.example` is the template.

To report a vulnerability, see [SECURITY.md](https://github.com/Ashin-06/rubicon/blob/main/SECURITY.md).

## Troubleshooting

| Problem | Cause and fix |
|---|---|
| `ValueError: step 'x': unknown tool 'y'` | The tool isn't in the registry used. Did you register it with `registry=` but call `run()` without it, or the other way round? |
| `ValueError: ... arguments [...] don't fit ...` | The plan's `args` don't match the tool's parameters. The message names the mismatch. |
| `RuntimeError: ... inside a running event loop ...` | You called `run()` (or an async tool from sync code) inside an event loop, for example in Jupyter or an async web server. Use `await arun(...)` or `async with Transaction()`. |
| `agent()` always declines | `ask_human` answers no when there is no terminal (CI, some IDE runners). Pass `confirm=`. |
| `JournalBusy` | Another `recover()` or `compact()` holds the journal's lock, or a writer waited more than 30 seconds. Retry later. |
| An undo ran twice | The process died after the undo but before it was journaled, and `recover()` ran it again. Make undos idempotent. |
| `recover()` returns nothing for a crashed run | Its process id is still alive on this machine (or was reused), or it belongs to another machine. Pass `include_live=True` once you know it is gone. |

## Testing

Each module has a self-check next to its code, and pytest runs all of them. No API keys or
network are needed.

```bash
pip install -e ".[all,dev]"
python rubicon.py              # core
python rubicon_mcp.py          # MCP adapter against an in-memory MCP server
python rubicon_llm.py          # planners with stub clients
pytest --cov                   # everything, with coverage
```

`tests/test_fuzz.py` adds randomised invariant tests. They generate random plans and agent loops
with tool failures, failing checks, declined confirmations, undo outages, parallel runs and
process deaths at every possible point, including in the middle of a recovery. Some scenarios
have another worker run `recover()`, `compact()` and `acknowledge()` on the same journal while a
transaction is running, including in the middle of an undo. After each one, a simulated operator
resolves the dead letters and recovery runs until done. Then the tests check the rules that must
always hold:

- an undoable step never runs after a permanent step unless it depends on it;
- nothing a completed permanent step relied on is ever undone;
- another worker never undoes a step of a transaction that is still running;
- everything after the last permanent step is cleaned up, by an undo or by the operator;
- no undo runs for something that never happened;
- failure reports (`committed`, `undone`, `undo_failed`, `not_run`) match what happened;
- in the end there are no dead letters, nothing left to recover, and compaction empties the
  journal.

```bash
pytest tests/test_fuzz.py                          # 400 of each scenario, as in CI
RUBICON_FUZZ=20000 pytest tests/test_fuzz.py       # deep run
RUBICON_FUZZ_SEED=1234 pytest tests/test_fuzz.py   # replay a reported failure
```

(On Windows PowerShell, set variables first: `$env:RUBICON_FUZZ=20000; pytest tests/test_fuzz.py`.)

CI runs the core with nothing installed, then the full suite on Python 3.10 to 3.14 on Linux,
Python 3.12 on Windows and macOS, and MCP SDK 1.x and 2.x. It also builds the package and runs
the self-checks from the installed wheel.

```
rubicon.py        core: registry, ordering, validation, gate, rollback, Transaction, journals
rubicon_mcp.py    MCP adapter
rubicon_llm.py    Gemini, Claude and OpenAI-compatible planners, plus bring-your-own-model helpers
demo.py           live demo with fake travel tools
tests/            pytest entry point for the self-checks, and the randomised invariant tests
```

## Contributing

Issues and pull requests are welcome; see
[CONTRIBUTING.md](https://github.com/Ashin-06/rubicon/blob/main/CONTRIBUTING.md). Bugs with a
small script that reproduces them get fixed fastest.

## License

[MIT](https://github.com/Ashin-06/rubicon/blob/main/LICENSE)
