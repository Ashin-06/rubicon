# Changelog

## 0.8.1

### Added
- Randomised invariant tests (`tests/test_fuzz.py`): random plans and agent loops with random
  failures, declines, undo outages, parallel runs and process deaths at every possible point,
  followed by recovery and a simulated operator, checked against the rules that must always
  hold. 400 scenarios of each kind run in CI; set `RUBICON_FUZZ` for deeper runs.

### Fixed
- A transaction that settled with a failed undo was skipped by `recover()` from another process
  for as long as the worker that ran it stayed alive. A settled transaction isn't running, so it
  is recovered right away.
- On 64-bit Windows the process liveness check passed 64-bit handles through ctypes without
  declaring their types; the Windows API signatures are now declared.
- A crashed transaction whose last open items were resolved with `acknowledge(journal, tx,
  step)` could not be compacted.
- On Linux an exited but not yet reaped (zombie) process counted as alive.
- A transaction whose first step failed stayed marked as live in its process; journal state is
  also updated only after a write succeeds.
- When several steps of one parallel wave failed together, only the first failure was read and
  asyncio reported the others as never retrieved (found by the randomised tests).
- Plan parsing stops at the first candidate that looks like a plan instead of parsing them all.

### Documented
- Undo functions must be safe to run twice: a crash after an undo but before it is journaled
  means `recover()` runs it again.

## 0.8.0

### Changed
- `recover()` leaves alone any transaction that may still be running: its process is alive on
  this machine, or it belongs to another machine. Before, a `recover()` started while other
  workers used the same journal (the multi-process setup `SQLiteJournal` is meant for) would undo
  their live work. Each transaction now records its process id and host. Pass
  `include_live=True` to recover other machines' transactions when you know they are gone.
- `acknowledge()` raises `ValueError` for a transaction or step the journal doesn't contain.

### Fixed
- `FileJournal.records()` read without the write lock. On Windows, `compact()` could fail with
  `PermissionError` while another process was reading, and a reader could see a line still being
  written. Reads now take the write lock.
- `compact()` never removed a rolled-back transaction whose last failed undo was resolved with
  `acknowledge(journal, tx, step)`. Such a transaction now counts as finished.
- An "uncertain" step was reported again, with another warning, on every `recover()` run while
  another undo in the same transaction kept failing. It is now reported once.
- A `with Transaction(journal=...)` block that committed or rolled back inside it wrote a second
  "finished" record on exit, and an empty block wrote records at all. Settle records are now
  written only when steps ran since the last one.
- `SQLiteJournal.drop()` now deletes all chunks in one transaction.
- Plan parsing: bracketed text in the prose before the plan (for example "See note [1]") no
  longer hides the real plan, for JSON and Python-style output alike.
- Plan repair: two argument keys that both match one parameter are left alone instead of
  renaming one of them; NaN no longer produces a bogus repair or becomes the string "nan";
  `"after"` values that are falsy but not a list (`""`, `0`, `{}`) are rejected instead of
  being read as an empty list; repairs are logged only for plans that are accepted.
- The demo reads `.env` as UTF-8.

## 0.7.0

### Added
- Pluggable journals: `FileJournal` (the default for a path), `SQLiteJournal` for many processes
  and threads on one machine, or any object with `append(records)` and `records()` (and
  optionally `lock()` and `drop()`) for shared stores such as Postgres or Redis.
- `recover()` takes an OS-level lock per journal, so two recoveries can't undo the same step
  twice; a second one raises `JournalBusy`. File journal writes from several processes are
  serialized with a file lock.
- Dead letters: failed undos, crashes during a step and permanent steps with an unknown outcome
  stay recorded until a person resolves them. `dead_letters()` lists them, `acknowledge()`
  resolves them, and `recover(..., on_dead_letter=)` reports each new one once.
- `compact()` removes settled transactions so the journal doesn't grow forever.
- Local plan repair in `parse_plan()` before asking the model again: tool and argument names
  that differ only in case or separators, values whose type plainly differs from the schema,
  Python-style literals and doubly-encoded `args_json`. Each repair is logged and can be
  collected with `repairs=[]`; `repair=False` keeps parsing strict.

### Changed
- The library logs to the standard `logging` module (logger `"rubicon"`) instead of printing.
  `log=` defaults to `None`; pass `log=print` to print progress as before.
- `recover()` no longer returns an empty report for a transaction it had nothing to do for.

### Fixed
- A failed undo could be lost: a second rollback (for example `with Transaction()` exiting after
  a failed `commit()` inside it) or a later commit wrote "finished", so `recover()` never retried
  the undo. Failed undos now stay owed until they succeed or are acknowledged.
- A sync tool's retry backoff blocked the event loop inside `arun()`; it now waits with
  `asyncio.sleep` except when sync code drives the call directly.
- The confirmation showed unresolved `$step` references for a permanent step that waits on
  another permanent step; everything already known is resolved now.
- The demo prints a clear message instead of a traceback when the planner fails (for example
  when the API quota is used up).

## 0.6.0

### Fixed
- `recover()` undid steps that a commit had made final. A `commit()` with no permanent steps
  waiting wrote "finished" without a "committed" record, so recovery still saw the earlier
  undoable steps as open. "finished" now settles everything before it.
- `recover()` skipped a `Transaction` that committed in stages and crashed in a later stage, so
  the later stage was never undone. New activity now reopens the transaction.
- A crash while a permanent step was running could make `recover()` undo the steps that step
  relied on, the same "no flight at all" outcome as the 0.4.0 fix, but after a crash. Steps are
  now journaled as "started" before they run; recovery leaves such a transaction untouched and
  reports it in `needs_attention` (with `kept`), and reports half-run undoable steps in
  `uncertain`.
- A step cancelled from inside (for example by a library's cancel scope) during a parallel wave
  was treated as finished. It is now a step failure and everything is rolled back.
- MCP schemas without `additionalProperties` accepted unknown arguments, so a hallucinated
  argument could pass validation and be silently ignored by the server. Extras are now accepted
  only when the schema allows them explicitly or declares no properties.
- A sync tool with `retries`, called with `tx.call()` from inside a running event loop, failed on
  its first retry because the backoff used `asyncio.sleep`. Sync tools now wait synchronously.
- An exception in an `on_event` or `log` callback turned a successful step into a failure after
  the tool had run. Callback errors are now written to stderr and execution carries on.
- `GEMINI_MODEL` and `CLAUDE_MODEL` set after importing `rubicon_llm` (for example by
  `load_dotenv()`) were ignored. They are now read each time a planner runs.
- `list_all_tools()` failed on session objects whose signature can't be inspected.
- The MCP self-check left two files open and read them with the platform encoding.

## 0.5.0

### Fixed
- The sentinel used for optional MCP parameters leaked through argument binding into checks,
  confirmations and the journal: checks that didn't take the parameter rejected the plan, the
  confirmation showed `<unset>`, and the journal lost a step's arguments so `recover()` couldn't
  undo it. Such parameters are now removed during binding.
- A check had to accept every optional parameter of its tool. Checks now receive only the
  parameters they declare.
- Cancelling or interrupting `arun(..., parallel=True)` could leave steps of the current wave
  running while the rollback happened, so they could finish afterwards. The wave is now stopped
  before rolling back.
- `recover()` journaled only at the end, so recovery interrupted halfway would undo some steps
  twice on the next run. Each undo is now journaled as it happens, and a transaction is marked
  finished only when every undo succeeded, so failed undos are retried later (also after an
  in-process rollback).
- The journal replaced a step's whole result or arguments when one nested value couldn't be
  serialized. Only that value is marked now, and recovery refuses to replay a step whose undo
  would need it.
- `agent(..., attempts=0)` raised `UnboundLocalError`; `attempts` must now be at least 1.
- The self-check for parallel runs measured wall-clock time and could fail on a slow machine. It
  now counts concurrent steps.

### Added
- `UNSET`, a default for optional parameters that should be left out of a call entirely.
- A rejected plan's retry now shows the model the plan it wrote, not only the error.
  `PlanRejected` (a `ValueError`) carries the model's text.
- `GEMINI_MODEL` and `CLAUDE_MODEL` environment variables override the default models.

## 0.4.0

### Fixed
- A permanent step failing after another permanent step had committed rolled back the undoable
  steps the committed one relied on (cancel the old booking, email fails, new seat released: no
  flight at all). Steps before a commit are now kept; only steps after the last commit are undone,
  and `PlanFailed.not_run` lists the permanent steps that never ran.
- Arguments are checked against each tool's signature (and its check's) before anything runs, so
  a hallucinated or missing argument name no longer fails halfway through a plan.
- `Transaction.wrap()` mishandled `*args`, `**kwargs`, positional-only parameters and defaults
  (which never reached the check), and could deadlock when a sync wrapper ran on an event-loop
  thread. Wrappers now bind calls properly, and are async when the tool or check is async.
- MCP tools had a `(**args)` signature and an unprefixed `__name__`. They now have real
  keyword-only signatures from their input schema, prefixed names, and optional parameters that
  are left out instead of sent as null.
- A typo in a `$step.field` reference inside text inserted the whole result instead of failing.
  Paths now resolve as far as they can (`$file.name.txt` gives `report.txt`) and a missing key or
  index is an error.
- Step ids may no longer be purely numeric, so `"$1"` in text is never mistaken for a reference.
- Dry runs ignored `after` dependencies, and a bad reference in a permanent step was blamed on
  the previous step.
- `Transaction` now records `undone` and `undo_failed`, and rollback keeps unwinding through a
  second Ctrl+C or a cancellation instead of stopping.
- Planner parsing: braces in trailing prose broke the JSON extraction, an empty `args_json`
  overwrote `args`, `"after": null` and bare step lists were rejected, and empty Claude or OpenAI
  responses raised `StopIteration` or `IndexError` instead of a clear error.
- `aagent()` blocked the event loop on the terminal prompt and couldn't use async planners.
- Type hints like `Optional[str]`, `Literal`, enums and `list[str]` produced empty schemas.
- MCP text results `"NaN"` and `"Infinity"` were turned into floats; non-text content was
  dropped.
- `plan_with_claude` and `plan_with_openai` no longer need their SDK installed when given a client.
- Demo: unused import removed, a clear message when `OPENAI_MODEL` is missing, and `.env.example`
  documents every variable.

### Added
- `undo(result, args)`: an undo can receive the call's arguments, for tools that return only "ok".
- Previews: a check's return value is shown at the confirmation and in the agent's
  "Scheduled ..." message.
- `parallel=True` runs independent undoable steps concurrently.
- `on_event` structured events with timings.
- `journal=` crash-safe log and `recover()` / `arecover()` to undo what a crash left unfinished.
- Per-tool `retries`, `undo_retries` (default 2, with backoff) and `timeout`.
- `agent()` and `aagent()` send a rejected plan's error back to the model for a corrected plan.
- `__all__` in every module, a pytest suite with coverage in CI, and a CI step that runs the core
  with nothing installed.

## 0.3.0

### Added
- `"$step.key"` and `"$step.list.0"` references pick a field or list item out of a step's result.
  A missing field is an error for whole-value references and is left as written inside text.

### Fixed
- `@tool` on a callable without `__name__` (such as `functools.partial`) now asks for `name=`
  instead of failing with an AttributeError.
- `tools_for_llm()` no longer fails on tools whose signature can't be inspected.
- `plan_with_gemini` retries rate limits and temporary overloads (429/5xx) instead of failing on
  the first one, matching the Anthropic and OpenAI SDKs.
- `plan_with_openai` no longer rejects valid plans from servers that report a normal finish as
  something other than `"stop"`.
- Packaging uses an SPDX license expression, so new setuptools versions build without warnings.

## 0.2.0

### Added
- `Transaction`: guard agents that call tools one at a time. Undoable calls run immediately;
  permanent calls are dry-run immediately and deferred to commit. Sync and async context manager,
  `wrap()` / `wrap_all()` keep tool names, docstrings and signatures.
- Async support everywhere: `arun`, `aagent`, and async tools, undo, check and confirm functions.
- `plan_with_openai` for any OpenAI-compatible API (OpenAI, Ollama, LM Studio, vLLM, Groq,
  OpenRouter, Gemini's OpenAI endpoint, ...), with strict schema, JSON-object and plain modes.
- `planning_request()` and `parse_plan()` to use any other model.
- `registry=` everywhere for isolated tool sets; `@tool` without arguments; `name=` override.
- JSON Schema parameters for local tools, inferred from type hints.
- `PlanFailed.undo_failed` and `PlanFailed.results`.
- CI on Python 3.10 to 3.13, Linux/Windows/macOS, MCP SDK 1.x and 2.x.

### Fixed
- MCP adapter failed on MCP SDK 2.x (renamed result and annotation fields).
- MCP tool lists were not paginated; only the first page was registered.
- Async tools were not awaited, so their steps were marked done without running.
- `KeyboardInterrupt` and task cancellation skipped rollback.
- Typos in `register_mcp(undo=..., checks=...)` were silently ignored.
- `$step` references inside nested lists and objects were not detected or resolved.
- Malformed model output (non-string ids, non-list `after`, list-valued tool names) crashed
  instead of being rejected with a clear error.
- `ask_human` crashed without a terminal; it now declines.
- Model output wrapped in code fences or prose is now parsed.

### Changed
- `tools_for_llm()` entries are `{name, description, reversible, parameters}`.
- `confirm` receives only permanent steps, with references resolved.
- MCP `call()` returned by `register_mcp` is now async.

## 0.1.0

- Initial release: reversibility-aware plan executor, MCP adapter, Gemini and Claude planners,
  live demo.
