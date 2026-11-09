# Changelog

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
