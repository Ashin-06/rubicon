# Changelog

## 0.12.0 - 2026-11-23

### Added
- `rubicon_openai_agents.function_tools(tx)` and `rubicon_pydantic_ai.tools(tx)`: a
  `Transaction`'s tools as OpenAI Agents SDK `FunctionTool`s and PydanticAI `Tool`s, with the
  new `openai-agents` and `pydantic-ai` extras and offline examples with scripted models.
- A command line, `python -m rubicon` or `rubicon`: `inspect JOURNAL` lists unfinished
  transactions and dead letters (`--json` too), `recover JOURNAL --module mytools` runs
  `recover()` with your tools, and `ack JOURNAL TX [STEP]` acknowledges a dead letter. Works with
  file, SQLite and PostgreSQL journals; standard library only.
- `checkpoint(value)`: an undoable tool records how to undo what it has done so far, before it
  finishes. If the step then fails, the rollback undoes it with that value, and after a crash
  mid-step `recover()` undoes it instead of reporting it as uncertain. Journaled as a
  `checkpoint` record; the crash fuzz covers a tool that checkpoints and then fails.

## 0.11.0 - 2026-11-23

### Added
- `dry_run(plan, registry)` / `adry_run`: what `run()` would do with nothing executed. It returns the
  order, which steps are undoable or permanent, the point of no return, and the result of each permanent
  step's check whose inputs are known now (`would_fail` lists those that already fail).
- Budgets: `max_calls` and `max_seconds` on `run()`, `arun()` and `Transaction`. Going over raises
  `BudgetExceeded` after rolling back. `run()` rejects a plan with more steps than `max_calls` before
  anything runs, and `max_seconds` is never enforced during a commit.

## 0.10.5 - 2026-11-23

### Documentation
- The README opens with a short recording of the MCP agent demo (a failure rolled back, then an
  approved send), now also on the PyPI page. No code changes.

## 0.10.4 - 2026-11-23

### Fixed
- On Windows, a script that made a sync call (`run()`, `tx.call()`) printed an asyncio
  traceback ("Error on reading from the event loop self pipe") when the interpreter exited, and
  each worker thread's event loop stayed open after the thread ended. The main thread's loop is
  now closed at exit and a worker's when its thread ends.

## 0.10.3 - 2026-11-23

### Added
- A `$reference` that is not a step id but is close to one (`$hodl` for `hold`) logs a warning,
  "did you mean $hold?", since it is sent as plain text.

### Fixed
- A `rubicon_langgraph.Guard` reused for another turn starts with `failure` cleared, so it no longer
  reports the previous turn's failure.

## 0.10.2 - 2026-11-17

### Fixed
- `rubicon_langgraph.Guard` works as `async with`: on an exception it awaits the rollback, so
  async undos run instead of failing inside the running event loop.
- `plan_with_claude(effort=..., fallback=...)`: both can be turned off for models or gateways
  (Bedrock, Vertex) that reject the effort setting or the server-side fallback beta.
- The `langgraph` extra declares `langchain-core`, which `rubicon_langgraph` imports directly.

### Documentation
- The README, CONTRIBUTING.md, RELEASING.md and the MCP demo README match 0.10.x: install from
  PyPI, every module, test file and environment variable listed, measured test count and
  coverage, and links that work on the PyPI page. No code changes.

## 0.10.1 - 2026-11-17

### Added
- `tools/rollback_benchmark.py` and `docs/BENCHMARK.md`: a rebooking agent against a real local
  HTTP service and a real SQLite database with injected timeouts, HTTP 500s, connection resets,
  slow responses, a locked database and crashes mid-step, comparing naive execution, a plain
  LangGraph ToolNode and Rubicon. `tests/test_failure_injection.py` checks the end state after
  rollback and recovery.
- `examples/mcp_agent_demo/`: a runnable agent against a local MCP stdio server.

### Fixed
- The `call` function returned by `register_mcp()` takes the tool name positionally only, so a
  tool argument named `name` no longer collides with it.

### Packaging
- A release workflow (`publish.yml`) uploads each published GitHub release to PyPI through
  trusted publishing, after checking that the tag matches the version and testing the built
  wheel. `RELEASING.md` describes the one-time PyPI setup.

## 0.10.0 - 2026-11-16

### Added
- `PostgresJournal(conninfo, table="rubicon_journal")`, for workers on several machines, with
  the new `postgres` extra (psycopg 3). Same contract as `SQLiteJournal`: ordered `append` and
  `records`, `lock()` through a PostgreSQL advisory lock, and `drop(tx_ids, upto)` for
  `compact()`. Tested in CI against PostgreSQL 17, including the crash and maintenance fuzz
  scenarios.
- `rubicon_otel.tracing()`, an `on_event` callback that records OpenTelemetry spans: one per
  transaction (or stage), with child spans for steps, checks, the gate, undos and the failure.
  New `otel` extra (`opentelemetry-api`).
- Two events: `tx_failed` (the failing step, the error, and what did not run) and `tx_settled`
  (once per commit or rollback, with its status and any undo still owed).
- `rubicon_langgraph.Guard`: Transaction-guarded tools for a LangGraph agent, a commit node that
  asks for confirmation as a human-in-the-loop interrupt, and rollback on errors. New
  `langgraph` extra and a runnable example, `examples/langgraph_agent.py`.

### Fixed
- Since 0.9.0, sync calls on Python 3.11+ ran with the context vars of the thread's first sync
  call, not the caller's current ones: a value set later (a request id, the current tracing
  span) was not seen by tools, checks, undos or confirm. Each call now runs in a copy of the
  caller's context, as with `asyncio.run()`.

## 0.9.0 - 2026-11-15

### Breaking
- The confirmation gate fails closed. `run()`, `arun()`, `agent()`, `aagent()` and
  `Transaction` given no `confirm=` used to approve permanent steps silently; they now refuse
  them. The plan fails with `PlanFailed` at step `"confirm"` after everything undoable is rolled
  back. Plans with only undoable steps still run without a confirm.
- Only a confirm that returns exactly `True` approves. Any other value (`1`, `"yes"`, a non-empty
  list, an object), an exception raised by the confirm, or a failed `await` declines, and the
  gate emits `gate_declined`.
- To keep the old behaviour for unattended runs, pass `confirm=APPROVE_ALL` (new export).
- The self-checks moved out of each module's `__main__` into pytest tests under `tests/`;
  `python rubicon.py` no longer runs anything. Run `pytest` instead.

### Added
- `APPROVE_ALL`, the explicit opt-in for running permanent steps with nobody asked.
- `tests/test_confirm.py`: missing, raising, truthy-but-not-`True` and async-failing confirms
  never run a permanent step, across `run`, `arun`, `Transaction` (sync and async) and `agent`.
- `demo.py --yes` approves unattended with `APPROVE_ALL`.

### Fixed
- Sync calls (`run()`, `tx.call()`, `tx.commit()` outside an event loop) reuse one event loop per
  thread instead of starting a new one each time. On Windows, thousands of new loops in a row
  could leave one stuck opening its loopback socket pair, and each call is now cheaper. Anything a
  call starts is still cancelled before it returns. Python 3.10 keeps a new loop per call.

### Changed
- Tests are grouped by area (`test_core.py`, `test_confirm.py`, `test_mcp.py`, `test_llm.py`,
  `test_fuzz.py`); the README lists the test count and coverage.
- A test that hangs fails after two minutes (`pytest-timeout`, in the `dev` extra), and a CI
  job after twenty.
- CI pins every action by commit SHA, runs the core tests with only pytest installed, and runs the
  suite against the installed wheel.

## 0.8.6 - 2026-11-10

Three independent audits (core concurrency and crashes; untrusted model output, signatures and
MCP; docs and packaging) plus an external review. Every fix below has a check that fails
without it.

### Fixed: crash safety and concurrency
- `compact()` could delete the next stage of a live `Transaction`: it read the journal, then
  dropped every record of the settled transactions, including records written in between. It
  now drops only the records it read (`drop(tx_ids, upto=...)` on both built-in journals).
- A line torn by a power cut or a full disk swallowed the next record in a `FileJournal`, which
  could make `recover()` undo what an in-flight permanent step relied on. A torn line is now
  ended before the next record is written.
- A `recover()` in the same process could undo a transaction that had just committed: the
  transaction left the in-process live set before its closing record was durable, and joined it
  only after its first record. Both orders are fixed, and `recover()` re-reads the journal before
  judging a transaction of its own process.
- `acknowledge(journal, tx)` settled a stage the worker started after the crash report. It now
  settles only what recovery kept or found in flight at that point.
- A journal write that failed during a rollback stopped the rollback, replaced the real error,
  and could report a step as both undone and failed. Every undo is now attempted, each step is
  reported once, and the tool's own error is kept.
- A step running on another thread when the transaction was rolled back stayed in place. It now
  undoes itself when it finishes.
- A step could be marked running only after its "started" record was written, so a settle on
  another thread could miss it. The order is now reversed.
- `with Transaction()` used from sync code on an event loop's thread could neither commit nor
  roll back. Commit and rollback now work there like `call()`.
- An async tool called from sync code inside a running loop ran up to its first `await`, then
  was abandoned without a trace. It is now refused before any of it runs.
- On Windows, a worker that exited with code 259 looked alive to `recover()` forever.
- A result with a cycle or very deep nesting failed the step, even without a journal. Such
  values are now marked in the journal, and nothing is copied when there is no journal.
- `on_dead_letter` was never called for dead letters recorded by a rollback in `run()` or a
  `Transaction`. `recover()` now passes every dead letter to it once.

### Fixed: plans and model output
- An answer containing two different plans ran the first one. It is now rejected as ambiguous.
- Deeply nested model output or arguments raised `RecursionError` instead of `PlanRejected`.
- Some short malformed answers took tens of seconds to reject; parsing is now linear.
- A step field under another name (`arguments`, `depends_on`) was silently dropped, so a tool ran
  without its arguments or before its dependency. Common names are now read as `args` and
  `after` (a repair), and refused with `repair=False`.
- `agent()` retried any `ValueError`, such as a missing API key, and sent its text to the model.
  Only rejected plans are retried now.
- `aagent()` took an async confirm reached through a lambda or an object for a yes without
  awaiting it. Its answer is now awaited, and so is a planner reached the same way.
- `args_json` written as the text `"null"` replaced valid `args`.
- Python-style model output could print a `SyntaxWarning`.
- `plan_with_openai` accepted any `json_mode`; it now raises for an unknown one.
- Model names from `GEMINI_MODEL` and `CLAUDE_MODEL` are stripped of spaces.

### Fixed: arguments and tools
- Argument names that aren't strings raised `TypeError` instead of a clear `ValueError`.
- A positional-only parameter or `*args` after an `UNSET` default failed at call time, after the
  plan had been accepted. Such calls are now bound correctly, or rejected at plan time.
- A reference path into an object that isn't a dict or list (`$hold.seat` on a dataclass)
  inserted the object's text. It is now a bad reference.
- An undo with an option of its own, such as `undo(result, force=False)`, received the step's
  arguments as `force`. A second parameter with a default is now left alone (unless it is named
  `args`).
- Tool descriptions lost the type hints of `functools.partial` objects and callable objects.
- `order()` took quadratic time on large plans.
- Registering a sync tool with `timeout=` now logs a warning, as the timeout can't apply.
- `run()` now shows its real keyword arguments instead of `**kwargs`.

### Fixed: MCP
- One property name that can't be a Python parameter (`"from"`, `"user-id"`) turned off all
  argument checks for that tool. Such tools are now checked by name against their schema, both
  in plans and in `Transaction` calls.
- Some valid schemas (boolean property schemas, a `required` that isn't a list, properties
  named `extra`, `extra_args` and `_extra`) crashed `register_mcp` for the whole server.
- A server listing one tool name twice is refused, so no tool ends up with the wrong label.
- A tool claimed both read-only and destructive is now treated as permanent.
- An error result with only structured content lost its message.
- A successful call whose text was nested too deeply to parse looked like a failure.
- A server handing out new pagination cursors forever is stopped after 1000 pages.

### Docs and packaging
- README rewritten for new users: a runnable quick start, a glossary, when to use it and when
  not, a comparison, troubleshooting, a runnable MCP example, and the API details that were
  missing (`Tool`, `SYSTEM`, `Transaction.id`, step ids, the custom journal contract).
- CONTRIBUTING.md and SECURITY.md added; package metadata completed; CI adds Python 3.14 and a
  job that builds the package and runs the self-checks from the installed wheel.
- Example addresses use example.com.

## 0.8.5

### Fixed
- `recover()` retrying the undos an earlier stage still owed wrote a closing record when it
  finished. If the worker that runs the transaction had started its next stage in the
  meantime, that record marked the new stage as finished, so a later crash left it in place and
  `compact()` could remove it while it ran. Recovery of a transaction that was already settled
  now writes only the per-step "undone" records.
- After a crash, `recover()` reported a read-only step (`undo=NOOP`) whose result wasn't
  JSON-serializable as an undo failure and raised a dead letter for it. Such a step needs
  nothing undone.
- Positional-only parameters given by name were bound in the wrong order when an earlier one
  was left out: `def f(a, b=2, /)` called with `{"b": 99}` set `a` to 99. They now bind in order,
  a left-out one takes its default, and a missing required one is an error.
- A reference like `$step.--0` or with a non-ASCII digit raised a bare `ValueError` from
  `int()` instead of being reported as a bad reference.
- A journal replay left a step open when its undo was recorded without a "done" record (the
  "done" write failed), so recovery could report it as uncertain.
- `parse_plan` repaired a cut-off `args_json` by reading an object nested inside it as the
  arguments. The field is now read whole, or refused.
- A list of objects in the prose before the plan, such as `Context: [{"flight": "AI101"}]`, was
  taken for the plan. A bare list counts as a plan only when its items name a tool, and an
  object only when its `"steps"` is a list.
- `agent(attempts=True)` was accepted as one attempt.
- An MCP tool that takes no arguments and declares `additionalProperties: false` accepted any
  arguments.
- `demo.py` set an empty environment variable from a blank `KEY=` line in `.env`.
- The self-checks left their temporary directories behind; `.gitignore` covers test and
  coverage artifacts.

### Tests
- A new randomised scenario has another worker's `recover()` retry owed undos while the live
  worker starts its next stage in the middle of an undo, then commits, rolls back or dies. It
  fails on 0.8.4. The concurrent-maintenance scenario does the same at random.
- Every fix above has a self-check that fails when the fix is reverted.

## 0.8.4

### Fixed
- A `Transaction` with a pending permanent step and a failed step, whose earlier stage owed an
  undo that another worker acknowledged and then compacted, lost the records that said who runs
  it, so another worker's `recover()` could undo its next stage. Every journal record now
  carries the pid and host of the process running the transaction.
- After a crash during a permanent step, `recover()` held back the undos an earlier,
  rolled-back stage still owed, and `acknowledge(journal, tx)` then wiped them, leaving those
  effects in place for good. Owed undos are now retried even while a permanent step is in
  flight, and acknowledging the whole transaction keeps them until they succeed or are
  acknowledged by name.
- A failed step in a plan that had already committed a permanent step settled the transaction
  from inside the step, ahead of the plan's own settle. The step now only records its failure.
- `parse_plan` read an `args_json` that was a list, number or boolean as no arguments. It is
  now refused like any other bad argument.

### Tests
- The randomised tests now crash right after a permanent step starts, run harsh rounds with
  many declined commits and undo outages, keep a `Transaction` going after a declined commit,
  and leave failed undos to `recover()` until the outage ends. Each fix above has a
  regression check that fails on 0.8.3.

## 0.8.3

### Changed
- `plan_with_gemini` caps the answer at 16,000 output tokens, as `plan_with_claude` already
  did, so a runaway response can't run up cost. A plan is far smaller; a cut-off one is
  rejected and retried.
- Every intentionally ignored exception now says why in a comment.

Both came from scanning Rubicon with vibeaudit (0 critical, 0 high; the one remaining medium
finding is a false alarm in vibeaudit: only "?" placeholders are formatted into the SQL in
`SQLiteJournal.drop()`).

## 0.8.2

### Fixed
- A `Transaction` that committed a stage, had that stage removed by `compact()` (from another
  worker), and then started a new stage wrote no new "begin" record, so another worker's
  `recover()` could not tell the new stage was running and undid it. The same happened when a
  stage ended with an owed undo that a person then acknowledged (found by the randomised tests
  at depth, seed 3079722). Every stage after a settle now writes its own "begin" with pid and
  host, and a transaction running in the current process is never treated as crashed.
- `acknowledge(journal, tx, step)` marked the whole transaction as not running, even if a later
  stage was still running, so another worker's `recover()` could undo that stage. A transaction
  now counts as settled only when nothing else of it is open.
- A transaction whose first step failed stayed unsettled in the journal, so it couldn't be
  compacted, and another worker skipped it for as long as its process lived. The transaction
  now settles itself when nothing is pending or running. (A journal-side rule for the same
  case was tried and rejected: the randomised tests showed that a live `Transaction` can hold
  permanent steps the journal doesn't know about yet.)
- A recovery attempt whose undo failed now records the transaction as settled, so later
  retries are not mistaken for a running transaction.

### Added
- Randomised tests for concurrent maintenance: another worker runs `recover()`, `compact()`
  and `acknowledge()` on the same journal while a transaction runs, and must never undo a
  running step.

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
