# Changelog

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
- CI on Python 3.10–3.13, Linux/Windows/macOS, MCP SDK 1.x and 2.x.

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
