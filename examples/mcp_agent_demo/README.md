# MCP agent demo

An agent works through a real MCP server over stdio, with every tool call guarded by Rubicon.
No API key is needed: the agent's tool calls are scripted.

```bash
pip install "rubicon-agents[mcp]"     # or, in a checkout: pip install -e ".[mcp]"
python examples/mcp_agent_demo/run_demo.py
```

## What is in it

`server.py` is an MCP server (the `mcp` package) that works inside one folder:

| Tool | Label in Rubicon | Why |
|---|---|---|
| `read_file` | undoable (nothing to undo) | `readOnlyHint=true`, and the client passes `trust_readonly=True` |
| `write_file` | undoable | the client maps it to `restore_file`, using the previous content `write_file` returns |
| `send_message` | permanent | appends to `outbox.jsonl`; nothing can unsend it |

`run_demo.py` starts the server, registers its tools with `rubicon_mcp.register_mcp()`, and runs
the agent inside a `Transaction`: it reads `notes.md`, writes `summary.md`, rewrites `index.md`,
asks to send a message, then writes `changelog.md`.

- **Run 1** starts the server with `--fail-on changelog.md`, so that last write fails as a full
  disk would. Rubicon restores `index.md`, removes `summary.md`, and drops the message, which was
  held for confirmation and never sent. The workspace ends exactly as it began, with no outbox.
- **Run 2** has no failure. At commit the message is shown for confirmation with its preview,
  approved, and sent.

## Options

```bash
python examples/mcp_agent_demo/run_demo.py --ask          # approve the message yourself
python examples/mcp_agent_demo/run_demo.py --workdir out  # keep the workspace in ./out
python examples/mcp_agent_demo/run_demo.py --llm          # Claude writes the plan instead
```

`--llm` needs `ANTHROPIC_API_KEY` in the environment and `pip install anthropic`. The plan Claude
writes runs through `rubicon.arun()`, so the message still waits for confirmation.

`tests/test_mcp_agent_demo.py` runs the scripted demo end to end.
