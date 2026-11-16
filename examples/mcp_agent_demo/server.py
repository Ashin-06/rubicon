"""A small MCP server over stdio: file tools inside one folder, plus a send-style tool.

    python server.py WORKDIR [--fail-on NAME]

  read_file      read-only (readOnlyHint)
  write_file     undoable: returns the previous content so the client can restore it
  restore_file   puts back what write_file replaced (used as write_file's undo)
  send_message   permanent: appends to WORKDIR/outbox.jsonl, which nothing can take back

--fail-on NAME makes write_file fail for that file name, as a full disk would.
"""
import json
import logging
import sys
from pathlib import Path

from mcp.types import ToolAnnotations

try:  # MCP SDK 2.x
    from mcp.server.mcpserver import MCPServer as Server
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # MCP SDK 1.x
    from mcp.server.fastmcp import FastMCP as Server
    from mcp.server.fastmcp.exceptions import ToolError

ROOT = Path(sys.argv[1]).resolve()
FAIL_ON = sys.argv[sys.argv.index("--fail-on") + 1] if "--fail-on" in sys.argv else None
srv = Server("workspace")
logging.disable(logging.INFO)  # keep the demo output readable; warnings still show


def _path(name):
    p = (ROOT / name).resolve()
    if p.parent != ROOT:
        raise ValueError(f"{name!r} is outside the workspace")
    return p


@srv.tool(annotations=ToolAnnotations(readOnlyHint=True))
def read_file(name: str) -> dict:
    """Read a file from the workspace (text is "" if it does not exist)."""
    p = _path(name)
    return {"name": name, "text": p.read_text(encoding="utf-8") if p.exists() else ""}


@srv.tool(annotations=ToolAnnotations(destructiveHint=False))
def write_file(name: str, text: str) -> dict:
    """Write a file in the workspace. Can be undone."""
    p = _path(name)
    if name == FAIL_ON:
        raise ToolError(f"no space left on device while writing {name}")
    previous = p.read_text(encoding="utf-8") if p.exists() else None
    p.write_text(text, encoding="utf-8")
    return {"name": name, "previous": previous}


@srv.tool()
def restore_file(name: str, previous: str | None = None) -> str:
    """Put back what write_file replaced; remove the file if it did not exist before."""
    p = _path(name)
    if previous is None:
        p.unlink(missing_ok=True)
    else:
        p.write_text(previous, encoding="utf-8")
    return "restored"


@srv.tool(annotations=ToolAnnotations(destructiveHint=True))
def send_message(to: str, text: str) -> str:
    """Send a message. PERMANENT: it cannot be unsent."""
    with open(ROOT / "outbox.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"to": to, "text": text}) + "\n")
    return f"sent to {to}"


if __name__ == "__main__":
    srv.run()
