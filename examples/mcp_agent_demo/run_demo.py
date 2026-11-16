"""An agent works through a real MCP server (stdio), guarded by Rubicon.

    python run_demo.py                # scripted agent, no API key needed
    python run_demo.py --ask          # you approve the permanent step on the terminal
    python run_demo.py --llm          # Claude writes the plan (needs ANTHROPIC_API_KEY)
    python run_demo.py --workdir DIR  # keep the workspace in DIR instead of a temp folder

Run 1 injects a failure mid-plan: the server's disk "fills up" on the third write. The two
earlier writes are rolled back and the message, held for confirmation, is never sent.
Run 2 has no failure: the message is shown for confirmation, approved, and sent.
"""
import argparse
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # run from a checkout without installing

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402

from rubicon import Transaction, arun  # noqa: E402
from rubicon_mcp import register_mcp  # noqa: E402


def show(workdir, title):
    print(f"\n  [{title}]")
    for p in sorted(Path(workdir).iterdir()):
        text = p.read_text(encoding="utf-8").strip().replace("\n", " | ")
        print(f"    {p.name}: {text[:70]}")


async def connect(workdir, fail_on, body):
    args = [str(HERE / "server.py"), str(workdir)] + (["--fail-on", fail_on] if fail_on else [])
    params = StdioServerParameters(command=sys.executable, args=args)
    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        reg = {}
        call = await register_mcp(
            session, registry=reg, trust_readonly=True,
            undo={"write_file": lambda r: call("restore_file", name=r["name"], previous=r["previous"])},
            checks={"send_message": lambda to, text: f"Will send to {to}: {text!r}"})
        return await body(reg)


def confirm_for(ask):
    def confirm(steps):
        print("  HELD for confirmation:")
        for s in steps:
            print(f"    {s['id']}: {s['tool']}  {s['preview']}")
        if ask:
            return input("  Send it? [y/N] ").strip().lower() == "y"
        print("  approved (scripted operator)")
        return True
    return confirm


async def agent(tx):
    """What a tool-calling agent does, one call at a time."""
    notes = (await tx.acall("read_file", name="notes.md"))["text"]
    print(f"  read notes.md ({len(notes)} chars)")
    summary = "Summary: " + notes.strip().splitlines()[0]
    print("  ", await tx.acall("write_file", name="summary.md", text=summary))
    print("  ", await tx.acall("write_file", name="index.md", text="- notes.md\n- summary.md\n"))
    print("  ", await tx.acall("send_message", to="team@example.com", text="The summary is ready."))
    print("  ", await tx.acall("write_file", name="changelog.md", text="Added summary.md\n"))


async def scripted(workdir, fail_on, ask):
    async def body(reg):
        tx = Transaction(registry=reg, confirm=confirm_for(ask), log=lambda m: print(f"  rubicon: {m}"))
        try:
            async with tx:
                await agent(tx)
        except Exception as e:
            print(f"  FAILED: {e}")
            print(f"  rolled back: {tx.undone}; never sent: the held message was dropped")
            return False
        print(f"  committed: {tx.committed}")
        return True
    return await connect(workdir, fail_on, body)


async def with_llm(workdir, ask):
    from rubicon_llm import plan_with_claude

    async def body(reg):
        goal = ("Read notes.md, write a one-line summary to summary.md, add it to index.md, "
                "and send team@example.com a message saying the summary is ready.")
        plan = await asyncio.to_thread(plan_with_claude, goal, None, reg)
        print("  plan:", json.dumps(plan, indent=1))
        return await arun(plan, registry=reg, confirm=confirm_for(ask), log=lambda m: print(f"  rubicon: {m}"))
    return await connect(workdir, None, body)


def seed(workdir):
    Path(workdir, "notes.md").write_text("Rubicon rolls back undoable steps.\nMore notes.\n", encoding="utf-8")
    Path(workdir, "index.md").write_text("- notes.md\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rubicon + MCP agent demo")
    ap.add_argument("--workdir")
    ap.add_argument("--ask", action="store_true", help="approve the permanent step yourself")
    ap.add_argument("--llm", action="store_true", help="let Claude plan (needs ANTHROPIC_API_KEY)")
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        workdir = Path(a.workdir or tmp)
        workdir.mkdir(parents=True, exist_ok=True)
        seed(workdir)
        if a.llm:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                sys.exit("--llm needs ANTHROPIC_API_KEY in the environment")
            print("== LLM plan, run through Rubicon")
            asyncio.run(with_llm(workdir, a.ask))
            show(workdir, "workspace after")
            return
        show(workdir, "workspace before")
        print("\n== Run 1: the disk fills up while writing changelog.md")
        asyncio.run(scripted(workdir, "changelog.md", a.ask))
        show(workdir, "workspace after run 1: as before, no outbox")
        print("\n== Run 2: no failure")
        asyncio.run(scripted(workdir, None, a.ask))
        show(workdir, "workspace after run 2")


if __name__ == "__main__":
    main()
