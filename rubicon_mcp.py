"""MCP adapter for rubicon: registers every tool of an MCP server, labelled undoable or permanent.

    call = await register_mcp(session, trust_readonly=True,
                              undo={"create_issue": lambda r: call("delete_issue", id=r["id"])},
                              checks={"send_email": validate_email})
    results = await asyncio.to_thread(run, plan)   # run() is sync; tools hop back onto this loop

Labelling:
  - tool listed in `undo`                 -> undoable with that undo(result)
  - readOnlyHint=true and trust_readonly  -> undoable (nothing to undo)
  - everything else                       -> permanent (MCP's own default: assume destructive)
MCP annotations are the server's *claims*. A lying server could mark a destructive tool read-only,
so read-only tools run before the safety gate only if you opt in with trust_readonly=True.
"""
import asyncio
import json

from rubicon import NOOP, TOOLS, Tool


async def register_mcp(session, *, prefix="", trust_readonly=False, undo=None, checks=None):
    undo, checks = undo or {}, checks or {}
    loop = asyncio.get_running_loop()

    async def acall(name, args):
        r = await session.call_tool(name, args)
        text = "\n".join(c.text for c in r.content if getattr(c, "text", None) is not None)
        if r.isError:
            raise RuntimeError(f"{name}: {text}")
        if r.structuredContent is not None:
            return r.structuredContent
        try:  # many servers return JSON as plain text
            return json.loads(text)
        except ValueError:
            return text

    def call(name, **args):  # sync; usable inside undo/check functions
        return asyncio.run_coroutine_threadsafe(acall(name, args), loop).result()

    for t in (await session.list_tools()).tools:
        readonly = bool(t.annotations and t.annotations.readOnlyHint)
        u = undo.get(t.name) or (NOOP if readonly and trust_readonly else None)
        fn = lambda _n=t.name, **args: call(_n, **args)
        fn.__doc__, fn.input_schema = t.description, t.inputSchema
        TOOLS[prefix + t.name] = Tool(fn, u, checks.get(t.name))
    return call


if __name__ == "__main__":
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.memory import create_connected_server_and_client_session
    from mcp.types import ToolAnnotations

    from rubicon import PlanFailed, order, run

    srv = FastMCP("travel")
    db = {"bookings": ["OLD1"], "seats": {"AI101": 1}, "holds": [], "outbox": []}
    RO = ToolAnnotations(readOnlyHint=True)

    @srv.tool(annotations=RO)
    def search(route: str) -> list[str]:
        return [f for f, n in db["seats"].items() if n > 0]

    @srv.tool(annotations=ToolAnnotations(destructiveHint=False))
    def hold_seat(flight: str) -> dict:
        if db["seats"].get(flight, 0) < 1:
            raise ValueError(f"no seats on {flight}")
        db["seats"][flight] -= 1
        db["holds"].append(flight)
        return {"flight": flight}

    @srv.tool()
    def release_seat(flight: str) -> str:
        db["holds"].remove(flight)
        db["seats"][flight] += 1
        return "ok"

    @srv.tool(annotations=ToolAnnotations(destructiveHint=True))
    def cancel_booking(ref: str) -> str:
        db["bookings"].remove(ref)
        return "cancelled"

    @srv.tool()
    def send_email(to: str, body: str) -> str:
        db["outbox"].append(to)
        return "sent"

    def plan(flight="AI101", to="me@x.com"):  # naive order: cancel first
        return [
            {"id": "cancel", "tool": "cancel_booking", "args": {"ref": "OLD1"}},
            {"id": "find", "tool": "search", "args": {"route": "BLR-DEL"}},
            {"id": "hold", "tool": "hold_seat", "args": {"flight": flight}},
            {"id": "mail", "tool": "send_email", "args": {"to": to, "body": "rebooked"}, "after": ["hold"]},
        ]

    def valid_to(to, body):
        if "@" not in to:
            raise ValueError(f"bad address {to}")

    quiet = lambda m: None

    async def main():
        async with create_connected_server_and_client_session(srv._mcp_server) as session:
            # Default: hints not trusted, so even the read-only tool is treated as permanent.
            await register_mcp(session)
            assert not any(TOOLS[n].reversible for n in ("search", "hold_seat", "cancel_booking"))

            call = await register_mcp(
                session, trust_readonly=True,
                undo={"hold_seat": lambda r: call("release_seat", flight=r["flight"])},
                checks={"send_email": valid_to})
            assert [s["id"] for s in order(plan())] == ["find", "hold", "cancel", "mail"]

            # Server-side error (no seats) -> nothing permanent happens.
            try:
                await asyncio.to_thread(run, plan(flight="XX9"), log=quiet)
                raise AssertionError("should fail")
            except PlanFailed as e:
                assert e.step == "hold" and "no seats" in str(e.cause) and e.committed == []
            assert db["bookings"] == ["OLD1"]

            # Dry-run check fails -> seat released over MCP, booking untouched.
            try:
                await asyncio.to_thread(run, plan(to="nope"), log=quiet)
                raise AssertionError("should fail")
            except PlanFailed as e:
                assert e.step == "mail" and e.undone == ["hold", "find"]
            assert db["bookings"] == ["OLD1"] and db["seats"]["AI101"] == 1 and db["holds"] == []

            # Happy path.
            await asyncio.to_thread(run, plan())
            assert db["bookings"] == [] and db["holds"] == ["AI101"] and db["outbox"] == ["me@x.com"]
            print("all MCP checks passed")

    asyncio.run(main())
