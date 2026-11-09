"""MCP adapter: register every tool of an MCP server, labelled undoable or permanent.

Works with the MCP Python SDK 1.x (ClientSession) and 2.x (ClientSession or Client).

    call = await register_mcp(session, trust_readonly=True,
                              undo={"create_issue": lambda r: call("delete_issue", id=r["id"])},
                              checks={"send_email": validate_email})
    results = await arun(plan)                      # or: async with Transaction() as tx

Labelling (names in `undo` / `checks` are the server's own tool names, without `prefix`):
  - listed in `undo`                     -> undoable with that undo(result) (sync or async)
  - readOnlyHint=true and trust_readonly -> undoable (nothing to undo)
  - everything else                      -> permanent (MCP's own default: assume destructive)
MCP annotations are the server's *claims*: a lying server could mark a destructive tool
read-only, so read-only tools run before the safety gate only with trust_readonly=True.
"""
import inspect
import json

from rubicon import NOOP, TOOLS, Tool


def _field(obj, *names):
    """Read a field under its MCP 2.x (snake_case) or 1.x (camelCase) name."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


async def list_all_tools(session):
    """Every tool on the server, following pagination (SDK 1.x and 2.x)."""
    by_params = "params" in inspect.signature(session.list_tools).parameters
    tools, cursor, seen = [], None, set()
    while True:
        if cursor is None:
            page = await session.list_tools()
        elif by_params:
            from mcp.types import PaginatedRequestParams
            page = await session.list_tools(params=PaginatedRequestParams(cursor=cursor))
        else:
            page = await session.list_tools(cursor=cursor)
        tools += page.tools
        cursor = _field(page, "next_cursor", "nextCursor")
        if not cursor or cursor in seen:  # guard against a server repeating a cursor
            return tools
        seen.add(cursor)


async def register_mcp(session, *, prefix="", trust_readonly=False, undo=None, checks=None,
                       registry=None):
    """Register the server's tools into `registry` (default: rubicon.TOOLS).

    Returns `call(name, **args)`, an async function for calling server tools directly, e.g.
    from undo or check functions.
    """
    undo, checks = undo or {}, checks or {}
    reg = TOOLS if registry is None else registry

    async def call(name, **args):
        r = await session.call_tool(name, args)
        text = "\n".join(c.text for c in (r.content or []) if getattr(c, "text", None) is not None)
        if _field(r, "is_error", "isError"):
            raise RuntimeError(f"{name}: {text}")
        structured = _field(r, "structured_content", "structuredContent")
        if structured is not None:
            return structured
        try:  # many servers return JSON as plain text
            return json.loads(text)
        except ValueError:
            return text

    def proxy(name, description, schema):
        async def fn(**args):
            return await call(name, **args)
        fn.__name__, fn.__doc__, fn.input_schema = name, description, schema
        return fn

    server_tools = await list_all_tools(session)
    names = {t.name for t in server_tools}
    unknown = sorted((undo.keys() | checks.keys()) - names)
    if unknown:  # a typo here would silently leave a tool without its undo or check
        raise ValueError(f"undo/checks name tools the server does not have: {unknown}")

    for t in server_tools:
        ann = t.annotations
        readonly = bool(ann and _field(ann, "read_only_hint", "readOnlyHint"))
        u = undo.get(t.name) or (NOOP if readonly and trust_readonly else None)
        fn = proxy(t.name, t.description, _field(t, "input_schema", "inputSchema"))
        reg[prefix + t.name] = Tool(fn, u, checks.get(t.name))
    return call


if __name__ == "__main__":
    import asyncio
    from types import SimpleNamespace as NS

    from mcp.types import ToolAnnotations

    from rubicon import PlanFailed, Transaction, arun, order, tools_for_llm

    try:  # MCP SDK 2.x
        from mcp import Client
        from mcp.server.mcpserver import MCPServer as Server

        def connect(srv):
            return Client(srv)
    except ImportError:  # MCP SDK 1.x
        from mcp.server.fastmcp import FastMCP as Server
        from mcp.shared.memory import create_connected_server_and_client_session

        def connect(srv):
            return create_connected_server_and_client_session(srv._mcp_server)

    srv = Server("travel")
    db = {"bookings": ["OLD1"], "seats": {"AI101": 1}, "holds": [], "outbox": []}

    @srv.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def search(route: str) -> list[str]:
        """Find flights with free seats."""
        return [f for f, n in db["seats"].items() if n > 0]

    @srv.tool(annotations=ToolAnnotations(destructiveHint=False))
    def hold_seat(flight: str) -> dict:
        """Hold a seat."""
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

    def clean():
        return db["bookings"] == ["OLD1"] and db["seats"]["AI101"] == 1 and db["holds"] == []

    async def main():
        async with connect(srv) as session:
            # Default: hints not trusted -> every tool is permanent.
            reg = {}
            await register_mcp(session, registry=reg)
            assert not any(t.reversible for t in reg.values())

            try:
                await register_mcp(session, registry={}, undo={"hold_sea": NOOP})
                raise AssertionError("typo should be rejected")
            except ValueError as e:
                assert "hold_sea" in str(e)

            reg = {}
            call = await register_mcp(
                session, registry=reg, trust_readonly=True,
                undo={"hold_seat": lambda r: call("release_seat", flight=r["flight"])},  # async undo
                checks={"send_email": valid_to})
            assert [s["id"] for s in order(plan(), reg)] == ["find", "hold", "cancel", "mail"]
            spec = {t["name"]: t for t in tools_for_llm(reg)}
            assert spec["hold_seat"]["parameters"]["required"] == ["flight"]
            assert spec["search"]["description"] == "Find flights with free seats."

            # Server-side error (no seats) -> nothing permanent happens.
            try:
                await arun(plan(flight="XX9"), registry=reg, log=quiet)
                raise AssertionError("should fail")
            except PlanFailed as e:
                assert e.step == "hold" and e.committed == []
            assert clean()

            # Dry-run check fails -> seat released over MCP, booking untouched.
            try:
                await arun(plan(to="nope"), registry=reg, log=quiet)
                raise AssertionError("should fail")
            except PlanFailed as e:
                assert e.step == "mail" and e.undone == ["hold", "find"]
            assert clean()

            # Transaction over MCP tools: agent crashes -> rolled back over MCP.
            try:
                async with Transaction(registry=reg, log=quiet) as tx:
                    await tx.acall("hold_seat", flight="AI101")
                    await tx.acall("cancel_booking", ref="OLD1")
                    raise RuntimeError("agent crashed")
            except RuntimeError:
                pass
            assert clean()

            # Happy path.
            await arun(plan(), registry=reg, log=quiet)
            assert db["bookings"] == [] and db["holds"] == ["AI101"] and db["outbox"] == ["me@x.com"]

    asyncio.run(main())

    # Pagination, both SDK styles, plus a server that repeats its cursor.
    class Pages:
        def __init__(self, style, loop_forever=False):
            self.loop_forever = loop_forever
            pages = {None: (["a", "b"], "c1"), "c1": (["c"], "c2"), "c2": (["d"], None)}
            self.pages = pages
            if style == "params":
                async def list_tools(*, params=None):
                    return self.page(params.cursor if params else None, "next_cursor")
            else:
                async def list_tools(cursor=None):
                    return self.page(cursor, "nextCursor")
            self.list_tools = list_tools

        def page(self, cursor, field):
            names, nxt = self.pages[cursor]
            nxt = "c1" if self.loop_forever and cursor == "c1" else nxt
            return NS(tools=[NS(name=n) for n in names], **{field: nxt})

    for style in ("params", "cursor"):
        assert [t.name for t in asyncio.run(list_all_tools(Pages(style)))] == ["a", "b", "c", "d"]
    assert [t.name for t in asyncio.run(list_all_tools(Pages("cursor", loop_forever=True)))] == ["a", "b", "c"]
    print("all MCP checks passed")
