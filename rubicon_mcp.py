"""MCP adapter: register every tool of an MCP server, labelled undoable or permanent.

Works with the MCP Python SDK 1.x (ClientSession) and 2.x (ClientSession or Client).

    call = await register_mcp(session, trust_readonly=True,
                              undo={"create_issue": lambda r: call("delete_issue", id=r["id"])},
                              checks={"send_email": validate_email})
    results = await arun(plan)                      # or: async with Transaction() as tx

Labelling (names in `undo` / `checks` are the server's own tool names, without `prefix`):
  - listed in `undo`                     -> undoable with undo(result) or undo(result, args)
  - readOnlyHint=true and trust_readonly -> undoable (nothing to undo)
  - everything else                      -> permanent (MCP's own default: assume destructive)
MCP annotations are the server's *claims*: a lying server could mark a destructive tool
read-only, so read-only tools run before the safety gate only with trust_readonly=True.

Each registered tool gets a real Python signature built from its input schema (keyword-only
parameters with type annotations), so agent frameworks that inspect signatures see the actual
parameters. Checks are called with keyword arguments.
"""
from __future__ import annotations

import inspect
import json
import keyword
import typing

from rubicon import NOOP, TOOLS, UNSET, Tool

__all__ = ["list_all_tools", "register_mcp"]

_PY_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


def _signature(schema):
    """A keyword-only Python signature for a JSON input schema, or None if the property names
    aren't valid Python identifiers."""
    schema = schema or {}
    props = dict(schema.get("properties") or {})
    for name in schema.get("required") or []:
        props.setdefault(name, {})
    if not all(n.isidentifier() and not keyword.iskeyword(n) for n in props):
        return None
    required = set(schema.get("required") or [])

    def annotation(name):
        t = (props[name] or {}).get("type")
        return _PY_TYPES.get(t, typing.Any) if isinstance(t, str) else typing.Any

    kw = inspect.Parameter.KEYWORD_ONLY
    params = [inspect.Parameter(n, kw, annotation=annotation(n)) for n in props if n in required]
    params += [inspect.Parameter(n, kw, default=UNSET, annotation=annotation(n)) for n in props if n not in required]
    # Extra arguments only when the schema explicitly allows them, or declares no properties at
    # all. Many servers leave out additionalProperties; accepting anything there would let a
    # hallucinated argument slip through, where the server may silently ignore it.
    extras = schema.get("additionalProperties")
    if extras is True or isinstance(extras, dict) or not props:
        extra = next(n for n in ("extra", "extra_args", "_extra") if n not in props)
        params.append(inspect.Parameter(extra, inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(params)


def _not_json(constant):
    raise ValueError(constant)


def _field(obj, *names):
    """Read a field under its MCP 2.x (snake_case) or 1.x (camelCase) name."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


async def list_all_tools(session):
    """Every tool on the server, following pagination (SDK 1.x and 2.x)."""
    try:
        by_params = "params" in inspect.signature(session.list_tools).parameters
    except (TypeError, ValueError):  # signature not inspectable: SDK 1.x/2.x sessions take params=
        by_params = True
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
        content = r.content or []
        text = "\n".join(c.text for c in content if getattr(c, "text", None) is not None)
        if _field(r, "is_error", "isError"):
            raise RuntimeError(f"{name}: {text}")
        structured = _field(r, "structured_content", "structuredContent")
        if structured is not None:
            return structured
        if not text and content:  # images, audio, resources: return them instead of ""
            return [c.model_dump(mode="json", exclude_none=True) if hasattr(c, "model_dump") else c
                    for c in content]
        try:  # many servers return JSON as plain text; "NaN"/"Infinity" stay text
            return json.loads(text, parse_constant=_not_json)
        except ValueError:
            return text

    def proxy(name, description, schema):
        sig = _signature(schema)

        async def fn(*a, **k):
            if sig is None:
                return await call(name, **k)
            bound = sig.bind(*a, **k)
            args = {}
            for key, value in bound.arguments.items():
                if sig.parameters[key].kind is inspect.Parameter.VAR_KEYWORD:
                    args.update(value)
                elif value is not UNSET:
                    args[key] = value
            return await call(name, **args)

        fn.__name__ = fn.__qualname__ = prefix + name
        fn.__doc__, fn.input_schema = description, schema
        if sig is not None:
            fn.__signature__ = sig
            fn.__annotations__ = {p.name: p.annotation for p in sig.parameters.values()
                                  if p.annotation is not inspect.Parameter.empty}
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
    import os
    import tempfile
    from pathlib import Path
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

    @srv.tool(annotations=ToolAnnotations(readOnlyHint=True))
    def tag_of(text: str, tag: str = "default") -> dict:
        return {"tag": tag}

    @srv.tool()
    def send_note(to: str, cc: str = "nobody") -> str:
        db["outbox"].append((to, cc))
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

            reg, noted = {}, []
            call = await register_mcp(
                session, registry=reg, trust_readonly=True,
                # async undo that uses the call's arguments rather than its result
                undo={"hold_seat": lambda result, args: call("release_seat", flight=args["flight"])},
                checks={"send_email": valid_to, "send_note": lambda to: noted.append(to)})

            # Optional MCP parameters left out never leak a sentinel into the check, the
            # confirmation or the journal.
            journal = os.path.join(tempfile.mkdtemp(), "j.jsonl")
            shown = []
            await arun([{"id": "t", "tool": "tag_of", "args": {"text": "x"}},
                        {"id": "n", "tool": "send_note", "args": {"to": "a@b"}}], registry=reg, log=quiet,
                       confirm=lambda st: shown.extend(st) or True, journal=journal)
            assert noted == ["a@b"] and shown[0]["args"] == {"to": "a@b"}
            assert db["outbox"].pop() == ("a@b", "nobody")  # server default applied
            text = Path(journal).read_text(encoding="utf-8")
            records = [json.loads(line) for line in text.splitlines()]
            assert next(r for r in records if r["event"] == "done")["args"] == {"text": "x"}
            assert "UNSET" not in text and "unserializable" not in text
            assert [s["id"] for s in order(plan(), reg)] == ["find", "hold", "cancel", "mail"]
            spec = {t["name"]: t for t in tools_for_llm(reg)}
            assert spec["hold_seat"]["parameters"]["required"] == ["flight"]
            assert spec["search"]["description"] == "Find flights with free seats."

            # Real signatures: keyword-only, annotated, visible through Transaction.wrap.
            sig = inspect.signature(reg["hold_seat"].fn)
            assert sig.parameters["flight"].kind is inspect.Parameter.KEYWORD_ONLY
            assert sig.parameters["flight"].annotation is str
            assert "flight" in str(inspect.signature(Transaction(registry=reg).wrap("hold_seat")))
            # Hallucinated or missing arguments are rejected before anything runs.
            for bad in ({"flght": "AI101"}, {}):
                try:
                    order([{"id": "h", "tool": "hold_seat", "args": bad}], reg)
                    raise AssertionError("bad args should be rejected")
                except ValueError:
                    pass
            # Optional parameters left out are omitted, so the server's default applies.
            assert await reg["tag_of"].fn(text="x") == {"tag": "default"}
            assert await reg["tag_of"].fn(text="x", tag="t") == {"tag": "t"}

            # Prefixes reach the function name, so wrapped tools from two servers don't collide.
            gh = {}
            await register_mcp(session, registry=gh, prefix="gh_")
            assert set(gh) == {"gh_" + n for n in reg} and gh["gh_hold_seat"].fn.__name__ == "gh_hold_seat"
            assert Transaction(registry=gh).wrap("gh_search").__name__ == "gh_search"

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

    # Result parsing: "NaN" stays text, JSON text is parsed, non-text content is returned.
    class Stub:
        def __init__(self, content):
            self.content = content

        async def list_tools(self):
            return NS(tools=[NS(name="t", description="", annotations=None, inputSchema=None)], nextCursor=None)

        async def call_tool(self, name, args):
            return NS(content=self.content, isError=False, structuredContent=None)

    async def result_of(content):
        call = await register_mcp(Stub(content), registry={})
        return await call("t")

    assert asyncio.run(result_of([NS(text="NaN")])) == "NaN"
    assert asyncio.run(result_of([NS(text="Infinity")])) == "Infinity"
    assert asyncio.run(result_of([NS(text='{"a": 1}')])) == {"a": 1}
    image = NS(type="image", data="aGk=", mimeType="image/png")
    assert asyncio.run(result_of([image])) == [image]

    # Property names that aren't Python identifiers fall back to **kwargs.
    assert _signature({"properties": {"user-id": {"type": "string"}}}) is None
    assert str(_signature({"properties": {"a": {"type": "integer"}}, "required": ["a"],
                           "additionalProperties": False})) == "(*, a: int)"
    # Without additionalProperties, unknown (hallucinated) arguments are rejected; extras only
    # when the schema allows them or declares no properties.
    plain = {"type": "object", "properties": {"flight": {"type": "string"}}, "required": ["flight"]}
    assert str(_signature(plain)) == "(*, flight: str)"
    assert "**extra" in str(_signature({**plain, "additionalProperties": True}))
    assert "**extra" in str(_signature({**plain, "additionalProperties": {"type": "string"}}))
    assert str(_signature({"type": "object"})) == "(**extra)"

    async def fake(*a, **k):
        pass
    fake.__signature__ = _signature(plain)
    try:
        order([{"id": "h", "tool": "hold", "args": {"flight": "AI101", "seat_class": "first"}}],
              {"hold": Tool(fake)})
        raise AssertionError("unknown argument should be rejected")
    except ValueError:
        pass

    # A session whose list_tools signature can't be inspected still lists every page.
    class Opaque:
        def __init__(self):
            self.inner = Pages("params")

        @property
        def list_tools(self):
            inner = self.inner.list_tools

            class Callable:
                __signature__ = property(lambda s: (_ for _ in ()).throw(ValueError("no signature")))

                async def __call__(self, **kw):
                    return await inner(**kw)
            return Callable()
    assert [t.name for t in asyncio.run(list_all_tools(Opaque()))] == ["a", "b", "c", "d"]
    print("all MCP checks passed")
