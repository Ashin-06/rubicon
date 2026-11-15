"""MCP adapter: tool registration, annotations, undo mapping, pagination and a live stdio server."""
import asyncio
import inspect
import json
import os
import tempfile
from pathlib import Path
from rubicon import NOOP
from rubicon import APPROVE_ALL, PlanFailed, Transaction, arun, order, tools_for_llm
from rubicon import Tool
from rubicon import Transaction as _Tx
from types import SimpleNamespace as NS

import pytest

pytest.importorskip("mcp")
from mcp.types import ToolAnnotations  # noqa: E402

from rubicon_mcp import (_signature, list_all_tools, register_mcp)

@pytest.fixture(autouse=True, scope="module")
def _scratch(tmp_path_factory):
    """Every mkdtemp() in this module lands in one scratch folder that pytest cleans up."""
    previous = tempfile.tempdir
    tempfile.tempdir = str(tmp_path_factory.mktemp("scratch"))
    yield
    tempfile.tempdir = previous


def test_live_server_round_trip():
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

    def plan(flight="AI101", to="me@example.com"):  # naive order: cancel first
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
            assert "flight" in str(inspect.signature(Transaction(registry=reg, confirm=APPROVE_ALL).wrap("hold_seat")))
            # Hallucinated or missing arguments are rejected before anything runs.
            for bad in ({"flght": "AI101"}, {}):
                try:
                    order([{"id": "h", "tool": "hold_seat", "args": bad}], reg)
                    raise AssertionError("bad args should be rejected")
                except ValueError:
                    pass  # expected: the plan was rejected
            # Optional parameters left out are omitted, so the server's default applies.
            assert await reg["tag_of"].fn(text="x") == {"tag": "default"}
            assert await reg["tag_of"].fn(text="x", tag="t") == {"tag": "t"}

            # Prefixes reach the function name, so wrapped tools from two servers don't collide.
            gh = {}
            await register_mcp(session, registry=gh, prefix="gh_")
            assert set(gh) == {"gh_" + n for n in reg} and gh["gh_hold_seat"].fn.__name__ == "gh_hold_seat"
            assert Transaction(registry=gh, confirm=APPROVE_ALL).wrap("gh_search").__name__ == "gh_search"

            # Server-side error (no seats) -> nothing permanent happens.
            try:
                await arun(plan(flight="XX9"), registry=reg, log=quiet, confirm=APPROVE_ALL)
                raise AssertionError("should fail")
            except PlanFailed as e:
                assert e.step == "hold" and e.committed == []
            assert clean()

            # Dry-run check fails -> seat released over MCP, booking untouched.
            try:
                await arun(plan(to="nope"), registry=reg, log=quiet, confirm=APPROVE_ALL)
                raise AssertionError("should fail")
            except PlanFailed as e:
                assert e.step == "mail" and e.undone == ["hold", "find"]
            assert clean()

            # Transaction over MCP tools: agent crashes -> rolled back over MCP.
            try:
                async with Transaction(registry=reg, log=quiet, confirm=APPROVE_ALL) as tx:
                    await tx.acall("hold_seat", flight="AI101")
                    await tx.acall("cancel_booking", ref="OLD1")
                    raise RuntimeError("agent crashed")
            except RuntimeError:
                pass  # expected: the simulated agent crash; the rollback is checked below
            assert clean()

            # Happy path.
            await arun(plan(), registry=reg, log=quiet, confirm=APPROVE_ALL)
            assert db["bookings"] == [] and db["holds"] == ["AI101"] and db["outbox"] == ["me@example.com"]

    asyncio.run(main())


def test_pagination_both_sdk_styles_plus_a_server_that_repeats():
    global Pages
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


def test_result_parsing():
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


def test_property_names_that_aren_t_python_identifiers_fall_back():
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
    assert str(_signature({"type": "object", "properties": {}, "additionalProperties": False})) == "()"

    async def fake(*a, **k):
        pass
    fake.__signature__ = _signature(plain)
    try:
        order([{"id": "h", "tool": "hold", "args": {"flight": "AI101", "seat_class": "first"}}],
              {"hold": Tool(fake)})
        raise AssertionError("unknown argument should be rejected")
    except ValueError:
        pass  # expected: the undeclared argument was rejected


def test_a_session_whose_list_tools_signature_can_t_be_inspected():
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


def test_odd_but_valid_schemas_hostile_servers_and_contradictory():
    global Server, bad, expect, spec
    # Odd but valid schemas, hostile servers and contradictory hints (independent audit, 0.8.6).
    def expect(exc, fn, *a, **k):
        try:
            fn(*a, **k)
        except exc as e:
            return e
        raise AssertionError(f"expected {exc.__name__}")

    class Server:
        def __init__(self, tools, result=None):
            self.tools, self.result, self.sent = tools, result, []

        async def list_tools(self):
            return NS(tools=self.tools, nextCursor=None)

        async def call_tool(self, name, args):
            self.sent.append((name, args))
            return self.result or NS(content=[NS(text="ok")], isError=False, structuredContent=None)

    def spec(name, schema=None, ro=None, destructive=None):
        ann = None if ro is None else ToolAnnotations(readOnlyHint=ro, destructiveHint=destructive)
        return NS(name=name, description="", annotations=ann, inputSchema=schema)

    # names that can't be Python parameters are still checked, before anything runs
    mail = Server([spec("send", {"type": "object", "properties": {"from": {"type": "string"},
                                                                   "to": {"type": "string"}},
                                 "required": ["to"], "additionalProperties": False})])
    odd = {}
    asyncio.run(register_mcp(mail, registry=odd))
    for bad in ({"to": "a@example.com", "bcc": "everyone@example.com"}, {"from": "x"}):
        expect(ValueError, order, [{"id": "s", "tool": "send", "args": bad}], odd)
    assert asyncio.run(arun([{"id": "s", "tool": "send", "args": {"from": "me", "to": "you"}}],
                            registry=odd, log=lambda m: None, confirm=APPROVE_ALL)) == {"s": "ok"}
    assert mail.sent == [("send", {"from": "me", "to": "you"})]
    try:
        asyncio.run(_Tx(registry=odd, log=lambda m: None).acall("send", to="you", bcc="everyone@example.com"))
        raise AssertionError("a Transaction call must check the names too")
    except TypeError:
        assert len(mail.sent) == 1  # nothing was sent
    try:
        asyncio.run(odd["send"].fn("positional"))
        raise AssertionError("positional arguments must be refused")
    except TypeError:
        pass


def test_boolean_property_schemas_a_required_that_isn_t_a_list():
    # boolean property schemas, a "required" that isn't a list, and taken **extra names
    weird = Server([spec("b", {"properties": {"a": True}}), spec("r", {"required": "flight"}),
                    spec("e", {"properties": {"extra": {}, "extra_args": {}, "_extra": {}},
                               "additionalProperties": True})])
    asyncio.run(register_mcp(weird, registry={}))
    assert str(_signature({"required": "flight"})) == "(**extra)"
    assert "**_extra0" in str(_signature({"properties": {"extra": {}, "extra_args": {}, "_extra": {}},
                                          "additionalProperties": True}))


def test_a_tool_listed_twice_can_t_be_labelled_reliably():
    # a tool listed twice can't be labelled reliably: refused
    expect(ValueError, lambda: asyncio.run(register_mcp(Server([spec("t", ro=True), spec("t", ro=False)]),
                                                        registry={}, trust_readonly=True)))


def test_read_only_and_destructive_at_once():
    # read-only and destructive at once: treated as permanent
    hints = {}
    asyncio.run(register_mcp(Server([spec("both", ro=True, destructive=True), spec("ro", ro=True)]),
                             registry=hints, trust_readonly=True))
    assert not hints["both"].reversible and hints["ro"].reversible


def test_an_error_result_with_only_structured_content_keeps_its():
    global call, e
    # an error result with only structured content keeps its message
    failing = Server([spec("q")], NS(content=[], isError=True, structuredContent={"error": "quota exceeded"}))
    call = asyncio.run(register_mcp(failing, registry={}))
    try:
        asyncio.run(call("q"))
        raise AssertionError("an error result must raise")
    except RuntimeError as e:
        assert "quota exceeded" in str(e)


def test_a_successful_call_whose_text_is_nested_too_deeply_to():
    global call
    # a successful call whose text is nested too deeply to parse is still a success
    deep = Server([spec("d")], NS(content=[NS(text="[" * 5000 + "]" * 5000)], isError=False, structuredContent=None))
    call = asyncio.run(register_mcp(deep, registry={}))
    deep_result = asyncio.run(call("d"))  # parsed where the stack allows it, else the text itself
    assert isinstance(deep_result, list) or deep_result.startswith("[[["), type(deep_result)


def test_a_server_that_hands_out_new_cursors_forever_is_stopped():
    # a server that hands out new cursors forever is stopped
    class Endless:
        def __init__(self):
            self.n = 0

        async def list_tools(self, cursor=None):
            self.n += 1
            return NS(tools=[], nextCursor=f"c{self.n}")

    expect(RuntimeError, lambda: asyncio.run(list_all_tools(Endless())))
