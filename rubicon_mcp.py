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

_MAX_PAGES = 1000
_PY_TYPES = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}


def _shape(schema):
    """(properties, required names, extras allowed) from an input schema, tolerating the odd
    but valid ones: boolean property schemas, a "required" that isn't a list of names."""
    schema = schema if isinstance(schema, dict) else {}
    raw = schema.get("properties")
    props = {n: (v if isinstance(v, dict) else {}) for n, v in (raw.items() if isinstance(raw, dict) else ())
             if isinstance(n, str)}
    required = schema.get("required")
    required = [n for n in required if isinstance(n, str)] if isinstance(required, list) else []
    for name in required:
        props.setdefault(name, {})
    extras = schema.get("additionalProperties")
    # Extra arguments only when the schema explicitly allows them, or declares no properties at
    # all. Many servers leave out additionalProperties; accepting anything there would let a
    # hallucinated argument slip through, where the server may silently ignore it.
    allowed = extras is True or isinstance(extras, dict) or (not props and extras is not False)
    return props, set(required), allowed


def _validator(schema, tool):
    """Argument checks for a schema whose names can't be Python parameters ("from", "user-id"):
    the same rules a signature would enforce, by name."""
    props, required, allowed = _shape(schema)

    def validate(args):
        unknown = sorted(set(args) - props.keys())
        if unknown and not allowed:
            raise TypeError(f"{tool} got unexpected arguments {unknown}")
        missing = sorted(required - set(args))
        if missing:
            raise TypeError(f"{tool} is missing required arguments {missing}")
    return validate


def _signature(schema):
    """A keyword-only Python signature for a JSON input schema, or None if the property names
    aren't valid Python identifiers."""
    props, required, allowed = _shape(schema)
    if not all(n.isidentifier() and not keyword.iskeyword(n) for n in props):
        return None

    def annotation(name):
        t = (props[name] or {}).get("type")
        return _PY_TYPES.get(t, typing.Any) if isinstance(t, str) else typing.Any

    kw = inspect.Parameter.KEYWORD_ONLY
    params = [inspect.Parameter(n, kw, annotation=annotation(n)) for n in props if n in required]
    params += [inspect.Parameter(n, kw, default=UNSET, annotation=annotation(n)) for n in props if n not in required]
    if allowed:
        extra = next(n for n in ("extra", "extra_args", *(f"_extra{i}" for i in range(len(props) + 1)))
                     if n not in props)
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
        if len(seen) >= _MAX_PAGES:  # or handing out new ones forever
            raise RuntimeError(f"the server's tool list did not end after {_MAX_PAGES} pages")


async def register_mcp(session, *, prefix="", trust_readonly=False, undo=None, checks=None,
                       registry=None):
    """Register the server's tools into `registry` (default: rubicon.TOOLS).

    Returns `call(name, **args)`, an async function for calling server tools directly, e.g.
    from undo or check functions.
    """
    undo, checks = undo or {}, checks or {}
    reg = TOOLS if registry is None else registry

    async def call(name, /, **args):
        r = await session.call_tool(name, args)
        content = r.content or []
        text = "\n".join(c.text for c in content if getattr(c, "text", None) is not None)
        structured = _field(r, "structured_content", "structuredContent")
        if _field(r, "is_error", "isError"):
            detail = text or (json.dumps(structured, default=repr) if structured is not None else "")
            raise RuntimeError(f"{name}: {detail or 'the server reported an error'}")
        if structured is not None:
            return structured
        if not text and content:  # images, audio, resources: return them instead of ""
            return [c.model_dump(mode="json", exclude_none=True) if hasattr(c, "model_dump") else c
                    for c in content]
        try:  # many servers return JSON as plain text; "NaN"/"Infinity" stay text
            return json.loads(text, parse_constant=_not_json)
        except (ValueError, RecursionError):  # not JSON, or nested too deeply to read: the call
            return text  # still succeeded, so its result must not look like a failure

    def proxy(name, description, schema):
        sig = _signature(schema)
        validate = _validator(schema, prefix + name) if sig is None else None

        async def fn(*a, **k):
            if sig is None:
                if a:
                    raise TypeError(f"{prefix + name} takes its arguments by name")
                validate(k)
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
        if sig is None:
            fn.validate_args = validate  # order() runs it, so a bad plan is refused before it starts
        else:
            fn.__signature__ = sig
            fn.__annotations__ = {p.name: p.annotation for p in sig.parameters.values()
                                  if p.annotation is not inspect.Parameter.empty}
        return fn

    server_tools = await list_all_tools(session)
    names = {t.name for t in server_tools}
    if len(names) != len(server_tools):  # one name, two labels: which one is true is unknowable
        seen, twice = set(), set()
        for t in server_tools:
            (twice if t.name in seen else seen).add(t.name)
        raise ValueError(f"the server lists these tools more than once: {sorted(twice)}")
    unknown = sorted((undo.keys() | checks.keys()) - names)
    if unknown:  # a typo here would silently leave a tool without its undo or check
        raise ValueError(f"undo/checks name tools the server does not have: {unknown}")

    for t in server_tools:
        ann = t.annotations
        # a tool claimed both read-only and destructive is treated as what costs least if wrong
        readonly = bool(ann and _field(ann, "read_only_hint", "readOnlyHint")
                        and not _field(ann, "destructive_hint", "destructiveHint"))
        u = undo.get(t.name) or (NOOP if readonly and trust_readonly else None)
        fn = proxy(t.name, t.description, _field(t, "input_schema", "inputSchema"))
        reg[prefix + t.name] = Tool(fn, u, checks.get(t.name))
    return call
