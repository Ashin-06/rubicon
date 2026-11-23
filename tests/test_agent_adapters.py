"""rubicon_openai_agents and rubicon_pydantic_ai: a Transaction's tools inside other agent SDKs."""
import asyncio
import json
import runpy
from pathlib import Path

import pytest

from rubicon import NOOP, Transaction, tool

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def registry():
    world = {"holds": [], "sent": []}
    reg = {}
    tool(lambda flight: world["holds"].append(flight) or flight, name="hold",
         undo=lambda r: world["holds"].remove(r), registry=reg)
    tool(lambda to: world["sent"].append(to), name="send", registry=reg)

    def broken(n: int):
        raise ConnectionError("service down")

    tool(broken, undo=NOOP, registry=reg)
    return world, reg


@pytest.mark.parametrize("approve", [True, False])
@pytest.mark.parametrize("example", ["openai_agents_agent.py", "pydantic_ai_agent.py"])
def test_the_examples(example, approve):
    pytest.importorskip("agents" if example.startswith("openai") else "pydantic_ai")
    module = runpy.run_path(str(EXAMPLES / example), run_name="example")  # a fresh world each time
    lines = []
    asyncio.run(module["main"](approve=approve, say=lines.append))
    asked = [line for line in lines if line.startswith("about to run")]
    assert asked == ["about to run cancel_booking({'ref': 'OLD1'})"]  # asked once, at the end, for the cancel
    if approve:
        assert module["world"] == {"bookings": set(), "holds": ["AI101"]}
    else:
        assert any(line.startswith("rolled back") for line in lines)
        assert module["world"] == {"bookings": {"OLD1"}, "holds": []}


def test_openai_function_tools():
    pytest.importorskip("agents")
    from agents.tool_context import ToolContext

    from rubicon_openai_agents import function_tools

    world, reg = registry()

    async def go():
        async with Transaction(reg, confirm=lambda st: True, log=lambda m: None) as tx:
            tools = {t.name: t for t in function_tools(tx)}
            assert tools["hold"].params_json_schema["required"] == ["flight"]

            async def invoke(name, **args):
                ctx = ToolContext(context=None, tool_name=name, tool_call_id="1", tool_arguments=json.dumps(args))
                return await tools[name].on_invoke_tool(ctx, json.dumps(args))

            assert await invoke("hold", flight="AI101") == "AI101"
            assert (await invoke("send", to="me")).startswith("Scheduled")
            assert await invoke("broken", n=1) == "Error: ConnectionError: service down"  # the model sees it
            assert world == {"holds": ["AI101"], "sent": []}
        assert world["sent"] == ["me"]  # committed when the block exited

    asyncio.run(go())


def test_pydantic_ai_tool_error_rolls_back():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from rubicon_pydantic_ai import tools

    world, reg = registry()

    def scripted(messages, info):
        return ModelResponse(parts=[ToolCallPart("hold", {"flight": "AI101"}), ToolCallPart("send", {"to": "me"}),
                                    ToolCallPart("broken", {"n": 1})])

    async def go():
        async with Transaction(reg, confirm=lambda st: True, log=lambda m: None) as tx:
            schemas = {t.name: t for t in tools(tx)}
            assert set(schemas) == {"hold", "send", "broken"}
            await Agent(FunctionModel(scripted), tools=list(schemas.values())).run("go")

    with pytest.raises(ConnectionError):  # PydanticAI ends the run; the block rolls back
        asyncio.run(go())
    assert world == {"holds": [], "sent": []}  # the hold was undone, the email never sent
