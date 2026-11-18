"""rubicon_langgraph: guarded tools in a LangGraph agent, the gate as an interrupt, commit at the end."""
import runpy
import sys
from pathlib import Path

import pytest

pytest.importorskip("langgraph")
from langchain_core.messages import HumanMessage  # noqa: E402
from langgraph.types import Command  # noqa: E402

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "langgraph_agent.py"


@pytest.fixture
def example():
    return runpy.run_path(str(EXAMPLE), run_name="example")  # a fresh world each time


def run_graph(example, resume):
    guard = example["Guard"](example["reg"], log=lambda m: None)
    graph = example["build"](guard)
    config = {"configurable": {"thread_id": "t"}}
    out = graph.invoke({"messages": [HumanMessage("Move OLD1 to AI101")]}, config)
    return guard, graph, config, out


def test_the_gate_is_an_interrupt_and_approval_commits(example):
    guard, graph, config, out = run_graph(example, True)
    (pending,) = out["__interrupt__"]
    assert pending.value["rubicon"] == "confirm"
    assert [(s["tool"], s["args"], s["preview"]) for s in pending.value["steps"]] == [
        ("cancel_booking", {"ref": "OLD1"}, "Booking OLD1 will be cancelled for good")]
    world = example["world"]
    assert world["bookings"] == {"OLD1"} and world["holds"] == ["AI101"]  # nothing permanent yet
    graph.invoke(Command(resume=True), config)
    assert guard.failure is None and guard.results["cancel_booking-1"] == "cancelled"
    assert world["bookings"] == set() and world["holds"] == ["AI101"]


@pytest.mark.parametrize("answer", [False, "yes", 1, {"approved": True}])
def test_anything_but_true_rolls_back(example, answer):
    guard, graph, config, _ = run_graph(example, answer)
    graph.invoke(Command(resume=answer), config)
    assert guard.failure is not None and guard.failure.step == "confirm"
    assert example["world"] == {"bookings": {"OLD1"}, "seats": {"AI101": 1}, "holds": []}


def test_a_failed_hold_never_reaches_the_gate(example):
    example["world"]["seats"]["AI101"] = 0  # the search finds nothing, the hold fails
    guard, graph, config, out = run_graph(example, True)
    assert out["__interrupt__"]  # the model carried on; the cancel is still only scheduled
    graph.invoke(Command(resume=False), config)
    assert example["world"]["bookings"] == {"OLD1"}


def test_an_error_in_the_graph_rolls_back(example):
    guard = example["Guard"](example["reg"], log=lambda m: None)
    hold = next(t for t in guard.tools() if t.name == "hold_seat")
    with pytest.raises(RuntimeError):
        with guard:
            hold.invoke({"flight": "AI101"})
            raise RuntimeError("the graph crashed")
    assert example["world"]["holds"] == [] and example["world"]["seats"]["AI101"] == 1


def test_tools_keep_names_descriptions_and_schemas(example):
    guard = example["Guard"](example["reg"], log=lambda m: None)
    tools = {t.name: t for t in guard.tools()}
    assert set(tools) == {"search_flights", "hold_seat", "cancel_booking"}
    assert "PERMANENT" in tools["cancel_booking"].description
    assert set(tools["hold_seat"].args) == {"flight"}
    assert tools["cancel_booking"].invoke({"ref": "OLD1"}).startswith("Scheduled")


def test_the_runnable_example(example):
    lines = []
    example["main"](approve=True, say=lines.append)
    assert lines[0].startswith("waiting for approval: cancel_booking")
    assert "'bookings': set()" in lines[-1]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="LangGraph's interrupt() in async nodes needs Python 3.11+")
def test_async_graph(example):
    import asyncio

    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    guard = example["Guard"](example["reg"], log=lambda m: None)
    builder = StateGraph(MessagesState)
    builder.add_node("agent", example["stub_model"])
    builder.add_node("tools", ToolNode(guard.tools(), handle_tool_errors=True))  # the model sees errors
    builder.add_node("commit", guard.acommit_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else "commit")
    builder.add_edge("tools", "agent")
    builder.add_edge("commit", END)
    graph = builder.compile(checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "a"}}

    async def go():
        out = await graph.ainvoke({"messages": [HumanMessage("Move OLD1 to AI101")]}, config)
        assert out["__interrupt__"]
        await graph.ainvoke(Command(resume=True), config)

    asyncio.run(go())
    assert guard.failure is None and example["world"]["bookings"] == set()


def test_async_with_rolls_back_async_undos():
    import asyncio
    from rubicon import tool
    from rubicon_langgraph import Guard
    undone = []
    reg = {}

    async def undo_book(result):
        undone.append("book")

    tool(lambda: "booked", undo=undo_book, name="book", registry=reg)

    async def main():
        with pytest.raises(ZeroDivisionError):
            async with Guard(reg, log=lambda m: None) as g:
                await g.tx.acall("book")
                1 / 0
    asyncio.run(main())
    assert undone == ["book"]
