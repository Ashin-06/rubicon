"""A LangGraph agent whose tools run through Rubicon, with the confirmation as an interrupt.

    pip install "rubicon-agents[langgraph]"
    python examples/langgraph_agent.py          # approve
    python examples/langgraph_agent.py --no     # decline: the held seat is released

The model is a stub that replays the classic mistake (cancel first, then look for a seat), so
this runs offline. Swap `stub_model` for a real chat model with `.bind_tools(guard.tools())`.
"""
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from rubicon import NOOP, tool
from rubicon_langgraph import Guard

world = {"bookings": {"OLD1"}, "seats": {"AI101": 1}, "holds": []}
reg = {}


@tool(undo=NOOP, registry=reg)
def search_flights(route: str) -> list:
    """Flights on a route that still have free seats."""
    return [f for f, free in world["seats"].items() if free]


def release(hold):
    world["holds"].remove(hold["flight"])
    world["seats"][hold["flight"]] += 1


@tool(undo=release, registry=reg)
def hold_seat(flight: str) -> dict:
    """Hold a seat on a flight; the hold can be released."""
    if not world["seats"].get(flight):
        raise RuntimeError(f"no seats on {flight}")
    world["seats"][flight] -= 1
    world["holds"].append(flight)
    return {"flight": flight}


def booking_exists(ref):
    if ref not in world["bookings"]:
        raise LookupError(f"no booking {ref}")
    return f"Booking {ref} will be cancelled for good"


@tool(check=booking_exists, registry=reg)
def cancel_booking(ref: str) -> str:
    """Cancel a booking. PERMANENT."""
    world["bookings"].remove(ref)
    return "cancelled"


SCRIPT = [  # what the stub model "decides" on each turn
    [("cancel_booking", {"ref": "OLD1"})],
    [("search_flights", {"route": "BLR-DEL"})],
    [("hold_seat", {"flight": "AI101"})],
    [],
]


def stub_model(state):
    turn = sum(isinstance(m, AIMessage) for m in state["messages"])
    calls = [{"name": n, "args": a, "id": f"call{turn}_{i}"} for i, (n, a) in enumerate(SCRIPT[turn])]
    return {"messages": [AIMessage(content="" if calls else "Done.", tool_calls=calls)]}


def build(guard):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", stub_model)
    builder.add_node("tools", ToolNode(guard.tools(), handle_tool_errors=True))  # the model sees errors
    builder.add_node("commit", guard.commit_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else "commit")
    builder.add_edge("tools", "agent")
    builder.add_edge("commit", END)
    return builder.compile(checkpointer=MemorySaver())


def main(approve=True, say=print):
    guard = Guard(reg, log=lambda m: None)
    graph = build(guard)
    config = {"configurable": {"thread_id": "rebook-1"}}
    with guard:  # an error anywhere in the graph rolls back what was done
        out = graph.invoke({"messages": [HumanMessage("Move OLD1 to AI101")]}, config)
        for pending in out.get("__interrupt__", []):
            for step in pending.value["steps"]:
                say(f"waiting for approval: {step['tool']}({step['args']}) -> {step['preview']}")
        graph.invoke(Command(resume=approve), config)
    say(f"results: {guard.results}" if guard.failure is None else f"rolled back: {guard.failure}")
    say(f"world: {world}")
    return guard


if __name__ == "__main__":
    main(approve="--no" not in sys.argv)
