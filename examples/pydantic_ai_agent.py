"""A PydanticAI agent whose tools run through Rubicon. Runs offline: a scripted model makes the
classic mistake of cancelling first, then looks for a seat.

    pip install "rubicon-agents[pydantic-ai]"
    python examples/pydantic_ai_agent.py          # approve
    python examples/pydantic_ai_agent.py --no     # decline: the held seat is released

Swap FunctionModel(scripted) for a real model, e.g. "openai:gpt-5.1".
"""
import asyncio
import sys

from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from rubicon import NOOP, Transaction, tool
from rubicon_pydantic_ai import tools

world = {"bookings": {"OLD1"}, "holds": []}
reg = {}


def release(flight):
    world["holds"].remove(flight)


@tool(undo=release, registry=reg)
def hold_seat(flight: str) -> str:
    """Hold a seat on a flight; the hold can be released."""
    world["holds"].append(flight)
    return flight


@tool(registry=reg)
def cancel_booking(ref: str) -> str:
    """Cancel a booking. PERMANENT."""
    world["bookings"].discard(ref)
    return "cancelled"


@tool(undo=NOOP, registry=reg)
def search_flights(route: str) -> list:
    """Flights on a route with free seats."""
    return ["AI101"]


def scripted(messages, info):
    if len(messages) == 1:  # first turn: cancel first, the classic mistake
        return ModelResponse(parts=[ToolCallPart("cancel_booking", {"ref": "OLD1"}),
                                    ToolCallPart("search_flights", {"route": "BLR-DEL"}),
                                    ToolCallPart("hold_seat", {"flight": "AI101"})])
    return ModelResponse(parts=[TextPart("Moved OLD1 to AI101.")])


async def main(approve=True, say=print):
    def ask(steps):
        for s in steps:
            say(f"about to run {s['tool']}({s['args']})")
        return approve  # a real app asks a person here

    try:
        async with Transaction(reg, confirm=ask, log=lambda m: None) as tx:  # commits on exit
            result = await Agent(FunctionModel(scripted), tools=tools(tx)).run("Move OLD1 to AI101")
            say(f"agent said: {result.output}")
    except Exception as e:  # PlanFailed: declined, everything undone
        say(f"rolled back: {e}")
    say(f"world: {world}")


if __name__ == "__main__":
    asyncio.run(main(approve="--no" not in sys.argv))
