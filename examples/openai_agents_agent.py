"""An OpenAI Agents SDK agent whose tools run through Rubicon. Runs offline: a scripted model
makes the classic mistake of cancelling first, then looks for a seat.

    pip install "rubicon-agents[openai-agents]"
    python examples/openai_agents_agent.py          # approve
    python examples/openai_agents_agent.py --no     # decline: the held seat is released

Drop `model=Scripted()` to use a real OpenAI model (needs OPENAI_API_KEY).
"""
import asyncio
import json
import sys

from agents import Agent, Runner, set_tracing_disabled
from agents.items import ModelResponse
from agents.models.interface import Model
from agents.usage import Usage
from openai.types.responses import ResponseFunctionToolCall, ResponseOutputMessage, ResponseOutputText

from rubicon import NOOP, Transaction, tool
from rubicon_openai_agents import function_tools

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


class Scripted(Model):
    """Stands in for a model: tool calls on the first turn, a short answer on the second."""

    async def get_response(self, system_instructions, input, *args, **kwargs):
        if not any(isinstance(i, dict) and i.get("type") == "function_call_output" for i in input):
            calls = [("cancel_booking", {"ref": "OLD1"}), ("search_flights", {"route": "BLR-DEL"}),
                     ("hold_seat", {"flight": "AI101"})]
            output = [ResponseFunctionToolCall(type="function_call", call_id=f"c{i}", name=n,
                                               arguments=json.dumps(a)) for i, (n, a) in enumerate(calls)]
        else:
            output = [ResponseOutputMessage(id="m", type="message", role="assistant", status="completed", content=[
                ResponseOutputText(type="output_text", text="Moved OLD1 to AI101.", annotations=[])])]
        return ModelResponse(output=output, usage=Usage(), response_id=None)

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError


set_tracing_disabled(True)  # the scripted model has no OpenAI account to send traces to


async def main(approve=True, say=print):
    def ask(steps):
        for s in steps:
            say(f"about to run {s['tool']}({s['args']})")
        return approve  # a real app asks a person here

    try:
        async with Transaction(reg, confirm=ask, log=lambda m: None) as tx:  # commits on exit
            agent = Agent(name="travel", instructions="Rebook the traveller.", tools=function_tools(tx),
                          model=Scripted())
            result = await Runner.run(agent, "Move OLD1 to AI101")
            say(f"agent said: {result.final_output}")
    except Exception as e:  # PlanFailed: declined, everything undone
        say(f"rolled back: {e}")
    say(f"world: {world}")


if __name__ == "__main__":
    asyncio.run(main(approve="--no" not in sys.argv))
