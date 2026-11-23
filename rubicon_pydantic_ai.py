"""PydanticAI integration: a Transaction's tools as PydanticAI Tools.

    from pydantic_ai import Agent
    from rubicon import Transaction
    from rubicon_pydantic_ai import tools

    async with Transaction(registry, confirm=ask_human) as tx:   # commits on a clean exit
        agent = Agent("openai:gpt-5.1", tools=tools(tx))
        await agent.run("Move OLD1 to AI101")

Undoable tools run at once; permanent ones are checked and scheduled, and the model is told so.
They run when the block exits, after one confirmation. A tool error ends the run, as in
PydanticAI, and the block rolls back what had run; raise pydantic_ai.ModelRetry from a tool to
let the model try again instead. Needs `pip install "rubicon-agents[pydantic-ai]"`.
"""
from __future__ import annotations

from pydantic_ai import Tool

from rubicon import tools_for_llm

__all__ = ["tools"]


def tools(tx):
    """One Tool per tool in the transaction's registry, each calling through tx."""
    def through(name):
        async def call(**args):
            return await tx.acall(name, **args)
        return call

    return [Tool.from_schema(through(s["name"]), name=s["name"], description=s["description"] or s["name"],
                             json_schema=s["parameters"]) for s in tools_for_llm(tx.tools)]
