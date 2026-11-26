"""OpenAI Agents SDK integration: a Transaction's tools as FunctionTools.

    from agents import Agent, Runner
    from rubicon import Transaction
    from rubicon_openai_agents import function_tools

    async with Transaction(registry, confirm=ask_human) as tx:   # commits on a clean exit
        agent = Agent(name="travel", instructions="...", tools=function_tools(tx))
        await Runner.run(agent, "Move OLD1 to AI101")

Undoable tools run at once; permanent ones are checked and scheduled, and the model is told so.
They run when the block exits, after one confirmation; an exception in the block rolls back.
A tool error goes back to the model as text, as with the SDK's own function_tool, so the agent
can react; the failed call changes nothing. BudgetExceeded is the exception: the
transaction has rolled back, so it ends the run. Needs `pip install "rubicon-agents[openai-agents]"`.
"""
from __future__ import annotations

import json

from agents import FunctionTool

from rubicon import BudgetExceeded, tools_for_llm

__all__ = ["function_tools"]


def _as_text(value):
    return value if isinstance(value, str) else json.dumps(value, default=str)


def function_tools(tx):
    """One FunctionTool per tool in the transaction's registry, each calling through tx."""
    out = []
    for spec in tools_for_llm(tx.tools):
        name = spec["name"]

        async def invoke(ctx, args_json, name=name):
            try:
                return _as_text(await tx.acall(name, **json.loads(args_json or "{}")))
            except BudgetExceeded:  # tx has rolled back: end the run, don't let the block commit
                raise
            except Exception as e:  # the model sees the error and can try something else
                return f"Error: {type(e).__name__}: {e}"

        out.append(FunctionTool(name=name, description=spec["description"] or name,
                                params_json_schema=spec["parameters"], on_invoke_tool=invoke,
                                strict_json_schema=False))
    return out
