"""LangGraph integration: Transaction-guarded tools, commit at the end of the graph run, and the
confirmation gate as a human-in-the-loop interrupt.

    guard = Guard(registry)
    builder.add_node("tools", ToolNode(guard.tools()))   # undoable tools run, permanent ones wait
    builder.add_node("commit", guard.commit_node)        # last node: asks, then commits
    graph = builder.compile(checkpointer=MemorySaver())  # interrupts need a checkpointer

    out = graph.invoke(inputs, config)                   # stops at the gate: out["__interrupt__"]
    graph.invoke(Command(resume=True), config)           # only True approves; anything else rolls back

The interrupt's value is {"rubicon": "confirm", "steps": [...]}, each step with its id, tool,
resolved args and check preview, as a confirm function would see them. After the commit node,
guard.results holds the results, or guard.failure the PlanFailed of a declined or failed commit.

The Transaction lives in this process, so resume the graph in the same process. If the graph
fails before the commit node, call guard.rollback(), or use the guard as a context manager.
Needs langgraph (`pip install "rubicon-agents[langgraph]"`).
"""
from __future__ import annotations

import inspect

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt

from rubicon import PlanFailed, Transaction

__all__ = ["Guard"]


class Guard:
    """One Transaction for one graph run, exposed as LangGraph tools and a commit node."""

    def __init__(self, registry=None, log=None, *, on_event=None, journal=None):
        self._approved = None
        self.tx = Transaction(registry, confirm=lambda steps: self._approved is True, log=log,
                              on_event=on_event, journal=journal)
        self.results, self.failure = None, None

    def tools(self):
        """LangChain tools for ToolNode or model.bind_tools(). Undoable tools run at once;
        permanent ones are checked, then scheduled for the commit node."""
        out = []
        for name in self.tx.tools:
            guarded = self.tx.wrap(name)
            description = inspect.getdoc(self.tx.tools[name].fn) or name
            kind = "coroutine" if inspect.iscoroutinefunction(guarded) else "func"
            out.append(StructuredTool.from_function(**{kind: guarded}, name=name, description=description))
        return out

    def _ask(self):
        steps = [{key: s[key] for key in ("id", "tool", "args", "preview")} for s in self.tx.pending]
        # on resume, LangGraph runs the node again and interrupt() returns the human's answer
        self._approved = interrupt({"rubicon": "confirm", "steps": steps}) if steps else True

    def commit_node(self, state):
        """Graph node: ask (as an interrupt) if permanent steps are waiting, then commit."""
        self._ask()
        try:
            self.results = self.tx.commit()
        except PlanFailed as e:
            self.failure = e
        return {}

    async def acommit_node(self, state):
        """commit_node for graphs run with ainvoke/astream."""
        self._ask()
        try:
            self.results = await self.tx.acommit()
        except PlanFailed as e:
            self.failure = e
        return {}

    def rollback(self):
        """Undo everything since the last commit, for a graph run that failed or was abandoned."""
        return self.tx.rollback()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.tx.rollback()
        return False
