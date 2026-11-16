"""OpenTelemetry spans for Rubicon: pass tracing() as on_event.

    from rubicon_otel import tracing
    run(plan, confirm=ask, on_event=tracing())          # or Transaction(..., on_event=tracing())

Each transaction (each stage of a Transaction that commits in stages) becomes a
"rubicon.transaction" span under whatever span is current when it starts. Inside it:

    rubicon.step <tool>      from step_started to step_done, step_committed or step_failed
    rubicon.check <tool>     a check that passed, with its preview
    rubicon.gate             the confirmation, with the steps it covered and the answer
    rubicon.undo <tool>      an undo that ran or failed
    rubicon.failure          the step that failed, the error, and what did not run

Spans carry rubicon.* attributes; failures set the span status to ERROR. Needs
opentelemetry-api (`pip install "rubicon-agents[otel]"`) and an SDK configured by the
application. Without an SDK the spans are no-ops.
"""
from __future__ import annotations

import threading

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

__all__ = ["tracing"]


def _ns(event):
    return int(event["time"] * 1e9)


def _attrs(event, *keys):
    out = {"rubicon.tx": event["tx"]}
    for key in keys:
        value = event.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = [str(v) for v in value]
        elif not isinstance(value, (str, bool, int, float)):
            value = str(value)
        out[f"rubicon.{key}"] = value
    return out


class _Tracing:
    def __init__(self, tracer):
        self.tracer = tracer
        self.lock = threading.Lock()
        self.tx = {}     # tx id -> transaction span
        self.steps = {}  # (tx id, step) -> step span

    def _tx_span(self, event):
        span = self.tx.get(event["tx"])
        if span is None:
            span = self.tx[event["tx"]] = self.tracer.start_span(
                "rubicon.transaction", start_time=_ns(event), attributes=_attrs(event))
        return span

    def _child(self, name, event, attributes, error=None):
        ctx = trace.set_span_in_context(self._tx_span(event))
        span = self.tracer.start_span(name, context=ctx, start_time=_ns(event), attributes=attributes)
        if error is not None:
            span.set_status(Status(StatusCode.ERROR, error))
        return span

    def __call__(self, event):
        kind, at = event["event"], _ns(event)
        with self.lock:
            if kind == "step_started":
                self.steps[event["tx"], event["step"]] = self._child(
                    f"rubicon.step {event['tool']}", event, _attrs(event, "step", "tool", "permanent"))
            elif kind in ("step_done", "step_committed", "step_failed"):
                span = self.steps.pop((event["tx"], event["step"]), None)
                if span is None:
                    span = self._child(f"rubicon.step {event['tool']}", event, _attrs(event, "step", "tool"))
                span.set_attributes(_attrs(event, "duration_ms"))
                span.set_attribute("rubicon.outcome", kind[len("step_"):])
                if kind == "step_failed":
                    span.set_status(Status(StatusCode.ERROR, event.get("error")))
                span.end(end_time=at)
            elif kind == "check_passed":
                self._child(f"rubicon.check {event['tool']}", event,
                            _attrs(event, "step", "tool", "preview")).end(end_time=at)
            elif kind in ("gate_confirmed", "gate_declined"):
                confirmed = kind == "gate_confirmed"
                span = self._child("rubicon.gate", event, {**_attrs(event, "steps"), "rubicon.confirmed": confirmed},
                                   error=None if confirmed else "declined")
                span.end(end_time=at)
            elif kind in ("step_undone", "undo_failed"):
                self._child(f"rubicon.undo {event['tool']}", event, _attrs(event, "step", "tool"),
                            error=event.get("error") if kind == "undo_failed" else None).end(end_time=at)
            elif kind == "tx_failed":
                self._child("rubicon.failure", event, _attrs(event, "step", "error", "not_run"),
                            error=event.get("error")).end(end_time=at)
                self._tx_span(event).set_status(Status(StatusCode.ERROR, event.get("error")))
            elif kind == "tx_settled":
                for key in [k for k in self.steps if k[0] == event["tx"]]:
                    self.steps.pop(key).end(end_time=at)  # cancelled mid-step
                span = self.tx.pop(event["tx"], None)
                if span is not None:
                    span.set_attributes(_attrs(event, "status", "undo_failed"))
                    if event.get("undo_failed"):
                        span.set_status(Status(StatusCode.ERROR, "an undo is still owed"))
                    span.end(end_time=at)


def tracing(tracer=None):
    """An on_event callback that records Rubicon's events as OpenTelemetry spans."""
    return _Tracing(tracer or trace.get_tracer("rubicon"))
