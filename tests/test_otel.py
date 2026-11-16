"""rubicon_otel turns on_event into OpenTelemetry spans, checked with the in-memory exporter."""
import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402

from rubicon import APPROVE_ALL, NOOP, PlanFailed, Transaction, run, tool  # noqa: E402
from rubicon_otel import tracing  # noqa: E402


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")

    def finished():
        return {s.name: s for s in exporter.get_finished_spans()}

    yield tracer, finished, exporter


@pytest.fixture
def reg():
    world = {"held": [], "sent": []}
    reg = {}
    tool(lambda f: world["held"].append(f) or f, name="hold", undo=lambda r: world["held"].remove(r),
         registry=reg)

    def must_be_address(to):
        if "@" not in to:
            raise ValueError(f"bad address {to}")
        return f"mail to {to}"

    def send(to):
        if to == "down@example.com":
            raise ConnectionError("smtp down")
        world["sent"].append(to)

    tool(send, name="send", check=must_be_address, registry=reg)
    tool(lambda: "ok", name="look", undo=NOOP, registry=reg)
    return reg


PLAN = [{"id": "h", "tool": "hold", "args": {"f": "AI101"}},
        {"id": "s", "tool": "send", "args": {"to": "me@example.com"}}]


def test_a_committed_plan(spans, reg):
    tracer, finished, exporter = spans
    run(PLAN, registry=reg, confirm=APPROVE_ALL, log=lambda m: None, on_event=tracing(tracer))
    got = finished()
    assert set(got) == {"rubicon.transaction", "rubicon.step hold", "rubicon.step send",
                        "rubicon.check send", "rubicon.gate"}
    tx = got["rubicon.transaction"]
    assert tx.attributes["rubicon.status"] == "committed" and tx.status.status_code != StatusCode.ERROR
    for name in ("rubicon.step hold", "rubicon.step send", "rubicon.check send", "rubicon.gate"):
        assert got[name].parent.span_id == tx.context.span_id
        assert got[name].attributes["rubicon.tx"] == tx.attributes["rubicon.tx"]
    assert got["rubicon.step hold"].attributes["rubicon.outcome"] == "done"
    assert got["rubicon.step send"].attributes["rubicon.outcome"] == "committed"
    assert got["rubicon.step send"].attributes["rubicon.permanent"] is True
    assert got["rubicon.check send"].attributes["rubicon.preview"] == "mail to me@example.com"
    assert got["rubicon.gate"].attributes["rubicon.confirmed"] is True
    assert list(got["rubicon.gate"].attributes["rubicon.steps"]) == ["s"]
    assert tx.start_time <= got["rubicon.step hold"].start_time <= got["rubicon.step hold"].end_time <= tx.end_time


def test_a_declined_gate_and_the_rollback(spans, reg):
    tracer, finished, _ = spans
    with pytest.raises(PlanFailed):
        run(PLAN, registry=reg, confirm=lambda st: False, log=lambda m: None, on_event=tracing(tracer))
    got = finished()
    assert got["rubicon.gate"].attributes["rubicon.confirmed"] is False
    assert got["rubicon.gate"].status.status_code == StatusCode.ERROR
    assert got["rubicon.undo hold"].attributes["rubicon.step"] == "h"
    failure = got["rubicon.failure"]
    assert failure.attributes["rubicon.step"] == "confirm" and list(failure.attributes["rubicon.not_run"]) == ["s"]
    tx = got["rubicon.transaction"]
    assert tx.status.status_code == StatusCode.ERROR and tx.attributes["rubicon.status"] == "rolled_back"


def test_a_failing_permanent_step(spans, reg):
    tracer, finished, _ = spans
    plan = [PLAN[0], {"id": "s", "tool": "send", "args": {"to": "down@example.com"}}]
    with pytest.raises(PlanFailed):
        run(plan, registry=reg, confirm=APPROVE_ALL, log=lambda m: None, on_event=tracing(tracer))
    got = finished()
    step = got["rubicon.step send"]
    assert step.status.status_code == StatusCode.ERROR and "smtp down" in step.status.description
    assert step.attributes["rubicon.outcome"] == "failed"
    assert got["rubicon.failure"].attributes["rubicon.step"] == "s"


def test_a_failed_undo_marks_the_transaction(spans):
    tracer, finished, _ = spans
    reg = {}

    def stuck(r):
        raise ConnectionError("still down")

    tool(lambda: "x", name="take", undo=stuck, undo_retries=0, registry=reg)
    tx = Transaction(registry=reg, log=lambda m: None, on_event=tracing(tracer))
    tx.call("take")
    tx.rollback()
    got = finished()
    assert got["rubicon.undo take"].status.status_code == StatusCode.ERROR
    assert got["rubicon.transaction"].status.status_code == StatusCode.ERROR
    assert list(got["rubicon.transaction"].attributes["rubicon.undo_failed"]) == ["take-1"]


def test_each_committed_stage_is_its_own_span(spans, reg):
    tracer, _, exporter = spans
    with Transaction(registry=reg, log=lambda m: None, on_event=tracing(tracer)) as tx:
        tx.call("look")
        tx.commit()
        tx.call("look")
    stages = [s for s in exporter.get_finished_spans() if s.name == "rubicon.transaction"]
    assert len(stages) == 2 and all(s.attributes["rubicon.status"] == "committed" for s in stages)


def test_the_transaction_nests_under_the_current_span(spans, reg):
    tracer, finished, _ = spans
    with tracer.start_as_current_span("agent turn") as outer:
        run([{"id": "l", "tool": "look"}], registry=reg, log=lambda m: None, on_event=tracing(tracer))
    assert finished()["rubicon.transaction"].parent.span_id == outer.get_span_context().span_id
