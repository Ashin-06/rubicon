"""Live demo: a real LLM plans against fake in-memory travel tools (no real bookings or emails).

    python demo.py                 # Gemini   (GEMINI_API_KEY or GOOGLE_API_KEY)
    python demo.py claude          # Claude   (ANTHROPIC_API_KEY)
    python demo.py openai          # any OpenAI-compatible API (OPENAI_MODEL, optional
                                   # OPENAI_BASE_URL / OPENAI_API_KEY; e.g. Ollama needs no key)

Keys can live in a .env file next to this script (git-ignored).
"""
import json
import os
import sys
from pathlib import Path

env = Path(__file__).with_name(".env")
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from rubicon import NOOP, PlanFailed, run, tool
from rubicon_llm import ask_human, plan_with_claude, plan_with_gemini, plan_with_openai

db = {"bookings": {"OLD1"}, "seats": {"AI101": 2, "AI202": 0}, "holds": [], "outbox": []}
reg = {}


@tool(undo=NOOP, registry=reg)
def search_flights(route: str) -> list:
    """List flights on a route that have free seats."""
    return [f for f, n in db["seats"].items() if n > 0]


def release(hold):
    db["holds"].remove(hold["flight"])
    db["seats"][hold["flight"]] += 1


@tool(undo=release, registry=reg)
def hold_seat(flight: str) -> dict:
    """Hold a seat on a flight. Can be released."""
    if db["seats"].get(flight, 0) < 1:
        raise RuntimeError(f"no seats on {flight}")
    db["seats"][flight] -= 1
    db["holds"].append(flight)
    return {"flight": flight}


def booking_exists(ref):
    if ref not in db["bookings"]:
        raise RuntimeError(f"no booking {ref}")
    return f"Booking {ref} will be cancelled for good."  # shown when you confirm


@tool(check=booking_exists, registry=reg)
def cancel_booking(ref: str) -> str:
    """Cancel an existing booking. PERMANENT."""
    db["bookings"].remove(ref)
    return "cancelled"


def valid_address(to, body):
    if "@" not in to:
        raise RuntimeError(f"bad address {to}")


@tool(check=valid_address, registry=reg)
def send_email(to: str, body: str) -> str:
    """Send an email. PERMANENT."""
    db["outbox"].append((to, body))
    return "sent"


def openai_planner(goal, client, registry):
    model = os.environ.get("OPENAI_MODEL")
    if not model:
        sys.exit("set OPENAI_MODEL (and OPENAI_BASE_URL for a non-OpenAI server) to use the openai planner")
    return plan_with_openai(goal, client, registry, model=model, base_url=os.environ.get("OPENAI_BASE_URL"))


provider = sys.argv[1] if len(sys.argv) > 1 else "gemini"
planners = {"gemini": plan_with_gemini, "claude": plan_with_claude, "openai": openai_planner}
if provider not in planners:
    sys.exit(f"unknown provider {provider!r}; choose one of {sorted(planners)}")

goal = "Move my BLR-DEL booking OLD1 to flight AI101 and email me@x.com to confirm."
plan = planners[provider](goal, None, reg)
print("Plan:\n" + json.dumps(plan, indent=1))
try:
    results = run(plan, registry=reg, confirm=ask_human)
    print("results:", results)
except PlanFailed as e:
    print("rolled back:", e)
print("final state:", db)
