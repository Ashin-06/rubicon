"""Live demo: a real LLM plans against fake in-memory travel tools (no real bookings or emails).

    python demo.py           # Gemini, needs GEMINI_API_KEY or GOOGLE_API_KEY
    python demo.py claude    # Claude, needs ANTHROPIC_API_KEY
"""
import json
import os
import sys
from pathlib import Path

# Load KEY=value lines from .env next to this file (so the key never goes in chat or code).
env = Path(__file__).with_name(".env")
if env.exists():
    for line in env.read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from rubicon import NOOP, tool
from rubicon_llm import ask_human, plan_with_claude, plan_with_gemini, run

planner = plan_with_claude if sys.argv[1:] == ["claude"] else plan_with_gemini

db = {"bookings": {"OLD1"}, "seats": {"AI101": 2, "AI202": 0}, "holds": [], "outbox": []}


@tool(undo=NOOP)
def search_flights(route: str) -> list:
    """List flights on a route that have free seats."""
    return [f for f, n in db["seats"].items() if n > 0]


@tool(undo=lambda r: (db["holds"].remove(r["flight"]), db["seats"].__setitem__(r["flight"], db["seats"][r["flight"]] + 1)))
def hold_seat(flight: str) -> dict:
    """Hold a seat on a flight. Can be released."""
    if db["seats"].get(flight, 0) < 1:
        raise RuntimeError(f"no seats on {flight}")
    db["seats"][flight] -= 1
    db["holds"].append(flight)
    return {"flight": flight}


def _must_exist(ref):
    if ref not in db["bookings"]:
        raise RuntimeError(f"no booking {ref}")


@tool(check=_must_exist)
def cancel_booking(ref: str) -> str:
    """Cancel an existing booking. PERMANENT."""
    db["bookings"].remove(ref)
    return "cancelled"


def _valid(to, body):
    if "@" not in to:
        raise RuntimeError(f"bad address {to}")


@tool(check=_valid)
def send_email(to: str, body: str) -> str:
    """Send an email. PERMANENT."""
    db["outbox"].append((to, body))
    return "sent"


plan = planner("Move my BLR-DEL booking OLD1 to flight AI101 and email me@x.com to confirm.")
print("Plan:\n" + json.dumps(plan, indent=1))
results = run(plan, confirm=ask_human)
print("results:", results)
print("final state:", db)
