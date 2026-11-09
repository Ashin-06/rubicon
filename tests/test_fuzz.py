"""Randomised invariant tests.

Example tests only cover situations someone thought of. These generate thousands of random
plans with random failures, declined confirmations, undo outages, parallel runs, crashes at every
possible point (including in the middle of a recovery) and restarts, and after each one check the
rules that must always hold.

    pytest tests/test_fuzz.py                        # quick, as in CI
    RUBICON_FUZZ=20000 pytest tests/test_fuzz.py     # deep
    RUBICON_FUZZ_SEED=1234 pytest tests/test_fuzz.py # replay a reported failure
"""
import asyncio
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rubicon import (  # noqa: E402
    _LIVE, PlanFailed, Transaction, acknowledge, arun, compact, dead_letters, recover, run, tool)

COUNT = int(os.environ.get("RUBICON_FUZZ", "400"))
SEED = int(os.environ.get("RUBICON_FUZZ_SEED", "0"))
UNDOABLE = ["u0", "u1", "u2", "ua"]  # ua is async
PERMANENT = ["p0", "p1", "pa"]  # pa is async


class Died(BaseException):
    """The process dies at this point."""


class Boom(Exception):
    """An ordinary failure."""


class World:
    """What the tools act on, with injected failures and a record of every effect."""

    def __init__(self):
        self.items = set()  # undoable effects currently in place
        self.ever = set()  # every undoable effect that ever happened
        self.timeline = []  # ("u" | "p", key) in the order effects happened
        self.violations = []
        self.dead = False
        self.fail_tool, self.fail_check, self.fail_undo = set(), set(), set()

    def registry(self):
        reg = {}

        def apply_u(key):
            if self.dead:
                raise Died()
            if key in self.fail_tool:
                raise Boom(f"{key} failed")
            if key in self.ever:
                self.violations.append(f"{key} ran twice")
            self.items.add(key)
            self.ever.add(key)
            self.timeline.append(("u", key))
            return {"key": key}

        def undo(result):
            if self.dead:
                raise Died()
            key = result["key"]
            if key in self.fail_undo:
                raise Boom(f"undo of {key} failed")
            if key not in self.ever:
                self.violations.append(f"undo of {key}, which never happened")
            self.items.discard(key)  # idempotent, as compensations must be

        def apply_p(key):
            if self.dead:
                raise Died()
            if key in self.fail_tool:
                raise Boom(f"{key} failed")
            if any(k == key for kind, k in self.timeline if kind == "p"):
                self.violations.append(f"permanent {key} ran twice")
            self.timeline.append(("p", key))
            return "ok"

        def check(key):
            if key in self.fail_check:
                raise Boom(f"check {key} failed")

        async def apply_u_async(key):
            await asyncio.sleep(0)
            return apply_u(key)

        async def apply_p_async(key):
            await asyncio.sleep(0)
            return apply_p(key)

        for name in ("u0", "u1", "u2"):
            tool(lambda key: apply_u(key), name=name, undo=undo, undo_retries=0, registry=reg)
        tool(apply_u_async, name="ua", undo=undo, undo_retries=0, registry=reg)
        for name in ("p0", "p1"):
            tool(lambda key: apply_p(key), name=name, check=check, registry=reg)
        tool(apply_p_async, name="pa", check=check, registry=reg)
        return reg

    def permanent_keys(self):
        return [k for kind, k in self.timeline if kind == "p"]

    def last_permanent(self):
        return max((i for i, (kind, _) in enumerate(self.timeline) if kind == "p"), default=-1)


class CrashJournal:
    """An in-memory journal whose process dies after a chosen number of writes."""

    def __init__(self, world, writes_until_crash):
        self.world, self.rows, self.left = world, [], writes_until_crash

    def append(self, records):
        if self.world.dead:
            raise Died()
        if self.left == 0:
            self.world.dead = True
            raise Died()
        self.left -= 1
        self.rows.extend(dict(r) for r in records)

    def records(self):
        return list(self.rows)

    def drop(self, tx_ids):
        self.rows = [r for r in self.rows if r.get("tx") not in set(tx_ids)]

    def restart(self, writes_until_crash=-1):
        """A new process starts: the world is reachable again, nothing in memory is live."""
        self.world.dead = False
        self.left = writes_until_crash
        for row in self.rows:
            _LIVE.discard(row.get("tx"))


def make_plan(rng):
    steps = []
    for i in range(rng.randint(1, 6)):
        after = [s["id"] for s in steps if rng.random() < 0.25]
        steps.append({"id": f"s{i}", "tool": rng.choice(UNDOABLE + PERMANENT), "args": {"key": f"s{i}"},
                      "after": after})
    rng.shuffle(steps)  # plan order must not matter for safety
    return steps


def depends_on_permanent(plan):
    """Steps that (transitively) depend on a permanent step: the only undoable steps allowed to
    run after a permanent one."""
    by_id = {s["id"]: s for s in plan}
    memo = {}

    def walk(sid):
        if sid not in memo:
            memo[sid] = False
            memo[sid] = any(by_id[d]["tool"] in PERMANENT or walk(d) for d in by_id[sid]["after"])
        return memo[sid]
    return {sid for sid in by_id if walk(sid)}


def inject(rng, world, plan):
    for s in plan:
        if rng.random() < 0.1:
            world.fail_tool.add(s["id"])
        if s["tool"] in PERMANENT and rng.random() < 0.1:
            world.fail_check.add(s["id"])
        if s["tool"] in UNDOABLE and rng.random() < 0.15:
            world.fail_undo.add(s["id"])


def execute(rng, plan, reg, **kwargs):
    decline = rng.random() < 0.15
    kwargs.update(registry=reg, confirm=lambda steps: not decline, log=lambda m: None,
                  parallel=rng.random() < 0.3)
    if rng.random() < 0.5:
        return run(plan, **kwargs)
    return asyncio.run(arun(plan, **kwargs))


def check_order(world, plan, problems):
    allowed = depends_on_permanent(plan)
    first_p = next((i for i, (kind, _) in enumerate(world.timeline) if kind == "p"), None)
    if first_p is not None:
        for kind, key in world.timeline[first_p:]:
            if kind == "u" and key not in allowed:
                problems.append(f"undoable {key} ran after a permanent step it doesn't depend on")


def scenario_in_process(rng):
    world = World()
    reg = world.registry()
    plan = make_plan(rng)
    inject(rng, world, plan)
    problems = []
    try:
        results = execute(rng, plan, reg)
    except PlanFailed as e:
        last_p = world.last_permanent()
        if e.committed != world.permanent_keys():
            problems.append(f"committed {e.committed} but permanent effects were {world.permanent_keys()}")
        after = {k for i, (kind, k) in enumerate(world.timeline) if kind == "u" and i > last_p}
        before = {k for i, (kind, k) in enumerate(world.timeline) if kind == "u" and i < last_p}
        if set(e.undone) | set(e.undo_failed) != after:
            problems.append(f"undone {e.undone} + undo_failed {e.undo_failed} != steps after last commit {after}")
        for key in before:
            if key not in world.items:
                problems.append(f"{key} was undone although a committed permanent step relied on it")
        for key in after:
            if (key in world.items) != (key in e.undo_failed):
                problems.append(f"{key}: in place={key in world.items}, undo_failed={key in e.undo_failed}")
        perms = {s["id"] for s in plan if s["tool"] in PERMANENT}
        if set(e.not_run) != perms - set(e.committed):
            problems.append(f"not_run {e.not_run} != {perms - set(e.committed)}")
    else:
        if set(results) != {s["id"] for s in plan}:
            problems.append("a successful run did not return every step")
        if world.items != {s["id"] for s in plan if s["tool"] in UNDOABLE}:
            problems.append("a successful run did not leave every undoable effect in place")
        if sorted(world.permanent_keys()) != sorted(s["id"] for s in plan if s["tool"] in PERMANENT):
            problems.append("a successful run did not apply every permanent step once")
    check_order(world, plan, problems)
    return problems + world.violations


def operator(journal, world, key_of=lambda step: step):
    """A person resolves every dead letter correctly, using what really happened.
    key_of maps a step label to what the step acted on (in plans they are the same)."""
    for letter in dead_letters(journal):
        tx, step, kind = letter["tx"], letter["step"], letter["kind"]
        key = key_of(step)
        if kind == "in_flight_permanent":
            if key in world.permanent_keys():
                acknowledge(journal, tx, note="it went through; keep everything")
            else:
                acknowledge(journal, tx, step, note="it never happened; undo the rest")
        else:  # uncertain or undo_failed: undo by hand if it is in place
            world.items.discard(key)
            acknowledge(journal, tx, step, note="handled by hand")


def scenario_crash(rng):
    world = World()
    reg = world.registry()
    plan = make_plan(rng)
    inject(rng, world, plan)
    journal = CrashJournal(world, writes_until_crash=rng.randint(0, 30))
    completed = False
    try:
        execute(rng, plan, reg, journal=journal)
        completed = True  # returned to the caller: everything it did is final
    except (PlanFailed, Died):
        pass
    check_order_problems = []
    check_order(world, plan, check_order_problems)
    last_p = len(world.timeline) if completed else world.last_permanent()
    keep = {k for i, (kind, k) in enumerate(world.timeline) if kind == "u" and i < last_p}

    # restart; recovery itself may die partway, and undos may keep failing for a while
    journal.restart(writes_until_crash=rng.randint(0, 12) if rng.random() < 0.4 else -1)
    for attempt in range(8):
        if attempt == 3:
            world.fail_undo.clear()  # the outage ends
        try:  # the process may die during recovery or while the operator works: just restart
            recover(journal, registry=reg, log=lambda m: None)
            operator(journal, world)
            if not dead_letters(journal) and not recover(journal, registry=reg, log=lambda m: None):
                break
        except Died:
            journal.restart()

    problems = check_order_problems + world.violations
    for key in keep:
        if key not in world.items:
            problems.append(f"{key} was undone although a permanent step that happened relied on it")
    for i, (kind, key) in enumerate(world.timeline):
        if kind == "u" and i > last_p and key in world.items:
            problems.append(f"{key} ran after the last permanent step and was never cleaned up")
    if dead_letters(journal):
        problems.append(f"dead letters left: {dead_letters(journal)}")
    leftover = recover(journal, registry=reg, log=lambda m: None)
    if leftover:
        problems.append(f"recover() still has work: {leftover}")
    compact(journal)
    if journal.rows:
        problems.append(f"compact() left {len(journal.rows)} records")
    return problems


def recover_until_clean(rng, journal, world, reg, key_of=lambda step: step):
    """Restart, recover (which may itself die), let an operator resolve dead letters, and retry
    failed undos once the outage ends."""
    journal.restart(writes_until_crash=rng.randint(0, 12) if rng.random() < 0.4 else -1)
    for attempt in range(8):
        if attempt == 3:
            world.fail_undo.clear()
        try:
            recover(journal, registry=reg, log=lambda m: None)
            operator(journal, world, key_of)
            if not dead_letters(journal) and not recover(journal, registry=reg, log=lambda m: None):
                return
        except Died:
            journal.restart()


def final_state_problems(world, journal, reg, final_upto):
    """Effects before `final_upto` (a commit that returned, or a permanent step that happened)
    must stay; effects after it must be cleaned up; nothing may be left open."""
    problems = list(world.violations)
    for i, (kind, key) in enumerate(world.timeline):
        if kind != "u":
            continue
        if i < final_upto and key not in world.items:
            problems.append(f"{key} was undone although it was final")
        if i >= final_upto and key in world.items:
            problems.append(f"{key} was never cleaned up")
    if dead_letters(journal):
        problems.append(f"dead letters left: {dead_letters(journal)}")
    leftover = recover(journal, registry=reg, log=lambda m: None)
    if leftover:
        problems.append(f"recover() still has work: {leftover}")
    compact(journal)
    if journal.rows:
        problems.append(f"compact() left {len(journal.rows)} records")
    return problems


def scenario_transaction(rng):
    """An agent loop through Transaction: calls, staged commits, agent crashes, process deaths."""
    world = World()
    reg = world.registry()
    journal = CrashJournal(world, writes_until_crash=rng.randint(0, 40))
    final_upto, completed, labels = 0, False, {}
    tx = Transaction(registry=reg, journal=journal, log=lambda m: None,
                     confirm=lambda steps: rng.random() > 0.15)
    try:
        with tx:
            for i in range(rng.randint(1, 8)):
                key, name = f"t{i}", rng.choice(UNDOABLE + PERMANENT)
                labels[f"{name}-{i + 1}"] = key  # Transaction labels calls "<tool>-<n>"
                if rng.random() < 0.1:
                    world.fail_tool.add(key)
                if rng.random() < 0.1:
                    world.fail_check.add(key)
                if rng.random() < 0.15:
                    world.fail_undo.add(key)
                try:
                    tx.call(name, key)
                except Boom:
                    pass  # the agent sees the tool error and carries on
                if rng.random() < 0.2:
                    tx.commit()
                    final_upto = len(world.timeline)
                if rng.random() < 0.08:
                    raise Boom("the agent crashed")
        completed = True
    except (PlanFailed, Boom, Died):
        pass
    if completed:
        final_upto = len(world.timeline)
    final_upto = max(final_upto, world.last_permanent())
    recover_until_clean(rng, journal, world, reg, key_of=labels.get)
    return final_state_problems(world, journal, reg, final_upto)


def _sweep(scenario, offset):
    failures = []
    for n in range(COUNT):
        seed = SEED + offset + n
        problems = scenario(random.Random(seed))
        if problems:
            failures.append(f"seed {seed}: {problems}")
            if len(failures) >= 5:
                break
    assert not failures, "\n".join(failures)


def test_in_process_invariants():
    _sweep(scenario_in_process, 0)


def test_crash_and_recovery_invariants():
    _sweep(scenario_crash, 1_000_000)


def test_transaction_invariants():
    _sweep(scenario_transaction, 2_000_000)


if __name__ == "__main__":
    test_in_process_invariants()
    test_crash_and_recovery_invariants()
    test_transaction_invariants()
    print(f"fuzz: {COUNT} scenarios each of plans, crash recovery and agent transactions passed")
