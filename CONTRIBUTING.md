# Contributing to Rubicon

Thanks for helping. Bug reports, fixes, documentation and new integrations are all welcome.

## Reporting a bug

Open an issue with:

- what you ran (ideally a short script that reproduces it),
- what you expected, and what happened instead,
- your Python version, operating system, and the Rubicon version (`pip show rubicon-agents`).

Rubicon's job is to never leave work half-done, so anything that does (a step left in place
after a failure, an undo that ran for something that never happened, a step undone that a
committed step relied on) is treated as a serious bug. For security problems, see
[SECURITY.md](SECURITY.md) instead of opening a public issue.

## Development setup

```bash
git clone https://github.com/Ashin-06/rubicon.git
cd rubicon
python -m venv .venv
. .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[all,dev]"
pytest                            # everything, no API keys or network needed
```

Each module carries its own self-check next to the code (`python rubicon.py`,
`python rubicon_mcp.py`, `python rubicon_llm.py`), and `tests/test_fuzz.py` holds the
randomised invariant tests. A deeper fuzz run before a pull request that touches execution or
recovery is appreciated:

```bash
RUBICON_FUZZ=5000 pytest tests/test_fuzz.py
```

## Pull requests

- Keep the core (`rubicon.py`) free of third-party dependencies.
- Add a check that fails without your change. For a bug, that is a reproduction of the bug.
- Keep public behaviour documented in README.md, and add an entry to CHANGELOG.md.
- Plain ASCII in code and docs.
- One logical change per pull request makes review fast.

By contributing, you agree that your contribution is licensed under the MIT License.
