# Security policy

## Supported versions

Security fixes go into the latest release. Rubicon is in beta (0.x), so please upgrade to the
newest version before reporting.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's private reporting
instead: on the repository's **Security** tab, choose **Report a vulnerability**. Include what
an attacker can do, the affected version, and a way to reproduce it.

You can expect a first answer within a week. Once a fix is released, the advisory is published
with credit to the reporter, unless you prefer otherwise.

## What counts

In scope, for example:

- a plan or model output that makes Rubicon run a tool that is not registered, or with arguments
  that don't fit it;
- a way for a permanent step to run without passing the confirmation;
- recovery undoing a step of a transaction that is still running, or one that a completed
  permanent step relied on;
- an MCP tool ending up labelled undoable or read-only when the rules in the README say it
  should be permanent.

Out of scope: what your own tools, undos and checks do with the arguments they receive (validate
at the tool boundary), and MCP servers that lie in their annotations when you pass
`trust_readonly=True` (see "Security and limitations" in the README).
