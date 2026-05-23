---
name: owner-review
description: Use when the user asks for an ownership-lens review from a senior engineer, sysadmin, data scientist, frontend owner, product owner, or similar role; review code, plans, docs, architecture, tests, operations, data quality, or launch readiness with narrow scope, direct findings, and actionable follow-ups.
---

# Owner Review

Perform a direct ownership review from the requested lens. This skill is for
prompts such as "assume the role of a senior engineer who needs to own this",
"have a data scientist review this", "sysadmin hat", or "give honest feedback".

## Scope

Default to the narrowest useful review scope.

- If the repo has uncommitted changes, review `git diff` and directly related
  tests/docs only.
- If the user asks about "this plan", "the proposal", or "what we just
  discussed", review the latest plan/proposal in conversation context.
- If the user names files, modules, commits, PRs, or work streams, review that
  target.
- Do not perform a broad repo-wide audit unless the user explicitly asks for one.
- When broader context is needed, inspect only supporting files required to
  evaluate the target.

## Workflow

1. Identify the requested lens. If unspecified, default to senior software
   engineer owning the codebase long term.
2. Inspect relevant artifacts before giving feedback when code, docs, plans, or
   tests are involved.
3. Lead with findings, risks, and tradeoffs. Do not start with praise.
4. Separate:
   - must-fix issues before commit/ship
   - acceptable temporary debt
   - follow-up work
   - missing tests, docs, monitoring, or operational checks
5. Use concrete file references for code/doc findings.
6. Recommend the next action.

## Lens Checklist

Senior engineer:

- maintainability, coupling, naming, type boundaries, migration risk, test
  quality, docs, and whether the change will be easy to own later.

Sysadmin/ops:

- deployment, rollback, routing, observability, secrets, permissions, external
  dependencies, failure modes, and on-call/debuggability.

Data scientist:

- data quality, leakage, validation, uncertainty, auditability, outlier handling,
  modeling/visualization boundaries, and whether conclusions are defensible.

Frontend owner:

- UX coherence, responsive layout, accessibility, browser behavior, state
  management, API contract usage, and frontend/browser test coverage.

Product owner:

- user workflow, feature sequencing, launch readiness, visible tradeoffs, scope
  control, and whether deferred work is captured clearly.

## Output Style

- Be direct, specific, and constructive.
- Avoid generic praise, reassurance, or recap unless it clarifies a decision.
- Prefer actionable feedback over abstract commentary.
- Do not pretend a literal external expert reviewed the work. Say "from this
  lens" or "as the owner, I would...".
- If there are no material issues, say so clearly and identify residual risk or
  test gaps.

## Follow-Ups

When a finding is real but not in scope for the current change:

- Say whether it belongs in `TODO.md`, a design doc, architecture doc, or a
  future-work section.
- If editing is requested or clearly part of the current task, update that
  artifact.
- If not editing, label it as "Follow-up to capture".
- Do not bury follow-ups in prose.
