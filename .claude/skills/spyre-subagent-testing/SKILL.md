---
name: spyre-subagent-testing
description: "Rules for orchestrators and subagents coordinating test execution in torch-spyre. Use whenever dispatching or writing subagent prompts that may involve running tests."
---

# Spyre Subagent Testing Coordination

## The Hard Constraint

**The Spyre device is exclusively owned by the first process that loads
`torch_spyre`.** No second process can acquire the device until the first
exits. This is a hardware/driver constraint, not a software choice.

This means:

- **Two pytest processes cannot run Spyre tests at the same time.**
- **Two subagents cannot run tests concurrently.**
- Even subagents working on completely unrelated test files will deadlock
  or fail if their test runs overlap.

This is not negotiable and cannot be worked around in software.

---

## Rules for Orchestrators

When using `dispatching-parallel-agents` or any multi-agent pattern in
torch-spyre, apply these rules:

### 1. Never dispatch parallel test-running agents

The standard "one agent per independent domain, dispatch in parallel"
pattern **does not apply to test execution** in this project.

```
# WRONG for torch-spyre:
Agent 1 → Fix and test feature A   ← runs pytest
Agent 2 → Fix and test feature B   ← runs pytest concurrently → DEADLOCK
```

### 2. Separate investigation from verification

Parallelize the investigation/coding phase; serialize the test phase:

```
# RIGHT for torch-spyre:
Phase 1 (parallel):
  Agent 1 → Investigate and fix feature A (no tests)
  Agent 2 → Investigate and fix feature B (no tests)

Phase 2 (sequential, one process):
  Run: python3 -m pytest tests/ -k "feature_a or feature_b"
```

### 3. Assign test execution to exactly one agent (or the orchestrator)

If subagents must run tests, designate a single agent responsible for all
test execution. Other agents return code changes only; the designated agent
(or the orchestrator itself) runs all tests after the others complete.

### 4. Never use parallel pytest options

Even within a single agent, never pass `-n`, `-n auto`, `--dist`, or any
pytest-xdist option. Always run pytest as a single sequential process.

---

## Rules for Subagents

If you are a subagent in torch-spyre and your prompt involves tests:

1. **Do not run tests unless you are the designated test agent.** If your
   prompt does not explicitly assign you test-running responsibility, return
   your code changes and let the orchestrator handle test execution.

2. **If you do run tests**, run a single sequential pytest process. No `-n`,
   no xdist, no background pytest processes.

3. **If tests fail with device acquisition errors**, another process likely
   holds the device. Do not retry in parallel — report back to the
   orchestrator to serialize.

---

## Structuring Subagent Prompts

When writing a subagent prompt that may involve tests, include this block:

```
IMPORTANT — Spyre device constraint: The Spyre device is exclusively owned
by the first process that loads torch_spyre. Do NOT run pytest in parallel
with any other process, and do NOT use pytest -n or xdist. If you need to
run tests, run a single sequential: python3 -m pytest <path>
[Only run tests if explicitly instructed to do so in this prompt.]
```

---

## Template: Multi-Agent Workflow for torch-spyre

```
Phase 1 — Parallel investigation (no test execution):
  Agent A: "Investigate and implement fix for X. Return the code changes
            only. Do NOT run tests."
  Agent B: "Investigate and implement fix for Y. Return the code changes
            only. Do NOT run tests."

Phase 2 — Sequential verification (orchestrator or single agent):
  After all agents complete:
  python3 -m pytest tests/ -k "relevant_tests"
```
