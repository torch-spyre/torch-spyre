---
name: weekly-doc-maintenance
description: "Weekly documentation maintenance: audit docs for drift, consistency, and flow, fix the safe issues, propose larger reorganizations for review, and open a humanized PR with no Claude attribution."
---

# Weekly Documentation Maintenance

This skill runs the weekly documentation sweep for torch-spyre. The weekly
GitHub Actions workflow (`.github/workflows/doc-drift-weekly.yaml`) invokes it,
and a maintainer can also run it by hand with `/weekly-doc-maintenance`.

The goal is to keep `docs/source/` matching the code, internally consistent, and
coherent to read, then open one pull request with the safe fixes applied and any
larger reorganization proposed for a human to decide.

## 0. Scope and safety

- Change only files under `docs/`. Never modify code, tests, or CI.
- Apply drift and consistency fixes directly. Do NOT restructure pages or move
  sections on your own; propose those in the PR body instead (section 4).
- If the sweep finds nothing worth changing, open no pull request and stop. Do
  not open an empty or cosmetic PR.
- Every change and every line of the PR follows the writing rules in section 5.

## 1. Preflight (read-only)

Fetch the latest upstream refs and read docs against the current code. Do not
merge, rebase, reset, or check anything out during the audit. The canonical
remote is `https://github.com/torch-spyre/torch-spyre.git`.

## 2. Drift audit

Run the full `check-docs` audit (see `.claude/skills/check-docs`) as the drift
pass, covering the supported-operations table, the API and environment-variable
reference, the compiler-pass docs, runtime, profiling, RFC links, the examples,
and the knowledge-graph extractor. Apply fixes for:

- Statements that no longer match the code, wrong function or file names, wrong
  signatures, and wrong pass order or table rows.
- Dead or wrong links, including example links and cross-references.
- Removed or renamed APIs (for example, a hint key or decorator that no longer
  exists), updated to the current replacement.

## 3. Consistency check

- Terminology guardrails: DMA and DCI describe host-memory to LPDDR5 transfers
  only; never label LPDDR5 to LX scratchpad transfers as DMA or DCI.
- Cross-page naming: the same component, pass, or variable is named the same way
  across pages.
- The environment-variable tables agree with `CLAUDE.md` and the code.
- Figures and captions match the text that references them.

## 4. Flow and organization

Read each area you touched for coherent flow. Apply only safe, local flow fixes
inline: a mislabeled heading, a sentence out of order, a missing lead-in to a
new section.

Do not reorganize on your own. When a larger change would improve coherence,
such as reordering sections, splitting or merging pages, or moving content
between pages, describe it in the PR body under a "Proposed reorganization (for
review)" heading. State the current structure, the proposed structure, and why,
and leave the content in place for a maintainer to decide.

## 5. Writing rules

Apply these to every change and to the PR text:

- No em dashes. Use commas or colons.
- No AI tells, no rhetoric, no metaphors. Plain declarative prose.
- Describe the current state only. No "previously", "used to", "now also", or
  before-and-after framing.
- Formal-technical voice: expand contractions.
- No period or middle dot as a separator inside a title, heading, or caption.

## 6. Build

Rebuild the docs (`make html` in `docs/`). The build must be free of warnings
and errors on the pages you changed. Fix anything the build reports.

## 7. Open the pull request

- Branch name: `docs/weekly-drift-<YYYY-MM-DD>`.
- Commit with sign-off (`git commit -s`).
- No Claude attribution anywhere: no "Generated with Claude Code" line and no
  `Co-Authored-By: Claude` trailer in the commit message or the PR body.
- Title: `docs: weekly drift and consistency maintenance (<date>)`.
- Body, written to the rules in section 5, with three parts: a "Drift fixes"
  list, a "Consistency fixes" list, and a "Proposed reorganization (for review)"
  section when there is one. Each fix names the file and what changed against
  the code.
- Open the PR against the canonical `main`.
