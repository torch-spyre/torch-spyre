---
name: inductor-overview
description: "Guidance specific to torch_spyre/_inductor: compilation pipeline internals, pass ordering, and conventions for code changes in this subtree. Use when working on files under torch_spyre/_inductor/."
---

# torch_spyre/_inductor Guidance

This skill is scoped to `torch_spyre/_inductor/` and is discovered
automatically by Claude Code for work under this subtree, independent of the
top-level `.claude/skills/` directory. It is owned and maintained by the
CODEOWNERS of this subtree.

For repo-wide context (Spyre hardware, device registration, general
compilation pipeline overview), see the top-level `project-overview` skill
first if you haven't already.

## Scope of this skill

This is scaffolding. Fill in sections below as inductor-specific guidance
accumulates — pass ordering rules, `OpSpec`/`LoopSpec` contract conventions,
coarse-tiling invariants, scratchpad/LX planning conventions, etc. Prefer
linking to `docs/source/compiler/` for material that's already documented
there rather than duplicating it here.

## Adding to this skill

- Keep guidance specific to `torch_spyre/_inductor/` here. Repo-wide
  conventions (license headers, commit signing, `import regex`, line length)
  stay in the top-level `CLAUDE.md` — don't duplicate them.
- Follow the top-level `CLAUDE.md` conventions for `SKILL.md` frontmatter:
  a quoted single-line `description`, not a multi-line `>-` block scalar.
- Companion reference files for this skill (decision trees, checklists,
  templates) belong alongside this file, under
  `torch_spyre/_inductor/.claude/skills/inductor-overview/`.
