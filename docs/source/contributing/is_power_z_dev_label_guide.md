# Label: `is_power_z_dev`

Marks issues and pull requests relating to **multi-platform enablement** — work
that exists because torch-spyre targets IBM Z (`s390x`) and POWER (`ppc64le`)
alongside x86.

---

## Why this label exists

torch-spyre runs on more than one hardware architecture, and platform
enablement work is spread thinly across every component — the compiler
backend, the runtime, operator coverage, CI, packaging, benchmarking. Very
little of it lives in one place, and much of it isn't obvious from the title.

The label makes that work **findable as a body of work** rather than as
scattered individual tickets. In practice it is used to:

- find open platform work to pick up, or check whether something is already in flight
- identify changes that need validating on more than one architecture
- review related work together when planning a release
- give newcomers to platform work a single entry point

If you maintain or contribute to a component, labelling your platform work is
what makes it visible to everyone else doing the same.

---

## When to apply it

**The test is why the work exists — not whether the title names an
architecture.** If the issue or PR would not exist were it not for
multi-platform support, apply the label.

Apply it to:

| | Example |
|---|---|
| Ports, builds, or fixes for `s390x` or `ppc64le` | *Fix alignment fault in codegen on ppc64le* |
| Sub-tasks of a platform workstream — CI, packaging, test fixes, automation — **even with no architecture in the title** | *Update the build to produce a manylinux wheel* |
| x86 work done as a **baseline for comparison** against another platform | *Run the benchmark suite on x86 to produce comparable output* |
| Planning, analysis, or design for platform enablement | *Analyse the pipeline architecture to identify integration points* |

Do **not** apply it to:

- ordinary single-platform work that would have happened regardless
- generic infrastructure with no platform driver
- endianness- or arch-adjacent code that isn't being changed for platform reasons

> **The common mistake is under-labelling.** Roughly half of correctly
> labelled work names no architecture anywhere in the title, because it's a
> supporting task within a larger platform effort. Those still count. If you
> are unsure, apply it — an over-applied label is easy to spot and remove; an
> omitted one is invisible.

---

## How to apply it

Add `is_power_z_dev` from the **Labels** menu on any issue or pull request, or:

```bash
gh issue edit <number> --add-label "is_power_z_dev"
gh pr edit <number> --add-label "is_power_z_dev"
```

Apply it to **both** the issue and the PR that resolves it where both exist.

### Finding existing work that should carry it

If you're tidying up a component's backlog, these searches surface most
candidates:

```
label:is_power_z_dev                          # already labelled
s390x OR ppc64le OR powerpc in:title,body     # names an architecture
"big-endian" OR "cross-compile" in:title,body # common platform concerns
```

Then check what **workstream** each result belongs to. Most platform tickets
sit inside a larger effort — open the epic or linked issues and label the whole
set, not only the one that happened to mention the architecture.

---

## Related labels

`is_power_z_dev` marks work as *platform enablement*. It is separate from any
`platform: …` labels, which indicate where something applies. A ticket may
carry both; neither implies the other.

---

## Questions

If a case is genuinely ambiguous, apply the label and raise it — borderline
examples are useful, and they're how this guidance gets sharpened.
