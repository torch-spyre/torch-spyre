# SDSC summary tooling

Reads the `sdsc_*.json` artifacts a Spyre compile writes and shows, per kernel and per op, what each tensor is, where it lives (LX or HBM), how much LX is in use, what the cost model predicted, and what the CP-SAT solver's objective said about each placement.

## Install

Nothing needs building: the scripts run from the checkout, with the repo's `.venv` if it has `tabulate` (and `textual` for the browser).
Set `SDSC_PYTHON` to use a different interpreter.

Put the launcher on your `PATH` (run from the repo root):

```bash
mkdir -p ~/.local/bin
ln -s "$PWD/tools/cost_model/summarize-sdsc/sdsc" ~/.local/bin/sdsc
```

The launcher follows the link back to the checkout, so it still finds its scripts and the repo's `.venv`.

To let Claude Code use the tool, link this directory in as a skill:

```bash
# for every project; compiles are often run from another checkout
mkdir -p ~/.claude/skills
ln -s "$PWD/tools/cost_model/summarize-sdsc" ~/.claude/skills/summarize-sdsc

# or for this checkout only
ln -s "$PWD/tools/cost_model/summarize-sdsc" .claude/skills/summarize-sdsc
```

The per-checkout link shows up as untracked in `git status`; do not commit it.
Start a new Claude Code session if the skill does not appear.

## Quick start

```bash
# 1. compile with artifacts + cost dumps collected in one place (~/sdsc_runs/<timestamp>)
sdsc capture -- python3 my_model.py

# 2. look at the newest run
sdsc tui                                     # interactive browser
sdsc report                                  # the box table, on stdout
sdsc report DIR --format github --out pr.md  # collapsible Markdown for a GitHub comment
sdsc report DIR --out summary.txt            # the box table, for a Slack file upload
sdsc shot                                    # <run>/summary.svg, a screenshot of the browser

# 3. compare two runs: what did the planner change?
sdsc diff ~/sdsc_runs/before ~/sdsc_runs/after
```

`capture` sets `TORCHINDUCTOR_CACHE_DIR`, `SPYRE_DUMP_COST=1`, `SPYRE_DUMP_COST_FILE` and `SPYRE_DUMP_COST_EXPR_FILE` for the command; everything else in your environment (`LAYOUT_SOLVER`, `CO_OPTIMIZING_LX_PLANNING`, ...) passes through. The other subcommands find the dumps by name (`cost_dump.log`, `cost_expr.jsonl`, `probe.json`) next to the artifacts. Any `/tmp/torchinductor_*` directory works as DIR too, without the cost columns unless you pass `--cost FILE` / `--cost-expr FILE`.

The objective dump (`SPYRE_DUMP_COST_EXPR_FILE`) needs a torch-spyre with the dump hook (PR #4595 or later). Without it the report still has everything except the `Objective` column.

The objective dump also records each buffer's residency *reason*: why the allocator kept it out of LX (PR #4738 or later).
Runs captured before that can still get reasons from an older residency probe JSON, passed with `--probe FILE` or dropped next to the artifacts as `probe.json`.

## Comparing two runs (`sdsc diff`)

```bash
sdsc diff ~/sdsc_runs/before ~/sdsc_runs/after
```

Prints the whole-run totals side by side (ops, LX tensors, peak LX live, predicted time, relayouts fired, outputs in LX), then one block per op whose plan differs: placement, cores, LX live, attributed cost, residency reason, the op's own objective terms and its core division.
An op planned identically is silence.

Kernels are paired by their op-name sequence, because the kernel directory is named by a content hash that changes with the compiler.
Ops within a kernel pair by position, or by a longest-common-subsequence on the op names when the counts differ, so an inserted relayout shuffle shifts nothing after it.
Only the kinds of fact **both** runs recorded are compared: diffing a run whose dump records residency reasons against an older one without them would otherwise report every op as changed, so the dropped kinds are named under the table instead.
The text tables fit the terminal: the widest cells wrap until the table fits `$COLUMNS` or the terminal's width.
Output to a pipe or `--out` keeps one row per line; `--width COLS` forces a width and `--width 0` turns wrapping off.

## The browser (`sdsc tui`)

Three panels, top to bottom: kernels and their ops, the selected op's tensors, and a detail pane with every field of the selected row.

| Key | Action |
|---|---|
| Enter on an op | show its tensor rows |
| `d` | toggle detail columns in the table |
| `/` then text, Enter / Esc | filter ops by name / clear the filter |
| `n` `p` | next / previous op |
| `g` `b` | jump to the selected tensor's consumer / producer |
| Tab | move focus between panels (the detail pane scrolls with arrows, PageUp, PageDown) |
| `]` `[` | grow / shrink the detail pane |
| `z` | zoom the focused panel to the whole screen, again to restore |
| `s` | save an SVG screenshot |
| `q` | quit |

## What the columns mean

- **What (-> consumer)**: the op's kind (index gather, LX relayout with its kind and slice counts, restickify, copy, matmul, compute), its input/output placement, and pointers to the op that reads its LX output (`->`) and the one that produced its LX input (`<-`).
- **Layout\* extent/wkSlices**: host extents in device layout order, `*` on the stick dimension, `/N` for a dimension split N ways across cores, `1` for a reduced or broadcast dimension.
- **Tile Shape / Tile Size**: the per-core tile in the allocation's own dimension order and its bytes.
- **LX footprint (stick padded)**: what LX actually holds for the tile.
- **Address (LX range)**: `start-end` for an LX tensor, `hbm#N` for an HBM placeholder.
- **coreIdToWkSlice**: which slice each core owns, compressed to per-dimension expressions.
- **Format**: SDSC data format and element width.
- **LX live / capacity**: LX bytes live while the op runs and their share of the per-core budget (1,625,344 B, the planner's; `--lx-capacity` to change).
- **Cost (SPYRE_DUMP_COST)**: the numeric cost model on the final plan. `85.8 us of 1311.4 us` is this op's HBM-byte share of the kernel's one prediction, an attribution, plus its read/write/LX bytes. `as <name>` when the dump named the op differently.
- **Objective term (solver, this plan)**: the CP-SAT objective as solved. `2057.4 us (75-op bundle)` is the bundle term the op belongs to; `if HBM: +40.0 us` is the same term with this op's residency flipped and every other decision held fixed. In the browser's detail pane the op's own terms are listed with other buffers substituted by their solved values, largest sensitivity first. A shuffle shows its relayout copy's charge and what the unfired alternatives would have cost.
- **Residency**: the allocator's own reason for a buffer that is not in LX, read from the objective dump (or the older probe JSON). The short form sits under the address on the op's OUTPUT row; the detail pane has the whole phrase. `spilled by solver (no residency benefit / no room)` means the solver weighed it and declined, so the objective is where to look; a gate reason (`op not allowed`, `partial/offset read`, `index tensor or indirectly accessed`) means it never reached the solver at all.
- **Core division (solver, this plan)**: the division the op's output buffer got, the candidates it was chosen from, and per producer edge the `(parent, consumer)` division pairs the residency gate admitted.
  An admitted pair the plan did not take is an LX residency the chosen division gave up.
  An edge with no admitted pair is one the gate forbids however either side divides.
  This is the part of the decision the objective's numbers cannot show: the objective prices the plan that was chosen, not the alternatives it beat.

The flip is not a feasible alternative plan: moving a buffer may force other divisions to change. Pricing the real alternative means re-solving with the buffer pinned, which this tool does not do.

## Files

- `sdsc`: the launcher.
- `batch_summarize_sdsc.py`: parsing (`build_report`) and the text / Markdown / HTML renderers; `--help` lists every option.
- `sdsc_tui.py`: the Textual browser; needs `textual` in the Python that runs it (`uv pip install --python $(which python3) textual`).
- `SKILL.md`: the instructions Claude Code follows when asked to summarize SDSC artifacts (see [Install](#install)).
