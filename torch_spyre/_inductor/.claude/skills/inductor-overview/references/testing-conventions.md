# Testing conventions for `_inductor` work

## Never run Spyre tests in parallel

Loading `torch_spyre` gives the process **exclusive access to the Spyre
device** until it exits. A second process that also loads `torch_spyre`
cannot acquire the device and will fail or hang.

Always run pytest as a single sequential process. Never use `-n`,
`-n auto`, `pytest-xdist`, or any other parallel/distributed test runner
for any suite in this repo — including in CI-adjacent local scripts.

## Local regression scope before pushing

The full `tests/` suite is slow to run locally and CI already covers it in
parallel infrastructure that isn't subject to the single-device
constraint above. For a local "did I break anything nearby" pass on
`_inductor` changes, you don't need the full suite — run:

- `tests/inductor/test_building_blocks.py` — always. It exercises the core
  op/lowering paths broadly enough to catch most collateral damage
  regardless of which pass you touched.
- The test file(s) that correspond to the specific pass/module you
  modified, if one exists — e.g. touching `wsr/coarse_tile.py` warrants
  `tests/inductor/test_coarse_tile_e2e.py`, `test_coarse_tiling.py`; a
  scratchpad allocator change warrants whatever exercises
  `torch_spyre/_inductor/scratchpad/`, etc. Match by name/import, not by
  habit — don't run a fixed list left over from someone else's feature
  work.
- `pre-commit run --all-files`

Don't run the entire `tests/` tree as a pre-push check — it's redundant
with what CI already runs on push, and the wall-clock cost isn't worth it
for local iteration.
