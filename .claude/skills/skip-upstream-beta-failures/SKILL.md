---
name: skip-upstream-beta-failures
description: "Triage a failing upstream-beta CI run and skip exactly the newly-failing tests in tests/configs/upstream_tests_beta/. Use when the user gives a GitHub Actions log download (logs_<run_id>) and/or a JUnit XML artifact download (upstream-beta-reports-<sha>) and asks to check/skip the failing tests."
---

# Skip Upstream-Beta Failures

`upstream_tests_beta` runs real upstream PyTorch test files against the
Spyre backend (see `tests/configs/upstream_tests_beta/*.yaml`). A PyTorch
version bump, or a change on our side, occasionally turns a few of those
tests from passing/xfail into unexpected failures. This skill is the
repeatable procedure for triaging one of those runs: find exactly which
tests regressed, understand *why*, and move only those tests to
`mode: skip` in the right config file — nothing more, nothing less.

## Inputs the user gives you

The user typically hands you one or two local download directories (from
a GitHub Actions run's log/artifact download):

1. **`logs_<run_id>/`** — the raw step logs of the `build_and_test` /
   "Run upstream-beta test matrix and report results" workflow run. This
   is a **dispatcher** job: it kicks off a separate matrix-of-shards
   workflow run via the GitHub API, waits for it, downloads each shard's
   JUnit XML, and prints one authoritative summary line:
   ```
   Test totals: {'total': N, 'passed': P, 'failed': F, 'errors': E, 'skipped': S, 'xfailed': 0, 'xpassed': 0}
   ```
   Grep for this (or `TEST_RESULTS_JSON:`) to get the real `failed`/`errors`
   counts. **Do not** try to grep raw `FAILED` lines out of this file for
   per-test names — this job does not contain the shard output inline, only
   the aggregated counts. If asked to find per-test detail here and it
   isn't present, tell the user and ask for the XML artifact bundle below
   instead of guessing from upload/download log noise.

2. **`upstream-beta-reports-<pytorch_sha>/`** — the actual downloaded
   artifact: one `<suite>.xml` JUnit file per test suite (e.g.
   `test_linalg.xml`, `nn-test_embedding.xml`, `inductor-test_config.xml`).
   **This is the authoritative source** for both counts and per-test
   failure detail. Always prefer this over scraping text logs.

   Sometimes the user gives you an older-style flat directory of `.txt`
   step logs (`N_<Suite Name>.txt`) from an earlier, non-dispatcher
   workflow shape instead. Same idea applies: look for each file's own
   `pytest` summary line (`X failed, Y passed, Z deselected in Ns`) and
   `FAILED [<time>] <path>::<Class>::<test>` lines — not the shell script's
   own `echo "=== Attempt N FAILED ..."` lines, which are retry-wrapper
   noise, not test results.

If the user gives only the `logs_<run_id>/` dispatcher bundle and it has
no inline per-test detail, ask them for the matching
`upstream-beta-reports-<sha>` artifact (same run, same SHA in the name)
before proceeding — do not fabricate or guess failing test names.

## Step 1 — Get the authoritative failure list

For an XML bundle, parse every file's `<testsuite>` for `failures=`/
`errors=` (never rely on `<skipped>` or passing `<testcase>` entries), then
pull the failing `<testcase>` detail:

```python
import xml.etree.ElementTree as ET, glob

total = 0
for f in sorted(glob.glob('*.xml')):
    root = ET.parse(f).getroot()
    for suite in root.iter('testsuite'):
        failures, errors = int(suite.get('failures', 0)), int(suite.get('errors', 0))
        if failures or errors:
            print(f, 'failures=', failures, 'errors=', errors)
            total += failures + errors
print('TOTAL', total)
```

Then, for each file with failures, print `classname`, `name`, and the
`<failure>`/`<error>` text (the `message` attribute for a one-line summary,
the element's full text for the traceback):

```python
for tc in root.iter('testcase'):
    for fail in tc.findall('failure') + tc.findall('error'):
        print(tc.get('classname'), '::', tc.get('name'))
        print(fail.text)  # full traceback + assertion/error message
```

Cross-check the XML-derived total against the dispatcher log's
`Test totals:` line (or the `.txt` shard's own summary line) if both are
available — they must agree. If the user previously asked "are you sure"
about a count, this cross-check is exactly why: always derive the number
from the structured summary (`failures=`/`errors=` attributes, or the
pytest summary line), never from a raw `grep -c FAILED` on a text log —
that double- or triple-counts (once in live run output, once in the
"short test summary info" section, plus any unrelated shell-script
`FAILED` echoes).

**If you are checking multiple log bundles from the same or overlapping
runs**, de-duplicate by `(classname, name)` before reporting a total or
editing configs — a regression that shows up in two bundles is one
regression, not two.

## Step 2 — For each failing test, find where it's configured

Each upstream test file has exactly one YAML under
`tests/configs/upstream_tests_beta/` (filename pattern: `test_<x>.yaml` for
`test/test_<x>.py`, `<dir>-test_<x>.yaml` for `test/<dir>/test_<x>.py`,
e.g. `nn-test_embedding.yaml` for `test/nn/test_embedding.py`,
`inductor-test_config.yaml` for `test/inductor/test_config.py`). Find the
test's current entry:

```bash
grep -n "<method_name>" tests/configs/upstream_tests_beta/<file>.yaml
```

The JUnit `name` is the OOT-wrapped name (e.g.
`test_offset_scalar_cast_spyre_float32`): strip the trailing `_spyre` and
any dtype suffix to get the config's bare method name
(`test_offset_scalar_cast`). The JUnit `classname` has `SPYRE` appended to
the upstream class (`TestTensorCreationSPYRE` → `TestTensorCreation`); the
config's `names:` entries are always `<UpstreamClassName>::<method_name>`.

Read enough surrounding context (the whole `tests:` list for that file, or
at least neighboring entries) to see:
- which `mode:` block it is currently in (`mandatory_success`, `xfail`,
  `xfail_strict`),
- whether it already has a `edits.dtypes.exclude` list you need to
  preserve or extend,
- whether *all* dtype variants of a parametrized test failed, or only
  some (check every testcase with that base name in the XML — do not
  assume one dtype's failure means every dtype failed).

## Step 3 — Classify the failure and write the skip

This repo's convention (confirmed with the user) is to move every newly
regressed test straight to `mode: skip`, regardless of failure flavor —
do not invent a different mode. The categories below are for writing a
useful comment, not for picking a different `mode:`.

Common shapes you'll see, each wants a comment that says *why*, concretely:

- **Unimplemented op**: `NotImplementedError: Could not run 'aten::X' with
  arguments from the 'spyre' backend`. Name the op.
- **Error-message mismatch**: the upstream test does
  `assertRaisesRegex(SomeError, "<pattern>")`, expecting a specific CPU
  error string, but Spyre raises a different (often backend-specific)
  error first — e.g. its own `NotImplementedError` before PyTorch's
  standard argument validation runs. Quote both the expected pattern and
  the actual message.
- **Wrong numeric result**: `AssertionError: Tensor-likes are not close!`
  / `Scalars are not close!` with a high mismatch percentage. This is a
  real correctness bug, not a missing feature — say so, and say what
  specifically was compared (e.g. "every element of each split chunk
  mismatches the numpy reference").
- **Unsupported dtype**: `RuntimeError: Spyre backend does not support
  dtype X`.

Before writing the comment, **search for an existing open issue** — this
repo cites issue numbers in skip reasons when one exists (see existing
`(#1588)`, `(#5817)` examples in `test_torch.yaml`):

```bash
gh issue list --repo torch-spyre/torch-spyre --search "<key phrase from the error>" --state all
```

If you find an open issue whose description matches, cite it:
`mode: skip  # skip: <short reason> (#NNNN)`. If you only find a *closed*
issue, do not cite it as if it's still open — say in your summary to the
user that the regression may have reappeared after that issue was closed,
and let them decide whether to reopen it. If you find nothing, use a
reason like `# skip: NotImplementedError on upstream PyTorch bump` (no
issue number) and mention to the user that no tracking issue exists yet.

### Editing the YAML

- Prefer adding a **new** `- names: [...] \n mode: skip` block near the
  test's old location over editing in place, with a comment explaining the
  regression — this preserves the history/comment of the block the test
  came out of for the tests that stay.
- If the failing test shares a `mode: mandatory_success` block with tests
  that are *not* failing, remove only the failing name(s) from that block
  and give the new skip block its own comment — don't skip tests that are
  still passing.
- If the test is dtype-parametrized and only some dtypes failed, keep
  `mandatory_success`/`xfail` for the surviving dtypes (narrow via
  `edits.dtypes.exclude`, matching how the rest of the file already
  excludes unsupported dtypes) rather than skipping the whole test name.
- If a test takes an `edits:` block (e.g. `dtypes.exclude`) in its
  original entry, carry that same `edits:` block onto the new skip entry
  only if it still applies (e.g. the dtype exclusion reason is unrelated
  to the new regression); drop it if irrelevant.
- **After editing, re-read the whole `tests:` list once more** and check
  for an accidental leftover duplicate block (an earlier edit pass in this
  skill's history once left a stale duplicate `test_vsplit` block behind
  after a cut/paste — always diff your own edit against the full file,
  not just the hunk you just wrote).

## Step 4 — Validate

Both of these must pass before you consider the edit done. Loading
`torch_spyre`'s native extension isn't needed for config validation, and
often isn't built in a dev checkout, so disable autoload:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -c "
import yaml, sys
sys.path.insert(0, 'tests')
from oot_framework.oot_test_config_models import OOTTestConfig
f = 'tests/configs/upstream_tests_beta/<file>.yaml'
cfg = OOTTestConfig(**yaml.safe_load(open(f)))
print(f, 'VALID')
"
```

If the sandboxed `python3` lacks `pyyaml`/`torch` (e.g. only Homebrew
Python is on `PATH`), find the interpreter that has them (the project's
venv, or `miniforge3`) and use that one explicitly rather than installing
packages into a throwaway interpreter.

```bash
pre-commit run --files tests/configs/upstream_tests_beta/<file>.yaml
```
Confirm "Format YAML files" passes (yamlfmt). Fix anything it reformats.

## Step 5 — Git workflow: commit and open a PR

1. **Check the current branch before committing anything.** A branch left
   over from a previous session's fix may already be merged — check:
   ```bash
   gh pr list --repo torch-spyre/torch-spyre --head <branch> --state all
   ```
   If it shows `MERGED`, do **not** keep committing onto it. Fetch
   `upstream/main`, diff your intended change against the *working tree*
   (not against the stale branch) to capture just your new edit as a
   patch, create a **fresh** branch off `upstream/main`, and re-apply only
   that patch there.
2. **Never run a destructive git command without checking `git status`
   first**, and never discard uncommitted work. This repo's working tree
   routinely has unrelated pre-existing modified/untracked scratch files
   (other in-progress work, CI workflow edits, experiment output) that are
   not part of this task — leave them alone. If you need to switch
   branches cleanly (e.g. moving off a stale branch per the point above):
   - `git stash push -u -m "<description>"` to park everything (tracked +
     untracked) before switching/rebasing,
   - do your branch/rebase work,
   - restore deliberately: `git stash pop` if nothing conflicts, or if you
     already hand-applied just your own diff via a saved patch, restore
     only the *untracked* files from the stash's untracked-files parent
     (`git log --parents` on the stash shows three parents; the third is
     the untracked tree) via `git checkout stash@{0}^3 -- .` followed by
     `git reset HEAD -- .` so they land back as untracked rather than
     staged, then `git stash drop`.
   - Only ever stage and commit the specific test-config file(s) you
     edited (`git add <file>` by name, never `-A`/`.`); confirm with
     `git status` that nothing else crept into the commit.
3. Commit with sign-off (DCO) and the required attribution trailer:
   ```bash
   git commit -s -m "$(cat <<'EOF'
   test(upstream-beta): skip <short description>

   <why: run/SHA, counts, what the test expected vs what Spyre does,
   issue number if one exists>

   Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
   EOF
   )"
   ```
4. Push to the user's fork (`origin`) and open the PR against
   `torch-spyre/torch-spyre:main`:
   ```bash
   git push -u origin <branch>
   gh pr create --repo torch-spyre/torch-spyre --base main \
     --head <github-user>:<branch> \
     --title "test(upstream-beta): skip <short description>" \
     --body "<Summary with run link/SHA/counts + Test plan checklist, ending with>

   🤖 Generated with [Claude Code](https://claude.com/claude-code)"
   ```
   The PR body should include: which run/SHA this was observed on, the
   authoritative failure count for that suite, what each failing test
   expected vs. what Spyre actually does, and the issue number if cited.

## Things that have gone wrong before (don't repeat them)

- Counting failures by `grep -c FAILED` on a text log — inflates the
  count by counting the same test's live-run line and its "short test
  summary info" line separately, plus unrelated shell-retry-wrapper
  `FAILED` echoes. Always use the structured `failures=`/`errors=`
  attributes (XML) or the pytest summary line (`X failed, Y passed`).
- Treating a dispatcher job's log (`logs_<run_id>/...Run upstream-beta
  test matrix and report results.txt`) as if it contains per-test detail —
  it only has the aggregate JSON; the real per-test failures are in the
  downloaded XML artifact.
- Leaving a duplicate block behind in the YAML after a cut/paste edit —
  always re-read the full file after editing, not just the diff hunk.
- Continuing to commit onto a branch whose earlier PR already merged —
  check `gh pr list --head <branch> --state all` before adding commits.
- Scooping up unrelated pre-existing working-tree changes into the commit
  (`git add -A`) — stage only the specific config file(s) you edited.
