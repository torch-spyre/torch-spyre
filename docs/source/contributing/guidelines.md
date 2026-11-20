# Contribution Guidelines

Thank you for your interest in contributing to Torch-Spyre! There are many
ways to contribute — bug reports, documentation improvements, new operation
support, and more.

For the full contribution guidelines, see the
[CONTRIBUTING.md](https://github.com/torch-spyre/torch-spyre/blob/main/CONTRIBUTING.md)
file in the repository root.

## Development Workflow

Torch-Spyre uses a **fork-based PR model**. All contributions come through a
personal fork rather than pushing branches directly to the upstream repository.

### Step 1: Fork and clone

Fork the repository on GitHub, then clone your fork locally:

```bash
git clone https://github.com/<your-username>/torch-spyre.git
cd torch-spyre
git remote add upstream https://github.com/torch-spyre/torch-spyre.git
```

### Step 2: Keep your fork in sync

Before starting new work, sync your local `main` with upstream:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### Step 3: Create a branch

Always branch off `main`:

```bash
git checkout -b my-feature-branch
```

### Step 4: Push to your fork and open a PR

Push your branch to your fork (not upstream):

```bash
git push origin my-feature-branch
```

Then open a PR on GitHub from `<your-username>/torch-spyre:my-feature-branch`
targeting `torch-spyre/torch-spyre:main`.

### Step 5: After your PR is merged

```bash
git checkout main
git fetch upstream
git merge upstream/main
git push origin main
git branch -d my-feature-branch
```

---

## Before You Start

* **Open an issue first** for large PRs so the team can align on the
  approach before you invest significant effort.
* **Sign off your commits** with `git commit -s` (Developer Certificate
  of Origin).

## Code Quality Standards

* Follow the **Google Python Style Guide** and **Google C++ Style Guide**.
* **Use `import regex` (aliased as `re` when needed), never `import re`.** The `enforce-import-regex-instead-of-re` pre-commit hook enforces this. Importing the standard-library `re` module will fail pre-commit.
* **Line length** is 88 characters, enforced by ruff.
* **Run pre-commit** before submitting to make sure linting passes:

  ```bash
  pip install pre-commit
  pre-commit run --all-files
  ```

  See the [pre-commit docs](https://pre-commit.com/#usage) if this is new to you. The configured hooks include `ruff-check`, `ruff-format`, `clang-format`, `cpplint`, `mypy`, `pymarkdown`, `yamlfmt`, and `actionlint`, plus local hooks for `forbid-pytest-ini` (pytest is configured in `[tool.pytest.ini_options]` in `pyproject.toml`), the `import regex` rule, DCO sign-off, filename checks, `requirements/dev.txt` synchronization with `pyproject.toml`, and `check-agent-context`, which validates `AGENTS.md`, `CLAUDE.md`, and the skill files under `.claude/skills/`.

* **Write tests**, both unit and integration, to keep the project correct and robust. The test suite is organized into four tiers. The first three run in CI on every pull request; the model-level configurations run on every push to `main`:

  | Tier | Where | What it covers |
  |---|---|---|
  | Upstream PyTorch compatibility | OpInfo-based tests instantiated against the `spyre` device via `instantiate_device_type_tests` (see `tests/oot_framework/`). `tests/test_spyre.py` covers Spyre tensor behavior (fill, copy, dtype, allocation). | Confirms Spyre tensors behave correctly at the PyTorch API level. An allowlist tracks which test variants pass; update it with each PR. |
  | Op-level | `tests/inductor/test_inductor_ops.py` | Each op compared against a CPU reference output. |
  | Building blocks | `tests/inductor/test_building_blocks.py` | Composed transformer subgraphs such as attention heads, FFN, and normalization. |
  | Model-level | `tests/models/test_model_ops.py`, `tests/models/test_model_ops_v2.py`, configured by `tests/configs/model_ops_tests/` | Operator cases taken from model configurations (Granite, Llama, Mistral, Qwen, Gemma, gpt-oss), each validated against the tolerances in its YAML configuration. |

* **Run the tests** with `python3 -m pytest tests/`. `pyproject.toml` sets `norecursedirs = ['tests/models']`, so the model-level tests are not collected by that command. `make tests` runs the configured shards, and `TEST_TYPE` (`smoke`, `unit`, `integration`, `regression`, `trunk`, `perf`, or `suite_<group>`) narrows the scope.

* **Follow the test invariants** in [`AGENTS.md`](https://github.com/torch-spyre/torch-spyre/blob/main/AGENTS.md#test-invariants). In short: an expected failure is declared as a strict xfail (`@pytest.mark.xfail(strict=True, raises=...)`, `strict_xfail(request, raises, reason)` from `tests/inductor/utils_inductor.py`, or the `expect_fail` key), never with `pytest.xfail(...)` in the test body, and gives `raises=<exception type>` wherever the marker is written; a deliberate rejection or a missing feature with a stable error is an assertion (`expect_raise`, or `pytest.raises(match=...)` with `@expects_raise`); and every xfail or skip for a bug cites an open issue. Every test must be selected by a shard configuration in `tests/configs/`, so run `make check-all-configs` after adding or renaming a test.

* **Run the Spyre integration suites on a pull request** by commenting `/spyre-test` on it. Any commenter can trigger the run, including on a pull request from a fork. It runs the hf-adapters and spyre-inference integration suites against the head of the pull request when downstream integration is enabled for the repository (`.github/workflows/downstream_integration.yaml`).

* **Document user-facing changes.** If your PR modifies how Torch-Spyre
  behaves from a user's perspective, add or update the relevant page under
  `docs/source/`. See the section structure in this documentation site for
  guidance on where to place new content.

* **Dev environment setup** — install development dependencies via:

  ```bash
  pip install -r requirements/dev.txt
  ```

## Building the Docs Locally

If your PR touches documentation, build and preview it locally before submitting:

```bash
pip install -r docs/requirements.txt
cd docs
make html SPHINXOPTS="-W --keep-going"
python -m http.server 8000 --directory build/html
```

Then open `http://localhost:8000` in your browser. For a live-reload
preview that rebuilds on save, run `make livehtml` instead (requires
`sphinx-autobuild`, already in `docs/requirements.txt`).

> **Note:** Do not open the HTML files directly from the filesystem (`file://`).
> Browsers block CSS and JavaScript when loading local files, resulting in an
> unstyled plain-text page. Always use the HTTP server above.

The `-W` flag turns Sphinx warnings into errors — the same check CI runs.
Fix any warnings before opening your PR.

## How to Extend the Compiler

The most common contribution is adding support for a new PyTorch operation.
See the [Spyre Inductor Operation Cookbook](../compiler/adding_operations.md)
for step-by-step patterns.

## Reporting Issues

Please open issues at
[github.com/torch-spyre/torch-spyre/issues](https://github.com/torch-spyre/torch-spyre/issues).
