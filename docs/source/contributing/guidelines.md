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

### 1. Fork and clone

Fork the repository on GitHub, then clone your fork locally:

```bash
git clone https://github.com/<your-username>/torch-spyre.git
cd torch-spyre
git remote add upstream https://github.com/torch-spyre/torch-spyre.git
```

### 2. Keep your fork in sync

Before starting new work, sync your local `main` with upstream:

```bash
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

### 3. Create a branch

Always branch off `main`:

```bash
git checkout -b my-feature-branch
```

### 4. Push to your fork and open a PR

Push your branch to your fork (not upstream):

```bash
git push origin my-feature-branch
```

Then open a PR on GitHub from `<your-username>/torch-spyre:my-feature-branch`
targeting `torch-spyre/torch-spyre:main`.

### 5. After your PR is merged

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
* **Use `import regex as re`, never `import re`.** The `enforce-import-regex-instead-of-re` pre-commit hook enforces this. Using the standard-library `re` module will fail pre-commit.
* **Line length** is 88 characters, enforced by ruff.
* **Run pre-commit** before submitting to make sure linting passes:

  ```bash
  pip install pre-commit
  pre-commit run --all-files
  ```

  See the [pre-commit docs](https://pre-commit.com/#usage) if this is new to you. The configured hooks include `ruff`, `clang-format`, `cpplint`, `mypy`, `pymarkdown`, `yamlfmt`, and `actionlint`, plus local hooks for the `import regex` rule, DCO sign-off, filename checks, and pinned `requirements/dev.txt` validation.

* **Write tests**, both unit and integration, to keep the project correct and robust. The test suite is organized into four tiers, all of which run in CI on every PR:

  | Tier | Where | What it covers |
  |---|---|---|
  | Upstream PyTorch compatibility | OpInfo-based tests instantiated against the `spyre` device via `instantiate_device_type_tests` (see `tests/test_spyre.py` and `tests/spyre_test_base_common.py`). | Confirms Spyre tensors behave correctly at the PyTorch API level. An allowlist tracks which test variants pass; update it with each PR. |
  | Op-level | `tests/inductor/test_inductor_ops.py` | Each op compared against a CPU reference output. |
  | Building blocks | `tests/inductor/test_building_blocks.py` | Composed transformer subgraphs such as attention heads, FFN, and normalization. |
  | Model-level | `tests/models/test_model_ops.py`, `tests/models/test_model_ops_v2.py` | Full model forward passes with real Granite weights, validated against YAML-specified tolerance profiles. |

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
python -m sphinx docs/source docs/build/html -W --keep-going
python -m http.server 8080 --directory docs/build/html
```

Then open `http://localhost:8080` in your browser.

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
