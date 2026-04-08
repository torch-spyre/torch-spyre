# Torch-Spyre Documentation — Contributor Guide

This guide is for anyone who wants to add, edit, or review the
Torch-Spyre documentation. It covers the site layout, local build
setup, writing conventions, and the pull-request workflow.

---

## Site Layout

```
docs/
├── README.md               ← this file
├── Makefile                ← build targets: html, livehtml, clean
├── requirements.txt        ← Python dependencies for building the docs
└── source/
    ├── conf.py             ← Sphinx configuration
    ├── index.rst           ← root table of contents
    ├── _static/
    │   ├── css/custom.css  ← theme overrides
    │   └── images/         ← all figures referenced by docs pages
    ├── architecture/       ← Spyre accelerator and dataflow architecture
    ├── getting_started/    ← Installation and Quickstart
    ├── user_guide/         ← Tensor layouts, running models, profiling, debugging
    ├── compiler/           ← Compiler architecture, passes, adding ops
    ├── runtime/            ← Runtime layer and device registration
    ├── rfcs/               ← RFC index and summaries
    ├── api/                ← Auto-generated API reference
    └── contributing/       ← Contribution guidelines, CI/CD
```

Each section has an `index.rst` that controls its sidebar entries.
Content pages are written in **Markdown** (`.md`) using
[MyST-Parser](https://myst-parser.readthedocs.io/), which supports
standard CommonMark plus RST-style cross-references and directives.

---

## Prerequisites

- Python 3.11 or later
- `pip`

---

## Local Setup

### 1. Install dependencies

From the repo root:

```bash
pip install -r docs/requirements.txt
```

This installs:

| Package | Purpose |
|---------|---------|
| `sphinx>=7.0` | Core documentation engine |
| `sphinx-rtd-theme>=2.0` | Read the Docs theme |
| `myst-parser>=3.0` | Markdown (`.md`) support |
| `sphinx-autobuild>=2024.0` | Live-reload dev server (optional) |

### 2. Build the HTML site

```bash
cd docs
make html
```

Output is written to `docs/build/html/`. Open `docs/build/html/index.html`
in your browser to preview.

### 3. Live-reload server (recommended for editing)

```bash
cd docs
make livehtml
```

This starts a local server at `http://127.0.0.1:8000` that automatically
rebuilds and refreshes the browser whenever you save a file.

### 4. Clean the build

```bash
cd docs
make clean
```

---

## Writing Documentation

### File format

- Use **Markdown** (`.md`) for all content pages.
- Use **reStructuredText** (`.rst`) only for `index.rst` toctree files
  and the API autodoc stub (`api/torch_spyre.rst`).

### Adding a new page

1. Create a `.md` file in the appropriate section directory,
   e.g. `docs/source/compiler/my_new_topic.md`.
2. Add the filename (without extension) to the section's `index.rst`
   toctree, e.g. add `my_new_topic` below the existing entries.
3. Rebuild to verify it appears in the sidebar.

### Adding a new section

1. Create a new directory under `docs/source/`, e.g. `docs/source/internals/`.
2. Add an `index.rst` in that directory with a title and toctree.
3. Add the section to the root `docs/source/index.rst` by adding
   `internals/index` to the main toctree.

### Cross-references

Link to another page using a Markdown link with a relative path:

```markdown
See [Compiler Architecture](../compiler/architecture.md) for details.
```

Link to a specific section heading by its anchor:

```markdown
See [Memory Hierarchy](../architecture/dataflow_architecture.md#memory-hierarchy).
```

The rendered version of this page is available at
[Dataflow Architecture on Read the Docs](https://torch-spyre.readthedocs.io/en/latest/architecture/dataflow_architecture.html).

### Figures

Place image files in `docs/source/_static/images/` and reference them
with a standard Markdown image link:

```markdown
![Description of the diagram](../_static/images/my-diagram.png)
```

For richer formatting (captions, sizing, alignment), use the MyST
`figure` directive:

```
:::{figure} ../_static/images/my-diagram.png
:alt: Description of the diagram
:width: 80%
:align: center

Caption text with attribution. *Source: ...*
:::
```

### Code blocks

Fenced code blocks with a language identifier are syntax-highlighted:

```python
import torch
model = torch.compile(model, backend="spyre")
```

---

## Conventions

- **Headings**: use `#` for page title, `##` for sections, `###` for
  subsections. Do not start a page at `##`.
- **Tone**: prefer active voice and concise sentences.
- **Links**: use GitHub URLs for source-code references, not relative
  paths from the repo root.
- **Figures**: always include an `:alt:` attribute and a caption with
  attribution.
- **TODOs**: mark incomplete stubs with a `## TODO` section listing
  what needs to be written. This makes gaps visible in the built site.

---

## Pull Request Workflow

### 1. Create a branch

```bash
git checkout main
git pull
git checkout -b docs/my-topic
```

Branch naming convention: `docs/<short-description>`.

### 2. Make your changes

Edit or create `.md` files under `docs/source/`. Run `make html` or
`make livehtml` to check the result locally.

### 3. Verify a clean build

```bash
cd docs && make html
```

The build must complete with **zero warnings** before submitting.
Check the last line of output:

```
build succeeded.
```

If warnings appear, fix them before opening a PR. Common causes:

| Warning | Fix |
|---------|-----|
| `undefined label` | Check the cross-reference path or anchor name |
| `document isn't included in any toctree` | Add the file to its section `index.rst` |
| `image file not readable` | Verify the image path under `_static/images/` |
| `Title underline too short` | Fix the RST heading underline length in `.rst` files |

### 4. Sign off your commit

Torch-Spyre uses the [Developer Certificate of Origin](https://developercertificate.org/).
All commits must be signed off:

```bash
git commit -s -m "docs: add section on tensor DMA encoding"
```

### 5. Open a pull request

Push your branch to your fork and open a PR against `main`:

```bash
git push -u fork docs/my-topic
```

Then open a PR on GitHub. In the PR description:

- Summarise what changed and why
- Link any related issues
- Include a screenshot of the rendered page if you added a new figure
  or significantly restructured a section

### 6. Review and merge

A maintainer will review the PR. Address any feedback, push additional
commits to the same branch, and the PR will be merged once approved.

---

## Useful References

- [MyST-Parser syntax guide](https://myst-parser.readthedocs.io/en/latest/syntax/syntax.html)
- [Sphinx directives reference](https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html)
- [sphinx-rtd-theme options](https://sphinx-rtd-theme.readthedocs.io/en/stable/configuring.html)
- [Torch-Spyre CONTRIBUTING.md](https://github.com/torch-spyre/torch-spyre/blob/main/CONTRIBUTING.md)
