# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
project = "Torch-Spyre"
copyright = "2025, Torch-Spyre Core Team"
author = "Torch-Spyre Core Team"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # auto-generate API docs from docstrings
    "sphinx.ext.autosummary",  # generate summary tables for modules
    "sphinx.ext.napoleon",  # Google/NumPy docstring support
    "sphinx.ext.intersphinx",  # cross-reference external docs (PyTorch, Python)
    "sphinx.ext.viewcode",  # add [source] links to API pages
    "sphinx.ext.todo",  # support .. todo:: directives
    "myst_parser",  # parse Markdown (.md) files
]

# MyST-Parser settings — allow RST-style cross-refs inside .md files
myst_enable_extensions = [
    "colon_fence",  # ::: fenced directives
    "deflist",  # definition lists
]

# Paths that contain templates
templates_path = ["_templates"]

# Patterns to exclude when looking for source files
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The suffix(es) of source filenames
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The master toctree document
master_doc = "index"

# -- Intersphinx mapping -----------------------------------------------------
# The PyTorch docs site moved its inventory and the canonical URL now 404s
# under -W. None of our pages use :py:class:`torch.X` style cross-references
# (links to pytorch.org are written as plain Markdown links instead), so we
# drop the torch entry until the upstream URL stabilizes.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "logo_only": False,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
}

html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

# -- Options for autodoc -----------------------------------------------------
# Mock the entire torch_spyre package. Pure-Python modules like streams.py
# cannot be imported without the C++ extensions and Spyre hardware, so
# autodoc cannot auto-generate their docs. The public API is documented
# manually in api/torch_spyre.rst instead.
#
# TODO: Revisit once the team decides on a pattern to make pure-Python
# modules importable without _C (e.g., lazy imports or conditional guards).
autodoc_mock_imports = ["torch", "torch_spyre"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

# -- todo extension ----------------------------------------------------------
todo_include_todos = True

# -- Suppress known non-critical warnings ------------------------------------
suppress_warnings = [
    "autodoc",  # torch_spyre is mocked; suppress all autodoc warnings
]
