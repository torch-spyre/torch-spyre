# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CI-agnostic local version scheme for torch-spyre.

Registered in ``pyproject.toml`` as::

    [tool.setuptools_scm]
    local_scheme = "_versioning:ci_local_scheme"

Appends ``+branch.gSHA[.bN][.dirty]`` to every non-tag build so each
CI artifact has a unique, traceable identity. Exact-tag builds return
an empty string — clean wheel names for releases and RCs.

CI environment variable priority
---------------------------------
Metadata       Jenkins              GitHub Actions        GitLab CI
-----------    -------------------  --------------------  -------------------
Build number   BUILD_NUMBER         GITHUB_RUN_NUMBER     CI_PIPELINE_IID
Branch name    BRANCH_NAME /        GITHUB_HEAD_REF  (PR) CI_COMMIT_REF_NAME
               GIT_BRANCH           GITHUB_REF_NAME  (push)
"""

from __future__ import annotations

import os

import regex as re


def _sanitize(text: str, max_len: int = 20) -> str:
    """Strip chars not permitted in a PEP 440 local identifier; keep dots."""
    return re.sub(r"[^a-zA-Z0-9.]", "", text)[:max_len]


def _get_build_number() -> str:
    return (
        os.getenv("BUILD_NUMBER")  # Jenkins
        or os.getenv("GITHUB_RUN_NUMBER")  # GitHub Actions
        or os.getenv("CI_PIPELINE_IID")  # GitLab CI
        or ""
    )


def _resolve_branch(version_branch: str | None) -> str:
    return (
        os.getenv("BRANCH_NAME")
        or os.getenv("GIT_BRANCH", "").split("/")[-1]  # strips 'origin/' prefix
        or os.getenv("GITHUB_HEAD_REF")  # PR events (avoids NNN/merge from REF_NAME)
        or os.getenv("GITHUB_REF_NAME")  # push events
        or os.getenv("CI_COMMIT_REF_NAME")  # GitLab CI
        or version_branch
        or ""
    )


def ci_local_scheme(version) -> str:  # type: ignore[no-untyped-def]
    """setuptools_scm ``local_scheme`` callable."""
    if version.exact:
        return ""

    node = (getattr(version, "node", None) or "").lstrip("g")[:8]
    branch = _resolve_branch(getattr(version, "branch", None))
    build_num = _get_build_number()

    branch_s = _sanitize(branch)
    parts: list[str] = []
    if branch_s:
        parts.append(branch_s)
    if node:
        parts.append(f"g{node}")
    if build_num:
        parts.append(f"b{build_num}")
    if getattr(version, "dirty", False):
        parts.append("dirty")

    return ("+" + ".".join(parts)) if parts else ""
