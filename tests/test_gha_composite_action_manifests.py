# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Guards against a manifest-load failure seen on recover-threaded-run-id (issue: CI's
ActionManifestManagerLegacy refused to load the action).

`name`, `description`, and the per-input/output `description` fields of a composite action
are plain metadata -- they sit outside `runs`, so the runner never binds a context (`inputs`,
`github`, ...) for them. A literal `${{ ... }}` there does not get templated; it makes the
loader choke while trying to read a StringToken and finding a BasicExpressionToken instead,
and the *whole* action.yml fails to load -- not just the offending field. Composite actions
sometimes need to show GitHub Actions expression syntax as a documentation example (e.g. to
tell a workflow_run caller how to shape an upload step); the syntax must be broken up (e.g.
`"$ {{ ... }}"`) rather than written literally or "escaped" via a nested expression, since the
loader rejects an expression token in these fields regardless of what it evaluates to.
"""

from pathlib import Path

import pytest
import regex as re
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_ACTION_MANIFESTS = sorted((_ROOT / ".github" / "actions").glob("*/action.y*ml"))

_EXPRESSION_OPEN = re.compile(r"\$\{\{")


def _untemplated_string_fields(manifest: dict):
    """Yield (field_path, value) for every plain-string field GH does not template."""
    if "name" in manifest:
        yield "name", manifest["name"]
    if "description" in manifest:
        yield "description", manifest["description"]
    for section in ("inputs", "outputs"):
        for key, spec in (manifest.get(section) or {}).items():
            if isinstance(spec, dict) and "description" in spec:
                yield f"{section}.{key}.description", spec["description"]


@pytest.mark.parametrize(
    "manifest_path", _ACTION_MANIFESTS, ids=lambda p: p.parent.name
)
def test_action_manifest_loads(manifest_path):
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    assert isinstance(manifest, dict), manifest_path


@pytest.mark.parametrize(
    "manifest_path", _ACTION_MANIFESTS, ids=lambda p: p.parent.name
)
def test_untemplated_fields_have_no_live_expression(manifest_path):
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    offenders = [
        field
        for field, value in _untemplated_string_fields(manifest)
        if isinstance(value, str) and _EXPRESSION_OPEN.search(value)
    ]
    assert not offenders, (
        f"{manifest_path.relative_to(_ROOT)}: {offenders} contain a literal '${{{{' -- "
        "GitHub's composite-action loader treats name/description (and inputs/outputs "
        "descriptions) as plain strings and refuses to load the whole file if one holds a "
        "template expression. Break up the sequence (e.g. '$ {{ ... }}') instead."
    )
