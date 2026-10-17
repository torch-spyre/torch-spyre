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

"""tests/oot_framework/utils/filter_configs.py: label/platform selection logic."""

import importlib.util
import platform
from pathlib import Path

import pytest
import yaml

HELPER = (
    Path(__file__).resolve().parent / "oot_framework" / "utils" / "filter_configs.py"
)
_spec = importlib.util.spec_from_file_location("filter_configs", HELPER)
assert _spec is not None and _spec.loader is not None
filter_configs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filter_configs)


def _write_config(tmp_path, *, labels=None, exclude_platforms=None):
    cfg = {"test_suite_config": {}}
    if labels is not None:
        cfg["test_suite_config"]["labels"] = labels
    if exclude_platforms is not None:
        cfg["test_suite_config"]["exclude_platforms"] = exclude_platforms
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg))
    return path


def test_load_tsc_dedup(tmp_path):
    path = _write_config(tmp_path, labels=["trunk"], exclude_platforms=["ppc64le"])
    tsc = filter_configs._load_tsc(path)
    assert filter_configs._load_labels(tsc) == ["trunk"]
    assert filter_configs._load_excluded_platforms(tsc) == ["ppc64le"]


@pytest.mark.parametrize(
    "exclude_platforms,machine,expected",
    [
        (["ppc64le"], "ppc64le", True),  # dropped: current arch is excluded
        (["ppc64le"], "x86_64", False),  # kept: current arch isn't excluded
        (None, "ppc64le", False),  # kept: no exclude_platforms set at all
    ],
)
def test_excluded_on_this_platform(
    tmp_path, monkeypatch, exclude_platforms, machine, expected
):
    monkeypatch.setattr(platform, "machine", lambda: machine)
    path = _write_config(tmp_path, exclude_platforms=exclude_platforms)
    excluded = filter_configs._load_excluded_platforms(filter_configs._load_tsc(path))
    assert filter_configs._excluded_on_this_platform(excluded) is expected
