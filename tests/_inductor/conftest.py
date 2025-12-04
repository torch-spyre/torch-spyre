# Copyright 2025 The Torch-Spyre Authors.
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

import pathlib
import os
import pytest
import yaml


# for test_models_ops.py
def pytest_collection_modifyitems(config, items):
    if os.getenv("TEST_MODELS_OPS_ONLY_SKIP_FILES") is not None:
        return False
    if os.getenv("TEST_MODELS_OPS_IGNORE_SKIP_FILES") is not None:
        return False

    yaml_path = pathlib.Path("models/skip_files.yaml")
    if not yaml_path.exists():
        return False

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    skip_markers = {}
    xfail_markers = {}

    reasons = data.get("skip", [])
    for reason in reasons:
        marker = pytest.mark.skip(reason=f"marked skip in skip_files.yaml: {reason}")
        skip_files = data["skip"][str(reason)]
        skip_paths = [pathlib.Path(p).resolve() for p in skip_files]
        for p in skip_paths:
            skip_markers[p] = marker

    reasons = data.get("xfail", [])
    for reason in reasons:
        marker = pytest.mark.xfail(reason=f"marked xfail in skip_files.yaml: {reason}")
        xfail_files = data.get(reason, [])
        xfail_paths = [pathlib.Path(p).resolve() for p in xfail_files]
        for p in xfail_paths:
            xfail_markers[p] = marker

    for item in items:
        test_file = pathlib.Path(item.fspath).resolve()
        if test_file in skip_markers:
            item.add_marker(skip_markers[test_file])
        if test_file in xfail_markers:
            item.add_marker(xfail_markers[test_file])


def pytest_ignore_collect(collection_path, config):
    if os.getenv("TEST_MODELS_OPS_ONLY_SKIP_FILES") is None:
        return False

    yaml_path = pathlib.Path("models/skip_files.yaml")
    if not yaml_path.exists():
        return False

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    exclude_paths = []
    reasons = data.get("skip", [])
    for reason in reasons:
        skip_files = data["skip"][str(reason)]
        exclude_paths.extend([pathlib.Path(p).resolve() for p in skip_files])

    reasons = data.get("xfail", [])
    for reason in reasons:
        xfail_files = data["xfail"][str(reason)]
        exclude_paths.extend([pathlib.Path(p).resolve() for p in xfail_files])
    print(exclude_paths)

    p = collection_path.resolve()
    print("CP", p, str(type(collection_path)), str(type(exclude_paths[0])))
    if os.path.isdir(p):
        return False
    return p not in exclude_paths
