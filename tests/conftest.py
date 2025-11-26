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

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    os.environ.setdefault("DTLOG_LEVEL", "error")
    os.environ.setdefault("DT_DEEPRT_VERBOSE", "-1")

# for test_models_ops.py
def pytest_collection_modifyitems(config, items):
    if os.getenv("TEST_MODELS_OPS_IGNORE_SKIP_FILES") is not None:
        return False

    yaml_path = pathlib.Path("models/skip_files.yaml")
    if not yaml_path.exists():
        return False

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f) or {}

    skip_files = data.get("skip", [])
    skip_paths = [pathlib.Path(p).resolve() for p in skip_files]
    skip_marker = pytest.mark.skip(reason="marked skip in skip_files.yaml")

    xfail_files = data.get("xfail", [])
    xfail_paths = [pathlib.Path(p).resolve() for p in xfail_files]
    xfail_marker = pytest.mark.xfail(reason="marked xfail in skip_files.yaml")

    for item in items:
        test_file = pathlib.Path(item.fspath).resolve()
        if test_file in skip_paths:
            item.add_marker(skip_marker)
        if test_file in xfail_paths:
            item.add_marker(xfail_marker)
