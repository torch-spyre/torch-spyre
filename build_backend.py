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
"""In-tree PEP 517 build backend wrapper (torch-spyre).

Adds the project root to sys.path so that _versioning.py is importable
by setuptools_scm when resolving the local_scheme entry-point string
"_versioning:ci_local_scheme" during an isolated build.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from setuptools.build_meta import *
