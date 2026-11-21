#!/usr/bin/env python3
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

"""Report which ops a model needs and how torch-spyre handles each one.

    python tests/op_coverage/op_coverage.py ultralytics yolov5n.yaml
    python tests/op_coverage/op_coverage.py --help

See tests/op_coverage/spyre_op_coverage/cli.py for more examples.
"""

import sys

# torch first: it autoloads the Spyre backend, which imports torch_spyre.
# Importing torch_spyre first re-enters it half-initialized.
import torch  # noqa: F401
import torch_spyre  # noqa: F401
from spyre_op_coverage.cli import main

if __name__ == "__main__":
    sys.exit(main())
