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

Command line (from a checkout)::

    python tests/op_coverage/op_coverage.py ultralytics yolov5n.yaml -o reports/

Python (import ``torch`` before this package)::

    import torch
    from spyre_op_coverage import analyze_model, to_markdown

    report = analyze_model(model, (example_input,), name="my-model")
    print(to_markdown(report))

The pipeline has four stages, one module each: ``capture`` (run the model on
the CPU under a recording Dynamo backend), ``trace`` (follow each call through
Spyre's decompositions on fake Spyre tensors), ``registry`` (judge each ATen op
against torch-spyre's lowerings, fallbacks and eager kernels) and ``render``.
``probe`` optionally compiles each signature on a Spyre device.
"""

from .analyze import analyze_calls, analyze_model, build_report
from .capture import CapturedCall, TensorMeta, capture
from .knowledge import Knowledge
from .model import CoverageReport, Support
from .registry import SpyreRegistry
from .render import to_json, to_markdown

__all__ = [
    "CapturedCall",
    "CoverageReport",
    "Knowledge",
    "SpyreRegistry",
    "Support",
    "TensorMeta",
    "analyze_calls",
    "analyze_model",
    "build_report",
    "capture",
    "to_json",
    "to_markdown",
]
