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

from .core import launch

__all__ = [
    "check_tensors_against_spec",
    "default_tensors",
    "launch",
    "load_kernelspec",
    "tensors_from_spec",
]


def __getattr__(name):
    """Resolve the kernelspec API on first use.

    ``launch_spec`` imports torch at module level, so importing it here would
    make ``spyre --help`` need torch-spyre present. Deferring keeps the import
    cost and the dependency at the point of use, while
    ``spyre_cli.load_kernelspec`` still reads as a plain attribute.
    """
    if name in (
        "load_kernelspec",
        "check_tensors_against_spec",
        "tensors_from_spec",
        "default_tensors",
    ):
        from . import launch_spec

        return getattr(launch_spec, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
