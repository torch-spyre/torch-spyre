# Copyright 2025-2026 The Torch-Spyre Authors.
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

"""
Spyre profiling package.

FFDC retrieval is the public API on this package
(``get_diagnostic_report``, also bound as ``torch.spyre.get_diagnostic_report``).
Device-side timing uses upstream ``torch.profiler``. Device presence is
``torch.spyre.is_available()``, not a flag on this package.
"""

from torch_spyre.profiler._ffdc import get_diagnostic_report

__all__ = ["get_diagnostic_report"]
