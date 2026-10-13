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

# No side effects: xdist's controller imports these with the Spyre backend unloaded.


class FallbackWarning(UserWarning):
    """
    Warning issued when an operator runs on a fallback device (e.g., CPU)
    instead of Spyre
    """


class RetileWarning(UserWarning):
    """Warning issued when an eager result had to be re-tiled to the layout a
    compiled graph assumes for it."""
