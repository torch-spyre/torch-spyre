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


from contextlib import contextmanager

import torch.fx.traceback

HINT_KEY = "work_division_hint"


@contextmanager
def work_division_hint(splits: list[int]):
    """Override core division splits for operations within this block.

    Args:
        splits: Split factors in iteration-space order — output dims first,
            then reduction dims. For a 2D matmul out = x @ y with x:(M,K)
            and y:(K,N), pass [M_split, N_split, K_split].

    Example::

        from torch_spyre._inductor.work_division_hint import work_division_hint

        @torch.compile
        def model(x, y):
            with work_division_hint([2, 1, 2]):
                out = x @ y  # M split by 2, N unsplit, K split by 2
            return out

    Note:
        Different hint values for the same compiled function may hit Dynamo's
        graph cache. Call ``torch._dynamo.reset()`` between experiments.
    """
    with torch.fx.traceback.annotate({HINT_KEY: list(splits)}):
        yield
