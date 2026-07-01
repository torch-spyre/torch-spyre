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

import unittest
import torch

from utils_inductor import compare_with_cpu


class TestMultiArgPointwiseBroadcastRank(unittest.TestCase):
    """Regression tests for #2961.

    PR #2948 removed the ``!= 0`` filter on ``stick_exprs`` in
    ``_multi_arg_pointwise_layouts``, letting broadcast/constant (stick expr 0)
    inputs flow into ``_pick_stick_dim``. When such an expr does not survive to
    any output dim, ``_pick_stick_dim`` returns ``-1`` and
    ``_compute_dim_order(-1, ...)`` appended a phantom trailing dimension,
    raising the rank by one (rank-6 -> rank-7) and tripping the C++ dim_map cap
    ("Unsupported tensor rank: 7") in spyre_tensor_impl.cpp.

    The fix skips non-surviving (-1) stick exprs in the multi-arg pointwise
    else-branch so the layout search falls through to valid candidates instead
    of synthesizing an invalid rank-7 layout.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_multi_arg_pointwise_broadcast_rank_regression(self):
        a, b, c, d, e = 2, 3, 4, 5, 64

        def fn(w, x, y, z):
            t = w + x
            t = t.view(1, a, b, d, e)
            t = t.unsqueeze(2) + y.unsqueeze(3)
            return t + z

        w = torch.randn(1, a, b * d * e, dtype=torch.float16) * 0.1
        x = torch.randn(1, a, b * d * e, dtype=torch.float16) * 0.1
        y = torch.randn(1, a, c, d, e, dtype=torch.float16) * 0.1
        # z broadcasts along dims 3,4,5 -> its stick expr does not survive to
        # any output dim, which is the (#2961) rank-7 trigger under compile.
        z = torch.randn(1, a, c, 1, 1, 1, dtype=torch.float16) * 0.1

        # Only the compiled path exercises the inductor layout propagation that
        # produced the rank-7 SpyreTensorLayout; assert it compiles, returns the
        # correct rank-6 shape, and matches CPU.
        compare_with_cpu(fn, w, x, y, z, run_eager=False)


if __name__ == "__main__":
    unittest.main()
