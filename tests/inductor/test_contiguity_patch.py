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

"""CPU checks of the stride-first contiguity patch (#6018).

``_contiguous_stride_first`` in ``torch_spyre/_inductor/patches.py`` is a copy
of torch's ``check_contiguous_sizes_strides`` with the stride compared before
``size == 1``. These tests need no device. They check that the copy still
answers like upstream, that upstream has not changed under it, and that
upstream still asks the size first (pytorch/pytorch#203781), which is the only
reason the patch exists.
"""

import hashlib
import inspect
import itertools
import unittest

import regex as re
import torch._prims_common as prims_common
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from torch_spyre._inductor.patches import _contiguous_stride_first

# sha256 of inspect.getsource(torch._prims_common.check_contiguous_sizes_strides)
# for the torch this copy was checked against. A torch bump that changes the
# upstream function changes this, so the copy is reviewed before it ships.
UPSTREAM_SOURCE_SHA256 = "eba4d232cddd0629030955b42e2b03dc3a7202c9fb71081a73cc7daaf9281b24"

TILE = 64


class TestContiguityPatch(unittest.TestCase):
    def test_copy_answers_like_upstream(self):
        """Same answer as upstream for every small concrete tensor shape."""
        upstream = prims_common.check_contiguous_sizes_strides
        sizes = (0, 1, 2, 3)
        strides = (0, 1, 2, 3, 6)
        for rank in (1, 2, 3):
            for size in itertools.product(sizes, repeat=rank):
                for stride in itertools.product(strides, repeat=rank):
                    for false_if_dde in (False, True):
                        self.assertEqual(
                            _contiguous_stride_first(size, stride, false_if_dde),
                            upstream(size, stride, false_if_dde),
                            f"sizes {size}, strides {stride}, "
                            f"false_if_dde {false_if_dde}",
                        )

    def test_upstream_function_is_unchanged(self):
        """Fails when torch changes the function the patch copies."""
        source = inspect.getsource(prims_common.check_contiguous_sizes_strides)
        actual = hashlib.sha256(source.encode()).hexdigest()
        self.assertEqual(
            actual,
            UPSTREAM_SOURCE_SHA256,
            "torch's check_contiguous_sizes_strides changed: review "
            "_contiguous_stride_first in patches.py, then update the hash",
        )

    def test_upstream_still_asks_size_first(self):
        """Fails if pytorch#203781 is fixed upstream; then delete the patch."""
        env = ShapeEnv()
        n = env.create_symintnode(
            env.create_symbol(2 * TILE, ConstantSource("n")), hint=2 * TILE
        )
        prims_common.check_contiguous_sizes_strides((n // TILE, TILE), (TILE, 1))
        guards = [str(g.expr) for g in env.guards]
        self.assertTrue(
            any(re.fullmatch(rf"Ne\(\(\w+//{TILE}\), 1\)", g) for g in guards),
            f"pytorch#203781 looks fixed (guards: {guards}): remove the patch",
        )


if __name__ == "__main__":
    unittest.main()
