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

"""The compiler writes a launch spec that matches what it compiled.

The producer's job is to record the HOST shape and dtype, which no compiled
artifact retains: device dtype is many-to-one and device shape is ceil-divided
into sticks. These tests pin that the recorded values are the host ones and that
``pool_size`` means "the pool tensor the caller must pass" rather than the
kernel's own pool extent.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch._inductor.utils import fresh_cache

from torch_spyre.execution.async_compile import _write_launch_spec
from torch_spyre._inductor.codegen.kernel_launchspec import (
    LAUNCH_SPEC_FILE,
    SPYRECODE_DIR,
    LaunchArg,
    LaunchArgLayout,
    check_launch_spec,
    load_launch_spec,
)


def _compiled_kernel_dirs(root):
    """Per-kernel output dirs under an inductor-spyre root, newest first."""
    spyre_dir = os.path.join(root, "inductor-spyre")
    if not os.path.isdir(spyre_dir):
        return []
    entries = [os.path.join(spyre_dir, d) for d in os.listdir(spyre_dir)]
    return sorted((d for d in entries if os.path.isdir(d)), key=os.path.getmtime)


class TestLaunchSpecEmission(unittest.TestCase):
    """End-to-end: compile something, then read the spec beside it."""

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def _compile_add(self, shape=(10, 512), dtype=torch.float16):
        """Compile a fresh ``a + b`` and report what landed beside the bundle.

        Returns ``(files, spec, tensors)``: the spyreCodeDir listing, the parsed
        launch spec (or None), and the tensors the kernel actually ran on.

        Everything is read INSIDE the ``fresh_cache`` context, which is why this
        returns data rather than a path -- the tree is deleted when the context
        exits, so a path handed back to the caller would already be gone. The
        per-test ``fresh_cache`` + ``_dynamo.reset()`` is what stops the second
        identical compile in a process being served from cache, which would
        write no new output dir for the test to inspect.
        """
        from torch._inductor.runtime.runtime_utils import cache_dir

        torch._dynamo.reset()
        with fresh_cache():
            root = cache_dir()
            before = set(_compiled_kernel_dirs(root))
            fn = torch.compile(lambda a, b: a + b, dynamic=False)
            a = torch.ones(shape, dtype=dtype, device="spyre")
            b = torch.ones(shape, dtype=dtype, device="spyre")
            out = fn(a, b)
            self.assertEqual(out.cpu().flatten()[0].item(), 2.0)

            new = [d for d in _compiled_kernel_dirs(root) if d not in before]
            self.assertEqual(
                len(new), 1, f"expected exactly one new kernel dir, got {new}"
            )
            kernel_dir = new[0]
            return (
                sorted(os.listdir(os.path.join(kernel_dir, SPYRECODE_DIR))),
                load_launch_spec(kernel_dir),
                [a, b, out],
            )

    def test_spec_describes_the_kernel_it_was_compiled_from(self):
        """One compile, then everything the spec has to get right.

        Kept as one test because each assertion needs the same real compile and
        they fail for the same reason -- a producer that recorded the wrong
        thing -- so splitting them would buy six compiles and no extra signal.
        """
        files, spec, tensors = self._compile_add(shape=(10, 512))

        # In spyreCodeDir/ alongside the backend compiler's own output, so the
        # folder the runtime loads also states what it expects to be called with.
        self.assertIn(LAUNCH_SPEC_FILE, files)
        self.assertIn("spyrecode.json", files)
        self.assertIn("init_binary.bin", files)
        self.assertIsNotNone(spec)

        # Binding is positional, so the indices must be contiguous from 0.
        args = sorted(spec["args"], key=lambda a: a["arg_index"])
        self.assertEqual([a["arg_index"] for a in args], list(range(len(args))))

        # The compiled signature has no output category; the spec must.
        roles = [a["role"] for a in args]
        self.assertEqual(roles, ["input", "input", "output"])

        for arg in args:
            # Host shape and dtype, not the tiled device view -- which is
            # recorded too and differs, the whole reason the host view needs
            # writing down separately.
            self.assertEqual(arg["shape"], [10, 512])
            self.assertEqual(arg["dtype"], "float16")
            self.assertNotEqual(arg["layout"]["device_size"], [10, 512])

        # The round trip that matters: the spec accepts the tensors the kernel
        # actually ran on, and rejects a transposed one.
        self.assertEqual(check_launch_spec(spec, tensors), [])
        bad = [torch.empty((512, 10), dtype=torch.float16)] + tensors[1:]
        problems = check_launch_spec(spec, bad)
        self.assertEqual(len(problems), 1)
        self.assertIn("transposed", problems[0])


class TestLaunchSpecPoolSize(unittest.TestCase):
    """``pool_size`` is the CALLER's pool tensor, not the kernel's pool.

    A bundle can allocate scratch internally, in which case it has a pool but no
    parameter for one. Recording the extent there would make a launcher prepend a
    tensor the kernel has no slot for. ``call_kernel`` gates on two conditions,
    and the producer has to reproduce both.
    """

    ARGS = [
        LaunchArg(
            arg_index=0,
            role="input",
            shape=[10, 512],
            dtype="float16",
            layout=LaunchArgLayout(
                device_size=[8, 10, 64],
                stride_map=[64, 512, 1],
                device_dtype="SEN169_FP16",
                element_arrangement="STANDARD",
            ),
        )
    ]
    KERNEL_POOL_EXTENT = 32768

    def _spec_pool_size(self, frontend_pool_allocation):
        with tempfile.TemporaryDirectory() as d:
            with patch(
                "torch_spyre._inductor.config.frontend_pool_allocation",
                frontend_pool_allocation,
            ):
                _write_launch_spec(
                    d, "k", self.ARGS, [], self.KERNEL_POOL_EXTENT, "sdsc"
                )
            with open(os.path.join(d, SPYRECODE_DIR, LAUNCH_SPEC_FILE)) as f:
                return json.load(f)["pool_size"]

    def test_backend_allocated_pool_records_zero(self):
        """Default path: device_mem_allocate inside the bundle, no caller pool."""
        self.assertEqual(self._spec_pool_size(False), 0)

    def test_frontend_allocated_pool_records_the_extent(self):
        """The caller really does prepend a pool tensor here."""
        self.assertEqual(self._spec_pool_size(True), self.KERNEL_POOL_EXTENT)

    def test_no_spec_is_written_without_launch_args(self):
        """Nothing to describe means no file, not an empty one."""
        with tempfile.TemporaryDirectory() as d:
            _write_launch_spec(d, "k", None, [], 0, "sdsc")
            self.assertFalse(
                os.path.exists(os.path.join(d, SPYRECODE_DIR, LAUNCH_SPEC_FILE))
            )


if __name__ == "__main__":
    unittest.main()
