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
    LaunchSpec,
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

        from torch_spyre._C import get_spyre_tensor_layout

        for arg, tensor in zip(args, tensors):
            # Host shape and dtype, not the tiled device view -- which is
            # recorded too and differs, the whole reason the host view needs
            # writing down separately.
            self.assertEqual(arg["shape"], [10, 512])
            self.assertEqual(arg["dtype"], "float16")
            self.assertEqual(arg["stride"], [512, 1])
            self.assertEqual(arg["storage_offset"], 0)
            self.assertNotEqual(arg["layout"]["device_size"], [10, 512])
            # valid_elements as [device dims, count] pairs, the same as the
            # tensor the kernel ran on: the layout constructor requires it.
            actual = get_spyre_tensor_layout(tensor).valid_elements
            self.assertEqual(
                arg["layout"]["valid_elements"],
                sorted([list(dims), count] for dims, count in actual.items()),
            )

    def _compile_specs(self, fn, *args, fullgraph=False):
        """Compile ``fn(*args)`` in a fresh cache; return its result and specs.

        Returns ``(result, specs, dirs)``: the specs are those of every new kernel
        dir, ``None`` where a dir has no spec, read inside the cache context for
        the same reason as ``_compile_add``.
        """
        from torch._inductor.runtime.runtime_utils import cache_dir

        torch._dynamo.reset()
        with fresh_cache():
            root = cache_dir()
            before = set(_compiled_kernel_dirs(root))
            result = torch.compile(fn, fullgraph=fullgraph)(*args)
            new = [d for d in _compiled_kernel_dirs(root) if d not in before]
            self.assertTrue(new, "the compile produced no kernel dir")
            files = [sorted(os.listdir(os.path.join(d, SPYRECODE_DIR))) for d in new]
            return result, [load_launch_spec(d) for d in new], files

    def test_non_contiguous_output_records_its_strides(self):
        """A transposed output keeps the graph's host strides, not contiguous ones.

        Rebuilding it with contiguous strides reads it back differently.
        """
        x = torch.randn(256, 128, dtype=torch.float16).to("spyre")
        out, specs, _ = self._compile_specs(lambda t: t.t() * 2, x)
        (spec,) = specs
        (output,) = [a for a in spec["args"] if a["role"] == "output"]
        self.assertEqual(output["shape"], [128, 256])
        self.assertEqual(output["stride"], list(out.stride()))
        self.assertEqual(output["stride"], [1, 128])

    def test_in_place_argument_is_input_output(self):
        """An argument that is both read and written is not just an output.

        Keeping only the last record made ``x.add_(y)``'s ``x`` an output, so the
        smoke-test fill never initialised it.
        """
        x = torch.randn(10, 512, dtype=torch.float16).to("spyre")
        y = torch.randn(10, 512, dtype=torch.float16).to("spyre")

        def add_in_place(p, q):
            p.add_(q)

        _, specs, _ = self._compile_specs(add_in_place, x, y)
        (spec,) = [s for s in specs if s is not None]
        self.assertEqual(
            sorted(a["role"] for a in spec["args"]), ["input", "input_output"]
        )

    def test_declared_loop_records_its_symbol(self):
        """A declared for_each_tile output's extent is recorded as the symbol.

        Inductor gives it as ``64*((s//64))``; the loop guards ``s % 64 == 0``, so
        the producer folds it, and the loop slot names the same symbol.
        """
        from torch_spyre._inductor.wsr import for_each_tile

        def tiled_gelu(t):
            rows = t.shape[0]
            torch._check(rows >= 128)
            torch._check(rows <= 512)
            torch._check(rows % 64 == 0)

            def body(_carry, tiles):
                (tile,) = tiles
                return None, torch.nn.functional.gelu(tile)

            _carry, o = for_each_tile(body, (t,), dims=(0,), tile_size=64, out_dim=0)
            return o

        host = torch.randn(512, 64, dtype=torch.float16)
        x = host.to("spyre", dynamic={0: {"min": 128, "max": 512, "granularity": 64}})
        x.resize_((256, 64))
        # fullgraph, as test_symbolic_loop_e2e compiles it: a graph break around
        # for_each_tile sends the loop through inductor's generic scan lowering.
        _, specs, _ = self._compile_specs(tiled_gelu, x, fullgraph=True)
        (spec,) = [s for s in specs if s is not None and "symbols" in s]

        (loop,) = [k for k in spec["symbol_kinds"] if k["kind"] == "loop_dimension"]
        sym = loop["pytorch_sym"]
        self.assertEqual((loop["granularity"], loop["max_value"]), (64, 512))
        self.assertEqual([a["shape"] for a in spec["args"]], [[sym, 64]] * 2)
        self.assertEqual(list(spec["symbols"]), [sym])

    def test_depthwise_conv_records_the_input_layouts(self):
        """A kernel compiled against a special packing records that packing.

        The depthwise conv2d puts the channels in one stick, so its inputs'
        recorded layouts must be the ones the tensors were given, not the
        default arrangement for their shapes.
        """
        from torch_spyre._C import (
            SpyreTensorLayout,
            get_device_dtype,
            get_spyre_tensor_layout,
        )

        fp16 = get_device_dtype(torch.float16)
        # valid_elements {} is "unknown", which the constructor accepts; only
        # device_size and stride_map decide the packing compared below.
        xl = SpyreTensorLayout(
            [32, 32, 1, 1, 64],
            [32, 1, -1, 65536, 1024],
            fp16,
            {},
        )
        wl = SpyreTensorLayout([3, 3, 1, 1, 64], [1, 3, -1, 9, 9], fp16, {})
        x = torch.randn(1, 64, 32, 32, dtype=torch.float16).to(device_layout=xl)
        w = torch.randn(64, 1, 3, 3, dtype=torch.float16).to(device_layout=wl)

        def conv(a, b):
            return torch.conv2d(a, b, None, stride=(1, 1), padding=(0, 0), groups=64)

        _, specs, _ = self._compile_specs(conv, x, w)
        (spec,) = [s for s in specs if s is not None]
        recorded = {
            tuple(a["shape"]): a["layout"] for a in spec["args"] if a["role"] == "input"
        }
        for tensor in (x, w):
            want = get_spyre_tensor_layout(tensor)
            got = recorded[tuple(tensor.shape)]
            self.assertEqual(got["device_size"], list(want.device_size))
            self.assertEqual(got["stride_map"], list(want.stride_map))

    def test_an_argument_that_cannot_be_recorded_writes_no_spec(self):
        """All or nothing: one unrecordable argument means no spec at all.

        A spec with a hole would put every later tensor in the wrong slot. The
        kernel itself still compiles and runs.
        """
        real = LaunchArgLayout.from_device_layout.__func__
        calls = []

        def fail_first(cls, device_layout):
            calls.append(device_layout)
            if len(calls) == 1:
                raise RuntimeError("simulated: cannot record this argument")
            return real(cls, device_layout)

        a = torch.ones(10, 512, dtype=torch.float16, device="spyre")
        with patch.object(
            LaunchArgLayout, "from_device_layout", classmethod(fail_first)
        ):
            out, specs, files = self._compile_specs(lambda p, q: p + q, a, a)
        self.assertTrue(calls, "the recorder never ran")
        self.assertEqual(out.cpu().flatten()[0].item(), 2.0)
        self.assertEqual(specs, [None] * len(specs))
        for listing in files:
            self.assertIn("spyrecode.json", listing)
            self.assertNotIn(LAUNCH_SPEC_FILE, listing)


class TestLaunchSpecSchema(unittest.TestCase):
    """Reading a spec refuses one that does not describe every argument."""

    def _raw(self, indices):
        return {
            "version": 1,
            "kernel_name": "k",
            "args": [
                {"arg_index": i, "role": "input", "shape": [1], "dtype": "float16"}
                for i in indices
            ],
        }

    def test_contiguous_indices_load(self):
        spec = LaunchSpec.from_dict(self._raw([1, 0]))
        self.assertEqual([a.arg_index for a in spec.args], [0, 1])

    def test_a_missing_index_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not describe every argument"):
            LaunchSpec.from_dict(self._raw([0, 2]))

    def test_a_repeated_index_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not describe every argument"):
            LaunchSpec.from_dict(self._raw([0, 0]))


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
