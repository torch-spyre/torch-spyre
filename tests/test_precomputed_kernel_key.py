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

"""CPU-only tests for TORCH_SPYRE_PRECOMPUTED_KERNEL_KEY.

Codegen stores the kernel-cache key in every generated
``async_compile.sdsc(..., precomputed_key="<key>|<fingerprint>")`` call; with
the switch on, sdsc() uses it instead of re-running compute_specs_hash().

Run with:
    python -m pytest tests/test_precomputed_kernel_key.py -v
"""

import contextlib
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import warnings

import sympy
import torch
from sympy import Integer, Symbol
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import NullHandler, V

import torch_spyre._inductor.config as spyre_config
from torch_spyre._inductor import scheduler as scheduler_mod
from torch_spyre._inductor.spyre_kernel import _codegen_op_spec_list
from torch_spyre._inductor.wrapper import spec_eval_namespace
from torch_spyre.execution import async_compile as async_compile_mod
from torch_spyre.execution import kernel_cache
from torch_spyre.execution.kernel_cache import (
    PrecomputedKernelKeyMismatch,
    _KernelHashRegistry,
    compute_specs_hash,
    kernel_key_fingerprint,
    precompute_kernel_key,
    resolve_precomputed_key,
)


# ---------------------------------------------------------------------------
# Spec fixtures: real OpSpecs that compile_op_spec handles on CPU.
# ---------------------------------------------------------------------------


def _make_op_spec(op: str = "add", size: int = 64, hbm_in: int = 0, hbm_out: int = 1):
    from torch_spyre._C import DataFormats
    from torch_spyre._inductor.core_mapping import derive_operation_mapping
    from torch_spyre._inductor.op_spec import OpSpec, TensorArg

    c0 = Symbol("c0")
    args = [
        TensorArg(
            is_input=is_input,
            arg_index=index,
            device_dtype=DataFormats.IEEE_FP16,
            device_size=[1, size, 1],
            device_coordinates=[Integer(0), c0 % size, Integer(0)],
            allocation={"hbm": hbm},
        )
        for is_input, index, hbm in ((True, 0, hbm_in), (False, 1, hbm_out))
    ]
    iteration_space = {c0: (Integer(size), 1)}
    spec = OpSpec(
        op=op,
        is_reduction=False,
        iteration_space=iteration_space,
        args=args,
        op_info={},
    )
    spec.core_id_to_work_slice = derive_operation_mapping(iteration_space)
    return spec


def _spec_sets():
    from torch_spyre._inductor.op_spec import LoopSpec

    return {
        "single": [_make_op_spec("add")],
        "two_ops": [_make_op_spec("add"), _make_op_spec("mul", size=128)],
        "loop": [LoopSpec(count=Integer(3), body=[_make_op_spec("exp")])],
    }


def _sympy_str(x) -> str:
    # SpyreKernel.codegen_kernel's sympy_str for non-indirect expressions.
    return "sympify('" + str(x) + "')"


def _src(specs) -> str:
    """The OpSpec list source exactly as SpyreKernel.codegen_kernel prints it."""
    buf = IndentedBuffer()
    buf.writeline("[")
    with buf.indent():
        _codegen_op_spec_list(specs, buf, _sympy_str)
    buf.writeline("]")
    return buf.getvalue()


@contextlib.contextmanager
def _env(**config):
    """Stable versions, a private registry and the given Spyre config."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.object(
                kernel_cache,
                "_get_backend_compiler_version",
                return_value="deeptools=test;flex=test",
            )
        )
        stack.enter_context(
            patch.object(
                kernel_cache, "_get_torch_spyre_version", return_value="0.0.test"
            )
        )
        stack.enter_context(
            patch.object(kernel_cache, "_registry", _KernelHashRegistry())
        )
        settings = {"spyre_kernel_cache": True}
        settings.update(config)
        stack.enter_context(spyre_config.patch(settings))
        yield


def _runtime_key(src: str, pool_size: int) -> str:
    """What sdsc() computes today for the specs the wrapper module builds."""
    specs = eval(src, spec_eval_namespace())  # noqa: S307
    return compute_specs_hash(specs, kernel_name="k", pool_size=pool_size)


# ---------------------------------------------------------------------------
# Codegen side
# ---------------------------------------------------------------------------


class TestPrecomputeKernelKey(unittest.TestCase):
    def test_equals_runtime_hash(self):
        for symbolic in (True, False):
            for name, specs in _spec_sets().items():
                for pool_size in (0, 8192):
                    with (
                        self.subTest(symbolic=symbolic, specs=name, pool=pool_size),
                        _env(bundle_symbolic_args=symbolic),
                    ):
                        src = _src(specs)
                        stored = precompute_kernel_key(src, "k", pool_size)
                        self.assertIsNotNone(stored)
                        key, sep, fingerprint = stored.rpartition("|")
                        self.assertEqual(sep, "|")
                        self.assertEqual(key, _runtime_key(src, pool_size))
                        self.assertEqual(fingerprint, kernel_key_fingerprint(pool_size))
                        self.assertEqual(
                            resolve_precomputed_key(stored, pool_size), key
                        )

    def test_distinct_specs_and_pools_get_distinct_keys(self):
        with _env():
            keys = set()
            for specs in _spec_sets().values():
                for pool_size in (0, 8192):
                    keys.add(precompute_kernel_key(_src(specs), "k", pool_size))
            self.assertEqual(len(keys), 6)

    def test_none_when_kernel_cache_off(self):
        with _env(spyre_kernel_cache=False):
            self.assertIsNone(precompute_kernel_key(_src(_spec_sets()["single"])))

    def test_none_when_versions_unknown(self):
        with (
            _env(),
            patch.object(
                kernel_cache,
                "_get_backend_compiler_version",
                side_effect=RuntimeError("LIB_VERSION_FILE is not set"),
            ),
        ):
            self.assertIsNone(precompute_kernel_key(_src(_spec_sets()["single"])))

    def test_none_when_source_does_not_evaluate(self):
        with _env():
            self.assertIsNone(precompute_kernel_key("[NotAName()]"))

    def test_does_not_register(self):
        with _env():
            precompute_kernel_key(_src(_spec_sets()["single"]), "k", 0)
            self.assertEqual(kernel_cache.get_kernel_registry().all_entries(), {})

    def test_hashes_without_the_graph_handler(self):
        # As on a cache reload, where no GraphLowering is installed.
        seen = []

        def fake_hash(*args, **kwargs):
            seen.append(V.graph)
            return "key"

        with (
            _env(),
            V.set_graph_handler(MagicMock()),
            patch.object(kernel_cache, "compute_specs_hash", side_effect=fake_hash),
        ):
            stored = precompute_kernel_key(_src(_spec_sets()["single"]), "k", 0)
        self.assertTrue(stored.startswith("key|"))
        self.assertEqual(len(seen), 1)
        self.assertIsInstance(seen[0], NullHandler)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------


class TestFingerprint(unittest.TestCase):
    def test_every_input_changes_it(self):
        with _env():
            base = kernel_key_fingerprint(0)
            self.assertEqual(base, kernel_key_fingerprint(0))
            changed = {
                "pool_size": kernel_key_fingerprint(64),
                "loop_unroll": kernel_key_fingerprint(
                    0, backend_loop_unroll=not spyre_config.backend_loop_unroll
                ),
            }
            for field, value in (
                ("bundle_symbolic_args", not spyre_config.bundle_symbolic_args),
                (
                    "frontend_pool_allocation",
                    not spyre_config.frontend_pool_allocation,
                ),
                ("sencores", spyre_config.sencores + 1),
                (
                    "disable_conv2d_spatial_split",
                    not spyre_config.disable_conv2d_spatial_split,
                ),
                ("backend_loop_unroll", not spyre_config.backend_loop_unroll),
            ):
                with spyre_config.patch({field: value}):
                    changed[field] = kernel_key_fingerprint(0)
            with patch.object(torch, "__version__", "0.0.other"):
                changed["torch"] = kernel_key_fingerprint(0)
            with patch.object(
                kernel_cache, "_get_torch_spyre_version", return_value="0.0.other"
            ):
                changed["torch_spyre"] = kernel_key_fingerprint(0)
            with patch.object(
                kernel_cache,
                "_get_backend_compiler_version",
                return_value="deeptools=other;flex=test",
            ):
                changed["backend"] = kernel_key_fingerprint(0)
            with patch.object(
                kernel_cache, "_get_key_source_digest", return_value="0" * 64
            ):
                changed["sources"] = kernel_key_fingerprint(0)
            with patch.object(
                kernel_cache, "_get_native_extension_digest", return_value="1" * 64
            ):
                changed["native_extension"] = kernel_key_fingerprint(0)
            with patch.object(sympy, "__version__", "0.0.other"):
                changed["sympy"] = kernel_key_fingerprint(0)
        for field, value in changed.items():
            self.assertNotEqual(value, base, field)

    def test_source_digest_is_stable(self):
        digest = kernel_cache._get_key_source_digest()
        self.assertEqual(len(digest), 64)
        kernel_cache._get_key_source_digest.cache_clear()
        self.assertEqual(kernel_cache._get_key_source_digest(), digest)

    def test_native_extension_digest_is_the_file_content_hash(self):
        import hashlib

        import torch_spyre._C as _C

        digest = kernel_cache._get_native_extension_digest()
        with open(_C.__file__, "rb") as f:
            self.assertEqual(digest, hashlib.sha256(f.read()).hexdigest())
        # Computed once per process.
        with patch("builtins.open", side_effect=AssertionError("re-read")):
            self.assertEqual(kernel_cache._get_native_extension_digest(), digest)

    def test_native_extension_file_change_changes_fingerprint(self):
        import torch_spyre._C as _C

        with tempfile.TemporaryDirectory() as tmp:
            fake = f"{tmp}/_C.so"
            digests = []
            for content in (b"build-a", b"build-b"):
                with open(fake, "wb") as f:
                    f.write(content)
                kernel_cache._get_native_extension_digest.cache_clear()
                with _env(), patch.object(_C, "__file__", fake):
                    digests.append(kernel_key_fingerprint(0))
        kernel_cache._get_native_extension_digest.cache_clear()
        self.assertNotEqual(digests[0], digests[1])

    def test_resolve_rejects_unusable_keys(self):
        with _env():
            fingerprint = kernel_key_fingerprint(0)
            self.assertEqual(resolve_precomputed_key(f"k|{fingerprint}", 0), "k")
            for stored, pool_size in (
                (None, 0),
                ("", 0),
                ("k", 0),
                ("k|", 0),
                (f"|{fingerprint}", 0),
                ("k|0123", 0),
                (f"k|{fingerprint}", 64),
            ):
                with self.subTest(stored=stored, pool=pool_size):
                    self.assertIsNone(resolve_precomputed_key(stored, pool_size))
            with patch.object(
                kernel_cache,
                "_get_backend_compiler_version",
                side_effect=RuntimeError("no versions"),
            ):
                self.assertIsNone(resolve_precomputed_key(f"k|{fingerprint}", 0))


# ---------------------------------------------------------------------------
# Runtime side: SpyreAsyncCompile.sdsc
# ---------------------------------------------------------------------------


def _runner(name, code_dir, kernel_provenance=None, symbol_kinds=None):
    return name, code_dir, kernel_provenance, symbol_kinds


class TestSdscPrecomputedKey(unittest.TestCase):
    def _sdsc(self, mode, stored, *, cached_dir=None, computed="computed", pool_size=0):
        """Run sdsc() with the backend stubbed; return the mocks."""
        compiler = async_compile_mod.SpyreAsyncCompile()
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        mocks = {}
        with contextlib.ExitStack() as stack:
            stack.enter_context(_env(precomputed_kernel_key=mode))
            stack.enter_context(
                torch._inductor.config.patch(
                    {"compile_threads": 1, "force_disable_caches": False}
                )
            )

            def mock(name, **kwargs):
                mocks[name] = stack.enter_context(
                    patch.object(async_compile_mod, name, **kwargs)
                )

            mock("compute_specs_hash", wraps=lambda *a, **k: computed)
            mock("get_cached_kernel_dir", return_value=cached_dir)
            mock("allocate_compile_dir", return_value=self.tmp)
            mock("commit_compile_dir", return_value=self.tmp)
            mock("generate_bundle", return_value=[])
            mock("save_symbol_kinds")
            mock("load_symbol_kinds", return_value=[])
            mock("record_kernel_name")
            mock("_run_backend_compiler")
            mock("find_unimplemented", return_value=None)
            mock("build_kernel_provenance_descriptor", return_value=None)
            mock("SpyreSDSCKernelRunner", side_effect=_runner)
            stored_key = stored() if callable(stored) else stored
            runner = compiler.sdsc(
                "sdsc_k_0", [], pool_size=pool_size, precomputed_key=stored_key
            )
            registry = kernel_cache.get_kernel_registry().all_entries()
        return runner, mocks, registry

    @staticmethod
    def _stored(key="pkey", pool_size=0):
        return lambda: f"{key}|{kernel_key_fingerprint(pool_size)}"

    def test_hit_uses_stored_key_without_hashing(self):
        runner, mocks, registry = self._sdsc(
            "1", self._stored(), cached_dir="/cache/pkey"
        )
        mocks["compute_specs_hash"].assert_not_called()
        mocks["get_cached_kernel_dir"].assert_called_once_with("pkey")
        mocks["generate_bundle"].assert_not_called()
        self.assertEqual(runner[1], "/cache/pkey")
        self.assertEqual(registry["pkey"]["hit_count"], 1)
        self.assertTrue(registry["pkey"]["precomputed"])
        self.assertEqual(registry["pkey"]["kernel_names"], ["sdsc_k_0"])

    def test_miss_compiles_and_commits_under_stored_key(self):
        runner, mocks, registry = self._sdsc("1", self._stored(), cached_dir=None)
        mocks["compute_specs_hash"].assert_not_called()
        mocks["allocate_compile_dir"].assert_called_once_with(
            "pkey", kernel_name="sdsc_k_0"
        )
        mocks["generate_bundle"].assert_called_once()
        mocks["commit_compile_dir"].assert_called_once_with(self.tmp, "pkey")
        self.assertEqual(registry["pkey"]["miss_count"], 1)

    def test_switch_off_hashes(self):
        for mode in ("0", "", "off"):
            with self.subTest(mode=mode):
                _, mocks, _ = self._sdsc(mode, self._stored(), cached_dir="/c")
                mocks["compute_specs_hash"].assert_called_once()
                mocks["get_cached_kernel_dir"].assert_called_once_with("computed")

    def test_no_stored_key_hashes(self):
        _, mocks, _ = self._sdsc("1", None, cached_dir="/c")
        mocks["compute_specs_hash"].assert_called_once()
        mocks["get_cached_kernel_dir"].assert_called_once_with("computed")

    def test_fingerprint_mismatch_hashes(self):
        # Stored for another pool size.
        _, mocks, _ = self._sdsc(
            "1", self._stored(pool_size=0), cached_dir="/c", pool_size=4096
        )
        mocks["compute_specs_hash"].assert_called_once()
        # Stored under another config.
        with _env(sencores=spyre_config.sencores + 1):
            stored = self._stored()()
        _, mocks, _ = self._sdsc("1", stored, cached_dir="/c")
        mocks["compute_specs_hash"].assert_called_once()
        mocks["get_cached_kernel_dir"].assert_called_once_with("computed")

    def test_kernel_cache_off_ignores_stored_key(self):
        compiler = async_compile_mod.SpyreAsyncCompile()
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        with (
            _env(precomputed_kernel_key="1", spyre_kernel_cache=False),
            patch.object(async_compile_mod, "find_unimplemented", return_value=None),
            patch.object(
                async_compile_mod,
                "build_kernel_provenance_descriptor",
                return_value=None,
            ),
            patch.object(async_compile_mod, "get_output_dir", return_value=tmp),
            patch.object(async_compile_mod, "generate_bundle", return_value=[]),
            patch.object(async_compile_mod, "_run_backend_compiler"),
            patch.object(
                async_compile_mod, "SpyreSDSCKernelRunner", side_effect=_runner
            ),
            patch.object(
                async_compile_mod.SpyreAsyncCompile, "_kernel_cache_key"
            ) as method,
            torch._inductor.config.patch({"compile_threads": 1}),
        ):
            runner = compiler.sdsc("sdsc_k_0", [], precomputed_key="pkey|x")
        method.assert_not_called()
        self.assertEqual(runner[1], tmp)

    def test_verify_match(self):
        runner, mocks, _ = self._sdsc(
            "verify", self._stored(), cached_dir="/cache/pkey", computed="pkey"
        )
        mocks["compute_specs_hash"].assert_called_once()
        mocks["get_cached_kernel_dir"].assert_called_once_with("pkey")
        self.assertEqual(runner[1], "/cache/pkey")

    def test_verify_mismatch_raises(self):
        with self.assertRaises(PrecomputedKernelKeyMismatch):
            self._sdsc("verify", self._stored(), cached_dir="/c", computed="other")

    def test_verify_without_stored_key_hashes(self):
        _, mocks, _ = self._sdsc("verify", None, cached_dir="/c")
        mocks["compute_specs_hash"].assert_called_once()


# ---------------------------------------------------------------------------
# Generated code
# ---------------------------------------------------------------------------


class TestEmittedSdscCall(unittest.TestCase):
    def _define(self, src, mode, *, uses_pool=False, pool_size=0, **config):
        wrapper = MagicMock()
        wrapper.src_to_kernel = {}
        wrapper.next_kernel_suffix.return_value = "0"
        graph = MagicMock()
        graph.wrapper_code = wrapper
        kernel = MagicMock()
        kernel._kernel_uses_hbm_pool.return_value = uses_pool
        kernel.pool_size = pool_size
        with (
            _env(precomputed_kernel_key=mode, ktir_emitter=False, **config),
            V.set_graph_handler(graph),
            patch.object(
                scheduler_mod, "get_fused_kernel_name", return_value="fused_add"
            ),
            patch.object(scheduler_mod, "get_kernel_metadata", return_value=("", "")),
        ):
            name = scheduler_mod.SuperDSCScheduling.define_kernel(
                MagicMock(), src, [], kernel
            )
        wrapper.define_kernel.assert_called_once()
        self.assertEqual(wrapper.define_kernel.call_args.args[0], name)
        return name, wrapper.define_kernel.call_args.args[1]

    @staticmethod
    def _call_kwargs(code):
        """Evaluate a generated sdsc(...) definition the way the wrapper does."""
        calls = []

        class _FakeAsyncCompile:
            def sdsc(self, kernel_name, specs, pool_size=0, precomputed_key=None):
                calls.append((kernel_name, specs, pool_size, precomputed_key))

        namespace = spec_eval_namespace()
        namespace["async_compile"] = _FakeAsyncCompile()
        eval(code, namespace)  # noqa: S307
        assert len(calls) == 1
        return calls[0]

    def test_code_is_identical_for_every_switch_value(self):
        src = _src(_spec_sets()["two_ops"])
        for uses_pool, pool_size in ((False, 0), (True, 8192)):
            codes = {
                mode: self._define(src, mode, uses_pool=uses_pool, pool_size=pool_size)
                for mode in ("0", "1", "verify")
            }
            with self.subTest(pool=pool_size):
                self.assertEqual(len(set(codes.values())), 1, codes)

    def test_emitted_key_is_the_runtime_key(self):
        src = _src(_spec_sets()["loop"])
        for uses_pool, pool_size in ((False, 0), (True, 8192)):
            with self.subTest(pool=pool_size):
                name, code = self._define(
                    src, "0", uses_pool=uses_pool, pool_size=pool_size
                )
                self.assertIn("precomputed_key='", code)
                self.assertEqual(f"pool_size={pool_size}" in code, uses_pool)
                with _env():
                    kernel_name, specs, passed_pool, stored = self._call_kwargs(code)
                    self.assertEqual(kernel_name, name)
                    self.assertEqual(passed_pool, pool_size)
                    expected = compute_specs_hash(
                        specs, kernel_name=name, pool_size=passed_pool
                    )
                    fingerprint = kernel_key_fingerprint(pool_size)
                    self.assertEqual(stored, f"{expected}|{fingerprint}")
                    # End to end through sdsc()'s key resolution.
                    for mode in ("1", "verify"):
                        with (
                            spyre_config.patch(precomputed_kernel_key=mode),
                            patch.object(
                                async_compile_mod,
                                "compute_specs_hash",
                                wraps=compute_specs_hash,
                            ) as hashed,
                        ):
                            got = async_compile_mod.SpyreAsyncCompile._kernel_cache_key(
                                MagicMock(),
                                kernel_name,
                                specs,
                                passed_pool,
                                spyre_config.backend_loop_unroll,
                                stored,
                            )
                        self.assertEqual(got, expected)
                        self.assertEqual(hashed.call_count, 0 if mode == "1" else 1)

    def test_kernel_cache_off_emits_none(self):
        _, code = self._define(
            _src(_spec_sets()["single"]), "1", spyre_kernel_cache=False
        )
        self.assertIn(", precomputed_key=None)", code)
        self.assertIsNone(self._call_kwargs(code)[3])

    def test_dedup_emits_once(self):
        src = _src(_spec_sets()["single"])
        wrapper = MagicMock()
        wrapper.src_to_kernel = {src: "sdsc_existing_0"}
        graph = MagicMock()
        graph.wrapper_code = wrapper
        with _env(), V.set_graph_handler(graph):
            name = scheduler_mod.SuperDSCScheduling.define_kernel(
                MagicMock(), src, [], MagicMock()
            )
        self.assertEqual(name, "sdsc_existing_0")
        wrapper.define_kernel.assert_not_called()


class TestSwitchValues(unittest.TestCase):
    _VAR = "TORCH_SPYRE_PRECOMPUTED_KERNEL_KEY"

    def _parse(self, value):
        env = dict(os.environ)
        env.pop(self._VAR, None)
        if value is not None:
            env[self._VAR] = value
        with patch.dict(os.environ, env, clear=True):
            return spyre_config._precomputed_kernel_key_from_env()

    def test_known_values(self):
        cases = {
            None: "0",
            "": "0",
            "0": "0",
            "false": "0",
            "no": "0",
            "1": "1",
            "true": "1",
            " YES ": "1",
            "verify": "verify",
            "Verify": "verify",
        }
        for value, expected in cases.items():
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                self.assertEqual(self._parse(value), expected, value)

    def test_unknown_value_warns_and_stays_off(self):
        for value in ("on", "2", "verfy"):
            with self.assertWarnsRegex(UserWarning, self._VAR):
                self.assertEqual(self._parse(value), "0", value)


if __name__ == "__main__":
    unittest.main()
