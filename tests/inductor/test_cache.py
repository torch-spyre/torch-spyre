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

import os
import pathlib
import shutil
import unittest
from unittest.mock import patch

import torch
from torch._inductor.utils import fresh_cache
from torch._dynamo.utils import counters

from torch_spyre.execution import async_compile as _async_compile
from torch_spyre.execution import kernel_cache


def _capture_op_specs(shape=(64, 64), dtype=torch.float16):
    """Return the real ``OpSpec`` list a ``torch.compile`` of ``abs`` emits.

    ``compute_specs_hash`` replays the full ``compile_op_spec`` ->
    ``parse_op_spec`` -> ``generate_sdsc`` pipeline, which is far stricter than
    a hand-built ``TensorArg`` can satisfy (this is exactly why
    ``test_kernel_provenance.py`` disables the kernel cache for its synthetic
    specs).  So the specs are captured from an actual compile instead of being
    constructed, mirroring the ``torch.abs`` / ``.to("spyre")`` pattern the
    rest of this file already uses.
    """
    captured = []
    original_sdsc = _async_compile.SpyreAsyncCompile.sdsc

    def _spy(self, kernel_name, specs):
        captured.append(list(specs))
        return original_sdsc(self, kernel_name, specs)

    with patch.object(_async_compile.SpyreAsyncCompile, "sdsc", _spy):
        tensor = torch.randn(shape, dtype=dtype).to("spyre")
        torch._dynamo.reset()
        torch.compile(torch.abs, dynamic=False)(tensor)

    assert captured, "no sdsc() call was observed while capturing OpSpecs"
    return captured[0]


class TestCache(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_cache(self):
        counters.clear()
        a = torch.randn((64, 64)).to("spyre")
        fn = torch.compile(torch.abs, dynamic=False)
        with fresh_cache():
            fn(a)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())
        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        torch._dynamo.reset()

        with fresh_cache():
            torch.compiler.load_cache_artifacts(artifact_bytes)
            fn(a)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())

    def test_cache_key_includes_spyre_layout(self):
        """
        Verify that FxGraphHashDetails includes SpyreTensorLayout in the cache key.
        Different layouts should produce different cache keys, preventing incorrect
        cache hits across layout changes.
        """
        from torch.spyre import SpyreTensorLayout

        x = torch.rand([64, 64], dtype=torch.float16)
        stl_a = SpyreTensorLayout(
            list(x.size()), list(x.stride()), torch.float16, [0, 1]
        )
        stl_b = SpyreTensorLayout(
            list(x.size()), list(x.stride()), torch.float16, [1, 0]
        )

        _ = x.to("spyre")  # wake up spyre
        tensor_a = x.to(device_layout=stl_a)
        tensor_b = x.to(device_layout=stl_b)

        fn = torch.compile(lambda a: a + a, dynamic=False)

        # ── Layout A — first compile, cache miss expected ─────────────────────────
        counters.clear()
        with fresh_cache():
            fn(tensor_a)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            artifacts_a = torch.compiler.save_cache_artifacts()
            self.assertIsNotNone(artifacts_a)
            artifact_bytes_a, _ = artifacts_a

        # ── Layout B — different layout, should NOT hit Layout A cache ────────────
        torch._dynamo.reset()
        counters.clear()
        with fresh_cache():
            # load Layout A artifact into cache
            torch.compiler.load_cache_artifacts(artifact_bytes_a)

            # compile with Layout B — should miss (different layout → different key)
            fn(tensor_b)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"],
                1,
                "Layout B should miss Layout A cache — different SpyreTensorLayout",
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"],
                0,
                "Layout B should not hit Layout A cache — SpyreTensorLayout differs",
            )

        # ── Layout A again — should hit its own cache ─────────────────────────────
        torch._dynamo.reset()
        counters.clear()
        with fresh_cache():
            torch.compiler.load_cache_artifacts(artifact_bytes_a)
            fn(tensor_a)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"],
                1,
                "Layout A should hit its own cache",
            )


class TestSpyreKernelCache(unittest.TestCase):
    """Unit tests for the persistent SDSC kernel-artifact cache.

    Everything here exercises ``torch_spyre.execution.kernel_cache`` directly.
    The cache root is derived from Inductor's ``cache_dir()``, so wrapping a
    test in ``fresh_cache()`` (which sets ``TORCHINDUCTOR_CACHE_DIR``) gives it
    a private cache root without touching the developer's real cache.
    """

    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)
        self._clear_kernel_cache_lru_caches()

    def tearDown(self):
        # Env-var-derived values are memoized, so a mutated env must not leak a
        # cached read into the next test.
        self._clear_kernel_cache_lru_caches()
        super().tearDown()

    @staticmethod
    def _clear_kernel_cache_lru_caches():
        kernel_cache._get_spyre_library_versions.cache_clear()
        kernel_cache._get_system_info.cache_clear()
        kernel_cache._get_compile_config.cache_clear()
        kernel_cache._iter_source_files.cache_clear()
        kernel_cache._get_torch_spyre_source_hash.cache_clear()

    # ------------------------------------------------------------------
    # Group A — cache-key sensitivity
    # ------------------------------------------------------------------

    def test_cache_key_changes_with_source_edit(self):
        """A codegen source edit must invalidate the key.

        This is the guarantee ``_get_torch_spyre_source_hash`` exists to
        provide: emitter changes downstream of ``compile_op_spec`` can alter
        ``bundle.mlir`` without changing any replayed value, so the key has to
        cover the sources themselves.
        """
        specs = _capture_op_specs()
        baseline = kernel_cache.compute_specs_hash(specs)

        edited = kernel_cache._package_root() / "_inductor" / "codegen" / "bundle.py"
        original_bytes = edited.read_bytes()
        try:
            edited.write_bytes(original_bytes + b"\n# kernel-cache test marker\n")
            self._clear_kernel_cache_lru_caches()
            mutated = kernel_cache.compute_specs_hash(specs)
        finally:
            edited.write_bytes(original_bytes)
            self._clear_kernel_cache_lru_caches()

        self.assertNotEqual(
            baseline,
            mutated,
            "editing a hashed codegen source must change the cache key",
        )
        # And the key must come back once the edit is reverted, otherwise the
        # hash depends on something other than file content.
        self.assertEqual(baseline, kernel_cache.compute_specs_hash(specs))

    def test_cache_key_changes_with_native_extension(self):
        """A C++ rebuild must invalidate the key even with no .py change."""
        specs = _capture_op_specs()
        baseline = kernel_cache.compute_specs_hash(specs)

        real_root = kernel_cache._package_root()
        with fresh_cache():
            fake_root = pathlib.Path(kernel_cache.get_cache_root_dir()) / "fake_pkg"
            # Mirror the real package tree so _iter_source_files still resolves
            # every hashed source, then vary only the extension's bytes.
            shutil.copytree(
                real_root,
                fake_root,
                ignore=shutil.ignore_patterns("__pycache__", "*.so"),
            )
            (fake_root / "_C.so").write_bytes(b"different native extension bytes")

            self._clear_kernel_cache_lru_caches()
            with patch.object(kernel_cache, "_package_root", return_value=fake_root):
                mutated = kernel_cache.compute_specs_hash(specs)

        self._clear_kernel_cache_lru_caches()
        self.assertNotEqual(
            baseline, mutated, "different _C.so bytes must change the cache key"
        )

    def test_cache_key_changes_with_lib_version_file(self):
        """A Spyre toolchain version bump must invalidate the key."""
        specs = _capture_op_specs()
        baseline = kernel_cache.compute_specs_hash(specs)

        with fresh_cache():
            lib_file = pathlib.Path(kernel_cache.get_cache_root_dir()) / "libs.txt"
            lib_file.write_text(
                "ibm-deeptools:0.0.0-test\nibm-senlib-core:0.0.0-test\n"
            )
            with patch.dict(os.environ, {"LIB_VERSION_FILE": str(lib_file)}):
                self._clear_kernel_cache_lru_caches()
                mutated = kernel_cache.compute_specs_hash(specs)

        self._clear_kernel_cache_lru_caches()
        self.assertNotEqual(
            baseline, mutated, "different Spyre library versions must change the key"
        )

    def test_cache_key_changes_with_torch_version(self):
        """A PyTorch upgrade must invalidate the key."""
        specs = _capture_op_specs()
        baseline = kernel_cache.compute_specs_hash(specs)

        with patch.object(torch, "__version__", f"{torch.__version__}.test"):
            mutated = kernel_cache.compute_specs_hash(specs)

        self.assertNotEqual(baseline, mutated)

    def test_cache_key_changes_with_op_spec_structure(self):
        """Different op structure (shape) must produce a different key."""
        baseline = kernel_cache.compute_specs_hash(_capture_op_specs(shape=(64, 64)))
        mutated = kernel_cache.compute_specs_hash(_capture_op_specs(shape=(128, 128)))
        self.assertNotEqual(baseline, mutated)

    def test_cache_key_changes_with_system_info(self):
        """Device/topology env vars are part of the key."""
        specs = _capture_op_specs()
        baseline = kernel_cache.compute_specs_hash(specs)

        different = "PF_TEST" if os.environ.get("FLEX_DEVICE") != "PF_TEST" else "AIU"
        with patch.dict(os.environ, {"FLEX_DEVICE": different}):
            kernel_cache._get_system_info.cache_clear()
            mutated = kernel_cache.compute_specs_hash(specs)

        kernel_cache._get_system_info.cache_clear()
        self.assertNotEqual(baseline, mutated)

    def test_cache_key_changes_with_compile_config(self):
        """Compiler-option env vars are part of the key."""
        specs = _capture_op_specs()
        baseline = kernel_cache.compute_specs_hash(specs)

        current = os.environ.get("LAYOUT_SOLVER", "greedy")
        different = "firstfit" if current != "firstfit" else "bestfit"
        with patch.dict(os.environ, {"LAYOUT_SOLVER": different}):
            kernel_cache._get_compile_config.cache_clear()
            mutated = kernel_cache.compute_specs_hash(specs)

        kernel_cache._get_compile_config.cache_clear()
        self.assertNotEqual(baseline, mutated)

    def test_cache_key_stable_across_calls(self):
        """The key must be deterministic for unchanged inputs."""
        specs = _capture_op_specs()
        self.assertEqual(
            kernel_cache.compute_specs_hash(specs),
            kernel_cache.compute_specs_hash(specs),
        )

    def test_cache_key_excludes_runtime_hbm_addresses(self):
        """The launch-supplied HBM base address must not reach the key.

        This is the assertion protecting the hit rate.  A kernel tensor's real
        base address is passed at launch as an ``!sdscbundle.input_arg``, so two
        runs that place the same tensor at different addresses have to share one
        cache entry; hashing the address would make every key allocation-
        specific.  The guarantee holds structurally: at compile time the base is
        the symbolic placeholder ``("kernel", arg_index)``, never a concrete
        address, and ``_symbol_kind_key`` keeps only the classification --
        ``kind``/``arg_index``/``offset`` -- while the parallel
        ``base_symbol_values`` entry that would carry an address is never hashed
        at all.
        """
        from torch_spyre._inductor.codegen.superdsc import compile_op_spec
        from torch_spyre._inductor.op_spec import LoopSpec, OpSpec

        specs = _capture_op_specs()

        def _flatten(entries):
            for entry in entries:
                if isinstance(entry, LoopSpec):
                    yield from _flatten(entry.body)
                elif isinstance(entry, OpSpec):
                    yield entry

        symbols: list = []
        symbol_id_offset = 0
        saw_kernel_symbol = False
        for idx, spec in enumerate(_flatten(specs)):
            _, base_symbol_values, _, symbol_kinds = compile_op_spec(
                idx, spec, symbols, symbol_id_offset
            )
            symbol_id_offset += len(base_symbol_values)

            for value, kind in zip(base_symbol_values, symbol_kinds):
                if kind.kind != "kernel":
                    continue
                saw_kernel_symbol = True
                # The value is a symbolic placeholder, not an address...
                self.assertEqual(
                    value,
                    ("kernel", kind.arg_index),
                    "a kernel base symbol must stay symbolic at compile time",
                )
                # ...and the hashed projection of the symbol drops the value
                # entirely, keeping only the address-independent structure.
                self.assertNotIn(value, kernel_cache._symbol_kind_key(kind))

        self.assertTrue(
            saw_kernel_symbol,
            "expected at least one launch-supplied kernel base symbol",
        )

    # ------------------------------------------------------------------
    # Group B — import-closure walker
    # ------------------------------------------------------------------

    def test_iter_source_files_contains_seeds_and_transitive_deps(self):
        files = kernel_cache._iter_source_files()

        self.assertEqual(list(files), sorted(files), "hashed paths must be sorted")
        self.assertEqual(len(files), len(set(files)), "hashed paths must be unique")

        seeds = {pathlib.PurePath(s) for s in kernel_cache._ARTIFACT_SOURCE_SEEDS}
        self.assertTrue(seeds.issubset(set(files)))

        # A known transitive dependency, reached only by following imports.
        self.assertIn(pathlib.PurePath("_inductor", "op_spec.py"), set(files))

        # Shrink detection: the closure is ~77 modules today.  A sudden drop
        # means the walker stopped following imports and the key silently got
        # weaker.
        self.assertGreaterEqual(len(files), 70)

    def test_iter_source_files_missing_seed_raises(self):
        with patch.object(
            kernel_cache, "_ARTIFACT_SOURCE_SEEDS", ["nonexistent/module.py"]
        ):
            kernel_cache._iter_source_files.cache_clear()
            with self.assertRaises(RuntimeError):
                kernel_cache._iter_source_files()
        kernel_cache._iter_source_files.cache_clear()

    def test_imported_modules_resolves_relative_imports(self):
        """Relative imports must resolve against the containing package.

        Note what the returned names are: for ``from X import y`` the walker
        yields ``X`` (the module imported *from*), not ``X.y``.  That is
        deliberate and sufficient -- ``from .foo import bar`` resolves to
        ``foo.py``, which is the file that needs hashing.  It does mean a bare
        ``from . import foo`` resolves only to the package ``__init__.py`` and
        not to ``foo.py``; no seed does that today, and a submodule reached
        *only* that way would have to be added as a seed.
        """
        import ast

        tree = ast.parse(
            "from . import sibling\n"
            "from .foo import bar\n"
            "from ..pkg.mod import baz\n"
            "from torch_spyre.absolute import qux\n"
            "import torch_spyre.plain\n"
        )
        modules = kernel_cache._imported_modules(tree, package="torch_spyre.x.y")

        # ``from .foo import bar`` -> the module ``torch_spyre.x.y.foo``.
        self.assertIn("torch_spyre.x.y.foo", modules)
        # ``from ..pkg.mod import baz`` -> one package up, then ``pkg.mod``.
        self.assertIn("torch_spyre.x.pkg.mod", modules)
        # ``from . import sibling`` -> the containing package itself.
        self.assertIn("torch_spyre.x.y", modules)
        # Absolute imports pass through untouched.
        self.assertIn("torch_spyre.absolute", modules)
        self.assertIn("torch_spyre.plain", modules)

    def test_imported_modules_ignores_non_torch_spyre_imports(self):
        """Only ``torch_spyre`` sources are hashable; the rest are version-keyed."""
        import ast

        root = kernel_cache._package_root()
        tree = ast.parse("import torch\nfrom sympy import Symbol\nimport os\n")

        for module in kernel_cache._imported_modules(tree, package="torch_spyre"):
            self.assertIsNone(
                kernel_cache._module_to_relpath(module, root),
                f"{module} is outside torch_spyre and must not be hashed as source",
            )

    # ------------------------------------------------------------------
    # Group C — artifact validation
    # ------------------------------------------------------------------

    @staticmethod
    def _populate_entry(cache_root, key, artifacts, with_sdsc=True):
        entry = pathlib.Path(cache_root) / key
        for rel in artifacts:
            target = entry / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.touch()
        entry.mkdir(parents=True, exist_ok=True)
        if with_sdsc:
            (entry / "sdsc_0.json").write_text("{}")
        return entry

    def test_get_cached_kernel_dir_returns_complete_entry(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            entry = self._populate_entry(
                root, "key_complete", kernel_cache._REQUIRED_ARTIFACTS
            )
            self.assertEqual(
                kernel_cache.get_cached_kernel_dir("key_complete"), str(entry)
            )

    def test_get_cached_kernel_dir_rejects_incomplete_entries(self):
        required = list(kernel_cache._REQUIRED_ARTIFACTS)
        for omitted in required:
            with self.subTest(missing=omitted), fresh_cache():
                root = kernel_cache.get_cache_root_dir()
                present = [a for a in required if a != omitted]
                self._populate_entry(root, "key_partial", present)
                self.assertIsNone(
                    kernel_cache.get_cached_kernel_dir("key_partial"),
                    f"a bundle missing {omitted} must not be served as a hit",
                )

    def test_get_cached_kernel_dir_rejects_entry_without_sdsc_json(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            self._populate_entry(
                root, "key_no_sdsc", kernel_cache._REQUIRED_ARTIFACTS, with_sdsc=False
            )
            self.assertIsNone(kernel_cache.get_cached_kernel_dir("key_no_sdsc"))

    def test_get_cached_kernel_dir_missing_entry(self):
        with fresh_cache():
            self.assertIsNone(kernel_cache.get_cached_kernel_dir("key_absent"))

    # ------------------------------------------------------------------
    # Group D — allocate / commit / failed-bundle retention
    # ------------------------------------------------------------------

    def test_allocate_compile_dir_is_inside_cache_root(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            tmp_dir = kernel_cache.allocate_compile_dir("key_alloc")

            self.assertEqual(os.path.dirname(tmp_dir), root)
            self.assertTrue(os.path.isdir(tmp_dir))
            # Same filesystem as the final entry is what makes the commit an
            # atomic rename rather than a copy.
            self.assertEqual(os.stat(tmp_dir).st_dev, os.stat(root).st_dev)

    def test_commit_compile_dir_happy_path(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            tmp_dir = kernel_cache.allocate_compile_dir("key_commit")
            pathlib.Path(tmp_dir, "marker").write_text("payload")

            committed = kernel_cache.commit_compile_dir(tmp_dir, "key_commit")

            self.assertEqual(committed, os.path.join(root, "key_commit"))
            self.assertEqual(pathlib.Path(committed, "marker").read_text(), "payload")
            self.assertFalse(os.path.exists(tmp_dir))

    def test_commit_compile_dir_race_existing_destination(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            winner = pathlib.Path(root, "key_race")
            winner.mkdir()
            (winner / "marker").write_text("winner")

            loser_dir = kernel_cache.allocate_compile_dir("key_race")
            pathlib.Path(loser_dir, "marker").write_text("loser")

            committed = kernel_cache.commit_compile_dir(loser_dir, "key_race")

            self.assertEqual(committed, str(winner))
            self.assertEqual((winner / "marker").read_text(), "winner")
            self.assertFalse(
                os.path.exists(loser_dir), "the loser's temp dir must be removed"
            )

    def test_commit_compile_dir_race_oserror_fallback(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            tmp_dir = kernel_cache.allocate_compile_dir("key_rename_fail")

            with patch.object(kernel_cache.os, "rename", side_effect=OSError("boom")):
                committed = kernel_cache.commit_compile_dir(tmp_dir, "key_rename_fail")

            # No exception propagates: a lost rename race is a normal outcome.
            self.assertEqual(committed, os.path.join(root, "key_rename_fail"))
            self.assertFalse(os.path.exists(tmp_dir))

    def test_retain_failed_compile_dir_happy_path(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            tmp_dir = kernel_cache.allocate_compile_dir("key_failed")
            pathlib.Path(tmp_dir, "bundle.mlir").write_text("broken bundle")

            retained = kernel_cache.retain_failed_compile_dir(tmp_dir, "key_failed")

            self.assertIsNotNone(retained)
            failed_root = os.path.join(root, kernel_cache._FAILED_DIR_NAME)
            self.assertEqual(os.path.dirname(retained), failed_root)
            self.assertTrue(os.path.basename(retained).startswith("key_failed."))
            # The artifacts are the point of retaining: dxp_standalone -d <dir>
            # on them is how the failure gets debugged.
            self.assertEqual(
                pathlib.Path(retained, "bundle.mlir").read_text(), "broken bundle"
            )
            self.assertFalse(os.path.exists(tmp_dir))
            # And the commit namespace is left clean.
            self.assertEqual(
                [d for d in os.listdir(root) if ".tmp." in d],
                [],
                "no half-written bundle may be left in the cache root",
            )

    def test_retain_failed_compile_dir_oserror_returns_none(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            tmp_dir = kernel_cache.allocate_compile_dir("key_retain_fail")

            with patch.object(kernel_cache.os, "rename", side_effect=OSError("boom")):
                retained = kernel_cache.retain_failed_compile_dir(
                    tmp_dir, "key_retain_fail"
                )

            self.assertIsNone(retained)
            # Load-bearing: retention is best-effort, but removing the temp dir
            # is not.  If it survived here, get_cached_kernel_dir could later
            # observe a half-written bundle in the commit namespace.
            self.assertFalse(os.path.exists(tmp_dir))
            self.assertEqual([d for d in os.listdir(root) if ".tmp." in d], [])

    def test_retain_failed_compile_dir_unique_per_failure(self):
        with fresh_cache():
            first_tmp = kernel_cache.allocate_compile_dir("key_repeat")
            pathlib.Path(first_tmp, "marker").write_text("first")
            first = kernel_cache.retain_failed_compile_dir(first_tmp, "key_repeat")

            second_tmp = kernel_cache.allocate_compile_dir("key_repeat")
            pathlib.Path(second_tmp, "marker").write_text("second")
            second = kernel_cache.retain_failed_compile_dir(second_tmp, "key_repeat")

            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertNotEqual(first, second)
            # A repeatedly-failing key must not overwrite its own evidence.
            self.assertEqual(pathlib.Path(first, "marker").read_text(), "first")
            self.assertEqual(pathlib.Path(second, "marker").read_text(), "second")

    # ------------------------------------------------------------------
    # Group E — LIB_VERSION_FILE handling
    # ------------------------------------------------------------------

    def test_get_spyre_library_versions_requires_env_var(self):
        with patch.dict(os.environ):
            os.environ.pop("LIB_VERSION_FILE", None)
            kernel_cache._get_spyre_library_versions.cache_clear()
            with self.assertRaises(RuntimeError):
                kernel_cache._get_spyre_library_versions()
        kernel_cache._get_spyre_library_versions.cache_clear()

    def test_get_spyre_library_versions_missing_file_raises(self):
        with fresh_cache():
            missing = os.path.join(kernel_cache.get_cache_root_dir(), "absent.txt")
            with patch.dict(os.environ, {"LIB_VERSION_FILE": missing}):
                kernel_cache._get_spyre_library_versions.cache_clear()
                with self.assertRaises(RuntimeError):
                    kernel_cache._get_spyre_library_versions()
        kernel_cache._get_spyre_library_versions.cache_clear()

    def test_get_spyre_library_versions_skips_malformed_lines(self):
        with fresh_cache():
            lib_file = pathlib.Path(kernel_cache.get_cache_root_dir()) / "libs.txt"
            lib_file.write_text(
                "deeptools:1.2.3\nthis line has no colon\n\n  ibm-senlib : 4.5.6  \n"
            )
            with patch.dict(os.environ, {"LIB_VERSION_FILE": str(lib_file)}):
                kernel_cache._get_spyre_library_versions.cache_clear()
                versions = kernel_cache._get_spyre_library_versions()

        kernel_cache._get_spyre_library_versions.cache_clear()
        self.assertEqual(versions, {"deeptools": "1.2.3", "ibm-senlib": "4.5.6"})

    # ------------------------------------------------------------------
    # Group F — stats / clear
    # ------------------------------------------------------------------

    def test_cache_stats_and_clear(self):
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()

            # Two committed entries, with known payload sizes.
            for key in ("key_a", "key_b"):
                tmp_dir = kernel_cache.allocate_compile_dir(key)
                pathlib.Path(tmp_dir, "bundle.mlir").write_bytes(b"x" * 1024)
                kernel_cache.commit_compile_dir(tmp_dir, key)

            # One retained failure.  Each non-entry below gets a distinct
            # payload size, so a regression that wrongly counts one of them
            # cannot land on the correct total by coincidence.
            failed_tmp = kernel_cache.allocate_compile_dir("key_failed")
            pathlib.Path(failed_tmp, "bundle.mlir").write_bytes(b"y" * 4096)
            self.assertIsNotNone(
                kernel_cache.retain_failed_compile_dir(failed_tmp, "key_failed")
            )

            # One in-flight compile dir that was never committed.
            in_flight = kernel_cache.allocate_compile_dir("key_in_flight")
            pathlib.Path(in_flight, "bundle.mlir").write_bytes(b"z" * 2048)

            # A stray file directly in the cache root: not an entry either.
            pathlib.Path(root, "stray.log").write_bytes(b"w" * 512)

            stats = kernel_cache.get_cache_stats()

            self.assertEqual(
                stats["total_cached_kernels"],
                2,
                "in-flight .tmp. dirs and failed/ are not cache entries",
            )
            self.assertEqual(stats["retained_failed_compiles"], 1)

            # Both statistics must describe the same set.  ``cache_size_mb`` is
            # read as "what clearing the cache would reclaim as cache content",
            # so it counts committed entries only -- not retained failures, not
            # a concurrent compile's in-flight dir, not stray root files.
            self.assertAlmostEqual(
                stats["cache_size_mb"],
                (2 * 1024) / (1024 * 1024),
                places=6,
                msg="cache_size_mb must count exactly the dirs "
                "total_cached_kernels counts",
            )

            kernel_cache.clear_cache()

            self.assertTrue(os.path.isdir(root))
            self.assertEqual(os.listdir(root), [])
            cleared = kernel_cache.get_cache_stats()
            self.assertEqual(cleared["total_cached_kernels"], 0)
            self.assertEqual(cleared["cache_size_mb"], 0.0)
            self.assertEqual(cleared["retained_failed_compiles"], 0)

    def test_cache_stats_reports_all_fields_when_root_absent(self):
        """The key set must not depend on whether the cache root exists yet.

        ``get_cache_stats`` has an early return for a missing root; a caller
        reading ``retained_failed_compiles`` should not have to guard against a
        KeyError just because nothing has been compiled yet.
        """
        with fresh_cache():
            root = kernel_cache.get_cache_root_dir()
            populated = set(kernel_cache.get_cache_stats())

            # get_cache_root_dir() recreates the root, so query the missing-root
            # branch with it patched to a path that does not exist.
            absent = os.path.join(root, "does_not_exist")
            with patch.object(kernel_cache, "get_cache_root_dir", return_value=absent):
                stats = kernel_cache.get_cache_stats()

        self.assertEqual(set(stats), populated)
        self.assertEqual(
            stats,
            {
                "total_cached_kernels": 0,
                "cache_size_mb": 0.0,
                "retained_failed_compiles": 0,
            },
        )

    # ------------------------------------------------------------------
    # Step 2 — end-to-end
    # ------------------------------------------------------------------

    def test_sdsc_cache_hit_numerics(self):
        """A second run served from the kernel cache must be numerically correct.

        Both runs share one cache root, so the second compile takes the cache-hit
        path in ``SpyreAsyncCompile.sdsc`` and skips ``generate_bundle`` and
        ``dxp_standalone`` entirely.  Comparing against eager is what makes this
        a correctness test rather than a plumbing test: a key that collided
        across genuinely different kernels would show up here as wrong numbers.
        """
        source = torch.randn((8, 8), dtype=torch.float16)
        expected = torch.abs(source)

        with fresh_cache():
            device_tensor = source.to("spyre")

            torch._dynamo.reset()
            first = torch.compile(torch.abs, dynamic=False)(device_tensor)
            first_cpu = first.cpu()

            stats_after_first = kernel_cache.get_cache_stats()
            self.assertGreaterEqual(
                stats_after_first["total_cached_kernels"],
                1,
                "the first compile should have populated the kernel cache",
            )

            torch._dynamo.reset()
            second = torch.compile(torch.abs, dynamic=False)(device_tensor)
            second_cpu = second.cpu()

            # No new cache entry: the second compile was served from the first.
            self.assertEqual(
                kernel_cache.get_cache_stats()["total_cached_kernels"],
                stats_after_first["total_cached_kernels"],
            )

        # The load-bearing assertion: the cached run must agree with the freshly
        # compiled run *exactly*.  A key that collided across genuinely
        # different kernels would serve the wrong bundle and show up here, and
        # unlike the eager comparison below this has no tolerance to hide in --
        # both values come from the same device kernel, so any difference at all
        # means the second run executed something else.
        torch.testing.assert_close(
            second_cpu,
            first_cpu,
            atol=0.0,
            rtol=0.0,
            msg="a kernel served from the cache must match the freshly compiled run",
        )

        # And both must be numerically right, at the fp16 device-vs-CPU
        # tolerance this repo uses elsewhere -- so a cache that consistently
        # returns the same wrong answer still fails.
        for actual in (first_cpu, second_cpu):
            torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
