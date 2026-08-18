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

import json
import os
import stat
import tempfile
from pathlib import Path

import pytest

from torch_spyre import make_spyre_module  # type: ignore[attr-defined]
from torch_spyre.profiler import get_diagnostic_report as profiler_get_diagnostic_report
from torch_spyre.profiler._ffdc import (
    CATEGORY_COMPILE_BACKEND,
    CATEGORY_COMPILE_FRONTEND,
    CATEGORY_RUNTIME_LAUNCH,
    CATEGORY_UNIMPLEMENTED,
    CATEGORY_UNKNOWN,
    _call_with_timeout,
    _default_output_dir,
    _MAX_REPORTS,
    _prune_old_reports,
    _report_sort_key,
    REQUIRED_FIELDS,
    collect,
    get_diagnostic_report,
    try_collect,
)

_VALID_REPORT_NAME = "ffdc_unknown_20250101T000001_000000_1.json"
_VALID_REPORT_JSON = f'{{"failure": {{"category": "{CATEGORY_UNKNOWN}"}}}}'
_NEWEST_VALID_REPORT_NAME = "ffdc_compile_20250101T000002_000000_1.json"


class _FrozenExc(Exception):
    def __setattr__(self, name, value):
        if name.startswith("_torch_spyre"):
            raise AttributeError(name)
        super().__setattr__(name, value)


class _UnhashableExc(Exception):
    def __hash__(self):
        raise RuntimeError("hash boom")


def _assert_one_backend_report(directory: str | Path) -> None:
    names = sorted(p.name for p in Path(directory).glob("ffdc_*.json"))
    assert len(names) == 1
    assert names[0].startswith("ffdc_compile_backend_")


def _write_ffdc_report(directory: Path, name: str, payload: str | bytes) -> Path:
    path = directory / name
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(payload)
    return path


def _assert_get_diagnostic_report_skips_newest(
    *,
    newest_name: str,
    newest_payload: str | bytes,
    get_report=get_diagnostic_report,
) -> None:
    """Assert a newer unusable report is skipped in favour of an older valid one.

    The invalid/newest fixture is written second and given a later st_mtime so an
    mtime-based implementation cannot pass by accidentally selecting the valid
    file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        # Write valid first so write order alone would give it the older mtime;
        # then pin mtimes explicitly (same pattern as the across-categories test).
        valid = _write_ffdc_report(d, _VALID_REPORT_NAME, _VALID_REPORT_JSON)
        newest = _write_ffdc_report(d, newest_name, newest_payload)
        os.utime(valid, (0, 0))
        os.utime(newest, (100, 100))
        result = get_report(output_dir=tmp)
        assert result is not None
        assert result["failure"]["category"] == CATEGORY_UNKNOWN
        assert result["_report_path"] == str(valid.resolve())


@pytest.fixture(autouse=True)
def _enable_ffdc(monkeypatch):
    monkeypatch.setenv("TORCH_SPYRE_FFDC", "1")


def _stub_module(monkeypatch, name, **attrs):
    """Insert a stub module; ``monkeypatch`` restores ``sys.modules`` after the test."""
    import sys
    import types

    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    return mod


def _patch_collect_raises(monkeypatch):
    """Force ``collect`` to raise so call sites exercise ``try_collect``."""
    import importlib

    ffdc_mod = importlib.import_module("torch_spyre.profiler._ffdc")

    def boom(*_args, **_kwargs):
        raise OSError("ffdc write failed")

    monkeypatch.setattr(ffdc_mod, "collect", boom)
    return ffdc_mod


def _reimport(monkeypatch, name):
    """Drop ``name`` from ``sys.modules`` and reimport; restore after the test."""
    import importlib
    import sys

    monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module(name)


class TestFfdcCollect:
    def _collect_to_tmpdir(self, exc=None, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            report = collect(exc, output_dir=tmp, **kwargs)
            # verify the JSON file was written and is valid
            path = report.get("_report_path")
            assert path is not None
            with open(path, encoding="utf-8") as f:
                on_disk = json.load(f)
            assert on_disk["failure"]["category"] == report["failure"]["category"]
        return report

    def test_collect_with_exception_is_complete(self):
        try:
            raise ValueError("test failure")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        assert report["collector"]["completeness_pct"] == 100.0
        assert report["collector"]["missing_fields"] == []
        assert report["collector"]["success"] is True

    def test_failure_fields_populated(self):
        try:
            raise RuntimeError("something went wrong")
        except RuntimeError as exc:
            report = self._collect_to_tmpdir(
                exc, failure_category=CATEGORY_COMPILE_FRONTEND
            )

        assert report["failure"]["category"] == CATEGORY_COMPILE_FRONTEND
        assert report["failure"]["exception_type"] == "RuntimeError"
        assert "something went wrong" in report["failure"]["message"]
        assert isinstance(report["failure"]["traceback"], str)
        assert "RuntimeError" in report["failure"]["traceback"]
        assert report["failure"]["file"] is not None
        assert isinstance(report["failure"]["lineno"], int)
        assert report["failure"]["lineno"] > 0

    def test_traceback_is_joined_string(self):
        try:
            raise TypeError("bad type")
        except TypeError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        tb = report["failure"]["traceback"]
        assert isinstance(tb, str)
        assert len(tb.splitlines()) > 1

    def test_runtime_context_passed_through(self):
        try:
            raise RuntimeError("kernel failed")
        except RuntimeError as exc:
            report = self._collect_to_tmpdir(
                exc,
                failure_category=CATEGORY_RUNTIME_LAUNCH,
                kernel_name="my_kernel",
                code_dir="/tmp/code",
            )

        assert report["runtime"]["kernel_name"] == "my_kernel"
        assert report["runtime"]["code_dir"] == "/tmp/code"

    def test_runtime_context_absent_is_none(self):
        try:
            raise RuntimeError("unimplemented")
        except RuntimeError as exc:
            report = self._collect_to_tmpdir(
                exc, failure_category=CATEGORY_UNIMPLEMENTED
            )

        assert report["runtime"]["kernel_name"] is None
        assert report["runtime"]["code_dir"] is None

    def test_call_with_timeout_raises_on_slow_work(self):
        import time

        with pytest.raises(TimeoutError):
            _call_with_timeout(lambda: time.sleep(5), 0.05)

    def test_collect_returns_early_when_disabled(self, monkeypatch):
        monkeypatch.setenv("TORCH_SPYRE_FFDC", "0")
        with tempfile.TemporaryDirectory() as tmp:
            report = collect(None, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)
        assert report["collector"]["disabled"] is True
        assert report["_report_path"] is None
        assert list(Path(tmp).glob("ffdc_*.json")) == []
        for key in (
            "metadata",
            "failure",
            "environment",
            "artifacts",
            "runtime",
            "hardware_state",
            "collector",
        ):
            assert key in report

    def test_use_spyre_profiler_does_not_enable_ffdc(self, monkeypatch):
        # FFDC must not share the CMake / Kineto USE_SPYRE_PROFILER build flag.
        monkeypatch.delenv("TORCH_SPYRE_FFDC", raising=False)
        monkeypatch.setenv("USE_SPYRE_PROFILER", "1")
        with tempfile.TemporaryDirectory() as tmp:
            report = collect(None, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)
        assert report["collector"].get("disabled") is True
        assert list(Path(tmp).glob("ffdc_*.json")) == []

    def test_collect_never_raises(self):
        # collect() must be best-effort; write failures must not propagate.
        # Use a plain file as output_dir so mkdir() raises NotADirectoryError —
        # a reliably unwritable path on every platform without root access.
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "not_a_dir"
            blocker.write_text("")  # create a file where a directory is expected
            report = collect(
                None,
                failure_category=CATEGORY_UNKNOWN,
                output_dir=str(blocker / "subdir"),
            )
        assert report is not None
        assert report["_report_path"] is None
        assert report["collector"]["success"] is False

    def test_try_collect_never_raises(self, monkeypatch):
        # Hook contract: serialization/I/O failures must not mask the original
        # exception that call sites are about to re-raise.
        _patch_collect_raises(monkeypatch)
        try_collect(ValueError("primary"), logger=None)

    def test_try_collect_keeps_inner_category_on_nested_hooks(self):
        # Backend sdsc/dbo-opt captures then re-raises into compile_fx.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("dxp_standalone failed")
            except RuntimeError as exc:
                try_collect(
                    exc,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    output_dir=tmp,
                )
                try_collect(
                    exc,
                    failure_category=CATEGORY_COMPILE_FRONTEND,
                    output_dir=tmp,
                )
            _assert_one_backend_report(tmp)

    def test_try_collect_skips_chained_wrapper_exception(self):
        # KTIR path: try_collect(inner) then raise RuntimeError(...) from inner.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("dbo-opt failed")
            except RuntimeError as inner:
                try_collect(
                    inner,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    output_dir=tmp,
                )
                try:
                    raise RuntimeError("wrapped frontend") from inner
                except RuntimeError as outer:
                    try_collect(
                        outer,
                        failure_category=CATEGORY_COMPILE_FRONTEND,
                        output_dir=tmp,
                    )
            _assert_one_backend_report(tmp)

    def test_try_collect_skips_implicit_context_exception(self):
        # ``raise New`` inside ``except`` sets __context__ without ``from``.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("inner backend")
            except RuntimeError as inner:
                try_collect(
                    inner,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    output_dir=tmp,
                )
                try:
                    raise RuntimeError("implicit wrap")
                except RuntimeError as outer:
                    try_collect(
                        outer,
                        failure_category=CATEGORY_COMPILE_FRONTEND,
                        output_dir=tmp,
                    )
            _assert_one_backend_report(tmp)

    def test_try_collect_skips_when_mark_is_on_context_not_cause(self):
        # ``raise New from other`` inside ``except marked``: __cause__ is
        # unmarked, __context__ is marked. Both links must be walked.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("marked context")
            except RuntimeError as marked:
                try_collect(
                    marked,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    output_dir=tmp,
                )
                other = ValueError("unrelated cause")
                try:
                    raise RuntimeError("outer") from other
                except RuntimeError as outer:
                    try_collect(
                        outer,
                        failure_category=CATEGORY_COMPILE_FRONTEND,
                        output_dir=tmp,
                    )
            _assert_one_backend_report(tmp)

    @pytest.mark.parametrize(
        "exc_cls",
        [_FrozenExc, _UnhashableExc],
        ids=["setattr_blocked", "hash_raises"],
    )
    def test_try_collect_survives_unmarkable_exceptions(self, exc_cls):
        # setattr or WeakSet hashing can fail; try_collect must not raise, and
        # the other mark path must still prevent a frontend relabel.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise exc_cls("primary")
            except exc_cls as caught:
                try_collect(
                    caught,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    output_dir=tmp,
                )
                try_collect(
                    caught,
                    failure_category=CATEGORY_COMPILE_FRONTEND,
                    output_dir=tmp,
                )
            _assert_one_backend_report(tmp)

    def test_try_collect_does_not_relabel_when_inner_collect_fails(self, monkeypatch):
        import torch_spyre.profiler._ffdc as ffdc_mod

        orig = ffdc_mod.collect
        calls = {"n": 0}

        def flaky(exc=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("ffdc write failed")
            return orig(exc, **kwargs)

        monkeypatch.setattr(ffdc_mod, "collect", flaky)
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("dxp_standalone failed")
            except RuntimeError as exc:
                try_collect(
                    exc,
                    failure_category=CATEGORY_COMPILE_BACKEND,
                    output_dir=tmp,
                )
                try_collect(
                    exc,
                    failure_category=CATEGORY_COMPILE_FRONTEND,
                    output_dir=tmp,
                )
            names = [p.name for p in Path(tmp).glob("ffdc_*.json")]
        assert names == []
        assert calls["n"] == 1

    def test_category_constants_match_report(self):
        for category in (
            CATEGORY_COMPILE_FRONTEND,
            CATEGORY_COMPILE_BACKEND,
            CATEGORY_RUNTIME_LAUNCH,
            CATEGORY_UNIMPLEMENTED,
            CATEGORY_UNKNOWN,
        ):
            try:
                raise ValueError("x")
            except ValueError as exc:
                report = self._collect_to_tmpdir(exc, failure_category=category)
            assert report["failure"]["category"] == category

    def test_report_filename_contains_category(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            with tempfile.TemporaryDirectory() as tmp:
                report = collect(
                    exc, failure_category=CATEGORY_COMPILE_FRONTEND, output_dir=tmp
                )
                fname = Path(report["_report_path"]).name
        assert fname.startswith("ffdc_compile_frontend_")
        assert ".json" in fname

    def test_empty_failure_category_normalizes_to_unknown(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            with tempfile.TemporaryDirectory() as tmp:
                report = collect(exc, failure_category="", output_dir=tmp)
                assert report["failure"]["category"] == CATEGORY_UNKNOWN
                fname = Path(report["_report_path"]).name
                assert fname.startswith("ffdc_unknown_")
                result = get_diagnostic_report(output_dir=tmp)
        assert result is not None
        assert result["failure"]["category"] == CATEGORY_UNKNOWN

    def test_collect_filename_has_report_sort_key(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            with tempfile.TemporaryDirectory() as tmp:
                report = collect(
                    exc, failure_category=CATEGORY_RUNTIME_LAUNCH, output_dir=tmp
                )
                path = Path(report["_report_path"])

        assert _report_sort_key(path) is not None
        assert path.name.startswith("ffdc_runtime_launch_")

    def test_completeness_pct_reflects_missing_fields(self):
        # Without an exception, failure.exception_type and failure.traceback are
        # None, so they appear in missing_fields.  This verifies that
        # completeness_pct is driven by REQUIRED_FIELDS programmatically:
        # any drift between the two would show up here as a wrong percentage.
        with tempfile.TemporaryDirectory() as tmp:
            report = collect(None, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)

        missing = report["collector"]["missing_fields"]
        assert "failure.exception_type" in missing
        assert "failure.traceback" in missing
        # REQUIRED_FIELDS has 11 entries; exc=None leaves exception_type and
        # traceback as None (2 missing, 9 present).
        # round(100 * 9 / 11, 1) == 81.8  — hardcoded to catch formula regressions.
        assert len(REQUIRED_FIELDS) == 11, (
            "Update the expected_pct below if REQUIRED_FIELDS changes"
        )
        assert report["collector"]["completeness_pct"] == 81.8
        assert report["collector"]["completeness_pct"] < 100.0

    def test_metadata_fields_present(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        meta = report["metadata"]
        for key in (
            "timestamp",
            "host",
            "pid",
            "python_version",
            "torch_version",
            "platform",
        ):
            assert key in meta

    def test_environment_keys_captured(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        env = report["environment"]
        for key in ("TORCH_COMPILE_DEBUG", "TORCH_SPYRE_DEBUG", "SPYRE_INDUCTOR_LOG"):
            assert key in env

    def test_capture_latency_is_positive(self):
        try:
            raise ValueError("x")
        except ValueError as exc:
            report = self._collect_to_tmpdir(exc, failure_category=CATEGORY_UNKNOWN)

        assert report["collector"]["capture_latency_ms"] > 0

    def test_get_diagnostic_report_returns_none_when_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            assert get_diagnostic_report(output_dir=tmp) is None

    def test_default_output_dir_uses_inductor_cache_root(self, monkeypatch, tmp_path):
        # Documents the real Inductor layout: cache_dir() → default_cache_dir()
        # → <tempdir>/torchinductor_<user>, not ~/.cache/torch/inductor.

        from torch._inductor.runtime.runtime_utils import (
            cache_dir as inductor_cache_dir,
        )

        monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False)
        expected = Path(inductor_cache_dir()) / "torch-spyre" / "ffdc_reports"
        assert _default_output_dir() == expected
        assert "torchinductor_" in str(expected)
        assert ".cache/torch/inductor" not in str(expected)

        custom = tmp_path / "custom_torchinductor"
        monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(custom))
        assert _default_output_dir() == custom / "torch-spyre" / "ffdc_reports"

    def test_get_diagnostic_report_returns_latest(self):
        # Second capture has the later embedded filename timestamp. Pin the first
        # report's st_mtime ahead so an mtime-based implementation cannot pass.
        with tempfile.TemporaryDirectory() as tmp:
            try:
                raise RuntimeError("first")
            except RuntimeError as exc:
                r1 = collect(
                    exc, failure_category=CATEGORY_COMPILE_FRONTEND, output_dir=tmp
                )
            try:
                raise RuntimeError("second")
            except RuntimeError as exc:
                r2 = collect(
                    exc, failure_category=CATEGORY_RUNTIME_LAUNCH, output_dir=tmp
                )
            os.utime(r1["_report_path"], (100, 100))
            os.utime(r2["_report_path"], (0, 0))

            result = get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert "failure" in result
            assert result["failure"]["category"] == CATEGORY_RUNTIME_LAUNCH
            assert result["_report_path"] == str(Path(r2["_report_path"]).resolve())

    def test_get_diagnostic_report_skips_corrupted_newest_report(self):
        _assert_get_diagnostic_report_skips_newest(
            newest_name=_NEWEST_VALID_REPORT_NAME,
            newest_payload="{not valid json",
        )

    def test_get_diagnostic_report_skips_non_utf8_newest_report(self):
        _assert_get_diagnostic_report_skips_newest(
            newest_name=_NEWEST_VALID_REPORT_NAME,
            newest_payload=b"\xff\xfe not utf-8",
        )

    @pytest.mark.parametrize(
        "malformed_name",
        [
            # Invalid date/time fields that would sort ahead of a real report.
            "ffdc_x_99999999T999999_999999_1.json",
            # Non-canonical (non-zero-padded) timestamp shape.
            "ffdc_unknown_9999999T010101_123456_1.json",
            # Arabic-Indic digits in ts_seconds (isdigit() true, isascii() false).
            "ffdc_unknown_٢٠٢٥٠١٠١T٠٠٠٠٠٢_000000_1.json",
            # Unicode digit PID (str.isdigit() true, but not ASCII).
            "ffdc_unknown_20250101T000002_000000_².json",
            # Non-ASCII category (str.isalnum() true, but not ASCII).
            "ffdc_未知_20250101T000002_000000_1.json",
        ],
    )
    def test_get_diagnostic_report_skips_invalid_filenames(self, malformed_name):
        _assert_get_diagnostic_report_skips_newest(
            newest_name=malformed_name,
            newest_payload='{"failure": {"category": "compile"}}',
        )

    @pytest.mark.parametrize(
        "payload",
        [
            "[]",
            "null",
            '"not a report"',
            "{}",
            '{"failure": null}',
            '{"failure": {"category": 42}}',
        ],
    )
    def test_get_diagnostic_report_skips_non_dict_newest_report(self, payload):
        _assert_get_diagnostic_report_skips_newest(
            newest_name=_NEWEST_VALID_REPORT_NAME,
            newest_payload=payload,
        )

    def test_get_diagnostic_report_returns_none_when_all_corrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_ffdc_report(Path(tmp), _VALID_REPORT_NAME, "{bad json")

            assert get_diagnostic_report(output_dir=tmp) is None

    @pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="os.mkfifo not available")
    def test_get_diagnostic_report_skips_fifo_newest(self):
        # Newest candidate is a FIFO with a valid report name; must not hang.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            valid = _write_ffdc_report(d, _VALID_REPORT_NAME, _VALID_REPORT_JSON)
            fifo = d / _NEWEST_VALID_REPORT_NAME
            os.mkfifo(fifo)
            assert stat.S_ISFIFO(os.lstat(fifo).st_mode)
            result = _call_with_timeout(
                lambda: get_diagnostic_report(output_dir=tmp), 1.0
            )
            assert result is not None
            assert result["failure"]["category"] == CATEGORY_UNKNOWN
            assert result["_report_path"] == str(valid.resolve())

    def test_get_diagnostic_report_skips_deeply_nested_json_newest(self):
        # json.load RecursionError must skip, not raise, so an older valid
        # report is still returned.
        _assert_get_diagnostic_report_skips_newest(
            newest_name=_NEWEST_VALID_REPORT_NAME,
            newest_payload="[" * 3000 + "]" * 3000,
        )

    def test_get_diagnostic_report_skips_symlink_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            valid = _write_ffdc_report(d, _VALID_REPORT_NAME, _VALID_REPORT_JSON)
            target = d / "other.json"
            target.write_text('{"failure": {"category": "compile_frontend"}}')
            link = d / _NEWEST_VALID_REPORT_NAME
            try:
                os.symlink(target, link)
            except OSError:
                pytest.skip("symlinks not supported")
            result = get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert result["_report_path"] == str(valid.resolve())

    def test_get_diagnostic_report_unreadable_dir_returns_none(self):
        if os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0):
            pytest.skip("chmod 0 is not meaningful on this platform")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "reports"
            d.mkdir()
            _write_ffdc_report(d, _VALID_REPORT_NAME, _VALID_REPORT_JSON)
            os.chmod(d, 0)
            try:
                assert get_diagnostic_report(output_dir=str(d)) is None
            finally:
                os.chmod(d, 0o700)

    def test_collect_writes_private_modes(self):
        if os.name == "nt":
            pytest.skip("POSIX file modes not enforced on Windows")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ffdc"
            collect(None, failure_category=CATEGORY_UNKNOWN, output_dir=str(out))
            assert stat.S_IMODE(out.stat().st_mode) == 0o700
            files = list(out.glob("ffdc_*.json"))
            assert len(files) == 1
            assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
            assert list(out.glob("*.tmp")) == []

    def test_collect_removes_tmp_on_write_failure(self, monkeypatch):
        import torch_spyre.profiler._ffdc as ffdc_mod

        def boom(*_args, **_kwargs):
            raise OSError("disk full")

        monkeypatch.setattr(ffdc_mod, "_dump_json", boom)
        with tempfile.TemporaryDirectory() as tmp:
            report = collect(None, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)
            assert report["_report_path"] is None
            assert report["collector"]["success"] is False
            assert list(Path(tmp).glob("ffdc_*.json")) == []
            assert list(Path(tmp).glob("*.tmp")) == []

    def test_get_diagnostic_report_includes_report_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            reports_dir = Path(tmp) / "reports"
            reports_dir.mkdir()
            cwd = os.getcwd()
            try:
                os.chdir(tmp)
                try:
                    raise RuntimeError("path test")
                except RuntimeError as exc:
                    written = collect(
                        exc,
                        failure_category=CATEGORY_COMPILE_FRONTEND,
                        output_dir="reports",
                    )

                result = get_diagnostic_report(output_dir="reports")
            finally:
                os.chdir(cwd)

            written_path = Path(written["_report_path"])
            assert written_path.is_absolute()
            assert written_path.is_file()
            assert result is not None
            report_path = Path(result["_report_path"])
            assert report_path.is_absolute()
            assert report_path.is_file()
            assert report_path.resolve().is_relative_to(reports_dir.resolve())
            assert report_path == written_path

    def test_get_diagnostic_report_works_when_capture_disabled(self, monkeypatch):
        # Retrieval is not gated on TORCH_SPYRE_FFDC; only collect() is.
        monkeypatch.setenv("TORCH_SPYRE_FFDC", "0")
        with tempfile.TemporaryDirectory() as tmp:
            report_file = _write_ffdc_report(
                Path(tmp), _VALID_REPORT_NAME, _VALID_REPORT_JSON
            )

            result = get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert result["failure"]["category"] == CATEGORY_UNKNOWN
            assert result["_report_path"] == str(report_file.resolve())

    def test_get_diagnostic_report_returns_latest_across_categories(self):
        # Selection sorts by the timestamp embedded in the filename, not mtime
        # or the full name. Pin unknown st_mtime ahead of compile so an
        # mtime-based implementation cannot pass; compile must still win via its
        # later embedded timestamp, and must not lose to a full-name lexical sort.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            compile_path = _write_ffdc_report(
                d,
                "ffdc_compile_20250101T000001_000000_1.json",
                '{"failure": {"category": "compile"}}',
            )
            unknown_path = _write_ffdc_report(
                d,
                "ffdc_unknown_20250101T000000_000000_1.json",
                '{"failure": {"category": "unknown"}}',
            )
            os.utime(compile_path, (0, 0))
            os.utime(unknown_path, (100, 100))

            result = get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert result["failure"]["category"] == "compile"

    def test_prune_old_reports_removes_oldest(self):
        # _prune_old_reports keeps the newest `keep` files by mtime, not by name.
        # compile sorts first lexically, so use compile as the NEWEST category —
        # a name-sort regression would evict these and wrongly keep the older files.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            # oldest → newest by mtime (index = mtime in seconds since epoch)
            files = [
                "ffdc_unknown_20250101T000000_000000_1.json",  # mtime 0 - oldest
                "ffdc_runtime_launch_20250101T000001_000000_1.json",  # mtime 1
                "ffdc_compile_20250101T000002_000000_1.json",  # mtime 2
                "ffdc_compile_20250101T000003_000000_1.json",  # mtime 3
                "ffdc_compile_20250101T000004_000000_1.json",  # mtime 4 - newest
            ]
            for i, name in enumerate(files):
                p = d / name
                p.write_text("{}")
                os.utime(p, (i, i))  # mtime = i seconds since epoch
            _prune_old_reports(d, keep=3)
            remaining = sorted(d.glob("ffdc_*.json"), key=lambda p: p.stat().st_mtime)
            assert len(remaining) == 3
            # The three newest by mtime must survive — all three are compile files
            # even though compile sorts first by name.
            assert [p.name for p in remaining] == [
                "ffdc_compile_20250101T000002_000000_1.json",
                "ffdc_compile_20250101T000003_000000_1.json",
                "ffdc_compile_20250101T000004_000000_1.json",
            ]

    def test_collect_succeeds_when_dir_has_dangling_ffdc_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            dangling = d / "ffdc_unknown_20990101T000000_000000_1.json"
            try:
                dangling.symlink_to(d / "missing.json")
            except OSError:
                pytest.skip("symlinks not supported")
            report = collect(None, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)
            assert report["_report_path"] is not None
            assert report["collector"]["success"] is True
            written = Path(report["_report_path"])
            assert written.is_file()
            assert dangling.is_symlink()

    def test_collect_prunes_beyond_max_reports(self):
        # After writing, collect() must not leave more than _MAX_REPORTS files.
        with tempfile.TemporaryDirectory() as tmp:
            # Pre-seed the directory with _MAX_REPORTS files so the next write
            # would exceed the cap.
            d = Path(tmp)
            for i in range(_MAX_REPORTS):
                (d / f"ffdc_unknown_20240101T{i:06d}_000000_1.json").write_text("{}")
            try:
                raise ValueError("x")
            except ValueError as exc:
                collect(exc, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)
            assert len(list(d.glob("ffdc_*.json"))) <= _MAX_REPORTS


class TestFfdcCompileFx:
    def test_compile_fx_spyre_failure_triggers_ffdc_frontend(self, monkeypatch):
        """Spyre compile_fx failures must try_collect with compile_frontend.

        Forces the Spyre branch to raise before real compile work so the test
        covers only the FFDC except-path. Also no-ops ``patch_inductor_fusions``
        because CI already applied it once and re-entering asserts.
        """
        import sys
        import types
        from enum import IntEnum

        import torch
        import torch._inductor.compile_fx as cfx

        from torch_spyre.constants import DEVICE_NAME

        class _ElementArrangement(IntEnum):
            STANDARD = 0
            DL16_TO_FP32 = 1
            FP32_TO_DL16 = 2
            EXX2 = 3
            QFP8CH = 4

        if "torch_spyre._C" not in sys.modules:
            _c = types.ModuleType("torch_spyre._C")
            _c.ElementArrangement = _ElementArrangement
            _c.launch_jobplan = lambda *a, **k: None
            _c.prepare_kernel = lambda *a, **k: None
            monkeypatch.setitem(sys.modules, "torch_spyre._C", _c)

        inductor = _reimport(monkeypatch, "torch_spyre._inductor")
        calls: list[dict] = []

        def fake_try_collect(exc, **kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(inductor, "try_collect", fake_try_collect)
        monkeypatch.setattr(inductor, "patch_inductor_fusions", lambda: None)

        def _fail_lazy_init():
            raise RuntimeError("frontend compile boom")

        monkeypatch.setattr(
            torch,
            "spyre",
            types.SimpleNamespace(
                _impl=types.SimpleNamespace(_lazy_init=_fail_lazy_init)
            ),
            raising=False,
        )
        monkeypatch.setattr(cfx, "_spyre_wrapped", False, raising=False)
        monkeypatch.setattr(cfx, "compile_fx", lambda *a, **k: None)
        inductor.enable_spyre_compile_fx_wrapper()

        class _DeviceNode:
            kwargs = {"device": DEVICE_NAME}

        class _FakeGM:
            class graph:
                nodes = [_DeviceNode()]

                @staticmethod
                def output_node():
                    class _Node:
                        args = ()

                    return _Node()

        with pytest.raises(RuntimeError, match="frontend compile boom"):
            cfx.compile_fx(_FakeGM(), [])

        assert len(calls) == 1
        assert calls[0]["failure_category"] == CATEGORY_COMPILE_FRONTEND


class TestFfdcAsyncCompile:
    def _load_async_compile(self, monkeypatch, tmp_path):
        """Stub inductor/extension imports and return ``(mod, out_dir)``."""
        import logging
        import sys

        out_dir = str(tmp_path / "bundle")

        inductor = _stub_module(monkeypatch, "torch_spyre._inductor")
        inductor.__path__ = []
        _stub_module(
            monkeypatch,
            "torch_spyre._inductor.logging_utils",
            get_inductor_logger=lambda name: logging.getLogger(name),
        )
        _stub_module(
            monkeypatch,
            "torch_spyre._inductor.op_spec",
            LoopSpec=object,
            OpSpec=object,
            UnimplementedOp=object,
            find_unimplemented=lambda specs: None,
        )
        _stub_module(
            monkeypatch,
            "torch_spyre._inductor.kernel_provenance",
            build_kernel_provenance_descriptor=lambda specs: None,
        )
        codegen = _stub_module(monkeypatch, "torch_spyre._inductor.codegen")
        codegen.__path__ = []
        _stub_module(
            monkeypatch,
            "torch_spyre._inductor.codegen.bundle",
            generate_bundle=lambda *a, **k: None,
        )
        if "torch_spyre._C" not in sys.modules:
            _stub_module(
                monkeypatch,
                "torch_spyre._C",
                launch_jobplan=lambda *a, **k: None,
                prepare_kernel=lambda *a, **k: None,
                register_kernel_provenance=lambda *a, **k: True,
            )

        class _Runner:
            def __init__(self, name, code_dir, kernel_provenance=None):
                self.kernel_name = name
                self.code_dir = code_dir
                self.kernel_provenance = kernel_provenance

        _stub_module(
            monkeypatch,
            "torch_spyre.execution.kernel_runner",
            SpyreSDSCKernelRunner=_Runner,
            SpyreUnimplementedRunner=object,
        )

        mod = _reimport(monkeypatch, "torch_spyre.execution.async_compile")
        monkeypatch.setattr(mod, "get_output_dir", lambda name: out_dir)
        monkeypatch.setattr(mod, "generate_bundle", lambda *a, **k: None)
        monkeypatch.setattr(mod, "find_unimplemented", lambda specs: None)
        return mod, out_dir

    def test_sdsc_dxp_failure_triggers_ffdc_collect(self, monkeypatch, tmp_path):
        """dxp_standalone failure must call try_collect then re-raise.

        Patch ``try_collect`` before reimporting ``async_compile`` so the
        module-level binding picks up the fake.
        """
        import importlib
        import subprocess

        ffdc_mod = importlib.import_module("torch_spyre.profiler._ffdc")
        calls: list[dict] = []

        def fake_try_collect(exc, **kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(ffdc_mod, "try_collect", fake_try_collect)

        mod, out_dir = self._load_async_compile(monkeypatch, tmp_path)

        def fail_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(mod.subprocess, "run", fail_run)

        with pytest.raises(subprocess.CalledProcessError):
            mod.SpyreAsyncCompile().sdsc("test_kernel", [])

        assert len(calls) == 1
        assert calls[0]["failure_category"] == CATEGORY_COMPILE_BACKEND
        assert calls[0]["kernel_name"] == "test_kernel"
        assert calls[0]["code_dir"] == out_dir

    def test_sdsc_dxp_failure_preserves_error_when_ffdc_raises(
        self, monkeypatch, tmp_path
    ):
        """FFDC collection failure must not replace CalledProcessError.

        Uses the real ``try_collect`` with a raising ``collect`` so the hook
        path is covered end-to-end (not a fake that swallows by construction).
        """
        import subprocess

        _patch_collect_raises(monkeypatch)
        mod, _out_dir = self._load_async_compile(monkeypatch, tmp_path)

        def fail_run(cmd, **kwargs):
            raise subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(mod.subprocess, "run", fail_run)

        with pytest.raises(subprocess.CalledProcessError) as ei:
            mod.SpyreAsyncCompile().sdsc("test_kernel", [])
        assert ei.value.returncode == 1


class TestFfdcKernelRunner:
    def _load_kernel_runner(self, monkeypatch, *, launch_side_effect=None):
        """Load ``kernel_runner`` without permanently shadowing real ``_C``.

        Prefer patching the module's bound ``launch_jobplan`` / ``prepare_kernel``.
        Stub ``_C`` only when it is absent (e.g. no extension on Mac).
        """
        import logging
        import sys

        def _launch(jobplan, args):
            if launch_side_effect is not None:
                raise launch_side_effect

        if "torch_spyre._C" not in sys.modules:
            _stub_module(
                monkeypatch,
                "torch_spyre._C",
                launch_jobplan=_launch,
                prepare_kernel=lambda path: "fake_jobplan",
                register_kernel_provenance=lambda *a, **k: True,
            )
        if "torch_spyre._inductor" not in sys.modules:
            inductor = _stub_module(monkeypatch, "torch_spyre._inductor")
            inductor.__path__ = []
        if "torch_spyre._inductor.logging_utils" not in sys.modules:
            _stub_module(
                monkeypatch,
                "torch_spyre._inductor.logging_utils",
                get_inductor_logger=lambda name: logging.getLogger(name),
            )
        if "torch_spyre._inductor.kernel_provenance" not in sys.modules:
            _stub_module(
                monkeypatch,
                "torch_spyre._inductor.kernel_provenance",
                KernelProvenanceDescriptor=object,
            )
        if "torch_spyre._inductor.profiler_event" not in sys.modules:
            _stub_module(
                monkeypatch,
                "torch_spyre._inductor.profiler_event",
                format_kernel_provenance_event_name=lambda descriptor: "event_name",
            )

        mod = _reimport(monkeypatch, "torch_spyre.execution.kernel_runner")
        monkeypatch.setattr(mod, "launch_jobplan", _launch)
        monkeypatch.setattr(mod, "prepare_kernel", lambda path: "fake_jobplan")
        monkeypatch.setattr(mod, "register_kernel_provenance", lambda *a, **k: True)
        return mod

    def test_unimplemented_preserves_error_when_ffdc_raises(self, monkeypatch):
        _patch_collect_raises(monkeypatch)
        mod = self._load_kernel_runner(monkeypatch)
        runner = mod.SpyreUnimplementedRunner("k", "aten::foo")

        with pytest.raises(RuntimeError, match="unimplemented operation") as ei:
            runner.run()
        assert "aten::foo" in str(ei.value)

    def test_launch_preserves_error_when_ffdc_raises(self, monkeypatch):
        _patch_collect_raises(monkeypatch)
        launch_exc = RuntimeError("launch_jobplan failed")
        mod = self._load_kernel_runner(monkeypatch, launch_side_effect=launch_exc)
        runner = mod.SpyreSDSCKernelRunner("k", "/tmp/code")

        with pytest.raises(RuntimeError, match="launch_jobplan failed"):
            runner.run()


class TestFfdcProfilerApi:
    def test_profiler_package_exports_get_diagnostic_report(self):
        import torch_spyre

        assert torch_spyre.profiler is not None
        assert not hasattr(torch_spyre.profiler, "is_available")
        assert "get_diagnostic_report" in torch_spyre.profiler.__all__
        assert hasattr(torch_spyre.profiler, "get_diagnostic_report")
        assert callable(torch_spyre.profiler.get_diagnostic_report)
        assert torch_spyre.profiler.get_diagnostic_report is get_diagnostic_report
        assert profiler_get_diagnostic_report is get_diagnostic_report

        with tempfile.TemporaryDirectory() as tmp:
            assert torch_spyre.profiler.get_diagnostic_report(output_dir=tmp) is None


class TestFfdcPublicApi:
    """Exercise the ``make_spyre_module()`` binding without claiming PrivateUse1.

    ``rename_privateuse1_backend()`` is process-wide and one-way, so public-API
    coverage uses a local module from ``make_spyre_module()`` rather than
    mutating ``torch.spyre`` for the worker.
    """

    def test_make_spyre_module_exposes_get_diagnostic_report(self):
        mod = make_spyre_module()
        assert hasattr(mod, "get_diagnostic_report")
        assert callable(mod.get_diagnostic_report)

    def test_make_spyre_module_get_diagnostic_report(self):
        mod = make_spyre_module()
        with tempfile.TemporaryDirectory() as tmp:
            assert mod.get_diagnostic_report(output_dir=tmp) is None
            try:
                raise ValueError("public api")
            except ValueError as exc:
                collect(exc, failure_category=CATEGORY_UNKNOWN, output_dir=tmp)
            result = mod.get_diagnostic_report(output_dir=tmp)
            assert result is not None
            assert result["failure"]["category"] == CATEGORY_UNKNOWN
            assert "public api" in result["failure"]["message"]
            report_path = Path(result["_report_path"])
            assert report_path.is_absolute()
            assert report_path.is_file()
            assert report_path.resolve().is_relative_to(Path(tmp).resolve())

    def test_make_spyre_module_skips_newer_invalid_report(self):
        _assert_get_diagnostic_report_skips_newest(
            newest_name=_NEWEST_VALID_REPORT_NAME,
            newest_payload="{}",
            get_report=make_spyre_module().get_diagnostic_report,
        )
