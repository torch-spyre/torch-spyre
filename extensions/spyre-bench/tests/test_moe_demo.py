"""The moe-demo example validates and expands through the engine itself, and the MoE pack additions load."""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

from spyre_bench import engine, trace_signals, validate

EX = Path(__file__).resolve().parent.parent / "examples"
MOE_ROLES = {
    "expert_id", "tile_index", "tokens", "cores", "start_us", "end_us", "e_sel", "loop_trips_max",
    "tokens_routed_hist",
}  # fmt: skip


def _cat():
    return engine.Catalog([str(EX / "moe-demo")], [], [str(EX / "central")])


def test_moe_demo_expands_and_validates_without_errors():
    cat = _cat()
    assert cat.errors == []
    names = {c["name"] for c in cat.cases}
    assert {f"moe_loop_c{c}_g{g}" for c in (1, 2, 4) for g in (256, 512)} <= names
    assert {"moe_loop_int8", "moe_loop_int32_index", "moe_serve_c1", "moe_serve_c8", "moe_serve_c32"} <= names
    assert len(cat.cases) == len(names) == 11


PINNED_IDS = {
    "moe_loop_c1_g256": "ec1cbeb4-0281-5d0d-8e9f-e116a12c4a70",
    "moe_loop_c1_g512": "055d4835-daf8-5c94-a175-2d7c7d2a2c04",
    "moe_loop_c2_g256": "87bd0adb-f8fb-5214-838d-28c13ad95aaf",
    "moe_loop_c2_g512": "dfacf45e-8778-53b6-8c5d-ec8bc741bfd9",
    "moe_loop_c4_g256": "5de2ce61-cf71-53f8-9206-5099c28c31b1",
    "moe_loop_c4_g512": "b6cd31ac-8b6d-5749-b307-64098d675c63",
    "moe_loop_int32_index": "dbaae63e-d034-5c34-8c4a-ce4d195b2f3c",
    "moe_loop_int8": "cc977a46-4115-558d-827b-07fc8f10ebe2",
    "moe_serve_c1": "73673257-3a50-5411-b798-baf796384c25",
    "moe_serve_c32": "7e68c66a-a504-5d6c-a387-bf1ae79df184",
    "moe_serve_c8": "f37caa5e-3b88-54cb-a25a-5fe783097420",
}


def test_case_ids_are_pinned():
    cat = _cat()
    assert {c["name"]: cat.id_token(c) for c in cat.cases} == PINNED_IDS


def test_sweep_axes_change_the_case_identity():
    cat = _cat()
    ids = {c["name"]: cat.id_token(c) for c in cat.cases}
    assert len(set(ids.values())) == len(ids)


def test_dtypes_int8_int32_validate_and_device_alias_does_not():
    cat = _cat()
    dtypes = {c["workload"].get("dtype") for c in cat.cases if c["name"].startswith("moe_loop")}
    assert {"int8", "int32", "fp16"} <= dtypes
    with tempfile.TemporaryDirectory(prefix="moe-demo-") as tmp:
        d = Path(tmp) / "moe-demo"
        shutil.copytree(EX / "moe-demo", d)
        f = d / "ops" / "moe-expert-loop-int8.yaml"
        f.write_text(f.read_text().replace("dtype: int8", "dtype: df16"))
        errors = engine.Catalog([str(d)], [], [str(EX / "central")]).errors
    assert any("df16" in e for e in errors)


def test_moe_metrics_registered_with_class():
    reg = _cat().reg
    for name in ("kernel.moe.fetch_bytes", "kernel.moe.loop_trips"):
        assert reg.metrics[name]["class"] == "kernel_time"
    assert reg.metrics["kernel.moe.loop_trips"]["compare"] == "exact"


def test_role_moe_attributes_registered_and_absent_signal_is_not_captured():
    reg = _cat().reg
    assert set(trace_signals.declared_roles(reg, "moe")) == MOE_ROLES
    got = trace_signals.resolve_roles(
        reg, "moe", {"role.moe.e_sel": "moe.e_sel", "role.moe.tokens": "gone"}, {"moe.e_sel": [3.0], "other": [1.0]}
    )
    assert got["e_sel"] == trace_signals.Signal((3.0,), ())
    assert got["tokens"] is trace_signals.NOT_CAPTURED and got["start_us"] is trace_signals.NOT_CAPTURED
    assert set(got) == MOE_ROLES


_REG = []


def _roles(props, meas):
    if not _REG:
        _REG.append(_cat().reg)
    return trace_signals.resolve_roles(_REG[0], "moe", props, meas)


def test_not_captured_is_a_distinct_falsy_sentinel():
    nc = trace_signals.NOT_CAPTURED
    assert nc is not None and not nc and repr(nc) == "NOT_CAPTURED"
    assert _roles({"role.moe.e_sel": "s"}, {"s": None})["e_sel"] is nc


def test_empty_array_is_captured_but_empty():
    got = _roles({"role.moe.e_sel": "s"}, {"s": []})["e_sel"]
    assert got == trace_signals.Signal((), ()) and got.complete


def test_partial_array_reports_missing_indices():
    got = _roles({"role.moe.e_sel": "s"}, {"s": [1, None, 3, float("nan")]})["e_sel"]
    assert got.missing_at == (1, 3) and not got.complete and got.values[0] == 1


def test_array_likes_nested_nan_and_none_inputs():
    class Arr:  # duck-typed array: no list base class
        def __init__(self, *v):
            self.v = v

        def __len__(self):
            return len(self.v)

        def __getitem__(self, i):
            return self.v[i]

    props = {"role.moe.e_sel": "a", "role.moe.tokens": "n", "role.moe.cores": "t", "role.moe.start_us": "d"}
    meas = {"a": Arr(1.0, float("nan")), "n": [[1, 2], [3, float("nan")], [4]], "t": (1, 2), "d": {"x": 1}}
    got = _roles(props, meas)
    assert got["e_sel"].missing_at == (1,)
    assert got["tokens"].missing_at == (1,) and got["tokens"].values[0] == [1, 2]
    assert got["cores"] == trace_signals.Signal((1, 2), ())
    assert got["start_us"] is trace_signals.NOT_CAPTURED  # a mapping is not an array
    assert all(v is trace_signals.NOT_CAPTURED for v in _roles(None, None).values())
    assert all(v is trace_signals.NOT_CAPTURED for v in _roles({"role.moe.e_sel": "s"}, None).values())


def test_unreadable_array_likes_are_not_captured():
    class ZeroD:  # like np.array(5.0): has len and getitem, len() raises
        def __len__(self):
            raise TypeError("len() of unsized object")

        def __getitem__(self, i):
            raise IndexError

    class KeyedByLabel:  # like a Series with a non-default index
        def __len__(self):
            return 2

        def __getitem__(self, i):
            raise KeyError(i)

    class Hostile:
        def __len__(self):
            raise RuntimeError("boom")

        def __getitem__(self, i):
            return 1

    class Nested:
        def __len__(self):
            return 1

        def __getitem__(self, i):
            return ZeroD()

    props = {"role.moe.e_sel": "z", "role.moe.tokens": "k", "role.moe.cores": "h", "role.moe.start_us": "n"}
    got = _roles(props, {"z": ZeroD(), "k": KeyedByLabel(), "h": Hostile(), "n": [1, 2]})
    assert got["e_sel"] is got["tokens"] is got["cores"] is trace_signals.NOT_CAPTURED
    assert got["start_us"] == trace_signals.Signal((1, 2), ())
    assert _roles({"role.moe.e_sel": "n"}, {"n": Nested()})["e_sel"].missing_at == (0,)


def test_numpy_arrays_when_numpy_is_importable():
    np = pytest.importorskip("numpy")
    props = {"role.moe.e_sel": "a", "role.moe.tokens": "z", "role.moe.cores": "m"}
    got = _roles(props, {"a": np.array([1.0, np.nan]), "z": np.array(5.0), "m": np.array([[1.0, 2.0], [np.nan, 1.0]])})
    assert got["e_sel"].missing_at == (1,)
    assert got["tokens"] is trace_signals.NOT_CAPTURED
    assert got["cores"].missing_at == (1,)


def test_malformed_property_values_do_not_raise():
    props = {"role.moe.e_sel": ["a"], "role.moe.tokens": {"x": 1}, "role.moe.cores": 7, "role.moe.start_us": "s"}
    got = _roles(props, {"s": 5, "a": [1]})
    assert all(v is trace_signals.NOT_CAPTURED for v in got.values())


def test_validate_cli_passes_with_no_warnings_for_moe_demo(capsys):
    argv = [str(EX / "moe-demo"), "--central", str(EX / "central"), "--json", "--lint-context"]
    rc = validate.main(argv)
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out[0]["ok"] and out[0]["cases"] == 11
    assert out[0]["warnings"] == []


def test_variants_state_their_own_flags():
    cat = _cat()
    flags = {c["name"]: c["workload"]["flags"] for c in cat.cases if "int" in c["name"]}
    assert flags["moe_loop_int8"] == {
        "experts": 128,
        "top_k": 8,
        "phase": "decode",
        "tile_tokens": 512,
        "expert_concurrency": 1,
    }
    assert flags["moe_loop_int32_index"] == {"experts": 128, "top_k": 8, "phase": "decode"}


def _fast(fn, limit=1.0):
    import time

    t0 = time.monotonic()
    out = fn()
    assert time.monotonic() - t0 < limit
    return out


def _one(entry, role="e_sel"):
    return _roles({f"role.moe.{role}": "s"}, {"s": [entry]})[role]


def test_pathological_nested_entries_finish_fast_and_read_as_missing():
    import yaml

    cyc = [1, 2, 3, 4, 5, 6, 7]
    cyc.append(cyc)
    anchored = yaml.safe_load("a: &a [1, 2, *a]")["a"]
    wide_deep = [[[[[[[[list(range(30))] * 30] * 30]] * 30] * 30]] * 30] * 30
    for junk in (cyc, anchored, wide_deep):
        got = _fast(lambda: _one(junk))
        assert got.missing_at == (0,)
    selfref = [None]
    selfref[0] = selfref
    assert _fast(lambda: _roles({"role.moe.e_sel": "s"}, {"s": selfref})["e_sel"]).missing_at == (0,)


def test_depth_cap_node_budget_and_empty_nested_are_pinned():
    def nest(levels, leaf=5):
        v = leaf
        for _ in range(levels):
            v = [v]
        return v

    assert trace_signals.MAX_DEPTH == 8 and trace_signals.MAX_NODES == 10_000
    assert _one(nest(8)).complete and _one(nest(9)).missing_at == (0,)
    assert _one([5] * 10_000).complete and _one([5] * 10_001).missing_at == (0,)
    assert _one([]).missing_at == (0,)  # an empty nested array is a missing entry
    assert _one([[]]).missing_at == (0,)
    assert _roles({"role.moe.e_sel": "s"}, {"s": []})["e_sel"].complete  # an empty top-level array is captured


def test_hostile_values_never_escape_trace_signals():
    class Hostile:
        def __init__(self, exc, where):
            self.exc, self.where = exc, where

        def _go(self, name):
            if name == self.where:
                raise self.exc("boom")

        def __float__(self):
            self._go("float")
            return 1.0

        def __len__(self):
            self._go("len")
            return 1

        def __getitem__(self, i):
            self._go("getitem")
            return 1

        def __iter__(self):
            self._go("iter")
            return iter([1])

        def __bool__(self):
            self._go("bool")
            return True

    class Mapping:
        def __getitem__(self, k):
            raise RuntimeError("boom")

    for exc in (RuntimeError, RecursionError, MemoryError, KeyError):
        for where in ("float", "len", "getitem", "iter", "bool"):
            h = Hostile(exc, where)
            _roles({"role.moe.e_sel": "s"}, {"s": h})
            _roles({"role.moe.e_sel": "s"}, {"s": [h]})
            _roles({"role.moe.e_sel": h}, {"s": [1]})
        _roles({"role.moe.e_sel": "s"}, Mapping())
        _roles(Hostile(exc, "getitem"), {"s": [1]})
    for exc in (KeyboardInterrupt, SystemExit):
        with pytest.raises(exc):
            _roles({"role.moe.e_sel": "s"}, {"s": [Hostile(exc, "getitem")]})
