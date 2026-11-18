"""The moe-demo example validates and expands through the engine itself, and the MoE pack additions load."""

import json
import shutil
import tempfile
from pathlib import Path

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


def test_role_moe_attributes_registered_and_absent_signal_is_not_captured():
    reg = _cat().reg
    assert set(trace_signals.declared_roles(reg, "moe")) == MOE_ROLES
    got = trace_signals.resolve_roles(
        reg, "moe", {"role.moe.e_sel": "moe.e_sel", "role.moe.tokens": "gone"}, {"moe.e_sel": [3.0], "other": [1.0]}
    )
    assert got["e_sel"] == [3.0]
    assert got["tokens"] is trace_signals.NOT_CAPTURED and got["start_us"] is trace_signals.NOT_CAPTURED
    assert set(got) == MOE_ROLES


def test_validate_cli_passes_with_no_context_warning_for_moe_demo(capsys):
    rc = validate.main([str(EX / "moe-demo"), "--central", str(EX / "central"), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out[0]["ok"] and out[0]["cases"] == 11
    assert not [w for w in out[0]["warnings"] if "moe" in w.lower() and "timer" in w]
