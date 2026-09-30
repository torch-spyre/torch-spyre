#!/usr/bin/env python3
"""Model-ops capability verdicts, read from the suites' JUnit reports.

tests/models/test_model_ops_v2.py records result.op / variant / args / backend / fallback_ops
on each test case. This folds them into CapabilityWriter entries, keeping
parse_model_ops_logs.py's op naming and shape rules so capability ids carry over.

    python3 model_ops_junit.py --xml gpt-oss-20b.xml ... [--parity model_ops_<run>.json]

--parity compares against the parser's JSON for the same run and exits 1 on a difference.
"""

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import regex as re

# Hashed into capability_id, in this order (ingest_model_ops.V2_DISC_KEYS).
DISC_KEYS = ("input_shapes", "input_dtypes")

_SIZE_OPS = {
    "torch.zeros",
    "torch.ones",
    "torch.empty",
    "torch.full",
    "torch.rand",
    "torch.randn",
    "torch.sym_sum",
}
# The parser only reads a shape/stride of digits, commas, spaces and minus signs.
_DIMS = re.compile(r"^\[[\d,\s\-]+\]$")
_TORCH_DTYPE = re.compile(r"^torch\.\w+$")


def normalize_op(op: str) -> str:
    """parse_model_ops_logs._normalize_op_name: one name for aliases of one op."""
    if not op:
        return op
    op = op.replace("nn_", "nn.").replace("functional_", "functional.")
    if op.startswith("aten."):
        op = op.replace("aten.", "torch.").split(".default")[0].split(".out")[0]
    if "embedding" in op.lower():
        return "torch.nn.functional.embedding"
    if "index_copy" in op.lower() or "index.copy" in op.lower():
        return "torch.index_copy_"
    return op


def _compact(dims) -> str:
    return re.sub(r"\s+", "", str(dims))


def signature(op: str, args: list) -> dict:
    """The parser's input_shapes/strides/dtypes, arg_values and target_shape for one test.

    Mirrors what it reads from the [INPUT SHAPES] block, including what it skips: a tensor
    without a stride, a 0-d tensor, or a dtype not spelled torch.<name>.
    """
    shapes, strides, dtypes, values = [], [], [], []
    target_shape = ""

    def _tensor(t: dict):
        shape, stride = _compact(t.get("shape")), _compact(t.get("stride"))
        dtype = str(t.get("dtype"))
        if _DIMS.match(shape) and _DIMS.match(stride) and _TORCH_DTYPE.match(dtype):
            shapes.append(shape)
            strides.append(stride)
            dtypes.append(dtype)

    for arg in args:
        if "tensor" in arg:
            _tensor(arg["tensor"])
        elif "tensor_list" in arg:
            for t in arg["tensor_list"]:
                _tensor(t)
        elif "value" in arg:
            raw = repr(arg["value"]).strip().strip("'\"")
            inner = raw[1:-1].strip() if raw[:1] + raw[-1:] in ("()", "[]") else ""
            if inner and ("view" in op or "reshape" in op):
                target_shape = raw
            elif inner and op in _SIZE_OPS and set(inner) <= set("0123456789,- "):
                shapes.append(
                    "["
                    + ",".join(d.strip() for d in inner.split(",") if d.strip())
                    + "]"
                )
            else:
                values.append(raw)
        elif "py" in arg:
            values.append(str(arg["py"]).strip().strip("'\""))

    if not shapes and values:
        dims = [v for v in values if re.match(r"^[\d,\s]+$", v)]
        if dims:
            shapes = [f"[{v.replace(' ', '')}]" for v in dims]
            values = [v for v in values if v not in dims]

    return {
        "input_shapes": shapes,
        "input_strides": strides,
        "input_dtypes": dtypes,
        "arg_values": values,
        "target_shape": target_shape,
    }


def _status(case: ET.Element) -> str:
    """passed | not_implemented | failed, or '' for a test that gave no verdict."""
    if case.find("failure") is not None or case.find("error") is not None:
        return "failed"
    skipped = case.find("skipped")
    if skipped is None:
        return "passed"
    return "not_implemented" if skipped.get("type") == "pytest.xfail" else ""


def capability_results(xml_path) -> list:
    """One CapabilityWriter entry per recorded test case in one suite's report."""
    out = []
    for case in ET.parse(xml_path).getroot().iter("testcase"):
        props, tags = {}, []
        for p in case.iter("property"):
            name, value = p.get("name", ""), p.get("value", "")
            if name == "tag":
                tags.append(value)
            elif name.startswith("result."):
                props[name[len("result.") :]] = value
        status = _status(case)
        op = normalize_op(props.get("op", ""))
        if not status or not op.startswith("torch."):
            continue
        model = next((t[len("model__") :] for t in tags if t.startswith("model__")), "")
        sig = signature(op, json.loads(props.get("args") or "[]"))
        backend = props.get("backend") or "spyre"
        out.append(
            {
                "subject": model,
                "name": op,
                "status": status,
                "backend": backend if status == "passed" else "spyre",
                "disc": {k: json.dumps(sig[k]) for k in DISC_KEYS},
                "tags": [props["variant"]] if props.get("variant") else [],
                "props": {
                    k: v
                    for k, v in (
                        ("test_name", case.get("name", "")),
                        ("input_strides", json.dumps(sig["input_strides"])),
                        ("fallback_ops", props.get("fallback_ops", "")),
                    )
                    if v and v != "[]"
                },
            }
        )
    return out


def _verdicts(entries) -> Counter:
    """(capability, backend, status) multiset, keyed as capability_id is: subject aside."""
    return Counter(
        (
            e["name"],
            e["disc"]["input_shapes"],
            e["disc"]["input_dtypes"],
            e["backend"],
            e["status"],
        )
        for e in entries
    )


def parity(xml_paths, parser_json) -> int:
    """Differences between the JUnit verdicts and the log parser's, per suite."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest_model_ops import _v2_capability_results

    old = {
        r["model_name"]: _v2_capability_results([r])
        for r in json.load(open(parser_json))
    }
    diffs = 0
    for path in xml_paths:
        new = _verdicts(capability_results(path))
        # Suites are named differently on each side, so pair by the verdicts they share.
        model, entries = max(
            old.items(), key=lambda kv: sum((_verdicts(kv[1]) & new).values())
        )
        prev = _verdicts(entries)
        only_new, only_old = new - prev, prev - new
        diffs += sum(only_new.values()) + sum(only_old.values())
        print(
            f"{Path(path).name} ~ {model}: {sum(new.values())} vs {sum(prev.values())} verdicts,"
            f" {sum((new & prev).values())} equal"
        )
        for sign, side in (("+", only_new), ("-", only_old)):
            for key, n in sorted(side.items()):
                print(f"  {sign} {n} x {key}")
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--xml", nargs="+", required=True, help="model-ops JUnit reports")
    ap.add_argument(
        "--parity", metavar="JSON", help="parse_model_ops_logs.py output to compare"
    )
    args = ap.parse_args()
    if args.parity:
        sys.exit(1 if parity(args.xml, args.parity) else 0)
    json.dump(
        [e for p in args.xml for e in capability_results(p)], sys.stdout, indent=1
    )


if __name__ == "__main__":
    main()
