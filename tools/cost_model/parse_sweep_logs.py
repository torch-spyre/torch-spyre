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

"""Parse profile_ops sweep logs into a unified record file (JSON + CSV).

Each profiled run emits a ``SUMMARY ...`` line (always), plus -- when the dumps are on
``op_it_space_splits=...`` line. Sweeps prefix each run with a ``## section`` header and
a ``-- label`` (or a ``=== ... ===`` run header). This walks those lines and emits ONE
record per SUMMARY, enriched with the section, label, forced + actual split, the MODEL
term breakdown (compute / base / turn / eff_underfill / c_loop / T_model), and the
per-tensor IO. Records from several logs MERGE into one file keyed by ``log:lineno``, so
re-parsing a log updates its rows in place (idempotent) without touching other logs.

PROVENANCE / VERSION HYGIENE. ``pred_us`` / ``T_model_us`` are baked into each log at RUN
time, so a log carries the predictions of whatever model version produced it -- mixing
logs mixes model generations. Every sweep header echoes ``git: <sha>``; we capture it as
``model_sha`` (and ``log_date`` from the filename) on each record, then flag ``is_current``
= (model_sha == the current-model SHA). The MEASURED ``kernel_us`` is version-independent
and always trustworthy; ``pred_us`` is only trustworthy for ``is_current`` rows. Use
``--current-sha`` to set the authoritative SHA (default: the SHA of the newest parsed log)
and ``--drop-ops`` to exclude retired ops (e.g. ``chain``) from the output entirely.

    python tools/cost_model/parse_sweep_logs.py examples/*.log
    python tools/cost_model/parse_sweep_logs.py                 # default: sweep_logs/*.log
    python tools/cost_model/parse_sweep_logs.py --out tools/cost_model/sweep_records.json <logs...>
    python tools/cost_model/parse_sweep_logs.py --drop-ops chain --current-sha c2bb9ce

Writes ``<out>.json`` (full structure) and ``<out>.csv`` (flat scalar columns).
"""

import argparse
import csv
import glob
import json
import os

import regex as re

_GIT = re.compile(r"^git:\s*([0-9a-f]{7,40})\b")
_LOG_DATE = re.compile(r"_(\d{8})_(\d{6})")  # <name>_YYYYMMDD_HHMMSS.log
_SECTION = re.compile(r"^##\s+(.*\S)")
_LABEL = re.compile(r"^--\s+(.*\S)")
_RUN_HDR = re.compile(r"^===\s+(.*?)\s+===\s*$")  # 3-equals run header (not the banner)
_KV = re.compile(r"(\w+)=(\S+)")
_SPLITS = re.compile(r"(d\d+):\s*(\d+)")
_LBL_MNK = re.compile(r"\bM=(\d+)\s+K=(\d+)\s+N=(\d+)")
_LBL_SPLIT = re.compile(r"\bsplit\s+m=(\d+)\s+n=(\d+)\s+k=(\d+)")
_LBL_B = re.compile(r"\bB=(\d+)")  # batch size for bmm_wd labels
_LBL_BMNK_SPLIT = re.compile(
    r"\bb=(\d+)\s+m=(\d+)\s+n=(\d+)\s+k=(\d+)"
)  # forced bmm split
_LBL_LAYOUT = re.compile(
    # TWO label forms exist. The original sweeps wrote `layoutA=0,1,2 layoutB=1,0,2`;
    # later ones write `A=0,1,2 B=0,1,2`. Matching only the first silently dropped the
    # layout on 84 of 138 bmm_layout rows (61 %) -- the two NEWEST logs -- so any figure
    # or fit keyed on layout quietly saw less than half the data.
    r"\blayoutA=([\d,]+)\s+layoutB=([\d,]+)|\bA=(\d,\d,\d)\s+B=(\d,\d,\d)"
)  # bmm_layout dim_orders
_LBL_RPC = re.compile(r"rows/core=(\d+)")
_LBL_FA = re.compile(  # flash_attn hint knobs from the label
    r"H=(\d+)\s+Lq=(\d+)\s+Lk=(\d+)\s+D=(\d+)\s+htiles=(\d+)\s+"
    r"qtiles=(\d+)\s+ktiles=(\d+)\s+wd=(\S+)"
)

_M_RW = re.compile(r"R=(\d+) B.*?W=(\d+) B.*?loop_trip L=(\d+)")
_M_BASE = re.compile(r"base =.*?=\s*([\d.]+) us")
_M_TURN = re.compile(r"turn =.*?=\s*([\d.]+) us")
_M_COMPUTE = re.compile(
    r"compute =.*?=\s*([\d.]+) us\s+\(M/m=([\d.]+), pt_eff=([\d.]+)\)"
)
_M_EFF = re.compile(r"eff_underfill = min\([^,]+,.*?\)\s*=\s*([\d.]+)\b")
_M_LOOP = re.compile(r"loop = c_loop\*L =.*?=\s*([\d.]+) us")
_M_TMODEL = re.compile(r"=> T_model =\s*([\d.]+) us")

_IO_OP = re.compile(r"^IO   op (\S+)(?:\s+\[(\w+)\])?")
_IO_TENSOR = re.compile(
    r"^IO     (input|output)\s+(\S+)\s+(?:torch (\[[^\]]*\]) -> )?"
    r"device (\[[^\]]*\]) in (\w+) = (\d+) elems x \d+B = (\d+) B"
    r"\s+\(hbm counted: (\d+) B\)(.*)$"
)
_IO_TOTAL = re.compile(r"=> HBM I/O total = (\d+) B\s+\(lx (\d+) B")
_FEATS = re.compile(
    r"^MODEL FEATS (.+)$"
)  # serialized OpFeatures bundle (offline scoring)


def _num(s):
    """``int`` for an integral literal, ``float`` for a decimal, else the string."""
    try:
        f = float(s)
    except (TypeError, ValueError):
        return s
    if "." not in s and "e" not in s.lower() and f.is_integer():
        return int(f)
    return f


def _parse_summary(line):
    d = {k: _num(v) for k, v in _KV.findall(line)}
    d["failed"] = "FAILED" in line or "kernel_us" not in d
    return d


def _parse_model(text):
    """Pull the term breakdown out of the joined ``MODEL`` lines of one run."""
    m = {}
    if g := _M_RW.search(text):
        m["R"], m["W"], m["loop_trip"] = int(g[1]), int(g[2]), int(g[3])
    for key, rx in (
        ("base_us", _M_BASE),
        ("turn_us", _M_TURN),
        ("eff_underfill", _M_EFF),
        ("loop_us", _M_LOOP),
        ("T_model_us", _M_TMODEL),
    ):
        if g := rx.search(text):
            m[key] = float(g[1])
    if g := _M_COMPUTE.search(text):
        m["compute_us"], m["m_over_m"], m["pt_eff"] = (
            float(g[1]),
            float(g[2]),
            float(g[3]),
        )
    return m


def _parse_io(io_lines):
    """List of per-tensor dicts + the HBM/LX totals from one run's ``IO`` block."""
    ops, cur, total = [], None, {}
    for ln in io_lines:
        if g := _IO_OP.match(ln):
            cur = {"op": g[1], "kind": g[2], "tensors": []}
            ops.append(cur)
        elif (g := _IO_TENSOR.match(ln)) and cur is not None:
            cur["tensors"].append(
                {
                    "role": g[1],
                    "name": g[2],
                    "logical": g[3],
                    "device": g[4],
                    "mem": g[5],
                    "elems": int(g[6]),
                    "bytes": int(g[7]),
                    "hbm_counted": int(g[8]),
                    "flags": g[9].strip(),
                }
            )
        elif g := _IO_TOTAL.search(ln):
            total = {"hbm_bytes": int(g[1]), "lx_bytes": int(g[2])}
    return ops, total


def _derive(rec):
    """Add op-specific derived fields (N, splits, MACs, rows/core) from label."""
    op, label = rec.get("op"), rec.get("label") or ""
    if op in ("mm", "mmwd"):
        rec["M"], rec["K"] = rec.get("rows"), rec.get("cols")
        if g := _LBL_MNK.search(label):
            rec["M"], rec["K"], rec["N"] = int(g[1]), int(g[2]), int(g[3])
        if all(rec.get(d) for d in ("M", "K", "N")):
            rec["macs"] = rec["M"] * rec["N"] * rec["K"]
        if g := _LBL_SPLIT.search(label):
            rec["split_forced"] = {"m": int(g[1]), "n": int(g[2]), "k": int(g[3])}
        # actual on-device split: d0=M(m), d1=N(n), d2=K(k)
        sp = rec.get("op_it_space_splits") or {}
        if sp:
            rec["split_actual"] = {
                "m": sp.get("d0", 1),
                "n": sp.get("d1", 1),
                "k": sp.get("d2", 1),
            }
    elif op in ("bmm_wd", "bmm_wd_3d2d", "bmm_layout"):
        # Forced-split bmm (bmm_layout also carries a device dim_order per operand). The
        # LABEL is authoritative (we set the split), so read b/m/n/k from it; MACs include
        # the batch (B*M*N*K). split_actual is left as the raw op_it_space_splits dict --
        # d0..dN mapping shifts with whether the batch dim collapses (3-dim when B=1, 4-dim
        # when B>=2), so we do NOT hard-map it here.
        if g := _LBL_MNK.search(label):
            rec["M"], rec["K"], rec["N"] = int(g[1]), int(g[2]), int(g[3])
        if g := _LBL_B.search(label):
            rec["B"] = int(g[1])
        if all(rec.get(d) for d in ("B", "M", "K", "N")):
            rec["macs"] = rec["B"] * rec["M"] * rec["N"] * rec["K"]
        if g := _LBL_BMNK_SPLIT.search(label):
            rec["split_forced"] = {
                "b": int(g[1]),
                "m": int(g[2]),
                "n": int(g[3]),
                "k": int(g[4]),
            }
        if g := _LBL_LAYOUT.search(label):  # bmm_layout: the two operand dim_orders
            # groups 1/2 = the `layoutA=`/`layoutB=` form, 3/4 = the newer `A=`/`B=` form
            rec["layout_a"], rec["layout_b"] = (g[1], g[2]) if g[1] else (g[3], g[4])
    elif op == "flash_attn":
        # Multi-op coarse-tiled flash attention. The label carries the hint knobs; derive
        # the coarse loop trip count (h_tiles * q_tiles * k_tiles) and the ~2 matmul MACs.
        if g := _LBL_FA.search(label):
            rec["H"], rec["Lq"], rec["Lk"], rec["D"] = (
                int(g[1]),
                int(g[2]),
                int(g[3]),
                int(g[4]),
            )
            rec["h_tiles"], rec["q_tiles"], rec["k_tiles"] = (
                int(g[5]),
                int(g[6]),
                int(g[7]),
            )
            rec["wd"] = g[8]
            rec["loop_trips"] = rec["h_tiles"] * rec["q_tiles"] * rec["k_tiles"]
            # ~2 batched matmuls (QK^T reduces D, PV reduces Lk); B assumed 1 (the example)
            rec["macs"] = 2 * rec["H"] * rec["Lq"] * rec["Lk"] * rec["D"]
    elif op == "chain":
        if g := _LBL_RPC.search(label):
            rec["rows_per_core"] = int(g[1])
    return rec


def parse_log(path):
    """Yield one record dict per SUMMARY line in ``path``."""
    base = os.path.basename(path)
    dm = _LOG_DATE.search(base)
    log_date = f"{dm[1]}_{dm[2]}" if dm else ""
    model_sha = ""  # set by the first ``git: <sha>`` header line (sweep provenance)
    section = label = None
    splits, io_lines, model_lines, feats_json = {}, [], [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if (g := _GIT.match(line)) and not model_sha:
                model_sha = g[1][:7]
            elif g := _SECTION.match(line):
                section = g[1]
            elif g := _LABEL.match(line):
                label = g[1]
            elif (
                (g := _RUN_HDR.match(line))
                and "sweep" not in line
                and "DONE" not in line
            ):
                label = g[1]
            elif (g := _SPLITS.findall(line)) and "op_it_space_splits" in line:
                splits = {k: int(v) for k, v in g}
            elif line.startswith("IO "):
                io_lines.append(line)
            elif g := _FEATS.match(line):  # MODEL FEATS <json> (before generic MODEL)
                feats_json = g[1]
            elif line.startswith("MODEL "):
                model_lines.append(line)
            elif line.startswith("SUMMARY"):
                rec = {
                    "id": f"{base}:{lineno}",
                    "log_file": base,
                    "lineno": lineno,
                    "model_sha": model_sha,
                    "log_date": log_date,
                    "section": section,
                    "label": label,
                    "op_it_space_splits": splits or None,
                }
                rec.update(_parse_summary(line))
                if model_lines:
                    rec["model"] = _parse_model("\n".join(model_lines))
                if io_lines:
                    ops, total = _parse_io(io_lines)
                    rec["io"] = ops
                    rec["io_totals"] = total
                if feats_json:
                    try:
                        rec["feats"] = json.loads(feats_json)
                    except json.JSONDecodeError:
                        rec["feats"] = None
                yield _derive(rec)
                label, splits, io_lines, model_lines, feats_json = (
                    None,
                    {},
                    [],
                    [],
                    None,
                )


_CSV_COLS = [
    "id",
    "log_file",
    "model_sha",
    "log_date",
    "is_current",
    "section",
    "op",
    "M",
    "K",
    "N",
    "rows",
    "cols",
    "tiles",
    "lx",
    "rows_per_core",
    "cores",
    "macs",
    "kernel_us",
    "kernel_us_min",
    "kernel_us_median",
    "kernel_us_std",
    "kernel_us_cv",
    "reps",
    "pred_us",
    "err_pct",
    "bw_gbps",
    "memset_us",
    "other_dev_us",
    "total_dev_us",
    "io_hbm_bytes",
    "layout_a",
    "layout_b",
    "h_tiles",
    "q_tiles",
    "k_tiles",
    "loop_trips",
    "wd",
    "failed",
]


def _flatten(rec):
    row = {c: rec.get(c, "") for c in _CSV_COLS}
    m = rec.get("model", {})
    for k in (
        "compute_us",
        "base_us",
        "turn_us",
        "eff_underfill",
        "pt_eff",
        "loop_us",
        "T_model_us",
    ):
        row[k] = m.get(k, "")
    sf, sa = rec.get("split_forced"), rec.get("split_actual")
    row["split_forced"] = "" if not sf else f"{sf['m']}x{sf['n']}x{sf['k']}"
    row["split_actual"] = "" if not sa else f"{sa['m']}x{sa['n']}x{sa['k']}"
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="*", help="log files (default: sweep_logs/*.log)")
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(here))
    ap.add_argument("--out", default=os.path.join(here, "sweep_records.json"))
    ap.add_argument(
        "--drop-ops",
        default="",
        help="comma list of ops to EXCLUDE from output (e.g. chain)",
    )
    ap.add_argument(
        "--current-sha",
        default="",
        help="model SHA whose pred_us is authoritative (default: newest "
        "parsed log's git sha). Flags is_current on every record.",
    )
    args = ap.parse_args()

    logs = args.logs or sorted(
        glob.glob(
            os.path.join(root, "docs", "source", "user_guide", "examples", "*.log")
        )
    )
    parsed = {}
    for path in logs:
        n = 0
        for rec in parse_log(path):
            parsed[rec["id"]] = rec
            n += 1
        print(f"  {os.path.basename(path)}: {n} runs")

    # Merge: keep records from logs not in this run, replace those that are.
    merged = {}
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            merged = {r["id"]: r for r in json.load(f).get("records", [])}
    touched = {os.path.basename(p) for p in logs}
    merged = {i: r for i, r in merged.items() if r["log_file"] not in touched}
    merged.update(parsed)

    # Drop retired ops entirely (e.g. chain), then flag the authoritative model version.
    drop = {o.strip() for o in args.drop_ops.split(",") if o.strip()}
    records = [r for r in merged.values() if r.get("op") not in drop]
    # current-sha default: the git sha of the newest parsed log (by log_date).
    cur_sha = args.current_sha
    if not cur_sha:
        dated = [r for r in records if r.get("model_sha") and r.get("log_date")]
        if dated:
            cur_sha = max(dated, key=lambda r: r["log_date"])["model_sha"]
    for r in records:
        r["is_current"] = bool(cur_sha) and r.get("model_sha") == cur_sha
    records = sorted(records, key=lambda r: (r["log_file"], r["lineno"]))
    n_cur = sum(1 for r in records if r["is_current"])
    print(
        f"  current-model sha={cur_sha or '(none)'}: {n_cur} current rows"
        + (f"; dropped ops {sorted(drop)}" if drop else "")
    )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"records": records}, f, indent=2)
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    cols = _CSV_COLS + [
        "split_forced",
        "split_actual",
        "compute_us",
        "base_us",
        "turn_us",
        "eff_underfill",
        "pt_eff",
        "loop_us",
        "T_model_us",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(_flatten(r) for r in records)

    ok = sum(1 for r in records if not r.get("failed"))
    print(f"-> {len(records)} records ({ok} ok) -> {args.out} + {csv_path}")


if __name__ == "__main__":
    main()
