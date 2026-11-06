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

"""Fit of the PT short-tile factor on measured matmul programs.

Does a short per-core M tile (8 to 40 rows) make a matmul program slower? This
tool answers with the planner's own price. It rebuilds each measured program in
``matmul_short_tile_records.json`` as an ``OpFeatures`` record, prices it with
``cost_model.predict_ops`` and the co-optimizing allocator's ``CostParams``, and
adds to the compute term a short-tile factor
``(5 / passes) ** e``, ``passes = rows per core / 8`` (the form of
``work_division._pt_underfill_factor``), for every exponent ``e`` from 0 to 0.5.

It prints the RMS of the relative error for each ``e``, the best ``e``, the best
``e`` with each shape left out, and, for each pair of cuts of one shape and one
placement, the measured difference beside the price difference at ``e = 0`` and
at the old ``e = 0.25``.

The records are measured device time of Granite 3.3 8B projections at 64 and
512 rows; see the file's ``note`` and ``build``. The measurement belongs to that
build: re-measure before trusting it on another one.

    python3 tools/cost_model/matmul_short_tile_fit.py
"""

import collections
import json
import math
import os
from contextlib import contextmanager

from torch_spyre._inductor import cost_model, work_division
from torch_spyre._inductor.scratchpad.allocator import _COST_PARAMS

_HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(_HERE, "matmul_short_tile_records.json")
EXPONENTS = [e / 100 for e in range(51)]


def features(r):
    """The program as the extractor records a 2-D shared-weight projection."""
    M, K, N = r["rows"], r["k"], r["n"]
    m, n, k = r["cut"]
    out = cost_model.ArgTraffic(
        name="out",
        role="output",
        is_lx=r["output"] == "lx",
        elems=M * N,
        dims=[N // 64, M, 64],
        logical=[1, M, N],
        is_boundary=False,
    )
    act = cost_model.ArgTraffic(
        name="x",
        role="input",
        is_lx=r["input"] == "lx",
        elems=M * K,
        broadcast=True,
        dims=[K // 64, M, 64],
        logical=[1, M, K],
        replication=n,
        is_boundary=False,
    )
    weight = cost_model.ArgTraffic(
        name="w",
        role="input",
        is_lx=r["weight"] == "lx",
        elems=K * N,
        broadcast=True,
        dims=[N // 64, K, 64],
        logical=[K, N],
        replication=m,
        is_boundary=True,
    )
    return cost_model.OpFeatures(
        name="mm",
        is_reduction=True,
        out_elems=M * N,
        cores=m * n * k,
        dtype_bytes=2,
        args=[out, act, weight],
        reduction_cores=k,
        is_matmul=True,
        matmul_macs=M * N * K,
        matmul_rows_per_core=M / m,
        matmul_cols_per_core=N / n,
        matmul_a_bytes=M * K * 2,
        matmul_b_bytes=K * N * 2,
        matmul_m_split=m,
        matmul_n_split=n,
    )


@contextmanager
def short_tile_factor(exponent):
    """Add (factor - 1) x compute to the planner's matmul compute term."""
    real = cost_model._matmul_execution_cost
    saved = work_division._PT_EFFICIENCY_EXPONENT

    def priced(b_axis, m_axis, n_axis, k_axis, max_cores, *args, **kwargs):
        (B, b), (M, m), (N, n), (K, k) = b_axis, m_axis, n_axis, k_axis
        compute = B * M * N * K / (b * m * n * k) / work_division._PEAK_MACS_US_CORE
        compute *= 2 if k > 1 else 1
        factor = work_division._pt_underfill_factor(M // m)
        return (
            real(b_axis, m_axis, n_axis, k_axis, max_cores, *args, **kwargs)
            + (factor - 1) * compute
        )

    work_division._PT_EFFICIENCY_EXPONENT = exponent
    cost_model._matmul_execution_cost = priced
    try:
        yield
    finally:
        cost_model._matmul_execution_cost = real
        work_division._PT_EFFICIENCY_EXPONENT = saved


def prices(records, exponent):
    with short_tile_factor(exponent):
        return [
            cost_model.predict_ops([features(r)], _COST_PARAMS) / 1000 for r in records
        ]


def rms(records, exponent):
    errors = [
        (p - r["us"]) / r["us"] for p, r in zip(prices(records, exponent), records)
    ]
    return 100 * math.sqrt(sum(x * x for x in errors) / len(errors))


def best_exponent(records):
    return min(EXPONENTS, key=lambda e: rms(records, e))


def main():
    with open(RECORDS, encoding="utf-8") as f:
        records = json.load(f)["records"]
    print(f"{len(records)} measured programs ({RECORDS})")
    for label, subset in (
        ("all", records),
        ("64 rows", [r for r in records if r["rows"] == 64]),
        ("512 rows", [r for r in records if r["rows"] == 512]),
    ):
        print(
            f"  {label:9s} RMS {rms(subset, 0.0):5.1f}% at e = 0, "
            f"{rms(subset, 0.25):5.1f}% at e = 0.25, best e = {best_exponent(subset)}"
        )
    families = collections.defaultdict(list)
    for r in records:
        families[(r["shape"], r["rows"], r["k"], r["n"])].append(r)
    loo = sorted(
        {
            best_exponent([r for r in records if r not in fam])
            for fam in families.values()
        }
    )
    print(f"best e with each shape left out: {loo}")
    print(
        "cut pairs of one shape and placement (us: measured, price at e = 0, at 0.25):"
    )
    price0, price25 = prices(records, 0.0), prices(records, 0.25)
    index = {id(r): i for i, r in enumerate(records)}
    groups = collections.defaultdict(list)
    for r in records:
        groups[
            (
                r["shape"],
                r["rows"],
                r["k"],
                r["n"],
                r["input"],
                r["weight"],
                r["output"],
            )
        ].append(r)
    for key, members in sorted(groups.items()):
        members.sort(key=lambda r: r["us"])
        for slow in members[1:]:
            fast = members[0]
            i, j = index[id(fast)], index[id(slow)]
            print(
                f"  {key[0]} {key[1]:3d}x{key[2]}x{key[3]} {fast['cut']} vs "
                f"{slow['cut']}: measured {slow['us'] - fast['us']:+6.1f}, "
                f"e=0 {price0[j] - price0[i]:+6.1f}, e=0.25 {price25[j] - price25[i]:+6.1f}"
            )


if __name__ == "__main__":
    main()
