# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

"""Find the first kernel where two captures of one program part ways.

Capture the program both ways with ``--save-inputs``, then compare::

    python tests/op_specs/capture.py prog.py --save-inputs --out good
    SOME_FLAG=1 python tests/op_specs/capture.py prog.py --save-inputs --out bad
    python tests/op_specs/compare.py good bad

Kernels are paired by name and argument shapes, in launch order among kernels
that share both (the ``_1`` suffix of a script name follows compile order, which
two runs need not share), and walked in the order they were first launched.

A kernel's inputs are the args it reads, as they were going in; its outputs are
the args its launch changed in either run. Values are compared as host tensors,
so two runs may lay a tensor out differently on the device as long as the copy
to the host reads both faithfully. It does not for every element arrangement
(#4393), so a pair whose arrangements differ is reported as not comparable
rather than compared. A NaN or an infinity in one run where the other has a
different value counts as an infinite difference.

The first kernel whose inputs agree but whose outputs do not, or whose launch
raised in one run only, is where the two runs part; the commands printed for it
replay the bad capture's script on the good capture's inputs, so its
``max|got - observed|`` lines compare against the good outputs.

Exit status:
    0   every kernel pairs, and every pair agrees
    3   a kernel parts (``EXIT_PARTS``)
    4   no kernel parts, but the runs are not shown to agree: a kernel only one
        run launched, a pair that could not be compared, or a pair whose inputs
        already differ (``EXIT_INCONCLUSIVE``); the first is named
    1   nothing to compare
"""

import argparse
from pathlib import Path

import torch

_SUFFIX = ".inputs.pt"

# Distinct from 1 (nothing to compare, or a traceback) and from argparse's 2.
EXIT_PARTS = 3
EXIT_INCONCLUSIVE = 4


def _load(directory: str) -> dict[tuple, list[tuple[str, dict]]]:
    """Every kernel in a capture directory that saved what its launch wrote, as
    (script name, record) lists in launch order, keyed by kernel name and
    argument shapes."""
    if not Path(directory).is_dir():
        raise SystemExit(f"no such directory: {directory}")
    saved: dict[tuple, list[tuple[str, dict]]] = {}
    for path in sorted(Path(directory).glob(f"*{_SUFFIX}")):
        # Mapped rather than read: a model's captures hold its weights many times.
        record = torch.load(path, mmap=True)
        if record.get("outputs") is None:
            continue
        stem = path.name[: -len(_SUFFIX)]
        shapes = tuple((tuple(t.shape), t.dtype) for t in record["tensors"])
        saved.setdefault((record.get("name", stem), shapes), []).append((stem, record))
    for kernels in saved.values():
        kernels.sort(key=lambda kernel: kernel[1]["launch"])
    return saved


def _pair(good: dict, bad: dict) -> tuple[list, list]:
    """(good stem, bad stem, good record, bad record) for every pair, in good's
    launch order, and (run, stem, record) for each kernel left over."""
    pairs, unpaired = [], []
    for key in good.keys() | bad.keys():
        g, b = good.get(key, []), bad.get(key, [])
        pairs += [(gs, bs, gr, br) for (gs, gr), (bs, br) in zip(g, b)]
        unpaired += [("good", stem, record) for stem, record in g[len(b) :]]
        unpaired += [("bad", stem, record) for stem, record in b[len(g) :]]
    pairs.sort(key=lambda pair: pair[2]["launch"])
    unpaired.sort(key=lambda kernel: (kernel[2]["launch"], kernel[0], kernel[1]))
    return pairs, unpaired


def _after(record: dict) -> list:
    """What each arg held once the launch returned: ``outputs`` holds None for an
    arg the launch left as it was."""
    return [t if o is None else o for t, o in zip(record["tensors"], record["outputs"])]


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest elementwise difference, relative to the largest finite magnitude in
    ``a``; infinite where the two disagree on a NaN or an infinity."""
    if a.shape != b.shape:
        return float("inf")
    a, b = a.float(), b.float()
    finite = a.isfinite() & b.isfinite()
    if not ((a == b) | (a.isnan() & b.isnan()))[~finite].all():
        return float("inf")
    a, b = a[finite], b[finite]
    if a.numel() == 0:
        return 0.0
    return (a - b).abs().max().item() / max(a.abs().max().item(), 1e-30)


def _worst(pairs) -> tuple[int | None, float]:
    """The arg index and relative difference of the pair that agrees least."""
    worst: tuple[int | None, float] = (None, 0.0)
    for i, (a, b) in pairs:
        diff = _rel_diff(a, b)
        if diff > worst[1]:
            worst = (i, diff)
    return worst


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("good", help="capture directory of the run that is right")
    parser.add_argument("bad", help="capture directory of the run that is wrong")
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-2,
        help="relative difference above which two tensors disagree (default 1e-2)",
    )
    args = parser.parse_args(argv)
    good, bad = _load(args.good), _load(args.bad)
    if not good or not bad:
        raise SystemExit("nothing to compare: capture both runs with --save-inputs")
    pairs, unpaired = _pair(good, bad)
    # (launch, kernel, why) for whatever leaves the runs' agreement unshown.
    loose = []
    for run, stem, record in unpaired:
        print(f"unpaired: {run}/{stem} (launch {record['launch']})")
        why = f"only the {run} run launched a kernel of its name and argument shapes"
        loose.append((record["launch"], f"{run}/{stem}", why))

    print(f"{'launch':>6}  {'inputs':>9}  {'outputs':>9}  kernel")
    parted = None
    for name, bad_name, g, b in pairs:
        label = name if bad_name == name else f"{name} (= {bad_name})"
        launch = g["launch"]
        arrangements = zip(g.get("arrangements", []), b.get("arrangements", []))
        differ = [i for i, (x, y) in enumerate(arrangements) if x != y]
        if differ:
            note = f"not comparable: arg{differ[0]}'s element arrangement differs"
            print(f"{launch:>6}  {'-':>9}  {'-':>9}  {label}  ({note})")
            loose.append((launch, name, note))
            continue
        everything = [True] * len(g["tensors"])
        reads = zip(g.get("reads", everything), b.get("reads", everything))
        read = [i for i, (x, y) in enumerate(reads) if x or y]
        _, inputs = _worst((i, (g["tensors"][i], b["tensors"][i])) for i in read)
        if inputs > args.tol:
            loose.append((launch, name, f"its inputs already differ by {inputs:.3g}"))
        agree = f"inputs agree to {inputs:.3g}"

        mark = ""
        raised = [run for run, r in (("good", g), ("bad", b)) if r.get("raised")]
        if raised:
            where = "both runs" if len(raised) == 2 else f"the {raised[0]} run only"
            note = f"launch raised in {where}"
            if parted is None and len(raised) == 1 and inputs <= args.tol:
                parted = (name, bad_name, launch, f"{agree}, but its {note}")
                mark = "  <-- parts here"
            print(f"{launch:>6}  {inputs:>9.3g}  {'-':>9}  {label}  ({note}){mark}")
            continue
        g_after, b_after = _after(g), _after(b)
        outs = zip(g["outputs"], b["outputs"])
        changed = [i for i, (x, y) in enumerate(outs) if x is not None or y is not None]
        arg, outputs = _worst((i, (g_after[i], b_after[i])) for i in changed)
        if parted is None and inputs <= args.tol < outputs:
            why = f"{agree}, output arg{arg} differs by {outputs:.3g}"
            parted = (name, bad_name, launch, why)
            mark = "  <-- parts here"
        print(f"{launch:>6}  {inputs:>9.3g}  {outputs:>9.3g}  {label}{mark}")

    if parted is not None:
        name, bad_name, launch, why = parted
        print(
            f"\nfirst to part: {name} (launch {launch}): {why}."
            "\nReplay the bad kernel on the good inputs, against the good outputs:"
            f"\n  mkdir -p replay && cp {args.bad}/{bad_name}.py replay/"
            f"\n  cp {args.good}/{name}{_SUFFIX} replay/{bad_name}{_SUFFIX}"
            f"\n  python replay/{bad_name}.py"
        )
        return EXIT_PARTS
    if loose:
        launch, kernel, why = min(loose)
        print(
            "\nno kernel parts, but the runs are not shown to agree; first at"
            f" {kernel} (launch {launch}): {why}."
        )
        return EXIT_INCONCLUSIVE
    print("\nno kernel parts: every kernel pairs, and every pair agrees")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
