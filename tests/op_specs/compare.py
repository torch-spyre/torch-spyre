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
two runs need not share), and walked in the order they were first launched. Values are compared as host tensors, so captures whose device layouts
differ still compare. The first kernel whose inputs agree but whose outputs do
not is where the two runs part; the commands printed for it replay the bad
capture's script on the good capture's inputs, so its ``max|got - observed|``
lines compare against the good outputs.

Exit status: 0 if no kernel parts, 1 if one does.
"""

import argparse
from pathlib import Path

import torch

_SUFFIX = ".inputs.pt"


def _load(directory: str) -> dict[tuple, list[tuple[str, dict]]]:
    """Every kernel in a capture directory that saved what its launch wrote, as
    (script name, record) lists in launch order, keyed by kernel name and
    argument shapes."""
    saved: dict[tuple, list[tuple[str, dict]]] = {}
    for path in sorted(Path(directory).glob(f"*{_SUFFIX}")):
        record = torch.load(path)
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
    launch order, and the stems of the kernels left over on either side."""
    pairs, unpaired = [], []
    for key in good.keys() | bad.keys():
        g, b = good.get(key, []), bad.get(key, [])
        pairs += [(gs, bs, gr, br) for (gs, gr), (bs, br) in zip(g, b)]
        unpaired += [stem for stem, _ in g[len(b) :] + b[len(g) :]]
    pairs.sort(key=lambda pair: pair[2]["launch"])
    return pairs, sorted(unpaired)


def _rel_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    """Largest elementwise difference, relative to the largest magnitude in ``a``."""
    if a.shape != b.shape:
        return float("inf")
    a, b = a.float(), b.float()
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
    for stem in unpaired:
        print(f"unpaired: {stem}")

    print(f"{'launch':>6}  {'inputs':>9}  {'outputs':>9}  kernel")
    parted = None
    for name, bad_name, g, b in pairs:
        _, inputs = _worst(enumerate(zip(g["tensors"], b["tensors"])))
        # The outputs are the args a launch changed, in either capture.
        written = [
            (i, (g["outputs"][i], b["outputs"][i]))
            for i in range(len(g["tensors"]))
            if not torch.equal(g["tensors"][i], g["outputs"][i])
            or not torch.equal(b["tensors"][i], b["outputs"][i])
        ]
        arg, outputs = _worst(written)
        mark = ""
        if parted is None and inputs <= args.tol < outputs:
            parted = (name, bad_name, g["launch"], arg, inputs, outputs)
            mark = "  <-- parts here"
        if bad_name != name:
            mark = f" (= {bad_name}){mark}"
        print(f"{g['launch']:>6}  {inputs:>9.3g}  {outputs:>9.3g}  {name}{mark}")

    if parted is None:
        print("\nno kernel parts: wherever the inputs agree, so do the outputs")
        return 0
    name, bad_name, launch, arg, inputs, outputs = parted
    print(
        f"\nfirst to part: {name} (launch {launch}): inputs agree to"
        f" {inputs:.3g}, output arg{arg} differs by {outputs:.3g}."
        "\nReplay the bad kernel on the good inputs, against the good outputs:"
        f"\n  mkdir -p replay && cp {args.bad}/{bad_name}.py replay/"
        f"\n  cp {args.good}/{name}{_SUFFIX} replay/{bad_name}{_SUFFIX}"
        f"\n  python replay/{bad_name}.py"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
