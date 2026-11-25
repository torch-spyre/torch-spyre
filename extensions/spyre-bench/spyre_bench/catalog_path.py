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

"""`spyre-bench catalogs`: the shared catalogs this package ships, and where they are installed.

A product catalog has to resolve every reference it makes or it does not validate at all, so the
nouns several products share (the recorded traces, the models, the card-count Environments) travel
with the engine instead of with one product. A caller passes the printed directory as `--central`,
where it composes with a deployment's own central catalog rather than replacing it.
"""

import argparse
from pathlib import Path

from spyre_bench.cliutil import UsageError, run_main

ROOT = Path(__file__).resolve().parent / "catalogs"


def available():
    return sorted(p.name for p in ROOT.iterdir() if p.is_dir() and any(p.glob("*.yaml"))) if ROOT.is_dir() else []


def path_of(name):
    directory = ROOT / name
    if not directory.is_dir() or not any(directory.glob("*.yaml")):
        raise UsageError(f"no packaged catalog {name!r} (have: {', '.join(available()) or 'none'})")
    return directory


def execute(a):
    if a.path:
        print(path_of(a.path))
        return 0
    names = available()
    if not names:
        print("this install ships no shared catalogs")
        return 0
    for name in names:
        print(f"{name}\t{ROOT / name}")
    return 0


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench catalogs", description=__doc__)
    ap.add_argument("--path", metavar="NAME", help="print just this catalog's directory, for --central")
    return ap


def main(argv=None):
    return run_main(execute, parser(), argv)


if __name__ == "__main__":
    raise SystemExit(main())
