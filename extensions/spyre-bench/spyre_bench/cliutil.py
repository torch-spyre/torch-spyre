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

"""Shared CLI plumbing for plan, run and verdict: BENCH_* env twins, catalog opening, exit codes, UTC stamps."""

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from spyre_bench import engine

EXIT_OK, EXIT_FAILED, EXIT_INVALID, EXIT_NOTHING, EXIT_STORE = 0, 1, 2, 4, 5
TRUE = ("1", "true", "yes", "on")


class UsageError(Exception):
    """A bad flag, request or config: exit 2 with the message."""


class Terminated(Exception):
    """SIGTERM arrived: unwind (killing children on the way) and exit 143."""


def env(name, default=None):
    value = os.environ.get(name)
    return value if value not in (None, "") else default


def pick(flag, name, default=None):
    """flag > BENCH_* env twin > default."""
    return flag if flag not in (None, "", []) else env(name, default)


def flag_on(flag, name):
    return bool(flag) or (env(name, "") or "").strip().lower() in TRUE


def csv(value):
    """A comma or whitespace separated list (flag, env or repeated flag values) as a list."""
    out = []
    for part in value if isinstance(value, (list, tuple)) else [value]:
        out += [x for x in str(part or "").replace(",", " ").split() if x]
    return out


def paths(flag, name):
    """Repeatable path flag, else the env twin split on the path separator."""
    return list(flag) if flag else [p for p in (env(name, "") or "").split(os.pathsep) if p]


def utc_stamp(when=None):
    return (when or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def host_arch():
    import platform

    m = platform.machine().lower()
    return {"amd64": "x86_64", "x86": "x86_64", "arm64": "aarch64"}.get(m, m)


def open_catalog(catalog, central, packs, required=True, implicit=True):
    """The Catalog, or None when no catalog directory exists and one is optional. Errors raise UsageError.

    With implicit=False only a named catalog (--catalog or BENCH_DIR) counts: no ./benchmarks default.
    """
    root = pick(catalog, "BENCH_DIR", "benchmarks" if implicit else None)
    if root is None or not Path(root).is_dir():
        if required:
            raise UsageError(f"catalog {root!r} is not a directory (--catalog or BENCH_DIR)")
        return None
    cat = engine.Catalog([root], paths(packs, "BENCH_PACK"), paths(central, "BENCH_CENTRAL"))
    if cat.errors:
        raise UsageError("\n".join(cat.errors))
    return cat


def add_catalog_flags(ap):
    ap.add_argument("--catalog", help="definitions directory (BENCH_DIR, default benchmarks)")
    ap.add_argument("--central", action="append", default=[], help="shared definitions (BENCH_CENTRAL)")
    ap.add_argument("--pack", action="append", default=[], help="extra pack directory (BENCH_PACK)")


def run_main(fn, ap, argv):
    """Parse and run a verb; UsageError and engine errors become exit 2 with the message on stderr."""
    a = ap.parse_args(argv)
    try:
        return fn(a)
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_INVALID
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except Terminated:
        print("terminated", file=sys.stderr)
        return 143
    except (FileNotFoundError, NotADirectoryError, ValueError, KeyError, TypeError, yaml.YAMLError) as e:
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)  # a missing or malformed input
        return EXIT_INVALID
    except OSError as e:  # the disk or the filesystem failed (a full disk after the ingest, say): not "invalid"
        print(f"error: I/O failure: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_STORE
