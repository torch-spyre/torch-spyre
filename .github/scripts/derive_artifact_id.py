#!/usr/bin/env python3
"""Derive the artifact_id for what this GHA leg actually ran.

Runs ON the image under test, because the base id lives in that image's filesystem and the
installed delta is known only to the workflow that installed it. Called by the
derive-gha-artifact-id composite action; see that action for how the result travels.

Prints a one-line record for the ingest to parse:

    <artifact_id>|<base_artifact_id>|<installed,comma,joined>

The two extra fields are the HASH INPUTS, carried because a uuid5 cannot be read back.

Never fails a build: any un-derivable id (an image baked before spyre-frameworks #1782, a
missing library, an unreadable file) exits 0 with empty outputs, and the ingest then writes
cases with no artifact row, exactly as it did before this existed.
"""

import argparse
import importlib
import os
import sys
import types

RECORD_SEP = "|"

_PKG = "_spyre_identity_only"  # real package's __init__ pulls clickhouse_connect


def _identity(library_dir: str):
    """Import the SHARED identity module -- a local copy would be a second definition."""
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [os.path.join(library_dir, "spyre_clickhouse_ingest")]
    sys.modules[_PKG] = pkg
    # Rebound every call, and submodules from an earlier one dropped, so a second call with a
    # different library_dir is honoured rather than silently served the first one's.
    for name in [n for n in sys.modules if n.startswith(f"{_PKG}.")]:
        del sys.modules[name]
    mod = importlib.import_module(f"{_PKG}.identity")
    return mod.BASE_ARTIFACT_ID_FILE, mod.base_artifact_id, mod.gha_artifact_id


def derive(
    component: str,
    arch: str,
    installed: str,
    base_id_file: str = "",
    library_dir: str = "",
) -> tuple[str, str, str]:
    """-> (record, artifact_id, base_artifact_id), each '' when not derivable."""
    default_file, read_base, derive_id = _identity(library_dir)
    base = read_base(base_id_file or default_file)
    # Comma-joined to survive the recovery's whitespace strip; the library splits on both.
    installed = ",".join((installed or "").replace(",", " ").split())
    if not base:
        # An image baked before the id was stamped: nothing to chain onto, and inventing a
        # coordinate would give every such leg one shared, wrong identity.
        return "", "", ""
    # No delta means the image ran UNCHANGED, so its own id is already correct.
    aid = base if not installed else derive_id(component, base, installed, arch)
    if not aid:
        return "", "", base
    return RECORD_SEP.join([aid, base, installed]), aid, base


def _emit(**outputs: str) -> None:
    """Append to $GITHUB_OUTPUT when the runner set it; a local run just skips this."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a") as fh:
        for name, value in outputs.items():
            fh.write(f"{name}={value}\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--component", default="", help="hash input; a wrong value re-keys the id"
    )
    ap.add_argument("--arch", default="", help="folded to x86_64 by the library")
    ap.add_argument(
        "--installed",
        default="",
        help="what this leg installed ON TOP of the image, space- or comma-separated. "
        "Empty means the image ran unchanged and its own id is used verbatim.",
    )
    ap.add_argument("--base-id-file", default="", help="override the in-image path")
    ap.add_argument(
        "--library-dir",
        default="",
        help="directory holding spyre_clickhouse_ingest, prepended to sys.path",
    )
    ap.add_argument(
        "--output-file", default="", help="write the record here when derived"
    )
    args = ap.parse_args(argv)

    try:
        record, aid, base = derive(
            args.component,
            args.arch,
            args.installed,
            args.base_id_file,
            args.library_dir,
        )
    except Exception as err:
        # Telemetry must never redden a test run, so a broken import or an unreadable file
        # degrades to no id rather than a non-zero exit.
        print(f"artifact_id: <none — derivation failed: {err!r}>", file=sys.stderr)
        _emit(artifact_id="", base_artifact_id="")
        return 0

    if aid and aid == base:
        print(f"artifact_id: {aid} (image ran unchanged — base id used verbatim)")
    elif aid:
        print(f"artifact_id: {aid} (base {base} + installed: {args.installed})")
    else:
        print("artifact_id: <none — this leg's rows will carry no artifact>")
        print(
            f"  base id read from: {args.base_id_file or '<library default>'} -> {base or '<absent>'}"
        )

    if record and args.output_file:
        with open(args.output_file, "w") as fh:
            fh.write(record)

    _emit(artifact_id=aid, base_artifact_id=base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
