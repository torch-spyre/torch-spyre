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

"""Applies schema/*.sql to ClickHouse in filename order; idempotent, never ALTERs."""

import argparse
import sys
from pathlib import Path

import regex as re


class SchemaApplier:
    """Locates, parses and applies the DDL files in schema/."""

    # A trailing ';' is optional, and two files carry ';' inside comment prose, so
    # statements are split after stripping comments, not on every ';' in the text.
    LINE_COMMENT = re.compile(r"--[^\n]*")
    BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
    # Minimum server version per file, from a `-- NEEDS CLICKHOUSE >= X.Y` header line,
    # declared in the file so the requirement travels with the DDL that has it.
    NEEDS_VERSION = re.compile(
        r"--\s*NEEDS CLICKHOUSE >=\s*(\d+)\.(\d+)", re.IGNORECASE
    )

    @staticmethod
    def schema_dir() -> Path:
        """Where the .sql files live: a checkout's sibling dir, else the install."""
        here = Path(__file__).resolve().parent
        sibling = here.parent / "schema"
        return sibling if sibling.is_dir() else here / "schema"

    @classmethod
    def sql_files(cls, schema_dir: Path) -> list:
        """The DDL files in apply order: filename order (prefix encodes dependency)."""
        return sorted(schema_dir.glob("*.sql"))

    @classmethod
    def statements(cls, text: str) -> list:
        """The executable statements in one file, comments removed."""
        stripped = cls.LINE_COMMENT.sub("", cls.BLOCK_COMMENT.sub("", text))
        return [s.strip() for s in stripped.split(";") if s.strip()]

    @classmethod
    def required_version(cls, text: str) -> tuple:
        """The (major, minor) floor this file declares, or () when it declares none."""
        m = cls.NEEDS_VERSION.search(text)
        return (int(m.group(1)), int(m.group(2))) if m else ()

    @staticmethod
    def version_tuple(text: str) -> tuple:
        return tuple(int(p) for p in re.findall(r"\d+", text)[:2])

    @classmethod
    def apply_file(
        cls, client, path: Path, dry_run: bool = False, text: str = ""
    ) -> int:
        """Execute one DDL file. Returns the number of statements applied."""
        stmts = cls.statements(text or path.read_text())
        for stmt in stmts:
            if dry_run:
                print(f"    {stmt.splitlines()[0][:100]}")
                continue
            client.command(stmt)
        return len(stmts)

    @classmethod
    def apply_all(cls, client, schema_dir: Path, dry_run: bool = False) -> int:
        """Apply every DDL file; skips one whose version floor the server misses."""
        files = cls.sql_files(schema_dir)
        if not files:
            # Raise, not return 0: "applied 0 statements" would read as success and
            # leave the caller believing a database it never created is ready.
            raise FileNotFoundError(f"no .sql files in {schema_dir}")
        server = ()
        if not dry_run:
            server = cls.version_tuple(client.command("SELECT version()"))
        total = 0
        for path in files:
            text = path.read_text()
            need = cls.required_version(text)
            if need and server and server < need:
                print(
                    f"  {path.name:34} SKIPPED -- needs ClickHouse >= "
                    f"{need[0]}.{need[1]}, server is {server[0]}.{server[1]}"
                )
                continue
            stmts = cls.apply_file(client, path, dry_run=dry_run, text=text)
            total += stmts
            print(f"  {path.name:34} {stmts} statement(s)")
        return total


SCHEMA_DIR = SchemaApplier.schema_dir()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the v2 schema DDL to a ClickHouse database"
    )
    parser.add_argument(
        "--schema-dir",
        type=Path,
        default=SCHEMA_DIR,
        help=f"Directory of *.sql files (default: {SCHEMA_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the statements that would run, without connecting",
    )
    args = parser.parse_args()

    files = SchemaApplier.sql_files(args.schema_dir)
    if not files:
        print(f"[error] No .sql files in {args.schema_dir}", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print(f"[dry-run] Would apply {len(files)} file(s) from {args.schema_dir}:")
        SchemaApplier.apply_all(None, args.schema_dir, dry_run=True)
        return

    from .client import client_summary, get_client

    print(f"[info] Applying {len(files)} file(s) to {client_summary()} ...")
    client = get_client()
    total = SchemaApplier.apply_all(client, args.schema_dir)
    print(f"[info] Applied {total} statement(s) from {len(files)} file(s).")


if __name__ == "__main__":
    main()
