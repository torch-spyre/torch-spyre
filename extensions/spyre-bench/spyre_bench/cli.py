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

"""spyre-bench command line: validate, explain, gen-schemas."""

import sys

VERBS = {
    "validate": ("spyre_bench.validate", "validate catalogs against the core schemas, packs and cross-resource rules"),
    "explain": ("spyre_bench.explain", "show a case with every reference resolved, its identity and its plan record"),
    "gen-schemas": ("spyre_bench.gen_schemas", "regenerate (or --check) the core JSON Schemas"),
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in VERBS:
        out = sys.stdout if not argv or argv[0] in ("-h", "--help") else sys.stderr
        print("usage: spyre-bench VERB [args]\n", file=out)
        for verb, (_, doc) in VERBS.items():
            print(f"  {verb:<12} {doc}", file=out)
        return 0 if out is sys.stdout else 2
    import importlib

    try:
        return importlib.import_module(VERBS[argv[0]][0]).main(argv[1:])
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 2
        return e.code if isinstance(e.code, int) else 0
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"error: {e.filename or e}: no such file or directory", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
