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

"""Golden vectors for the on-demand flag grammar and host policy.

Other implementations of the grammar (a comment bot that cannot import the engine) copy golden/ondemand-vectors.json
and must reproduce every `out`. Rewrite after an intended change:
    PYTHONPATH=extensions/spyre-bench python extensions/spyre-bench/tests/test_ondemand_vectors.py --refresh
"""

import json
import sys
import unittest
from pathlib import Path

from spyre_bench import ondemand as od

GOLDEN = Path(__file__).resolve().parent / "golden" / "ondemand-vectors.json"
COMMANDS_GOLDEN = Path(__file__).resolve().parent / "golden" / "ondemand-command-vectors.json"
COMMAND_TEXTS = [
    "/spyre-bench sanity",
    "/bench sanity",
    "please\n/spyre-bench decoder+startup@sanity reuse=only force=a_1\nthanks",
    "/spyre-bench torch-spyre:a,b budget=3 sha=abcdef123456",
    "/spyre-bench sanity type=quality",
    "/spyre-bench sanity TEST_TYPE=quality",
    "/spyre-bench sanity test_type=perf",
    "/spyre-bench sanity arch=ppc64le",
    "/spyre-bench confirm",
    "/spyre-bench cancel",
    "/spyre-bench confirm now",
    "/spyre-bench sanity reuse",
    "/spyre-bench sanity reuse=only reuse=never",
    "/spyre-benchmark sanity",
    "  /spyre-bench sanity",
    "> /spyre-bench sanity",
    "```\n/spyre-bench sanity\n```",
    "<!-- /spyre-bench sanity -->",
    "no command here",
    "/spyre-bench",
]
WRITER = {"login": "dev1", "permission": "write", "association": "COLLABORATOR"}
READER = {"login": "rando", "permission": "read", "association": "NONE"}

CASES = [
    ({"bench": "sanity"}, "github.com", WRITER),
    ({"bench": "sanity"}, "ghe", READER),
    ({"bench": "decoder+startup@sanity"}, "github.com", WRITER),
    ({"bench": "decoder+startup"}, "ghe", WRITER),
    ({"bench": "spyre-inference:decoder@core;torch-spyre:ladder@core"}, "ghe", WRITER),
    ({"bench": "spyre-inference:decoder@core;torch-spyre:ladder@core"}, "github.com", WRITER),
    ({"bench": "a_1,b-2"}, "github.com", WRITER),
    ({"bench": "#pr-sanity-with-reuse"}, "ghe", WRITER),
    ({"bench": "hf-adapters:hf-models@sanity", "bench-reuse": "only"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-reuse": "never"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-reuse": "never"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-reuse": "alwys"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-force": "all"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-force": "all"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-force": "x_1,y", "bench-reuse": "prefer"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-force": "x", "bench-reuse": "never"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-force": "all,x"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-profile": "light"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-profile": "light"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-profile": "full"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "main"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-vs": "build:orch#12"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "tomorrow"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "nightly"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "tag:release-1"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "tag:release-1"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-vs": "build:orch#12"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-vs": "artifact:0b7f1c52-3d1e-4a39-9c21-5f6a8e2d4b10"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "artifact:0b7f1c52-3d1e-4a39-9c21-5f6a8e2d4b10"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-vs": "artifact:0B7F1C52-3D1E-4A39-9C21-5F6A8E2D4B10"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "artifact:not-an-id"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "ref:ref-a"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "ref:ref-a"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-vs": "ref:"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "ref:a b"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "ref:$(id)"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "tag:x;rm"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "build:a/../b#1"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "main:x"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "tag:[x](http://e.co)"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-clean": "true"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-clean": "true"}, "github.com", READER),
    ({"bench": "sanity", "bench-clean": "true", "bench-max-wait": "45m"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-clean": "true", "bench-max-wait": "2h"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-max-wait": "45m"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-clean": "maybe"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-env": "env-a"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-env": "env-a"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-budget": "3"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-budget": "50"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-budget": "nan"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-budget": "-1"}, "ghe", WRITER),
    ({"bench": "core"}, "github.com", WRITER),
    ({"bench": "core"}, "ghe", WRITER),
    ({"bench": "extended"}, "ghe", WRITER),
    ({"bench": "decoder@sanitty"}, "ghe", WRITER),
    ({"bench": "spyre-infernce:sanity"}, "ghe", WRITER),
    ({"bench": "torch-spyre:a;torch-spyre:b"}, "ghe", WRITER),
    ({"bench": "bad name"}, "ghe", WRITER),
    ({"bench": "a$(id)"}, "ghe", WRITER),
    ({"bench": ""}, "ghe", WRITER),
    ({"bench": "a,b@core"}, "ghe", WRITER),
    ({"purpose": "pr", "bench": "sanity"}, "ghe", WRITER),
    ({"bench": "sanity", "purpose": "nightly"}, "ghe", WRITER),
    ({"bench-reuse": "only"}, "ghe", WRITER),
    ({"bench": "sanity", "sha": "abcdef1"}, "github.com", WRITER),
    ({"bench": "sanity", "sha": "ABC"}, "github.com", WRITER),
    ({"bench": "@org/team"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-vs": "build:[click](http://e.co)#1"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-sha": "https://evil.example/x"}, "github.com", WRITER),
    ({"bench": "www.evil.com"}, "github.com", WRITER),
    ({"bench": "sanity", "bench-reuse": "![i](http://e/p.png)"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-budget": "1\u202e2"}, "ghe", WRITER),
    ({"bench": "sanity", "bench-budjet": "3"}, "ghe", WRITER),
]


def evaluate(flags, host, actor):
    try:
        req = od.parse_flags(flags)
    except od.RequestError as e:
        return {"error": str(e)}
    return {"request": req.to_dict(), "violations": od.check_policy(req, od.POLICIES[host], actor)}


def build():
    return [
        {"in": {"flags": flags, "host": host, "actor": actor}, "out": evaluate(flags, host, actor)}
        for flags, host, actor in CASES
    ]


def command_vector(text):
    try:
        cmd = od.parse_command(text)
    except od.RequestError as e:
        return {"error": str(e)}
    return {"kind": None} if cmd is None else {"kind": cmd.kind, "flags": cmd.flags}


REWRITE_GOLDEN = Path(__file__).resolve().parent / "golden" / "ondemand-rewrite-vectors.json"
REWRITE_TEXTS = [t for t in COMMAND_TEXTS if "confirm" not in t and "cancel" not in t and "reuse\n" not in t] + [
    "/spyre-bench sanity preset=nightly",
    "/spyre-bench sanity arch=amd64,s390x type=quality",
    "/spyre-bench sanity vs=ref:ref-a",
    "/spyre-bench sanity vs=artifact:0b7f1c52-3d1e-4a39-9c21-5f6a8e2d4b10",
    "/spyre-bench sanity max-wait=45m clean=true budget=3 sha=abcdef1 env=e1 profile=light vs=main purpose=pr",
]


def build_rewrites():
    out = []
    for t in REWRITE_TEXTS:
        try:
            cmd = od.parse_command(t)
            out.append({"in": {"text": t}, "out": None if cmd is None else od.to_spyre_test(cmd)})
        except od.RequestError as e:
            out.append({"in": {"text": t}, "out": {"error": str(e)}})
    return out


def build_commands():
    return [{"in": {"text": t}, "out": command_vector(t)} for t in COMMAND_TEXTS]


class Vectors(unittest.TestCase):
    def test_rewrite_golden_matches_the_engine(self):
        self.assertEqual(json.loads(REWRITE_GOLDEN.read_text()), build_rewrites(), "run with --refresh")

    def test_default_preset_and_explicit_preset_win(self):
        by = {v["in"]["text"]: v["out"] for v in build_rewrites()}
        self.assertEqual(by["/spyre-bench sanity"], "/spyre-test preset=perf-only bench=sanity TEST_TYPE=perf")
        self.assertEqual(
            by["/spyre-bench sanity preset=nightly"], "/spyre-test bench=sanity preset=nightly TEST_TYPE=perf"
        )
        self.assertNotIn("preset=perf-only", by["/spyre-bench sanity arch=amd64,s390x type=quality"])

    def test_command_golden_matches_the_engine(self):
        self.assertEqual(json.loads(COMMANDS_GOLDEN.read_text()), build_commands(), "run with --refresh")

    def test_golden_matches_the_engine(self):
        self.assertEqual(
            json.loads(GOLDEN.read_text()), build(), "run this file with --refresh after an intended change"
        )

    def test_every_branch_is_represented(self):
        outs = build()
        self.assertTrue(any("error" in o["out"] for o in outs))
        self.assertTrue(any(o["out"].get("violations") for o in outs))
        self.assertTrue(any("request" in o["out"] and not o["out"]["violations"] for o in outs))


if __name__ == "__main__":
    if "--refresh" in sys.argv:
        GOLDEN.write_text(json.dumps(build(), indent=1, sort_keys=True) + "\n")
        REWRITE_GOLDEN.write_text(json.dumps(build_rewrites(), indent=1, sort_keys=True) + "\n")
        COMMANDS_GOLDEN.write_text(json.dumps(build_commands(), indent=1, sort_keys=True) + "\n")
        print(f"wrote {GOLDEN}")
    else:
        unittest.main()
