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

"""The artifact flags, declared once for every surface (`artifacts resolve|ensure`, `results`).

`add_artifact_options(parser, ...)` adds them; `artifact_options(args)` turns the parsed flags
into the keyword arguments of resolver.resolve / resolver.ensure. A surface sets its own
defaults (origin, the --tag-date help) through the parameters, never by redeclaring a flag.
"""

import argparse
from datetime import date

from .resolver import LOOKUP_MODES, REGISTRY_MODES


def source(value: str) -> tuple:
    """`repo@ref@sha` or `repo@sha` -> (repo, git_ref, git_sha)."""
    parts = value.split("@")
    if len(parts) == 2:
        return parts[0], "", parts[1]
    if len(parts) == 3:
        return tuple(parts)
    raise argparse.ArgumentTypeError(f"--source wants repo@[ref@]sha, got {value!r}")


def _pair(value: str) -> tuple:
    key, sep, val = value.partition("=")
    if not (sep and key):
        raise argparse.ArgumentTypeError(f"wants key=value, got {value!r}")
    return key, val


def add_artifact_options(
    parser,
    *,
    origin: str = "built",
    tag_date_help: str = "default with a dated --tag-family: today (UTC)",
    platform_alias: bool = False,
    arch_required: bool = True,
    tag_family: str = "",
    group_title: str = "artifact",
):
    """Add every artifact flag to `parser` (one argparse group)."""
    g = parser.add_argument_group(group_title)
    g.add_argument(
        "--artifact",
        default="",
        help="what it names: <artifact_id> | id:<id> | image:<ref>[@digest] | rpm:<file|url|nevra|glob> "
        "| wheel:<name==version|file|url> | generic:<url>[#sha256] | gha:<id>|<base>|<installed>, "
        "each optionally + ;component=;name=;id12= (an authoritative identity)",
    )
    g.add_argument(
        "--artifact-id",
        default="",
        help="alias of --artifact for an id: a bare artifact_id, or derive-gha-artifact-id's "
        "<id>|<base>|<installed> record (= gha:<record>)",
    )
    names = ["--arch", "--platform"] if platform_alias else ["--arch"]
    g.add_argument(
        *names,
        dest="arch",
        default="",
        required=arch_required,
        help="the artifact's arch, or multi for a manifest list"
        + ("; --platform is a deprecated alias" if platform_alias else ""),
    )
    g.add_argument(
        "--tag-family",
        default=tag_family,
        help="tag it in this tag_families.yaml family: its registry tag, else dated by --tag-date; "
        "also the family of each --tag whose prefix names none",
    )
    g.add_argument(
        "--tag",
        dest="tags",
        action="append",
        default=[],
        help="a full tag, repeatable; it takes the family its prefix names, else --tag-family "
        "(refused with neither), and replaces the resolved tag of that family",
    )
    g.add_argument(
        "--tag-date",
        type=date.fromisoformat,
        default=None,
        help=f"YYYY-MM-DD; {tag_date_help}",
    )
    g.add_argument(
        "--lookup",
        default="auto",
        choices=LOOKUP_MODES,
        help="auto: an existing record wins; off: derive only, read no database; only: it must be recorded",
    )
    g.add_argument(
        "--registry",
        default="auto",
        choices=REGISTRY_MODES,
        help="auto: call the registry/Artifactory only when the spec needs it; off: never (fails if needed)",
    )
    g.add_argument(
        "--origin",
        default=origin,
        help=f"artifacts.origin of a new record (default {origin})",
    )
    g.add_argument(
        "--source",
        dest="sources",
        action="append",
        type=source,
        default=[],
        help="repo@[ref@]sha, repeatable",
    )
    g.add_argument(
        "--identity-dep",
        dest="identity_deps",
        action="append",
        default=[],
        help="repeatable, e.g. base=<sha256>",
    )
    g.add_argument(
        "--context-dep",
        dest="context_deps",
        action="append",
        default=[],
        help="repeatable",
    )
    g.add_argument(
        "--prop",
        dest="props",
        action="append",
        type=_pair,
        default=[],
        help="artifact prop k=v, repeatable",
    )
    g.add_argument(
        "--tag-prop",
        dest="tag_props",
        action="append",
        type=_pair,
        default=[],
        help="tag prop k=v, repeatable",
    )
    g.add_argument("--run-url", default="", help="the CI run behind the rows")
    g.add_argument(
        "--dry-run", action="store_true", help="resolve and report; write nothing"
    )
    return g


def ci_tags(
    event: str, repository: str, branch: str, sha: str, pr_number, day=None
) -> list:
    """The (tag, family) pairs an artifact built by this CI event carries, spelled as Jenkins
    tags its own builds: `<repo>@<sha12>` (main) for a push to main, plus `nightly-<day>`
    (nightly) for a scheduled run of main; `<repo>#<pr>` and `<repo>#<pr>@<sha12>` (pr) for a
    pull request; none for any other event."""
    name, sha12 = (repository or "").rstrip("/").rsplit("/", 1)[-1], (sha or "")[:12]
    pr = str(pr_number or "").strip()
    if not (name and len(sha12) == 12):
        return []
    if event == "pull_request" and pr not in ("", "0"):
        return [(f"{name}#{pr}", "pr"), (f"{name}#{pr}@{sha12}", "pr")]
    if event in ("push", "schedule") and branch == "main":
        pin = [(f"{name}@{sha12}", "main")]
        return (
            pin + [(f"nightly-{day}", "nightly")]
            if event == "schedule" and day
            else pin
        )
    return []


def artifact_spec(args) -> str:
    """--artifact, else --artifact-id as a spec: a `|` record is gha:, a bare id stays one."""
    if getattr(args, "artifact", ""):
        return args.artifact
    record = (getattr(args, "artifact_id", "") or "").strip()
    return f"gha:{record}" if "|" in record else record


def artifact_options(args) -> dict:
    """The parsed flags as resolve/ensure keyword arguments."""
    return {
        "lookup": args.lookup,
        "registry": args.registry,
        "tag_family": args.tag_family,
        "tags": list(args.tags),
        "tag_date": args.tag_date,
        "origin": args.origin,
        "sources": list(args.sources),
        "identity_deps": list(args.identity_deps),
        "context_deps": list(args.context_deps),
        "props": dict(args.props),
        "tag_props": dict(args.tag_props),
        "run_url": args.run_url,
        "dry_run": args.dry_run,
    }


ENSURE_ONLY = (
    "origin",
    "sources",
    "identity_deps",
    "context_deps",
    "props",
    "tag_props",
    "run_url",
    "dry_run",
)
