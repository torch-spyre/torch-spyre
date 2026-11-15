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

"""Post benchmark verdicts as an advisory GitHub check `Spyre Perf (<arch>)` / `Spyre Quality (<arch>)`.

Input is the JSON report of `python -m spyre_clickhouse_ingest bench-verdicts` (schema
bench-verdicts/v1); this script needs only the standard library. A verdict is never a `failure`:
regressed and suspected rows give `neutral`. `failure` is reserved for `--run-broken` (the run
itself did not complete). Token: GITHUB_TOKEN or GH_TOKEN, never printed. `--dry-run` prints the
payload and needs no token.

    bench_check.py --report r.json --repo owner/name --sha <sha> --arch x86_64 [--kind perf]

`Spyre Quality (<arch>)` shows state and counts only unless `--detail` is given: no score, no baseline, and no
verdict word that reveals a movement, because the check is read on public pull requests. It is `success`
when results were recorded and `neutral` when there was nothing to record; only `--run-broken` makes it
`failure`. `--detail` (internal repositories) lists the flagged rows with their verdict words; measured
values, baselines and notes appear only when `--show-values` is also given, as on the perf check. A quality row
that is invalid or suspect is never reported as regressed.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

KINDS = {"perf": "Spyre Perf", "quality": "Spyre Quality"}
MAX_ANNOTATIONS = 50  # the API's per-request limit
SUMMARY_LIMIT = 60000  # the API's cap is 65535 characters
# The dashboard puts the arch in the path (`/x86/performance/run-details/<run_id>`).
ARCH_SLUGS = {"x86_64": "x86"}
REASON_TEXT = {
    "single_run": "one run over the floor; regressed needs the previous run over as well",
    "suspect_data": "the run is flagged suspect and is kept out of baselines",
    "purpose_not_baseline": "the run's purpose is not baseline-eligible and it is kept out of baselines",
    "few_repeats": "too few valid repeats to call a regression",
    "suspect_repeats": "some repeats are flagged suspect",
    "validity_unknown": "no policy judged the data behind this score, so it cannot be called a regression",
    "trust_external": "the score came from an external source and is not trusted to call a regression",
}
TITLES = {
    "success": "No regressions",
    "neutral": "Advisory: review the flagged benchmarks",
    "failure": "The benchmark run did not complete",
}


def check_name(kind, arch):
    return f"{KINDS[kind]} ({arch})"


TRUSTED = ("unqualified_env", "shared_env", "valid")


def rows_of(report, kind, arch=None):
    """Live rows of one kind. Defence in depth: a quality row is never `regressed` unless its data was judged
    and trusted, whatever the report says (an older reader or a hand-built report may not have enforced it)."""
    return [
        _guard(r)
        for r in report["results"]
        if r["kind"] == kind
        and not r["invalidated"]
        and not r.get("diagnostic")
        and (arch is None or r["series"]["arch"] in ("", arch))
    ]


def _guard(r):
    if r["kind"] == "quality" and r["verdict"] == "regressed":
        if r.get("validity") not in TRUSTED or r.get("trust") == "external":
            return {**r, "verdict": "suspected", "reason": "validity_unknown"}
    return r


def flagged(rows):
    flagged_rows = [
        r
        for r in rows
        if r["verdict"] in ("regressed", "suspected") and r["tier"] != "informational"
    ]
    return sorted(
        flagged_rows,
        key=lambda r: (r["verdict"] != "regressed", -(r["worse_pct"] or 0)),
    )


def persistent(rows):
    """Live rows regressed against the accepted baseline (schema 88): the rolling verdict may have absorbed the shift."""
    found = [
        r
        for r in rows
        if (r.get("anchored") or {}).get("verdict") == "regressed"
        and r["tier"] != "informational"
    ]
    return sorted(
        found,
        key=lambda r: (
            -(r["anchored"].get("days_exceeding") or 0),
            -(r["anchored"].get("worse_pct") or 0),
        ),
    )


def conclusion(rows, run_broken="", kind="perf", detail=True):
    if run_broken:
        return "failure"
    if kind == "quality" and not detail:
        return "success" if rows else "neutral"
    if flagged(rows) or persistent(rows):
        return "neutral"
    return (
        "neutral"
        if not rows or all(r["verdict"] == "no_baseline" for r in rows)
        else "success"
    )


def link(template, row, report, arch):
    """The dashboard URL for a row; '' when no template is configured or it does not format."""
    if not template:
        return ""
    ids = {
        "run_id": row["run_id"],
        "benchmark_id": row["benchmark_id"],
        "artifact_id": report["query"].get("artifact_id") or "",
        "arch": arch,
        "arch_slug": ARCH_SLUGS.get(arch, arch),
    }
    url = template
    for key, value in ids.items():
        url = url.replace("{" + key + "}", urllib.parse.quote(str(value), safe=""))
    return "" if "{" in url else url


def cell(text):
    """Text safe for a markdown table cell or line: no table, link, image, HTML or mention syntax."""
    text = "".join(c if c.isprintable() else " " for c in str(text))
    for ch in "`[]<>*~&":
        text = text.replace(ch, "_")
    # Zero-width spaces break autolinks (`://`, `www.`), issue refs (`#`) and mentions (`@`).
    for ch, broken in (
        ("://", ":\u200b//"),
        ("www.", "www\u200b."),
        ("#", "#\u200b"),
        ("@", "@\u200b"),
    ):
        text = text.replace(ch, broken)
    return text.replace("\\", "\\\\").replace("|", "\\|")


def fmt(v):
    return "" if v is None else f"{v:.4g}"


def counts_summary(report, rows, name, run_broken, kind, arch):
    """State and counts only: how many cases were recorded and how many results were set aside."""
    cases = len({r["benchmark_id"] for r in rows})
    lines = [f"**{name}** is advisory and not a required check."]
    if run_broken:
        lines.append(f"The run did not complete: {cell(run_broken)}")
    lines += ["", f"{cases} case(s) recorded." if rows else "No results for this run."]
    set_aside = sum(
        1
        for r in report["results"]
        if r["invalidated"]
        and r["kind"] == kind
        and not r.get("diagnostic")
        and r["series"]["arch"] in ("", arch)
    )
    if set_aside:
        lines.append(f"{set_aside} invalidated result(s) are excluded.")
    lines.append("Details are on the dashboard.")
    return "\n".join(lines)


def summary_text(
    report,
    rows,
    name,
    template,
    arch,
    limit,
    run_broken,
    show_values=False,
    kind="perf",
):
    counts = {}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    lines = [f"**{name}** is advisory and not a required check."]
    if run_broken:
        lines.append(f"The run did not complete: {cell(run_broken)}")
    lines.append("")
    lines.append(
        ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
        or "No results for this run."
    )
    own = report["summary"].get("by_kind", {}).get(kind, {}).get("invalidated")
    if own is None:
        own = report["summary"]["invalidated"]
    if own:
        lines.append(f"{own} invalidated result(s) are excluded.")
    # Run-wide lines name runs and views of every kind: kept off a check when the report also holds quality rows.
    if not any(r["kind"] == "quality" for r in report["results"]):
        if report.get("excluded_runs"):
            lines.append(
                f"{len(report['excluded_runs'])} invalidated run(s) have no verdicts."
            )
        if show_values:
            lines += [f"Note: {cell(n)}" for n in report["notes"]]
    pers = persistent(rows)
    if pers:
        lines += [
            "",
            f"**Persistent regressions** (worse than the accepted baseline, still open): {len(pers)}",
        ]
        for r in pers[:limit]:
            a = r["anchored"]
            since = f" since {cell(a['since_ts'][:10])}" if a.get("since_ts") else ""
            worse = f", {fmt(a.get('worse_pct'))}% worse" if show_values else ""
            lines.append(
                f"- {cell(r['name'])} {cell(r['metric'])}: regressed vs anchor v{cell(a.get('version') or '?')}{since} "
                f"({a.get('n_exceeding_runs', 0)} runs){worse}, rolling verdict `{cell(r['verdict'])}`"
            )
        if len(pers) > limit:
            lines.append(f"- {len(pers) - limit} more not shown.")
    top = flagged(rows)
    reasons = sorted(
        {r["reason"] for r in top if r["verdict"] == "suspected" and r["reason"]}
    )
    lines += [
        f"- suspected ({k}): {REASON_TEXT[k]}" for k in reasons if k in REASON_TEXT
    ]
    if top:
        head = ["verdict", "benchmark", "metric"]
        if show_values:
            head += ["value", "baseline median", "worse %"]
        head += ["repeats", "link"]
        lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for r in top[:limit]:
            n = r["noise"] or {}
            url = link(template, r, report, arch)
            cells = [
                f"{cell(r['verdict'])}{' (' + r['reason'] + ')' if r['reason'] in REASON_TEXT or r['reason'].startswith('validity_') and cell(r['reason']) == r['reason'] else ''}",
                cell(r["name"]),
                cell(r["metric"]),
            ]
            if show_values:
                cells += [
                    fmt(r["value"]),
                    fmt(r["baseline"]["median"]),
                    fmt(r["worse_pct"]),
                ]
            cells += [str(n.get("n_repeats", "")), f"[run]({url})" if url else ""]
            lines.append("| " + " | ".join(cells) + " |")
        if len(top) > limit:
            lines.append(f"\n{len(top) - limit} more flagged result(s) not shown.")
    return "\n".join(lines)[:SUMMARY_LIMIT]


def annotations(rows, path, cap, show_values=False):
    """Warnings for regressed (and persistent), notices for suspected; none without a file path to attach them to."""
    if not path:
        return []
    out = []
    top = flagged(rows)
    for r in top:
        out.append(
            {
                "path": path,
                "start_line": 1,
                "end_line": 1,
                "annotation_level": "warning"
                if r["verdict"] == "regressed"
                else "notice",
                "title": f"{r['verdict']}: {r['name']} {r['metric']}"[:255],
                "message": (
                    f"{fmt(r['worse_pct'])}% worse than the baseline median "
                    f"{fmt(r['baseline']['median'])} (value {fmt(r['value'])})."
                    if show_values
                    else f"{r['verdict']} against the baseline."
                ),
            }
        )
    shown = {id(r) for r in top}
    for r in persistent(rows):
        if id(r) in shown:
            continue
        a = r["anchored"]
        out.append(
            {
                "path": path,
                "start_line": 1,
                "end_line": 1,
                "annotation_level": "warning",
                "title": f"regressed vs anchor: {r['name']} {r['metric']}"[:255],
                "message": (
                    f"{fmt(a.get('worse_pct'))}% worse than the accepted baseline {fmt(a.get('anchor_value'))} "
                    if show_values
                    else "regressed against the accepted baseline "
                )
                + f"since {(a.get('since_ts') or '?')[:10]} ({a.get('n_exceeding_runs', 0)} runs).",
            }
        )
    return out[: max(0, min(cap, MAX_ANNOTATIONS))]


def title(concl, rows, kind="perf", detail=True):
    if concl == "failure":
        return TITLES[concl]
    if not rows:
        return "No benchmark results to judge"
    if kind == "quality" and not detail:
        return "Quality results recorded"
    if concl == "neutral" and not flagged(rows) and persistent(rows):
        return "Advisory: persistent regression against the accepted baseline"
    if concl == "neutral" and not flagged(rows):
        return "No baseline yet"
    return TITLES[concl]


def build_payload(report, kind, arch, sha, args):
    rows = rows_of(report, kind, arch)
    name = check_name(kind, arch)
    detail = kind != "quality" or getattr(args, "detail", False)
    shown = getattr(args, "show_values", False)
    concl = conclusion(rows, args.run_broken, kind, detail)
    template = args.dashboard_url_template or os.environ.get(
        "BENCH_DASHBOARD_URL_TEMPLATE", ""
    )
    top = flagged(rows) if detail else []
    url = link(template, top[0], report, arch) if top else ""
    return {
        "name": name,
        "head_sha": sha,
        "status": "completed",
        "conclusion": concl,
        "output": {
            "title": title(concl, rows, kind, detail),
            "summary": summary_text(
                report,
                rows,
                name,
                template,
                arch,
                args.top,
                args.run_broken,
                shown,
                kind,
            )
            if detail
            else counts_summary(report, rows, name, args.run_broken, kind, arch),
            "annotations": annotations(
                rows, args.annotation_path, args.max_annotations, shown
            )
            if detail
            else [],
        },
        **({"details_url": url} if url else {}),
    }


class GitHub:
    def __init__(self, token, api=None):
        self.token = token
        self.api = (
            api or os.environ.get("GITHUB_API_URL") or "https://api.github.com"
        ).rstrip("/")

    def call(self, method, path, body=None):
        req = urllib.request.Request(
            self.api + path,
            method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = (
                exc.read().decode(errors="replace")[:300].replace(self.token, "***")
            )
            raise SystemExit(
                f"GitHub API {method} {path} failed: HTTP {exc.code} {detail}"
            ) from None
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"GitHub API {method} {path} failed: {exc.reason}"
            ) from None

    def upsert_check(self, repo, payload):
        """PATCH the check of the same name on this sha when one exists, else create it."""
        q = urllib.parse.urlencode({"check_name": payload["name"], "per_page": 100})
        found = self.call(
            "GET", f"/repos/{repo}/commits/{payload['head_sha']}/check-runs?{q}"
        )
        runs = sorted(found.get("check_runs", []), key=lambda c: c.get("id", 0))
        if runs:
            # GitHub appends annotations on update, so repeated runs would pile up duplicates.
            patch = {k: v for k, v in payload.items() if k not in ("name", "head_sha")}
            patch["output"] = {
                k: v for k, v in patch["output"].items() if k != "annotations"
            }
            return "updated", self.call(
                "PATCH", f"/repos/{repo}/check-runs/{runs[-1]['id']}", patch
            )
        return "created", self.call("POST", f"/repos/{repo}/check-runs", payload)

    def post_status(self, repo, payload):
        """Commit-status fallback (no neutral state exists, so a verdict is `success` with the text)."""
        state = "failure" if payload["conclusion"] == "failure" else "success"
        desc = f"{payload['output']['title']}"[:140]
        body = {"state": state, "context": payload["name"], "description": desc}
        if payload.get("details_url"):
            body["target_url"] = payload["details_url"]
        return "status", self.call(
            "POST", f"/repos/{repo}/statuses/{payload['head_sha']}", body
        )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--report", required=True, help="bench-verdicts JSON file, or - for stdin"
    )
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--sha", required=True)
    ap.add_argument("--arch", required=True)
    ap.add_argument("--kind", choices=[*KINDS, "both"], default="perf")
    ap.add_argument("--mode", choices=("check", "status"), default="check")
    ap.add_argument(
        "--dashboard-url-template",
        default="",
        help="may use {run_id} {benchmark_id} {artifact_id} {arch} {arch_slug}",
    )
    ap.add_argument(
        "--annotation-path",
        default="",
        help="file the annotations attach to; none when empty",
    )
    ap.add_argument("--max-annotations", type=int, default=10)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument(
        "--run-broken",
        default="",
        help="reason the run itself failed; makes the conclusion `failure`",
    )
    ap.add_argument(
        "--show-values",
        action="store_true",
        help="include measured values, percentages and report notes; off by default so a public check carries none",
    )
    ap.add_argument(
        "--detail",
        action="store_true",
        help="quality check: list flagged rows (verdict words, no values unless --show-values too); never on public pull requests",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    try:
        return post(args)
    except (Exception, SystemExit) as exc:
        # Advisory: nothing here may fail the job except the explicit --run-broken conclusion.
        print(f"::warning::bench check not posted: {escape(exc)}")
        return 0


def load_report(args):
    try:
        if args.report == "-":
            report = json.load(sys.stdin)
        else:
            with open(args.report) as fh:
                report = json.load(fh)
        if report.get("schema") != "bench-verdicts/v1":
            raise ValueError(f"unsupported report schema {report.get('schema')!r}")
        return report
    except (Exception, SystemExit):
        if not args.run_broken:
            raise
        # The run itself failed: still post that, with nothing to judge.
        return {"schema": "bench-verdicts/v1", "query": {}, "runs": [], "notes": [],
                "results": [], "summary": {"invalidated": 0}}  # fmt: skip


def post(args):
    report = load_report(args)
    kinds = list(KINDS) if args.kind == "both" else [args.kind]
    payloads = []
    for kind in kinds:
        payload = build_payload(report, kind, args.arch, args.sha, args)
        # "both" skips a kind the run has no results for; an explicit kind always posts.
        if (
            args.kind == "both"
            and not rows_of(report, kind, args.arch)
            and not args.run_broken
        ):
            continue
        payloads.append(payload)
    if args.dry_run:
        print(json.dumps(payloads, indent=2))
        return 0
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN (or GH_TOKEN) is not set")
    gh = GitHub(token)
    for payload in payloads:
        try:
            print(post_one(gh, args, payload))
        except (Exception, SystemExit) as exc:
            print(f"::warning::{payload['name']} not posted: {escape(exc)}")
    return 0


def escape(exc):
    return str(exc).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def post_one(gh, args, payload):
    try:
        what, _ = (
            gh.post_status(args.repo, payload)
            if args.mode == "status"
            else gh.upsert_check(args.repo, payload)
        )
    except (Exception, SystemExit):
        if args.mode != "check" or not payload["output"].get("annotations"):
            raise
        # An annotation on a path missing at this sha is rejected whole; keep the check.
        payload["output"]["annotations"] = []
        what, _ = gh.upsert_check(args.repo, payload)
    return f"{what}: {payload['name']} {payload['conclusion']} on {args.sha[:12]}"


if __name__ == "__main__":
    sys.exit(main())
