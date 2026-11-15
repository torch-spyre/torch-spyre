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

"""On-demand benchmarks for a pull request: the comment grammar, admission policy and fork safety.

Everything here is a pure function over plain data: no network, no clock (callers pass `now`), no files. The
callers (a PR-comment workflow, the internal comment bot) fetch the facts, call these, and act on the decision.

  spyre-bench ondemand parse     --comment-file F --host H            comment -> request.json
  spyre-bench ondemand authorize --facts F                            request + actor + PR facts -> decision.json
  spyre-bench ondemand preview   --plan P --request R [--format md]   plan.json -> preview comment / JSON
  spyre-bench ondemand engine-request | env | state | sanitize         request -> plan request / BENCH_* / hidden state / inert text

Grammar, policy table and the threat model: docs/README.md ("On-demand benchmarks for a PR").
"""

import argparse
import base64
import difflib
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import regex

from spyre_bench import cliutil
from spyre_bench.cliutil import UsageError

PRODUCTS = ("torch-spyre", "hf-adapters", "spyre-inference", "spyre-comms")
DEFAULT_PRODUCT = "spyre-inference"
TIERS = ("sanity", "core", "extended")
REUSE_MODES = ("if_valid", "never", "prefer", "only")
PROFILES = ("off", "light", "full")
VS_FIXED = ("main", "nightly")
HOSTS = ("github.com", "ghe")
PURPOSE = "pr"
PREVIEW_SCHEMA = "bench.preview/1"
PREVIEW_MARKER = "<!-- spyre-bench-preview/1 -->"
WRITE_PERMISSIONS = ("write", "maintain", "admin")
MEMBER_ASSOCIATIONS = ("OWNER", "MEMBER", "COLLABORATOR")

NAME = regex.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DURATION = regex.compile(r"([1-9][0-9]{0,3})([smh])")
SHA = regex.compile(r"[0-9a-f]{40}")
_KEYS = ("bench", "purpose", "reuse", "force", "profile", "vs", "clean", "env", "max-wait", "budget", "sha")
SHA_PREFIX = regex.compile(r"[0-9a-f]{7,40}")
_UNIT = {"s": 1, "m": 60, "h": 3600}


_CONTROL = regex.compile(r"[\x00-\x1f\x7f`<>\\]")
_INVISIBLE = regex.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")
_MARKUP = regex.compile(r"[|\[\]()!*~]")
_REF = regex.compile(r"#(?=[0-9])")


def neutral(text, limit=300):
    """Untrusted text made inert in a GitHub comment: no control or bidi characters, tags, backticks or mentions."""
    s = _CONTROL.sub(" ", _INVISIBLE.sub("", str(text))).replace("@", "@\u200b").replace("\n", " ").strip()
    return s if len(s) <= limit else s[: limit - 1] + "…"


def safe(text, limit=160):
    """`neutral` plus what a table cell must not carry: pipes, links, images, emphasis, issue refs and URL schemes."""
    s = neutral(text, 10**6)
    s = _REF.sub("#\u200b", _MARKUP.sub(" ", s)).replace("://", ":\u200b//")
    return s if len(s) <= limit else s[: limit - 1] + "…"


class RequestError(UsageError):
    """A comment that cannot become a request: the message is safe to post back to the PR."""

    def __init__(self, message):
        super().__init__(neutral(message))


@dataclass(frozen=True)
class Policy:
    """What a host lets a commenter ask for. Numbers are proposals until real `est` values exist."""

    host: str
    tiers: tuple
    reuse_modes: tuple
    force_all: bool
    profiles: tuple
    custom_env: bool
    budget_device_hours: dict  # tier -> cap for one request
    pr_window_device_hours: float  # all requests on one PR within `window`
    per_user_per_hour: int
    max_wait_default_s: int
    max_wait_cap_s: int
    clean_per_user: int
    clean_per_user_per_day: int
    clean_cluster: int
    confirm_fraction: float = 0.5
    window: timedelta = timedelta(hours=24)
    approval_ttl: timedelta = timedelta(hours=72)


POLICIES = {
    "github.com": Policy(
        host="github.com",
        tiers=("sanity",),
        reuse_modes=("if_valid", "prefer", "only"),
        force_all=False,
        profiles=("off",),
        custom_env=False,
        budget_device_hours={"sanity": 4.0},
        pr_window_device_hours=8.0,
        per_user_per_hour=6,
        max_wait_default_s=1800,
        max_wait_cap_s=3600,
        clean_per_user=1,
        clean_per_user_per_day=3,
        clean_cluster=2,
    ),
    "ghe": Policy(
        host="ghe",
        tiers=("sanity", "core"),
        reuse_modes=REUSE_MODES,
        force_all=True,
        profiles=("off", "light"),
        custom_env=True,
        budget_device_hours={"sanity": 12.0, "core": 48.0},
        pr_window_device_hours=96.0,
        per_user_per_hour=6,
        max_wait_default_s=1800,
        max_wait_cap_s=3600,
        clean_per_user=1,
        clean_per_user_per_day=3,
        clean_cluster=2,
    ),
}


@dataclass(frozen=True)
class OnDemandRequest:
    """A parsed `/bench` or `/spyre-test bench=` request, before any policy or catalog check."""

    selections: tuple  # ((product, {"tier"?, "suites"?, "cases"?}, as_asked), ...)
    purpose: str = PURPOSE
    reuse: str = "if_valid"
    force: tuple = ()
    profile: str = "off"
    vs: str = "nightly"
    clean: bool = False
    env: str = ""
    max_wait_s: int = 0
    budget: float = 0.0
    sha: str = ""  # head commit (7-40 hex) the commenter reviewed; required for a fork head
    asked: dict = field(default_factory=dict)  # the flags as written, for the "asked vs planned" diff

    def to_dict(self):
        return {
            "purpose": self.purpose,
            "selections": [{"product": p, "selection": s, "as_asked": a} for p, s, a in self.selections],
            "reuse": self.reuse,
            "force": list(self.force),
            "profile": self.profile,
            "vs": self.vs,
            "clean": self.clean,
            "env": self.env,
            "max_wait_s": self.max_wait_s,
            "budget": self.budget,
            "sha": self.sha,
            "asked": self.asked,
        }

    @staticmethod
    def from_dict(d):
        if d.get("purpose", PURPOSE) != PURPOSE:
            raise RequestError(f"purpose {d.get('purpose')!r}: a comment can only ask for purpose 'pr'")
        return OnDemandRequest(
            selections=tuple((s["product"], s["selection"], s.get("as_asked", "")) for s in d["selections"]),
            purpose=d.get("purpose", PURPOSE),
            reuse=d.get("reuse", "if_valid"),
            force=tuple(d.get("force") or ()),
            profile=d.get("profile", "off"),
            vs=d.get("vs", "nightly"),
            clean=bool(d.get("clean")),
            env=d.get("env", ""),
            max_wait_s=int(d.get("max_wait_s") or 0),
            budget=float(d.get("budget") or 0.0),
            sha=str(d.get("sha") or ""),
            asked=d.get("asked") or {},
        )

    def tiers_for(self, policy):
        """Tier per pick for budgeting. Cases and named selections carry no tier of their own: the plan decides."""
        top = max(policy.tiers, key=TIERS.index)
        return [s.get("tier") or ("sanity" if s.get("suites") else top) for _, s, _ in self.selections]


def _did_you_mean(word, choices):
    hit = difflib.get_close_matches(word, list(choices), n=1, cutoff=0.6)
    return f"; did you mean '{hit[0]}'" if hit else ""


def _name(word, what):
    if not NAME.fullmatch(word or ""):
        raise RequestError(f"{what} {word!r}: letters, digits, '.', '_' and '-' only")
    return word


def parse_one_selection(text):
    """`tier` | `suite[+suite][@tier]` | `case[,case]` | `#name` -> (selection dict, as_asked text)."""
    text = text.strip()
    if not text:
        raise RequestError("empty selection")
    if text.startswith("#"):
        return {"cases": [_name(text[1:], "selection name")]}, text
    if "@" in text or "+" in text:
        suites, _, tier = text.partition("@")
        if "," in suites or "@" in tier:
            raise RequestError(f"selection {text!r}: use 'suite+suite@tier' or a comma list of cases, not both")
        out = {"suites": [_name(s, "suite") for s in suites.split("+")]}
        if "@" in text:
            if tier not in TIERS:
                raise RequestError(f"tier {tier!r} is not one of {', '.join(TIERS)}{_did_you_mean(tier, TIERS)}")
            out["tier"] = tier
        else:
            out["tier"] = "sanity"
        return out, text
    if text in TIERS:
        return {"tier": text}, text
    return {"cases": [_name(c, "case") for c in text.split(",")]}, text


def parse_selection(text, default_product=DEFAULT_PRODUCT):
    """`[product:]selection[;[product:]selection]` -> ((product, selection, as_asked), ...)."""
    out, seen = [], set()
    for part in str(text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        product, sep, rest = part.partition(":")
        if sep:
            if product not in PRODUCTS:
                raise RequestError(f"unknown product {product!r}{_did_you_mean(product, PRODUCTS)}")
        else:
            product, rest = default_product, part
        if product in seen:
            raise RequestError(f"product {product!r} appears twice: join its selections in one")
        seen.add(product)
        sel, asked = parse_one_selection(rest)
        out.append((product, sel, asked))
    if not out:
        raise RequestError("bench= needs a selection, for example bench=sanity")
    return tuple(out)


def parse_duration(text):
    m = DURATION.fullmatch(str(text or ""))
    if not m:
        raise RequestError(f"duration {text!r}: use a number and s, m or h, such as 45m")
    return int(m.group(1)) * _UNIT[m.group(2)]


def _truthy(text, key):
    word = str(text).strip().lower()
    if word in ("true", "yes", "1", "on"):
        return True
    if word in ("false", "no", "0", "off"):
        return False
    raise RequestError(f"{key}={text!r}: use true or false")


def _canonical_key(raw):
    key = raw[len("bench-") :] if raw.startswith("bench-") else raw
    if key not in _KEYS:
        raise RequestError(f"unknown flag {raw!r}{_did_you_mean(key, _KEYS)}")
    return key


def parse_flags(flags, default_product=DEFAULT_PRODUCT):
    """The flag family as a mapping (`bench`, `bench-reuse`/`reuse`, ...) -> OnDemandRequest. Shape checks only."""
    got = {}
    for raw, value in dict(flags).items():
        key = _canonical_key(str(raw))
        if key in got:
            raise RequestError(f"flag {key!r} given twice")
        got[key] = str(value)
    if "bench" not in got:
        raise RequestError("bench-* flags need a selection: bench=<selection>")
    purpose = got.get("purpose", PURPOSE)
    if purpose != PURPOSE:
        raise RequestError(f"purpose={purpose!r}: a comment can only ask for purpose 'pr'")
    reuse = got.get("reuse", "if_valid")
    if reuse not in REUSE_MODES:
        raise RequestError(f"reuse={reuse!r}: one of {', '.join(REUSE_MODES)}{_did_you_mean(reuse, REUSE_MODES)}")
    force = tuple(dict.fromkeys(_name(c, "force case") for c in cliutil.csv(got.get("force", ""))))
    if "all" in force and len(force) > 1:
        raise RequestError("force=all cannot be combined with case names")
    if force and reuse == "never":
        raise RequestError("force has no effect with reuse=never: everything runs already")
    profile = got.get("profile", "off")
    if profile not in PROFILES:
        raise RequestError(f"profile={profile!r}: one of {', '.join(PROFILES)}")
    vs = got.get("vs", "nightly")
    if vs not in VS_FIXED and not regex.fullmatch(r"build:[A-Za-z0-9._/-]{1,80}#[0-9]{1,9}", vs):
        raise RequestError(f"vs={vs!r}: main, nightly or build:<job>#<n>")
    env = got.get("env", "")
    if env:
        _name(env, "env")
    budget = 0.0
    if "budget" in got:
        try:
            budget = float(got["budget"])
        except ValueError:
            raise RequestError(f"budget={got['budget']!r}: a number of device-hours") from None
        if not budget > 0 or budget == float("inf"):
            raise RequestError("budget must be a positive, finite number of device-hours")
    sha = got.get("sha", "")
    if sha and not SHA_PREFIX.fullmatch(sha):
        raise RequestError(f"sha={sha!r}: 7 to 40 lower-case hex characters of the head commit")
    max_wait = parse_duration(got["max-wait"]) if "max-wait" in got else 0
    clean = _truthy(got["clean"], "clean") if "clean" in got else False
    if max_wait and not clean:
        raise RequestError("max-wait only applies with clean=true")
    return OnDemandRequest(
        selections=parse_selection(got["bench"], default_product),
        reuse=reuse,
        force=force,
        profile=profile,
        vs=vs,
        clean=clean,
        env=env,
        max_wait_s=max_wait,
        budget=budget,
        sha=sha,
        asked={k: got[k] for k in sorted(got)},
    )


@dataclass(frozen=True)
class Command:
    kind: str  # run | confirm | cancel
    flags: dict
    line: str


def parse_command(text, command="/bench"):
    """The first `command` line outside code fences and quotes -> Command, or None when there is none.

    `/bench <selection> [key=value ...]`, `/bench confirm`, `/bench cancel`. A first token without `=` is the selection.
    """
    fence, hidden = "", False
    for line in str(text or "").replace("\r\n", "\n").split("\n"):
        stripped = line.strip()
        mark = stripped[:3] if stripped.startswith(("```", "~~~")) else ""
        if mark and (not fence or mark == fence):
            fence = "" if fence else mark
            continue
        if "<!--" in line and "-->" not in line.split("<!--", 1)[1]:
            hidden = True
        if hidden:
            hidden = "-->" not in line
            continue
        if fence or "<!--" in line or not (line.startswith(command + " ") or line.rstrip() == command):
            continue
        tokens = line[len(command) :].split()
        if len(tokens) > 40 or len(line) > 1000:
            raise RequestError("command too long")
        if tokens[:1] in (["confirm"], ["cancel"]):
            if len(tokens) > 1:
                raise RequestError(f"'{tokens[0]}' takes no arguments")
            return Command(tokens[0], {}, line.strip())
        flags = {}
        for i, tok in enumerate(tokens):
            key, sep, value = tok.partition("=")
            if not sep:
                if i != 0:
                    raise RequestError(f"{tok!r}: expected key=value")
                key, value = "bench", tok
            if key in flags:
                raise RequestError(f"flag {key!r} given twice")
            flags[key] = value
        return Command("run", flags, line.strip())
    return None


def check_policy(req, policy, actor=None):
    """Host and actor gates on a parsed request. Returns a list of violation messages (empty = allowed)."""
    bad = []
    for product, sel, _ in req.selections:
        tier = sel.get("tier") or "sanity"
        if tier not in policy.tiers:
            where = "internal requests only" if tier == "core" else "ops requests only"
            bad.append(f"tier '{tier}' for {product} is not available here ({where})")
    if req.reuse not in policy.reuse_modes:
        bad.append(f"reuse={req.reuse} is not available here")
    if "all" in req.force and not policy.force_all:
        bad.append("force=all is not available here: name the cases")
    if req.profile not in policy.profiles:
        bad.append(f"profile={req.profile} is not available here")
    if req.env and not policy.custom_env:
        bad.append("env= is not available here: the environment is fixed by the workflow")
    if req.max_wait_s > policy.max_wait_cap_s:
        bad.append(f"max-wait is capped at {policy.max_wait_cap_s // 60}m")
    if req.clean and actor is not None and not has_write(actor):
        bad.append("clean=true needs write access to the repository")
    if req.budget and req.budget > max(policy.budget_device_hours.values()):
        bad.append(f"budget {req.budget:g} exceeds the host cap {max(policy.budget_device_hours.values()):g}")
    return bad


def has_write(actor):
    return str((actor or {}).get("permission", "")).lower() in WRITE_PERMISSIONS


def invoker_allowed(actor, policy):
    """Who may ask at all. github.com: write permission, checked by API; GHE: any org member association."""
    if has_write(actor):
        return True
    return policy.host == "ghe" and str((actor or {}).get("association", "")).upper() in MEMBER_ASSOCIATIONS


def budget_limit(req, policy):
    """Device-hours one request may plan: the largest tier asked, capped by an explicit budget= (never raised by it)."""
    cap = max(policy.budget_device_hours.get(t, 0.0) for t in req.tiers_for(policy))
    return min(cap, req.budget) if req.budget else cap


def check_budget(estimate_hours, limit, policy):
    """-> (ok, needs_confirm). Over the limit never runs; above `confirm_fraction` of it needs a confirm."""
    if estimate_hours > limit:
        return False, False
    return True, estimate_hours > limit * policy.confirm_fraction


@dataclass(frozen=True)
class Approval:
    """A pinned-sha approval: who allowed which head sha to run, for which request, and until when."""

    sha: str
    approver: str
    permission: str
    request_digest: str
    granted_at: str
    request_id: str = ""

    def to_dict(self):
        return dict(self.__dict__)

    @staticmethod
    def from_dict(d):
        keys = {"sha", "approver", "permission", "request_digest", "granted_at", "request_id"}
        if (
            not isinstance(d, dict)
            or not {"sha", "approver", "permission", "request_digest", "granted_at"} <= set(d) <= keys
        ):
            raise RequestError("the stored approval is malformed")
        return Approval(**{k: str(v) for k, v in d.items()})


def request_digest(req):
    body = json.dumps({k: v for k, v in req.to_dict().items() if k != "asked"}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:16]


def make_approval(actor, head_sha, req, now, request_id=""):
    """The approval a write-access commenter gives by commenting. Refuses anything else or a non-sha."""
    if not has_write(actor):
        raise RequestError("only a commenter with write access can approve a run")
    if not SHA.fullmatch(str(head_sha or "")):
        raise RequestError("the head sha must be a full 40-character commit id")
    return Approval(head_sha, actor["login"], actor["permission"], request_digest(req), _iso(now), request_id)


def approval_valid(approval, head_sha, req, now, policy, request_id=""):
    """-> (ok, reason). The approval must match the exact head sha, the exact request, and be fresh."""
    if approval is None:
        return False, "no approval on record"
    if approval.sha != head_sha:
        return False, "the PR head moved since the approval: a new request is needed"
    if approval.request_digest != request_digest(req):
        return False, "the approval was given for a different request"
    if str(approval.permission).lower() not in WRITE_PERMISSIONS:
        return False, "the approver did not have write access"
    age = _parse(now) - _parse(approval.granted_at)
    if age > policy.approval_ttl:
        return False, "the approval expired"
    if age < -timedelta(minutes=5):
        return False, "the approval is dated in the future"
    if approval.request_id and request_id and approval.request_id != request_id:
        return False, "the approval belongs to another request"
    return True, ""


def sandbox_profile(is_fork):
    """What the measuring job may touch. Constant for forks; the same floor applies to same-repo runs."""
    return {
        "secrets": False,
        "registry_write": False,
        "store_credentials": False,
        "checkout": "pinned_sha",
        "engine_source": "base_default_branch",
        "provenance": "fork" if is_fork else "branch",
        "baseline_eligible": False,
        "results_via": "trusted_post_step",
    }


def _parse(ts):
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))


def _iso(ts):
    return _parse(ts).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Decision:
    state: str  # admit | needs_confirm | deny
    reasons: list = field(default_factory=list)
    cancel: list = field(default_factory=list)  # request ids to cancel (superseded)
    pinned_sha: str = ""
    approval: dict = None
    sandbox: dict = None
    limit_device_hours: float = 0.0

    @property
    def allowed(self):
        return self.state in ("admit", "needs_confirm")

    def to_dict(self):
        return dict(self.__dict__, allowed=self.allowed)


LIVE_STATES = ("active", "previewed", "launched", "admit", "needs_confirm")
DEAD_STATES = ("cancelled", "denied", "rejected", "deny")


def _hours(value):
    """A device-hour figure from stored data: finite and >= 0, else infinite so that a corrupt row fails closed."""
    try:
        x = float(value or 0)
    except (TypeError, ValueError):
        return math.inf
    return x if math.isfinite(x) and x >= 0 else math.inf


def admission(history, new, now, policy):
    """Supersede and rate rules over earlier requests. `history` rows: id, pr, user, ts, state, est, clean.

    Row states: active, previewed, launched, admit, needs_confirm count as live; cancelled, denied, rejected, deny
    count for nothing except the rate limit and the per-day clean quota (a submit-and-cancel loop still costs).
    One live request per PR: a new one cancels the requester's own, or anyone's when the requester has write access;
    otherwise it is refused. Window cap on estimated device-hours per PR, per-user hourly rate and clean quotas.
    Returns (reasons, cancel_ids).
    """
    now = _parse(now)
    live = [h for h in history if h.get("state") not in DEAD_STATES]
    here = [h for h in live if h["pr"] == new["pr"] and h.get("state") in LIVE_STATES]
    reasons = []
    cancel = [h["id"] for h in here if h["user"] == new["user"] or new.get("can_supersede")]
    if len(cancel) < len(here):
        reasons.append("another user's benchmark request is still running on this pull request")
    kept = [h for h in live if h["id"] not in cancel]
    in_window = [h for h in kept if h["pr"] == new["pr"] and now - _parse(h["ts"]) <= policy.window]
    spent = sum(_hours(h.get("est")) for h in in_window)
    if spent + new["est"] > policy.pr_window_device_hours:
        reasons.append(
            f"PR window budget: {spent:g} device-hours already planned in {policy.window.total_seconds() / 3600:g}h, "
            f"cap {policy.pr_window_device_hours:g}"
        )
    hour = [h for h in history if h["user"] == new["user"] and now - _parse(h["ts"]) <= timedelta(hours=1)]
    if len(hour) >= policy.per_user_per_hour:
        reasons.append(f"rate limit: {policy.per_user_per_hour} requests per user per hour")
    if new.get("clean"):
        mine = [
            h for h in history if h["user"] == new["user"] and h.get("clean") and h.get("state") not in DEAD_STATES[1:]
        ]
        if sum(1 for h in mine if h.get("state") in LIVE_STATES and h["id"] not in cancel) >= policy.clean_per_user:
            reasons.append(f"clean quota: {policy.clean_per_user} concurrent clean request per user")
        if sum(1 for h in mine if now - _parse(h["ts"]) <= timedelta(days=1)) >= policy.clean_per_user_per_day:
            reasons.append(f"clean quota: {policy.clean_per_user_per_day} clean requests per user per day")
        if sum(1 for h in kept if h.get("clean") and h.get("state") in LIVE_STATES) >= policy.clean_cluster:
            reasons.append(f"clean quota: {policy.clean_cluster} clean requests running at once")
    return reasons, cancel


def plan_violations(plan, policy):
    """Cases of a plan that the host does not allow. Cases and named selections carry their own tier, so this is
    where a tier limit is enforced for them; a case without a tier cannot be judged and is refused."""
    bad = []
    for c in plan.get("cases") or []:
        if c.get("action") != "run":
            continue
        tier = (c.get("run_props") or {}).get("run.tier")
        if tier not in policy.tiers:
            bad.append(f"case {safe(c.get('name'), 60)} has tier '{safe(tier or 'none', 20)}', not available here")
    return bad[:5]


def _same_repo(a, b):
    return str(a).lower() == str(b).lower()


def _finite_hours(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) and v >= 0 else None


def authorize(facts):
    """The whole gate for one request, from facts the caller fetched from the API.

    facts: host, now, actor{login, permission, association}, pr{number, base_repo, head_repo, head_sha},
    request (to_dict form), history (rows for `admission`), request_id.
    Two stages: `request` (default; policy, permission, fork, sha, rates) and `planned` (also needs
    estimate_device_hours, finite and >= 0, and optionally the plan, whose cases are checked against the host tiers).
    A confirm adds confirmed=true, pinned_sha (the sha recorded with the request, required), requester (the actor
    dict stored with the request) and approval (the record stored with it; required for a fork).
    """
    policy = POLICIES[facts["host"]]
    req = OnDemandRequest.from_dict(facts["request"])
    actor, pr, now = facts.get("actor") or {}, facts["pr"], facts["now"]
    confirmed = bool(facts.get("confirmed"))
    who = (facts.get("requester") or actor) if confirmed else actor
    head = str(pr.get("head_sha", ""))
    is_fork = not _same_repo(pr.get("head_repo", ""), pr.get("base_repo", "-"))

    def deny(*reasons):
        return Decision("deny", list(reasons), sandbox=sandbox_profile(is_fork), pinned_sha=head)

    if not invoker_allowed(who, policy):
        return deny("the commenter is not allowed to run benchmarks on this repository")
    if confirmed and not has_write(actor):
        return deny("only a commenter with write access can confirm a run")
    if not SHA.fullmatch(head):
        return deny("the PR head is not a full commit id")
    if confirmed and not facts.get("pinned_sha"):
        return deny("nothing records which commit this request was made for: comment again")
    if facts.get("pinned_sha") and facts["pinned_sha"] != head:
        return deny("the PR head moved since this request was made: comment again to run the new commit")
    violations = check_policy(req, policy, who)
    if violations:
        return deny(*violations)
    approval = None
    if is_fork:
        # A fork runs only the exact commit a write-access commenter named; the comment is the approval.
        if not (req.sha and head.startswith(req.sha)):
            return deny("fork pull request: add sha=<the first 7 or more characters of the head you reviewed>")
        recorded = facts.get("approval")
        if confirmed and not recorded:
            return deny("fork pull request: no approval is recorded for this request: comment again")
        approval = Approval.from_dict(recorded) if recorded else None
        if approval is None and has_write(actor):
            approval = make_approval(actor, head, req, now, str(facts.get("request_id", "")))
        ok, why = approval_valid(approval, head, req, now, policy)
        if not ok:
            return deny(f"fork pull request: {why}")
    limit = budget_limit(req, policy)
    planned = facts.get("stage") == "planned"
    est = _finite_hours(facts["estimate_device_hours"]) if facts.get("estimate_device_hours") is not None else None
    if planned and est is None:
        return deny("the plan has no usable estimate in device-hours")
    if facts.get("estimate_device_hours") is not None and est is None:
        return deny("the estimate is not a finite number of device-hours")
    needs_confirm = False
    if planned:
        if facts.get("plan"):
            bad = plan_violations(facts["plan"], policy)
            if bad:
                return deny(*bad)
        ok, needs_confirm = check_budget(est, limit, policy)
        if not ok:
            return deny(f"plan needs {est:g} device-hours, over the {limit:g} limit for this request")
        needs_confirm = needs_confirm and not confirmed
    reasons, cancel = admission(
        facts.get("history") or [],
        {
            "pr": pr["number"],
            "user": who.get("login", ""),
            "est": est or 0.0,
            "clean": req.clean,
            "can_supersede": has_write(actor),
        },
        now,
        policy,
    )
    if reasons:
        return deny(*reasons)
    state = "needs_confirm" if needs_confirm else "admit"
    why = (
        [f"estimate above {int(policy.confirm_fraction * 100)}% of the limit: confirm to run"] if needs_confirm else []
    )
    return Decision(state, why, cancel, head, approval.to_dict() if approval else None, sandbox_profile(is_fork), limit)


def engine_request(req, product, subject=None, environment=""):
    """The engine Run request for one product of the parsed request (what `spyre-bench plan --request` reads)."""
    for p, sel, _ in req.selections:
        if p == product:
            break
    else:
        raise KeyError(product)
    selection = {k: v for k, v in sel.items() if v}
    out = {"purpose": req.purpose, "selection": {**selection, "type": "perf"}}
    if subject:
        out["subject"] = subject
    if req.env or environment:
        out["environment"] = req.env or environment
    out["reuse"] = {"mode": req.reuse, "override": True, **({"force": list(req.force)} if req.force else {})}
    if req.budget:
        out["budget"] = {"device_hours": req.budget}
    if req.profile != "off":
        out["overrides"] = {"profile_level": req.profile}
    return out


def bench_env(req, product, subject_id="", environment=""):
    """The BENCH_* twins of engine_request, for `make bench-plan` / `make bench-run`."""
    for p, sel, _ in req.selections:
        if p == product:
            break
    else:
        raise KeyError(product)
    env = {"BENCH_PURPOSE": req.purpose, "BENCH_REUSE": req.reuse}
    if sel.get("cases"):
        env["BENCH_SELECTION"] = ",".join(sel["cases"])
    if sel.get("tier"):
        env["BENCH_TIER"] = sel["tier"]
    if sel.get("suites"):
        env["BENCH_SUITES"] = ",".join(sel["suites"])
    if req.force:
        env["BENCH_FORCE"] = ",".join(req.force)
    if req.budget:
        env["BENCH_BUDGET"] = repr(req.budget)
    if req.clean:
        env["BENCH_CLEAN"] = "1"
        if req.max_wait_s:
            env["BENCH_MAX_WAIT"] = f"{req.max_wait_s}s"
    if req.env or environment:
        env["BENCH_ENVIRONMENT"] = req.env or environment
    if req.profile != "off":
        env["BENCH_SET"] = f"run.profile_level={req.profile}"
    if subject_id:
        env["BENCH_SUBJECT_ID"] = subject_id
    return env


STATE_OPEN, STATE_CLOSE, STATE_MAX = "<!-- spyre-bench-state:", "-->", 8192


def encode_state(state):
    """A request's state as a hidden comment line, so a bot comment is the only store a confirm needs."""
    raw = base64.b64encode(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).decode()
    if len(raw) > STATE_MAX:
        raise RequestError("the request is too large to store for a later confirm")
    return f"{STATE_OPEN} {raw} {STATE_CLOSE}"


def decode_state(text):
    """The state in a comment body, or None. The caller must only pass comments the bot itself wrote."""
    start = str(text or "").find(STATE_OPEN)
    if start < 0:
        return None
    end = text.find(STATE_CLOSE, start)
    raw = text[start + len(STATE_OPEN) : end].strip() if end > 0 else ""
    if not raw or len(raw) > STATE_MAX:
        return None
    try:
        out = json.loads(base64.b64decode(raw, validate=True))
    except (ValueError, RecursionError):
        return None
    return out if isinstance(out, dict) else None


# ---------------------------------------------------------------------------------------------------------------
# Preview: what the PR comment and the dashboard show before cards are taken.

_ORDER = {"run": 0, "reuse": 1, "skip": 2}
_MAX_ROWS = 40
_REASON_TEXT = {"reuse_needs_subject": "runs unless reusable; reuse is looked up when the run starts"}


def _asked_vs_planned(req, plan):
    """Rows where what the commenter wrote is not what will happen, each with the reason."""
    rows = []
    s = plan["summary"]
    planned_req = plan.get("request") or {}
    for product, sel, as_asked in req.selections:
        if sel.get("suites") and "@" not in as_asked:
            rows.append(
                {"field": f"{product} tier", "asked": "(none)", "planned": "sanity", "why": "suites default to sanity"}
            )
        if sel.get("cases"):
            rows.append(
                {
                    "field": f"{product} tier",
                    "asked": "(cases)",
                    "planned": "not filtered",
                    "why": "named cases run whatever their tier; the host limit is checked on the plan",
                }
            )
    mode = ((planned_req.get("reuse") or {}).get("mode")) or "never"
    if mode != req.reuse:
        rows.append({"field": "reuse", "asked": req.reuse, "planned": mode, "why": "the engine did not apply it"})
    forced = set(req.force) - {"all"}
    known = {c["name"] for c in plan["cases"]}
    for name in sorted(forced - known):
        rows.append({"field": "force", "asked": name, "planned": "(not in selection)", "why": "no such case planned"})
    for name in sorted(forced & known):
        case = next(c for c in plan["cases"] if c["name"] == name)
        if case["action"] != "run":
            rows.append(
                {"field": "force", "asked": name, "planned": case["action"], "why": safe(case.get("reason", ""))}
            )
    if req.clean:
        not_clean = [
            c["name"] for c in plan["cases"] if c["action"] == "run" and not (c.get("clean") or {}).get("requested")
        ]
        if not_clean:
            rows.append(
                {
                    "field": "clean",
                    "asked": "true",
                    "planned": "not requested on %d case(s)" % len(not_clean),
                    "why": "the plan carries no clean request for them",
                }
            )
    if req.profile != "off":
        levels = {c.get("profile_level", "off") for c in plan["cases"] if c["action"] == "run"}
        if levels and levels != {req.profile}:
            rows.append(
                {"field": "profile", "asked": req.profile, "planned": ",".join(sorted(levels)), "why": "per case"}
            )
    if s["skip"]:
        by = ", ".join(f"{n} {safe(r, 60)}" for r, n in sorted(s["reasons"].items()))
        rows.append({"field": "cases", "asked": f"{s['cases']} cases", "planned": f"{s['skip']} skipped", "why": by})
    return rows


def build_preview(plan, req, policy=None, queue=None, dashboard_url="", notes=()):
    """The preview as data (the dashboard consumes this; `render_markdown` is the same data as text)."""
    s, est = plan["summary"], plan.get("estimate") or {}
    known = _finite_hours(est.get("device_hours"))
    hours = known if known is not None else 0.0
    limit = budget_limit(req, policy) if policy else None
    needs_confirm = False
    over = known is None
    violations = plan_violations(plan, policy) if policy else []
    if policy and limit is not None and known is not None:
        ok, needs_confirm = check_budget(hours, limit, policy)
        over = not ok
    cases = sorted(plan["cases"], key=lambda c: (_ORDER.get(c["action"], 9), c["name"]))
    body = {
        "schema": PREVIEW_SCHEMA,
        "purpose": plan.get("purpose"),
        "counts": {"cases": s["cases"], "run": s["run"], "reuse": s["reuse"], "skip": s["skip"]},
        "estimate": {"device_hours": hours, "wall_min": float(est.get("wall_min") or 0.0)},
        "reasons": dict(sorted(s["reasons"].items())),
        "queue": queue or {},
        "budget": {
            "limit_device_hours": limit,
            "needs_confirm": needs_confirm,
            "over": over,
            "estimate_known": known is not None,
            "violations": violations,
        },
        "asked": req.asked,
        "diff": _asked_vs_planned(req, plan),
        "cases": [
            {
                "name": c["name"],
                "action": c["action"],
                "reason": c.get("reason", ""),
                "repeats": c.get("repeats"),
                "est_min": c.get("est_min"),
                "reused_from": c.get("reused_from"),
                "environment": c.get("environment"),
            }
            for c in cases
        ],
        "dashboard_url": dashboard_url,
        "notes": [safe(n, 200) for n in notes],
    }
    body["digest"] = hashlib.sha256(
        json.dumps({k: body[k] for k in ("counts", "estimate", "diff", "cases")}, sort_keys=True).encode()
    ).hexdigest()[:16]
    return body


def render_markdown(preview, public=True, command="/bench", state=None):
    """The PR comment. Stable layout, one marker so the same comment is edited in place."""
    c, e, b = preview["counts"], preview["estimate"], preview["budget"]
    out = [PREVIEW_MARKER] + ([encode_state(state)] if state else []) + ["### Benchmark plan"]
    out.append(f"**run {c['run']} · reuse {c['reuse']} · skip {c['skip']}** of {c['cases']} cases")
    out.append(f"Estimate: {e['device_hours']:g} device-hours (about {e['wall_min']:g} min of measuring).")
    if b.get("limit_device_hours") is not None:
        out.append(f"Limit for this request: {b['limit_device_hours']:g} device-hours.")
    q = preview.get("queue") or {}
    if q:
        out.append("Queue: " + ", ".join(f"{safe(k, 30)} {safe(v, 40)}" for k, v in sorted(q.items())) + ".")
    for v in b.get("violations") or []:
        out.append(f"**Not allowed:** {safe(v, 160)}")
    if not b.get("estimate_known", True):
        out.append("**The plan has no usable estimate: nothing will run.**")
    elif b.get("over"):
        out.append("**Over the limit: nothing will run.** Narrow the selection or ask for a smaller tier.")
    elif b.get("violations"):
        out.append("**Nothing will run** until the selection fits this host.")
    elif b.get("needs_confirm"):
        out.append(f"**Confirm to run:** reply `{command} confirm` (write access needed). Nothing runs until then.")
    if preview["diff"]:
        out += ["", "**You asked for / will run**", "", "| item | asked | planned | why |", "|---|---|---|---|"]
        out += [
            f"| {safe(d['field'], 40)} | {safe(d['asked'], 60)} | {safe(d['planned'], 60)} | {safe(d['why'], 100)} |"
            for d in preview["diff"]
        ]
    if preview["cases"]:
        out += ["", "| action | case | repeats | note |", "|---|---|---|---|"]
        for case in preview["cases"][:_MAX_ROWS]:
            note = _REASON_TEXT.get(case["reason"]) or safe(case["reason"], 100)
            if not note and case["reused_from"]:
                note = f"reused from {safe(case['reused_from'], 40)}"
            out.append(f"| {case['action']} | {safe(case['name'], 90)} | {case['repeats'] or ''} | {note} |")
        if len(preview["cases"]) > _MAX_ROWS:
            out.append(f"| | …and {len(preview['cases']) - _MAX_ROWS} more | | |")
    url = str(preview.get("dashboard_url") or "")
    if url.startswith("https://") and " " not in url:
        out += ["", f"{'Dashboard (sign-in required)' if public else 'Details'}: {neutral(url, 200)}"]
    out += ["", *[f"_{safe(n, 200)}_" for n in preview.get("notes", [])]]
    out += ["", f"<sub>plan {preview['digest']}</sub>"]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# CLI


def _load(path):
    with open(path) as f:
        return json.load(f)


def _cmd_parse(a):
    with open(a.comment_file) as f:
        cmd = parse_command(f.read(), a.command)
    if cmd is None:
        print(json.dumps({"command": None}))
        return cliutil.EXIT_NOTHING
    if cmd.kind != "run":
        print(json.dumps({"command": cmd.kind}))
        return cliutil.EXIT_OK
    req = parse_flags(cmd.flags, a.default_product)
    print(
        json.dumps(
            {"command": "run", "request": req.to_dict(), "policy_violations": check_policy(req, POLICIES[a.host])}
        )
    )
    return cliutil.EXIT_OK


def _cmd_authorize(a):
    facts = _load(a.facts)
    decision = authorize(facts)
    print(json.dumps(decision.to_dict(), sort_keys=True))
    return cliutil.EXIT_OK if decision.allowed else cliutil.EXIT_INVALID


def _cmd_preview(a):
    req = OnDemandRequest.from_dict(_load(a.request))
    queue = json.loads(a.queue) if a.queue else None
    pv = build_preview(_load(a.plan), req, POLICIES[a.host], queue, a.dashboard_url or "", a.note)
    state = json.loads(a.state) if a.state else None
    print(
        render_markdown(pv, public=a.host == "github.com", command=a.command, state=state)
        if a.format == "md"
        else json.dumps(pv, sort_keys=True)
    )
    return cliutil.EXIT_OK


def _cmd_engine_request(a):
    req = OnDemandRequest.from_dict(_load(a.request))
    subject = {"artifact_id": a.subject_id} if a.subject_id else None
    print(json.dumps(engine_request(req, a.product, subject, a.environment or ""), sort_keys=True))
    return cliutil.EXIT_OK


def _cmd_env(a):
    req = OnDemandRequest.from_dict(_load(a.request))
    print(json.dumps(bench_env(req, a.product, a.subject_id or "", a.environment or ""), sort_keys=True))
    return cliutil.EXIT_OK


def _cmd_sanitize(a):
    print(safe(sys.stdin.read(a.limit * 4), a.limit))
    return cliutil.EXIT_OK


def _cmd_state(a):
    state = decode_state(sys.stdin.read())
    print(json.dumps(state))
    return cliutil.EXIT_OK if state else cliutil.EXIT_NOTHING


def parser():
    ap = argparse.ArgumentParser(prog="spyre-bench ondemand", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    p = sub.add_parser("parse", help="comment text -> request")
    p.add_argument("--comment-file", required=True)
    p.add_argument("--host", choices=HOSTS, default="github.com")
    p.add_argument("--command", default="/bench")
    p.add_argument("--default-product", default=DEFAULT_PRODUCT)
    p.set_defaults(fn=_cmd_parse)
    p = sub.add_parser("authorize", help="facts.json -> decision (exit 0 allowed, 2 denied)")
    p.add_argument("--facts", required=True)
    p.set_defaults(fn=_cmd_authorize)
    p = sub.add_parser("engine-request", help="request.json -> the Run request for `spyre-bench plan --request`")
    p.add_argument("--request", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--subject-id")
    p.add_argument("--environment")
    p.set_defaults(fn=_cmd_engine_request)
    p = sub.add_parser("env", help="request.json -> the BENCH_* variables for `make bench-plan` as JSON")
    p.add_argument("--request", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--subject-id")
    p.add_argument("--environment")
    p.set_defaults(fn=_cmd_env)
    p = sub.add_parser("sanitize", help="stdin -> the same text made inert for a PR comment")
    p.add_argument("--limit", type=int, default=600)
    p.set_defaults(fn=_cmd_sanitize)
    p = sub.add_parser("state", help="comment body on stdin -> the hidden state, exit 4 when there is none")
    p.set_defaults(fn=_cmd_state)
    p = sub.add_parser("preview", help="plan.json + request.json -> preview")
    p.add_argument("--plan", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--host", choices=HOSTS, default="github.com")
    p.add_argument("--format", choices=("md", "json"), default="md")
    p.add_argument("--queue", help="JSON object of queue facts to show")
    p.add_argument("--dashboard-url")
    p.add_argument("--command", default="/bench")
    p.add_argument("--note", action="append", default=[], help="a line shown under the table (repeatable)")
    p.add_argument("--state", help="JSON object stored in the comment for a later /bench confirm")
    p.set_defaults(fn=_cmd_preview)
    return ap


def main(argv=None):
    a = parser().parse_args(argv)
    try:
        return a.fn(a)
    except UsageError as e:
        print(f"error: {safe(e, 300)}", file=sys.stderr)
        return cliutil.EXIT_INVALID
    except (OSError, ValueError, KeyError, TypeError, RecursionError) as e:
        print(f"error: {type(e).__name__}: {safe(e, 300)}", file=sys.stderr)
        return cliutil.EXIT_INVALID


if __name__ == "__main__":
    sys.exit(main())
