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
  spyre-bench ondemand compare   --plan P --request R --product X    what vs= resolves to, per case (read-only store)
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
from pathlib import Path
from datetime import datetime, timedelta, timezone

import regex

from spyre_bench import cliutil, compare
from spyre_bench.cliutil import UsageError

PRODUCTS = ("torch-spyre", "hf-adapters", "spyre-inference", "spyre-comms")
DEFAULT_PRODUCT = "spyre-inference"
TIERS = ("sanity", "core", "extended")
REUSE_MODES = ("if_valid", "never", "prefer", "only")
PROFILES = ("off", "light", "full")
HOSTS = ("github.com", "ghe")
PURPOSE = "pr"
PREVIEW_SCHEMA = "bench.preview/1"
PREVIEW_MARKER = "<!-- spyre-bench-preview/1 -->"
WRITE_PERMISSIONS = ("write", "maintain", "admin")
MEMBER_ASSOCIATIONS = ("OWNER", "MEMBER", "COLLABORATOR")

NAME = regex.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
DURATION = regex.compile(r"([1-9][0-9]{0,3})([smh])")
SHA = regex.compile(r"[0-9a-f]{40}")
_KEYS = ("bench", "purpose", "reuse", "force", "profile", "vs", "clean", "env", "max-wait", "budget", "sha", "type")
TYPES = ("perf", "quality")
COMMANDS = ("/spyre-bench", "/bench")  # /bench is the deprecated spelling on github.com
SHA_PREFIX = regex.compile(r"[0-9a-f]{7,40}")
FORK_SHA_MIN = 12
CONFIRM_TTL = timedelta(hours=24)
_UNIT = {"s": 1, "m": 60, "h": 3600}


_CONTROL = regex.compile(r"[\x00-\x1f\x7f`<>\\]")
_INVISIBLE = regex.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]")
_MARKUP = regex.compile(r"[|\[\]()!*~&]")
_LINKISH = regex.compile(r"[\[\]!&]")
_REF = regex.compile(r"#(?=[0-9])")


def neutral(text, limit=300):
    """Untrusted text made inert in a GitHub comment: no control or bidi characters, tags, backticks or mentions."""
    s = _CONTROL.sub(" ", _INVISIBLE.sub("", str(text))).replace("@", "@\u200b").replace("\n", " ").strip()
    s = _LINKISH.sub(" ", s).replace("://", ":\u200b//").replace("www.", "www\u200b.")
    return s if len(s) <= limit else s[: limit - 1] + "…"


def safe(text, limit=160):
    """`neutral` plus what a table cell must not carry: pipes, links, images, emphasis, issue refs and URL schemes."""
    s = neutral(text, 10**6)
    s = _REF.sub("#\u200b", _MARKUP.sub(" ", s))
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
    types: tuple
    vs_kinds: tuple
    vs_relax: bool
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


def _load_policies():
    """The limits from ondemand_policy.json, the single place they are set."""
    with open(Path(__file__).with_name("ondemand_policy.json"), encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for host in HOSTS:
        d = raw[host]
        for key in ("tiers", "reuse_modes", "profiles", "types", "vs_kinds"):
            d[key] = tuple(d[key])
        out[host] = Policy(host=host, **d)
    return out


def _load_settings():
    with open(Path(__file__).with_name("ondemand_policy.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return {"enabled": bool(raw["enabled"]), "enforce_budgets": bool(raw["enforce_budgets"])}


POLICIES = _load_policies()
SETTINGS = _load_settings()  # tests and callers may override per process; the file is the default


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
    type: str = "perf"  # perf | quality (TEST_TYPE)
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
            **({"type": self.type} if self.type != "perf" else {}),
        }

    @staticmethod
    def from_dict(d):
        """A request from stored or supplied JSON: every field is checked again, as `parse_flags` would."""
        if not isinstance(d, dict):
            raise RequestError("the request is not an object")
        if d.get("purpose", PURPOSE) != PURPOSE:
            raise RequestError(f"purpose {d.get('purpose')!r}: a comment can only ask for purpose 'pr'")
        req = OnDemandRequest._build(d)
        req._validate()
        return req

    def _validate(self):
        if self.type not in TYPES:
            raise RequestError("the stored request has an unknown type")
        if self.reuse not in REUSE_MODES or self.profile not in PROFILES:
            raise RequestError("the stored request has an unknown reuse mode or profile")
        try:
            compare.parse_vs(self.vs)
        except ValueError:
            raise RequestError("the stored request has a bad vs") from None
        if self.sha and not SHA_PREFIX.fullmatch(self.sha):
            raise RequestError("the stored request has a bad sha")
        if self.env:
            _name(self.env, "env")
        for name in self.force:
            _name(name, "force case")
        if not (math.isfinite(self.budget) and self.budget >= 0) or self.max_wait_s < 0:
            raise RequestError("the stored request has a bad budget or wait")
        if not self.selections:
            raise RequestError("the stored request has no selection")
        seen = set()
        for product, sel, _ in self.selections:
            if product not in PRODUCTS or product in seen or not isinstance(sel, dict):
                raise RequestError("the stored request names a bad product")
            seen.add(product)
            if sel.get("tier") is not None and sel["tier"] not in TIERS:
                raise RequestError("the stored request has a bad tier")
            if set(sel) - {"tier", "suites", "cases"}:
                raise RequestError("the stored request has an unknown selection key")
            for key in ("suites", "cases"):
                if sel.get(key) is not None and not isinstance(sel[key], list):
                    raise RequestError("the stored request has a bad selection list")
                for name in sel.get(key) or ():
                    _name(name, key)

    @staticmethod
    def _build(d):
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
            type=str(d.get("type") or "perf"),
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
    key = raw.lower().replace("_", "-")
    key = key[len("bench-") :] if key.startswith("bench-") else key
    key = "type" if key == "test-type" else key
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
    try:
        compare.parse_vs(vs)
    except ValueError as e:
        raise RequestError(str(e)) from None
    env = got.get("env", "")
    if env:
        _name(env, "env")
    budget = 0.0
    if "budget" in got:
        try:
            if not got["budget"].isascii() or "_" in got["budget"] or got["budget"] != got["budget"].strip():
                raise ValueError
            budget = float(got["budget"])
        except ValueError:
            raise RequestError(f"budget={got['budget']!r}: a number of device-hours") from None
        if not budget > 0 or budget == float("inf"):
            raise RequestError("budget must be a positive, finite number of device-hours")
    ttype = got.get("type", "perf").lower()
    if ttype not in TYPES:
        raise RequestError(f"type={got['type']!r}: one of {', '.join(TYPES)}")
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
        type=ttype,
        asked={k: got[k] for k in sorted(got)},
    )


@dataclass(frozen=True)
class Command:
    kind: str  # run | confirm | cancel
    flags: dict
    line: str


def parse_command(text, command=None):
    """The first `command` line outside code fences and quotes -> Command, or None when there is none.

    `/spyre-bench <selection> [key=value ...]`, `/spyre-bench confirm`, `/spyre-bench cancel`; `/bench` is the deprecated
    spelling. A first token without `=` is the selection. `command` (one spelling) overrides the defaults.
    """
    commands = (command,) if command else COMMANDS
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
        if fence or "<!--" in line:
            continue
        used = next((c for c in commands if line.startswith(c + " ") or line.rstrip() == c), None)
        if used is None:
            continue
        tokens = line[len(used) :].split()
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


def _bench_family(key):
    k = key.lower().replace("_", "-")
    k = k[len("bench-") :] if k.startswith("bench-") else k
    return k in _KEYS or k == "test-type"


def to_spyre_test(cmd):
    """The internal `/spyre-test` line a /spyre-bench command stands for (the comment bot does the rewrite; kept here so the
    two stay equal by the shared rewrite vectors). Perf defaults to the existing forced perf-only preset on amd64, ppc64le
    and s390x unless the commenter chose a preset; there is no quality-only preset, so quality keeps the default preset."""
    out = ["/spyre-test"]
    out += [f"{k}={v}" for k, v in cmd.flags.items() if k.lower() in ("arch", "archs")]
    ttype = "perf"
    for key, value in cmd.flags.items():
        k = key.lower().replace("_", "-")
        if k in ("arch", "archs"):
            continue
        if k in ("type", "test-type", "bench-type"):
            ttype = value
        elif k == "bench":
            out.append(f"bench={value}")
        elif _bench_family(key):
            out.append(f"bench-{k[len('bench-') :] if k.startswith('bench-') else k}={value}")
        else:
            out.append(f"{key}={value}" if value != "" else key)
    if ttype == "perf" and not any(k.lower() == "preset" for k in cmd.flags):
        out.insert(1 + sum(1 for k in cmd.flags if k.lower() in ("arch", "archs")), "preset=perf-only")
    out.append(f"TEST_TYPE={ttype}")
    return " ".join(out)


def check_policy(req, policy, actor=None):
    """Host and actor gates on a parsed request. Returns a list of violation messages (empty = allowed)."""
    bad = []
    for product, sel, _ in req.selections:
        tier = sel.get("tier") or "sanity"
        if tier not in policy.tiers:
            where = "internal requests only" if tier == "core" else "ops requests only"
            bad.append(f"tier '{tier}' for {product} is not available here ({where})")
    if req.type not in policy.types:
        bad.append(f"type={req.type} is not available here")
    if req.reuse not in policy.reuse_modes:
        bad.append(f"reuse={req.reuse} is not available here")
    if "all" in req.force and not policy.force_all:
        bad.append("force=all is not available here: name the cases")
    if req.profile not in policy.profiles:
        bad.append(f"profile={req.profile} is not available here")
    if compare.parse_vs(req.vs).kind not in policy.vs_kinds:
        bad.append(f"vs={compare.parse_vs(req.vs).kind} is not available here")
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
    out = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    return out if out.tzinfo else out.replace(tzinfo=timezone.utc)


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
    would_refuse: list = field(default_factory=list)  # budget findings reported but not enforced

    @property
    def allowed(self):
        return self.state in ("admit", "needs_confirm")

    def to_dict(self):
        return dict(self.__dict__, allowed=self.allowed)


SOFT_PREFIXES = ("PR window budget", "another user's benchmark request")
LIVE_STATES = ("active", "previewed", "launched", "admit", "needs_confirm")
DEAD_STATES = ("cancelled", "denied", "rejected", "deny")
FINISHED_STATES = ("done", "completed", "finished", "failed")


def _hours(value):
    """A device-hour figure from stored data: finite and >= 0, else infinite so that a corrupt row fails closed."""
    try:
        x = float(value or 0)
    except (TypeError, ValueError):
        return math.inf
    return x if math.isfinite(x) and x >= 0 else math.inf


def admission(history, new, now, policy):
    """Supersede and rate rules over earlier requests. `history` rows: id, pr, user, ts, state, est, clean.

    Row states: any state that is not dead or finished (active, previewed, launched, admit, needs_confirm, or
    anything unknown) is live, case-insensitively; done, completed and finished are never cancelled but still count
    toward the window; cancelled, denied, rejected and deny count for nothing except the rate limit and the
    per-day clean quota (a submit-and-cancel loop still costs).
    One live request per PR: a new one cancels the requester's own, or anyone's when the requester has write access;
    otherwise it is refused. Window cap on estimated device-hours per PR, per-user hourly rate and clean quotas.
    Returns (reasons, cancel_ids).
    """
    now = _parse(now)
    history = [dict(h, pr=str(h.get("pr")), state=str(h.get("state", "")).lower()) for h in history]
    live = [h for h in history if h["state"] not in DEAD_STATES]

    def mine(h):
        return (
            f"{str(h.get('repo') or new.get('repo', '')).lower()}#{h['pr']}"
            == f"{str(new.get('repo', '')).lower()}#{new['pr']}"
        )

    here = [h for h in live if mine(h) and h["state"] not in FINISHED_STATES]
    reasons = []
    cancel = [h["id"] for h in here if h["user"] == new["user"] or new.get("can_supersede")]
    if len(cancel) < len(here):
        reasons.append("another user's benchmark request is still running on this pull request")
    kept = [h for h in live if h["id"] not in cancel]
    in_window = [h for h in kept if mine(h) and now - _parse(h["ts"]) <= policy.window]
    spent = sum(_hours(h.get("est")) for h in in_window)
    if spent + new["est"] > policy.pr_window_device_hours:
        reasons.append(
            f"PR window budget: {spent:g} device-hours already planned in {policy.window.total_seconds() / 3600:g}h, "
            f"cap {policy.pr_window_device_hours:g}"
        )
    hour = [
        h
        for h in history
        if h["user"] == new["user"] and h.get("rate", True) and now - _parse(h["ts"]) <= timedelta(hours=1)
    ]
    if len(hour) >= policy.per_user_per_hour:
        reasons.append(f"rate limit: {policy.per_user_per_hour} requests per user per hour")
    if new.get("clean"):
        mine = [h for h in history if h["user"] == new["user"] and h.get("clean") and h["state"] not in DEAD_STATES[1:]]
        if sum(1 for h in mine if h["state"] in LIVE_STATES and h["id"] not in cancel) >= policy.clean_per_user:
            reasons.append(f"clean quota: {policy.clean_per_user} concurrent clean request per user")
        if sum(1 for h in mine if now - _parse(h["ts"]) <= timedelta(days=1)) >= policy.clean_per_user_per_day:
            reasons.append(f"clean quota: {policy.clean_per_user_per_day} clean requests per user per day")
        if sum(1 for h in kept if h.get("clean") and h["state"] in LIVE_STATES) >= policy.clean_cluster:
            reasons.append(f"clean quota: {policy.clean_cluster} clean requests running at once")
    return reasons, cancel


def plan_violations(plan, policy):
    """Cases of a plan that the host does not allow (always enforced). Cases and named selections carry their own tier,
    so this is where a tier limit is enforced for them; a case without a tier cannot be judged and is refused. A reused
    result can turn out invalid by the time the run starts, and then the case runs: judge it as if it will."""
    bad = []
    for c in plan.get("cases") or []:
        if str(c.get("action", "")).lower() == "skip":
            continue
        tier = (c.get("run_props") or {}).get("run.tier")
        if tier not in policy.tiers:
            bad.append(f"case {safe(c.get('name'), 60)} has tier '{safe(tier or 'none', 20)}', not available here")
    return bad[:5]


def plan_cost_issues(plan):
    """Run cases whose cost is unknown (no run estimate). Refused when budgets are enforced, reported otherwise."""
    bad = []
    for c in plan.get("cases") or []:
        action = str(c.get("action", "")).lower()
        est = c.get("est_min")
        if action not in ("reuse", "skip") and not (isinstance(est, (int, float)) and math.isfinite(est) and est > 0):
            bad.append(f"case {safe(c.get('name'), 60)} declares no run estimate (run.est), so its cost is unknown")
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

    would = []

    def deny(*reasons):
        return Decision(
            "deny", list(reasons), sandbox=sandbox_profile(is_fork), pinned_sha=head, would_refuse=list(would)
        )

    if not SETTINGS["enabled"]:
        return deny("on-demand benchmarks are not enabled yet")
    enforce = SETTINGS["enforce_budgets"]

    def soft(msg):
        """A budget finding: refuses when budgets are enforced, otherwise recorded as 'would have been refused'."""
        would.append(msg)
        return deny(msg) if enforce else None

    if not invoker_allowed(who, policy):
        return deny("the commenter is not allowed to run benchmarks on this repository")
    if confirmed and not has_write(actor):
        return deny("only a commenter with write access can confirm a run")
    if not SHA.fullmatch(head):
        return deny("the PR head is not a full commit id")
    if confirmed and not facts.get("pinned_sha"):
        return deny("nothing records which commit this request was made for: comment again")
    if confirmed:
        try:
            age = _parse(now) - _parse(facts.get("requested_at"))
        except (TypeError, ValueError):
            return deny("nothing records when this request was made: comment again")
        if age > CONFIRM_TTL or age < -timedelta(minutes=5):
            return deny("this request is too old to confirm: comment again")
    if facts.get("pinned_sha") and facts["pinned_sha"] != head:
        return deny("the PR head moved since this request was made: comment again to run the new commit")
    violations = check_policy(req, policy, who)
    if violations:
        return deny(*violations)
    if req.sha and not head.startswith(req.sha):
        return deny(
            "sha= does not match the current head of the pull request: comment again with the commit you reviewed"
        )
    approval = None
    if is_fork:
        # A fork runs only the exact commit a write-access commenter named; the comment is the approval.
        if not (len(req.sha) >= FORK_SHA_MIN and head.startswith(req.sha)):
            return deny(
                f"fork pull request: add sha=<the first {FORK_SHA_MIN} or more characters of the head you reviewed>"
            )
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
    stage = facts.get("stage", "request")
    if stage not in ("request", "planned"):
        return deny(f"unknown stage {neutral(stage, 20)!r}")
    planned = stage == "planned"
    est = _finite_hours(facts["estimate_device_hours"]) if facts.get("estimate_device_hours") is not None else None
    if facts.get("estimate_device_hours") is not None and est is None:
        return deny("the estimate is not a finite number of device-hours")
    if planned and isinstance(facts.get("plan"), dict):
        plan_est = _finite_hours((facts["plan"].get("estimate") or {}).get("device_hours"))
        if plan_est is None:
            denied = soft("the plan has no usable estimate in device-hours")
            if denied:
                return denied
            plan_est = 0.0
        est = plan_est if est is None else max(est, plan_est)
    if planned and est is None:
        denied = soft("the plan has no usable estimate in device-hours")
        if denied:
            return denied
        est = 0.0
    if facts.get("estimate_device_hours") is not None and est is None:
        return deny("the estimate is not a finite number of device-hours")
    needs_confirm = False
    if planned:
        plan = facts.get("plan")
        if not isinstance(plan, dict) or not plan.get("cases"):
            return deny("the planned stage needs the plan, with its cases")
        bad = plan_violations(plan, policy)
        if bad:
            return deny(*bad)
        for msg in plan_cost_issues(plan):
            denied = soft(msg)
            if denied:
                return denied
        # The estimate is declared by the catalog; the timeouts bound what a case can actually take.
        worst = sum(
            float(c.get("timeout_min") or 0) * float(c.get("accelerators") or 1) / 60
            for c in plan.get("cases") or []
            if str(c.get("action", "")).lower() not in ("reuse", "skip")
        )
        if worst > limit:
            denied = soft(
                f"the plan's timeouts allow up to {worst:g} device-hours, over the {limit:g} limit for this request"
            )
            if denied:
                return denied
        ok, needs_confirm = check_budget(est, limit, policy)
        if not ok:
            denied = soft(f"plan needs {est:g} device-hours, over the {limit:g} limit for this request")
            if denied:
                return denied
        needs_confirm = needs_confirm and not confirmed and enforce
    own = str(facts.get("request_id") or "")
    reasons, cancel = admission(
        [h for h in facts.get("history") or [] if not own or str(h.get("id")) != own],
        {
            "pr": pr["number"],
            "repo": pr.get("base_repo", ""),
            "user": who.get("login", ""),
            "est": est or 0.0,
            "clean": req.clean,
            "can_supersede": has_write(actor),
        },
        now,
        policy,
    )
    hard = [r for r in reasons if not r.startswith(SOFT_PREFIXES)]
    if hard:
        return deny(*hard)
    for r in (r for r in reasons if r.startswith(SOFT_PREFIXES)):
        denied = soft(r)
        if denied:
            return denied
    if not enforce:
        cancel = [c for c in cancel]  # supersede of the requester's own live request still applies
    state = "needs_confirm" if needs_confirm else "admit"
    why = (
        [f"estimate above {int(policy.confirm_fraction * 100)}% of the limit: confirm to run"] if needs_confirm else []
    )
    return Decision(
        state, why, cancel, head, approval.to_dict() if approval else None, sandbox_profile(is_fork), limit, would
    )


def engine_request(req, product, subject=None, environment=""):
    """The engine Run request for one product of the parsed request (what `spyre-bench plan --request` reads)."""
    for p, sel, _ in req.selections:
        if p == product:
            break
    else:
        raise KeyError(product)
    selection = {k: v for k, v in sel.items() if v}
    out = {"purpose": req.purpose, "selection": {**selection, "type": req.type}}
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
    env = {
        "BENCH_PURPOSE": req.purpose,
        "BENCH_REUSE": req.reuse,
        **({"BENCH_TYPE": req.type} if req.type != "perf" else {}),
    }
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


# ---------------------------------------------------------------------------------------------------------------
# Request log: the ci_requests row for a request, and admission history read back from v_ci_requests.

CI_STATES = ("previewed", "launched", "denied", "rejected", "failed", "cancelled")
CI_CHANNEL = "comment"
HISTORY_SQL = (
    # Straight from the log, not v_ci_requests: the estimate of the latest event that carries one (a later event
    # written without a plan must not turn a 90 device-hour request into 0).
    "SELECT toString(request_id) AS id, argMin(actor, ts) AS user, argMax(state, ts) AS state, "
    "argMin(subjects, ts) AS subjects, argMaxIf(estimate, ts, estimate != '') AS estimate, min(ts) AS ts "
    "FROM ci_requests WHERE channel = 'comment' AND target_kind = 'pr' AND ts >= now() - INTERVAL 2 DAY "
    "GROUP BY request_id"
)
LAUNCH_TTL = timedelta(hours=6)  # a launched request has no completion event; past the longest job it counts as done


def _uuid7(now=None):
    import os
    import uuid

    ms = int((_parse(now) if now else datetime.now(timezone.utc)).timestamp() * 1000)
    b = bytearray(ms.to_bytes(6, "big") + os.urandom(10))
    b[6] = (b[6] & 0x0F) | 0x70
    b[8] = (b[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(b)))


def ci_request_row(
    req,
    state,
    actor,
    *,
    host,
    pr,
    repo,
    head_sha,
    plan=None,
    request_id="",
    comment_url="",
    error="",
    now=None,
    would_refuse=(),
    cmp=None,
):
    """One ci_requests row (append-only; one per event). `plan` is a plan.json, `actor` the login that caused the event."""
    if state not in CI_STATES:
        raise RequestError(f"state {state!r}: one of {', '.join(CI_STATES)}")
    cases = (plan or {}).get("cases") or []
    est = (plan or {}).get("estimate") or {}
    pv = build_preview(plan, req, POLICIES[host]) if plan else None
    brief = lambda c: {"name": str(c.get("name")), "action": c.get("action"), "reason": str(c.get("reason") or "")}  # noqa: E731
    return {
        "request_id": request_id or _uuid7(now),
        "plan_id": (pv or {}).get("digest", ""),
        "state": state,
        "actor": neutral(actor, 200),
        "channel": CI_CHANNEL,
        "target_kind": "pr",
        "subjects": json.dumps(
            {"repo": repo, "pr": int(pr), "head_sha": head_sha, "comment_url": comment_url, "host": host},
            sort_keys=True,
        ),
        "preset": "bench-pr",
        "arches": [str(a) for a in ((plan or {}).get("leg") or {}).get("arch", "").split()] or [],
        "modes": [req.type],
        "cells_requested": json.dumps([brief(c) for c in cases if c.get("action") == "run"]),
        "cells_skipped": json.dumps([brief(c) for c in cases if c.get("action") == "reuse"]),
        "cells_blocked": json.dumps([brief(c) for c in cases if c.get("action") not in ("run", "reuse")]),
        "estimate": json.dumps(
            {
                "device_hours": est.get("device_hours"),
                "wall_min": est.get("wall_min"),
                "clean": req.clean,
                "limit": budget_limit(req, POLICIES[host]),
            },
            sort_keys=True,
        )
        if plan
        else "",
        "canonical": json.dumps(req.to_dict(), sort_keys=True),
        "error": neutral(error, 500),
        "props": {
            "bench.host": host,
            "bench.purpose": req.purpose,
            "bench.digest": (pv or {}).get("digest", ""),
            "bench.enforced": str(SETTINGS["enforce_budgets"]).lower(),
            # What the budgets would have refused, recorded whether or not they are enforced: this is the usage record.
            "bench.would_refuse": neutral("; ".join(map(str, would_refuse)), 500),
            **({k: neutral(v, 200) for k, v in compare.request_props(cmp).items()} if cmp else {}),
        },
    }


_HISTORY_STATE = {"previewed": "previewed", "launched": "launched", "denied": "denied", "rejected": "rejected",
                  "cancelled": "cancelled", "failed": "done"}  # fmt: skip


def history_from_ci_requests(rows, host, now=None):
    """HISTORY_SQL rows -> the history rows `admission` reads, for one host only. A row it cannot read raises, so a
    damaged log blocks admission rather than waving a request through. A launched request older than LAUNCH_TTL is done."""
    now = _parse(now) if now else datetime.now(timezone.utc)
    out = []
    for r in rows:
        try:
            subj = json.loads(r.get("subjects") or "{}")
            if subj["host"] != host:
                continue
            est = json.loads(r.get("estimate") or "{}")
            ts = _parse(r["ts"])
            state = _HISTORY_STATE.get(str(r["state"]).lower(), "unknown")
            if state == "launched" and now - ts > LAUNCH_TTL:
                state = "done"
            hours = est.get("device_hours")
            out.append(
                {"id": str(r["id"]), "pr": subj["pr"], "repo": str(subj["repo"]).lower(), "user": str(r["user"]),
                 "ts": _iso(ts), "state": state, "est": hours if hours is not None else 0,
                 "clean": bool(est.get("clean"))}
            )  # fmt: skip
        except (KeyError, TypeError, ValueError, AttributeError):
            raise RequestError(
                "the request log has a row that cannot be read: not admitting anything until it is fixed"
            ) from None
    return out


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


def compare_view(cmp):
    """The compare document as the preview carries it: states, reasons and short ids, never a measured value."""
    t = cmp.get("target") or {}
    return {
        "schema": cmp["schema"],
        "vs": cmp["vs"]["text"],
        "status": cmp["status"],
        "refusal": cmp.get("refusal"),
        "target": {k: t.get(k) for k in ("artifact_id", "source", "ref_key", "role", "baseline_eligible")}
        if t
        else None,
        "cap": cmp["cap"],
        "subject_artifact": cmp.get("subject_artifact", ""),
        "counts": cmp["counts"],
        "estimate": cmp["estimate"],
        "cases": [
            {
                "name": c["name"],
                "benchmark_id": c.get("benchmark_id") or "",
                "state": c["state"],
                "why": c["why"],
                "subject": c["subject"]["action"],
                "subject_run": c["subject"].get("reused_from") or "",
                "other_run": c["other"]["run_id"],
                "other_validity": c["other"]["validity"],
            }
            for c in cmp["cases"]
        ],
        "digest": cmp["digest"],
    }


def build_preview(plan, req, policy=None, queue=None, dashboard_url="", notes=(), cmp=None):
    """The preview as data (the dashboard consumes this; `render_markdown` is the same data as text).

    `cmp` is a compare document: the other side's device-hours join the estimate the budget is judged on."""
    s, est = plan["summary"], plan.get("estimate") or {}
    known = _finite_hours(est.get("device_hours"))
    if known is not None and cmp:
        other = cmp.get("estimate") or {}
        known = known + float(other.get("other_device_hours") or 0.0) if other.get("known", True) else None
    hours = known if known is not None else 0.0
    limit = budget_limit(req, policy) if policy else None
    needs_confirm = False
    over = known is None
    violations = plan_violations(plan, policy) if policy else []
    if policy and limit is not None and known is not None:
        ok, needs_confirm = check_budget(hours, limit, policy)
        over = not ok
    enforced = SETTINGS["enforce_budgets"]
    needs_confirm = needs_confirm and enforced
    would = plan_cost_issues(plan) if policy else []
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
            "enforced": enforced,
            "cost_issues": would,
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
    keyed = {k: body[k] for k in ("counts", "estimate", "diff", "cases")}
    if cmp:
        body["compare"] = compare_view(cmp)
        keyed["compare"] = cmp["digest"]
    body["digest"] = hashlib.sha256(json.dumps(keyed, sort_keys=True).encode()).hexdigest()[:16]
    return body


_REFUSAL_PUBLIC = {  # fixed wording: a public comment never repeats an artifact id or a stored name
    "vs_not_found": "nothing matches that",
    "vs_not_allowed": "that kind of comparison is not available here",
    "target_moved": "the comparison target changed since the preview",
    "store_unavailable": "the result store did not answer",
    "no_store": "no result store is configured",
}
_CMP_WORD = {"ready": "other side stored", "will_run_other": "other side runs", "cannot_compare": "cannot compare"}


def _render_compare(cv, public):
    """Compare section of the PR comment. A public host names no artifact, run or reference, only states."""
    c = cv["counts"]
    t = cv.get("target") or {}
    where = "" if public or not t else f" ({safe(t.get('source'), 20)} {safe(str(t.get('artifact_id'))[:8], 8)})"
    out = [
        f"**Compared with {safe(cv['vs'], 80)}**{where}: other side stored {c['ready']} · other side runs {c['will_run_other']} · cannot compare {c['cannot_compare']}"
    ]
    out.append("A PR run is advisory (`suspected` at most) and never becomes a baseline.")
    if cv["status"] != "ok":
        code = (cv.get("refusal") or {}).get("code")
        out.append(
            f"**Nothing to compare:** {_REFUSAL_PUBLIC.get(code, 'the comparison is not available') if public else safe((cv.get('refusal') or {}).get('message'), 160)}"
        )
    bad = [k for k in cv["cases"] if k["state"] != "ready"][:10]
    if bad:
        out += ["", "| case | state | why |", "|---|---|---|"]
        out += [
            f"| {safe(k['name'], 90)} | {_CMP_WORD.get(k['state'], safe(k['state'], 20))} | {safe(k['why'], 100)} |"
            for k in bad
        ]
    return out


def render_markdown(preview, public=True, command="/spyre-bench", state=None):
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
    report_only = not b.get("enforced", True)
    issues = [safe(i, 160) for i in b.get("cost_issues") or []]
    if b.get("violations"):
        out.append("**Nothing will run** until the selection fits this host.")
    elif report_only and (b.get("over") or issues or not b.get("estimate_known", True)):
        out.append(
            "**Report only:** budgets are not enforced yet. This would have been refused (over the limit or cost unknown); it runs anyway."
        )
        out += [f"- {i}" for i in issues]
    elif not b.get("estimate_known", True):
        out.append("**The plan has no usable estimate: nothing will run.**")
    elif b.get("over"):
        out.append("**Over the limit: nothing will run.** Narrow the selection or ask for a smaller tier.")
    elif b.get("violations"):
        out.append("**Nothing will run** until the selection fits this host.")
    elif b.get("needs_confirm"):
        out.append(f"**Confirm to run:** reply `{command} confirm` (write access needed). Nothing runs until then.")
    cv = preview.get("compare")
    if cv:
        out += ["", *_render_compare(cv, public)]
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
            out.append(
                f"| {safe(case['action'], 10)} | {safe(case['name'], 90)} | {safe(case['repeats'] or '', 6)} | {note} |"
            )
        if len(preview["cases"]) > _MAX_ROWS:
            out.append(f"| | …and {len(preview['cases']) - _MAX_ROWS} more | | |")
    url = str(preview.get("dashboard_url") or "")
    if regex.fullmatch(r"https://[A-Za-z0-9._~/?&=%#:-]{1,300}", url):
        out += ["", f"{'Dashboard (sign-in required)' if public else 'Details'}: {url}"]
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
    pv = build_preview(
        _load(a.plan),
        req,
        POLICIES[a.host],
        queue,
        a.dashboard_url or "",
        a.note,
        _load(a.compare) if a.compare else None,
    )
    state = json.loads(a.state) if a.state else None
    print(
        render_markdown(pv, public=a.host == "github.com", command=a.command or "/spyre-bench", state=state)
        if a.format == "md"
        else json.dumps(pv, sort_keys=True)
    )
    return cliutil.EXIT_OK


def _cmd_compare(a):
    from spyre_bench import store as store_mod

    req = OnDemandRequest.from_dict(_load(a.request))
    plan = _load(a.plan)
    policy = POLICIES[a.host]
    relax = tuple(cliutil.csv(a.relax_labels or ""))
    if relax and not policy.vs_relax:
        raise RequestError("relaxed environment matching is not available here")
    if a.store_file:
        spec = _load(a.store_file)
        for r in spec.get("results", []):
            r["ts"] = _parse(r["ts"])
        for t in spec.get("targets", []):
            t["ts"] = _parse(t["ts"])
        st = store_mod.MemoryStore(
            results=spec.get("results", []), targets=spec.get("targets", []), events=spec.get("events", [])
        )
    else:
        st = store_mod.open_store(a.store)
    doc = compare.plan_compare(
        plan,
        compare.parse_vs(req.vs),
        st,
        product=a.product,
        host_kinds=policy.vs_kinds,
        relax=relax,
        purposes=tuple(cliutil.csv(a.purposes)) if a.purposes else compare.MAIN_PURPOSES,
        expect_artifact=a.expect_artifact or "",
    )
    print(json.dumps(doc, sort_keys=True))
    if (a.expect_digest and doc["digest"] != a.expect_digest) or (a.expect_artifact and doc["status"] != "ok"):
        print("error: the comparison changed since the preview", file=sys.stderr)
        return cliutil.EXIT_INVALID
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


def _cmd_record(a):
    req = OnDemandRequest.from_dict(_load(a.request))
    plan = _load(a.plan) if a.plan else None
    row = ci_request_row(
        req,
        a.state,
        a.actor,
        host=a.host,
        pr=a.pr,
        repo=a.repo,
        head_sha=a.head_sha,
        plan=plan,
        request_id=a.request_id or "",
        comment_url=a.comment_url or "",
        error=a.error or "",
        cmp=_load(a.compare) if a.compare else None,
    )
    print(json.dumps(row, sort_keys=True))
    return cliutil.EXIT_OK


def _cmd_history(a):
    print(json.dumps(history_from_ci_requests(_load(a.rows), a.host), sort_keys=True))
    return cliutil.EXIT_OK


def _cmd_limit(a):
    pol = POLICIES[a.host]
    print(
        json.dumps(
            {
                "request_device_hours": max(pol.budget_device_hours.values()),
                "pr_window_device_hours": pol.pr_window_device_hours,
            }
        )
    )
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
    p.add_argument("--command", default=None)
    p.add_argument("--default-product", default=DEFAULT_PRODUCT)
    p.set_defaults(fn=_cmd_parse)
    p = sub.add_parser("authorize", help="facts.json -> decision (exit 0 allowed, 2 denied)")
    p.add_argument("--facts", required=True)
    p.set_defaults(fn=_cmd_authorize)
    p = sub.add_parser("compare", help="plan.json + request.json (+ the store) -> what vs= resolves to, per case")
    p.add_argument("--plan", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--host", choices=HOSTS, default="github.com")
    p.add_argument("--store", default="none", help="none | clickhouse (read-only)")
    p.add_argument("--store-file", help="a JSON file standing in for the store (tests and demos)")
    p.add_argument(
        "--relax-labels", help="env label patterns to match instead of the environment id (where the host allows)"
    )
    p.add_argument("--purposes", help="purposes whose results vs=main may use")
    p.add_argument("--expect-artifact", help="refuse when the target is not this artifact (the one the preview showed)")
    p.add_argument("--expect-digest", help="exit 2 when the comparison differs from the one the preview showed")
    p.set_defaults(fn=_cmd_compare)
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
    p = sub.add_parser("record", help="request (+plan) -> the ci_requests row for one event, as JSON")
    p.add_argument("--request", required=True)
    p.add_argument("--plan")
    p.add_argument("--state", required=True)
    p.add_argument("--actor", required=True)
    p.add_argument("--host", choices=HOSTS, required=True)
    p.add_argument("--pr", type=int, required=True)
    p.add_argument("--repo", required=True)
    p.add_argument("--head-sha", required=True)
    p.add_argument("--request-id")
    p.add_argument("--comment-url")
    p.add_argument("--error")
    p.add_argument("--would-refuse", action="append", help="a budget finding to record (repeatable)")
    p.add_argument("--compare", help="compare.json: records the comparison target in the row")
    p.set_defaults(fn=_cmd_record)
    p = sub.add_parser("history", help="v_ci_requests rows (JSON file) -> admission history rows")
    p.add_argument("--rows", required=True)
    p.add_argument("--host", choices=HOSTS, required=True)
    p.set_defaults(fn=_cmd_history)
    p = sub.add_parser("limit", help="the per-request and per-PR window device-hour limits of a host, as JSON")
    p.add_argument("--host", choices=HOSTS, required=True)
    p.set_defaults(fn=_cmd_limit)
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
    p.add_argument("--compare", help="compare.json from `ondemand compare`: adds its section and cost")
    p.add_argument("--command", default=None)
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
