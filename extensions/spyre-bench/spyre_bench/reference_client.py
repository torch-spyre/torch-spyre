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

"""The dashboard side of `reference submit|approve|reject|retract|list|show`: an HTTP client and an in-memory fake.

Both implement the same methods. The server owns the approval rules; FakeClient reproduces them so tests can prove them.
"""

import hashlib
import http.client
import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from spyre_bench import cliutil, reference

URL_ENV, TOKEN_ENV = "BENCH_DASHBOARD_URL", "BENCH_DASHBOARD_TOKEN"
LIST_PATH = "/api/v2/bench/references"
UPLOADS_PATH = "/api/v2/uploads"
RETRACT_PATH = "/api/v2/uploads/{id}/retract"
NEEDS_TOKEN = "this verb needs a dashboard bearer token: set {token} (and {url})"
UI_ONLY = (
    "approval is UI-only in v1: no programmatic identity is configured (set {token} to an approval service token "
    "when one exists); use the dashboard's approval page"
)
TIMEOUT_S = 30


class ClientError(RuntimeError):
    """The service is unreachable or answered with an error."""


class Refused(ClientError):
    """The service (or its rules) refused the request."""


class NoCredentials(ClientError):
    """No dashboard URL or bearer token is configured."""


class InsecureUrl(NoCredentials):
    """The dashboard URL would carry the bearer token in clear text to another host."""


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect would carry the bearer token to another host; treat it as an error instead."""

    def redirect_request(self, *args, **kwargs):
        return None


def _default_open(req, timeout):
    return urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout)


def _q(part):
    return urllib.parse.quote(str(part), safe="")


class HttpClient:
    """Calls the dashboard API with a bearer token; `opener(request, timeout)` is injectable for tests."""

    def __init__(self, base_url, token, opener=None):
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme == "https" and parsed.hostname:
            pass
        elif parsed.scheme == "http" and parsed.hostname in LOCAL_HOSTS:
            pass
        else:
            raise InsecureUrl(
                f"{base_url!r}: the dashboard URL must be https:// (http:// only for localhost); "
                "the bearer token is never sent in clear text to another host"
            )
        self.base = base_url.rstrip("/")
        self.token = token
        self.opener = opener or _default_open

    @classmethod
    def from_env(cls, env=None, opener=None, write=True):
        url, token = cliutil.env(URL_ENV), cliutil.env(TOKEN_ENV)
        if env is not None:
            url, token = env.get(URL_ENV), env.get(TOKEN_ENV)
        if not token:
            raise NoCredentials((UI_ONLY if write else NEEDS_TOKEN).format(token=TOKEN_ENV, url=URL_ENV))
        if not url:
            raise NoCredentials(f"set {URL_ENV} to the dashboard base URL")
        return cls(url, token, opener)

    def call(self, method, path, body=None, query=None, raw=None, ctype=None):
        url = self.base + path + (("?" + urllib.parse.urlencode(query)) if query else "")
        data = raw
        if body is not None:
            data, ctype = json.dumps(body).encode(), "application/json"
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json")
        if ctype:
            req.add_header("Content-Type", ctype)
        try:
            with self.opener(req, TIMEOUT_S) as resp:
                text = resp.read().decode()
        except urllib.error.HTTPError as e:
            message = _error_text(e.read().decode(errors="replace"))
            if e.code == 409 and path.startswith(UPLOADS_PATH):
                message = f"duplicate content: this reference is already submitted or published ({message})"
            hint = HTTP_HINTS.get(e.code)
            message = f"{hint}: {message}" if hint and message else (hint or message)
            raise (Refused if 400 <= e.code < 500 else ClientError)(f"HTTP {e.code}: {message}") from None
        except (urllib.error.URLError, OSError, http.client.HTTPException) as e:
            raise ClientError(f"cannot reach the dashboard: {type(e).__name__}: {e}") from None
        except UnicodeDecodeError:
            raise ClientError("the dashboard answered with bytes that are not text") from None
        if not text.strip():
            return {}
        try:
            return json.loads(text)
        except ValueError:
            raise ClientError(
                f"the dashboard answered with something that is not JSON: {text.strip()[:120]!r}"
            ) from None

    def submit(self, bundle, doc=None, dry_run=False, inspect=False):
        """POST the sealed .tgz to the upload route as multipart (bundle, format=reference, submit_token)."""
        content = Path(bundle).read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        boundary = digest[:32]
        fields = {"format": "reference", "submit_token": digest[:48]}  # a retry of the same bundle is the same submit
        parts = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in fields.items()
        ]
        head = f'--{boundary}\r\nContent-Disposition: form-data; name="bundle"; filename="bundle.tgz"\r\n'
        parts.append(head.encode() + b"Content-Type: application/gzip\r\n\r\n" + content + b"\r\n")
        raw = b"".join(parts) + f"--{boundary}--\r\n".encode()
        path = UPLOADS_PATH + ("/inspect" if inspect else "")
        return self.call("POST", path, query={"dry_run": 1} if dry_run and not inspect else None, raw=raw,
                         ctype=f"multipart/form-data; boundary={boundary}")  # fmt: skip

    def retire(self, key):
        return self.call("POST", f"{LIST_PATH}/{_q(key)}/retire", {})

    def rollback(self, key, to_artifact_id):
        return self.call("POST", f"{LIST_PATH}/{_q(key)}/rollback", {"to_artifact_id": to_artifact_id})

    def me(self):
        return self.call("GET", f"{LIST_PATH}/me")

    def approve(self, upload_id, reason="", terms_ack=False, promote=True, register_missing=False, ack_not_clean=False):
        body = {
            "terms_ack": terms_ack, "promote": promote, "register_missing": register_missing,
            "ack_not_clean": ack_not_clean,
        }  # fmt: skip
        return self.call(
            "POST", f"{LIST_PATH}/{_q(upload_id)}/approve", {**body, **({"reason": reason} if reason else {})}
        )

    def reject(self, upload_id, reason="", request_changes=False):
        verb = "request-changes" if request_changes else "reject"
        return self.call("POST", f"{LIST_PATH}/{_q(upload_id)}/{verb}", {"reason": reason})

    def retract(self, upload_id, reason=""):
        return self.call("POST", RETRACT_PATH.format(id=_q(upload_id)), {"reason": reason} if reason else {})

    def list(self, state=None, klass=None, origin=None, component=None, pending=False, mine_approvable=False):
        query = {"state": state, "class": klass, "origin": origin, "component": component}
        query = {k: v for k, v in query.items() if v}
        query.update({"pending": 1} if pending else {}, **({"mine_approvable": 1} if mine_approvable else {}))
        out = self.call("GET", LIST_PATH, query=query)
        return out.get("references", out) if isinstance(out, dict) else out

    def show(self, ident):
        return self.call("GET", f"{LIST_PATH}/{_q(ident)}")


HTTP_HINTS = {
    401: "the token was rejected",
    403: "not allowed",
    404: "not found, or hidden from this identity",
    503: "the cluster could not verify the token; retry",
}


def _error_text(body):
    try:
        doc = json.loads(body)
    except ValueError:
        return body.strip()[:300]
    if not isinstance(doc, dict):
        return str(body).strip()[:300]
    text = str(doc.get("error") or doc.get("message") or body)
    if doc.get("missing_cases"):
        text += f" (missing cases: {doc['missing_cases'][:5]}; re-run with --register-missing)"
    return text


class FakeServer:
    """In-memory approval service: uploads, bindings, group members, admins and registered cases."""

    def __init__(self, bindings=(), members=None, admins=(), registered=None):
        self.bindings = list(bindings)
        self.members = dict(members or {})
        self.admins = {reference.norm_login(a) for a in admins}
        self.registered = None if registered is None else set(registered)
        self.uploads = {}
        self.tags = []
        self.seq = 0

    @property
    def current(self):
        """key -> upload id: the newest of the retire rows and the current rows of published uploads."""
        out = {}
        for kind, key, uid in self.tags:
            if kind == "retired":
                out[key] = None
            elif self.uploads[uid]["state"] == "published":
                out[key] = uid
        return {k: v for k, v in out.items() if v}

    def client(self, user):
        return FakeClient(self, user)


class FakeClient:
    """The HttpClient methods against a FakeServer, acting as one user."""

    def __init__(self, server=None, user="submitter"):
        self.server = server or FakeServer()
        self.user = reference.norm_login(user)

    def _upload(self, upload_id):
        try:
            return self.server.uploads[upload_id]
        except KeyError:
            raise Refused(f"no upload {upload_id!r}") from None

    def _is_admin(self):
        return self.user in self.server.admins

    def _approvers(self, up):
        return reference.holders(
            self.server.bindings, reference.PERMISSION_APPROVE, reference.facts_of(up["doc"]), self.server.members
        )

    def _missing(self, doc):
        ids = [bid for _, bid in reference.case_ids(doc)]
        if self.server.registered is None:
            return []
        return [i for i in ids if i not in self.server.registered]

    def submit(self, bundle, doc=None, dry_run=False, inspect=False):
        doc = doc or reference.bundle_doc(bundle)
        problems = reference.validate(doc, label_deny=reference.denylist_of())
        if problems:
            raise Refused("; ".join(problems))
        if dry_run or inspect:
            return {
                "state": "dry_run",
                "upload_id": None,
                "report": {"key": doc["key"], "cases": len(reference.case_ids(doc))},
            }
        s = self.server
        s.seq += 1
        uid = f"up-{s.seq:04d}"
        s.uploads[uid] = {
            "upload_id": uid, "state": "pending", "key": doc["key"], "submitter": self.user, "doc": doc,
            "version": None, "digest": reference.content_digest(doc), "props": {},
        }  # fmt: skip
        return {"upload_id": uid, "state": "pending", "key": doc["key"], "report": {"requires_approval": True}}

    def approve(self, upload_id, reason="", terms_ack=False, promote=True, register_missing=False, ack_not_clean=False):
        up = self._upload(upload_id)
        if up["state"] == "published" and up.get("approved_by"):
            if not terms_ack:
                raise Refused("terms_ack is required")
            again = reference.decide_approval(
                submitter=up["submitter"], approver=self.user, is_admin=self._is_admin(),
                approvers=self._approvers(up), reason=reason,
            )  # fmt: skip
            if not again.allowed:
                raise Refused(again.error)
            return {
                "status": "duplicate",
                "state": "published",
                "version": up["version"],
                "current": self.server.current.get(up["key"]) == upload_id,
            }
        if up["state"] != "pending":
            raise Refused(f"cannot approve an upload that is {up['state']!r}")
        if not terms_ack:
            raise Refused("terms_ack is required")
        verdict = reference.decide_approval(
            submitter=up["submitter"], approver=self.user, is_admin=self._is_admin(),
            approvers=self._approvers(up), reason=reason,
        )  # fmt: skip
        if not verdict.allowed:
            raise Refused(verdict.error)
        twin = next(
            (u for u in self.server.uploads.values() if u is not up and u["version"] and u["digest"] == up["digest"]),
            None,
        )
        if twin:
            raise Refused(f"duplicate content: {twin['upload_id']} holds this content, retracted or not")
        if up["doc"].get("clean") is not True and not ack_not_clean:
            raise Refused("acknowledge its warnings (ack_not_clean): the reference is not known to be clean")
        absent = self._missing(up["doc"])
        if absent and not register_missing:
            raise Refused(f"{len(absent)} case(s) are not registered: {absent[:5]}; approve with register_missing")
        if absent:
            self.server.registered |= set(absent)
        s = self.server
        up["version"] = 1 + sum(1 for u in s.uploads.values() if u["key"] == up["key"] and u["version"])
        up["state"], up["approved_by"], up["props"] = "published", self.user, verdict.props
        if promote:
            s.tags.append(("current", up["key"], up["upload_id"]))
        return {
            "status": "ingested", "state": "published", "version": up["version"], "current": promote,
            "registered_now": absent,
            "approval_override": verdict.override, "approval_self": verdict.self_approved,
        }  # fmt: skip

    def reject(self, upload_id, reason="", request_changes=False):
        if not reason.strip():
            raise Refused("a reason is required")
        up = self._upload(upload_id)
        if up["state"] != "pending":
            raise Refused(f"cannot reject an upload that is {up['state']!r}")
        verdict = reference.decide_reject(
            submitter=up["submitter"], actor=self.user, is_admin=self._is_admin(), approvers=self._approvers(up)
        )  # fmt: skip
        if not verdict.allowed:
            raise Refused(verdict.error)
        outcome = "changes_requested" if request_changes else "rejected"
        up["state"], up["props"] = "rejected", {"review_outcome": outcome, "reason": reason}
        return {"state": "rejected", "review_outcome": outcome}

    def retract(self, upload_id, reason=""):
        up = self._upload(upload_id)
        verdict = reference.decide_retract(
            submitter=up["submitter"], actor=self.user, is_admin=self._is_admin(), state=up["state"]
        )  # fmt: skip
        if not verdict.allowed:
            raise Refused(verdict.error)
        was_current = self.server.current.get(up["key"]) == upload_id
        up["state"] = "retracted"  # no tag is written: the current version is read from the tags that remain
        after = self.server.current.get(up["key"])
        return {"state": "retracted", "current_after": after, "retired": was_current and after is None}

    def _guard_admin(self):
        if not self._is_admin():
            raise Refused("only an administrator can retire or roll back a reference")

    def retire(self, key):
        self._guard_admin()
        self.server.tags.append(("retired", key, ""))
        return {"key": key, "state": "retired"}

    def rollback(self, key, to_artifact_id):
        self._guard_admin()
        up = self._upload(to_artifact_id)
        if up["key"] != key or up["state"] != "published":
            raise Refused(f"{to_artifact_id} is not a published version of {key}")
        self.server.tags.append(("current", key, to_artifact_id))
        return {"key": key, "current": to_artifact_id}

    def list(self, state=None, klass=None, origin=None, component=None, pending=False, mine_approvable=False):
        rows = []
        for up in self.server.uploads.values():
            doc = up["doc"]
            if (
                (state and up["state"] != state)
                or (klass and doc["class"] != klass)
                or (origin and doc["origin"] != origin)
                or (component and component not in reference.facts_of(doc)["component"])
                or (pending and up["state"] != "pending")
                or (
                    mine_approvable
                    and (up["state"] != "pending" or self.user not in self._approvers(up) - {up["submitter"]})
                )
            ):
                continue
            rows.append({k: up[k] for k in ("upload_id", "state", "key", "version", "submitter")})
        return rows

    def show(self, ident):
        up = self._upload(ident)
        approvers = self._approvers(up) - {up["submitter"]}
        return {
            "upload_id": up["upload_id"], "state": up["state"], "key": up["key"], "version": up["version"],
            "submitter": up["submitter"], "digest": up["digest"], "warnings": up["doc"].get("warnings", []),
            "comparability": up["doc"].get("comparability", {}), "missing_cases": self._missing(up["doc"]),
            "eligible_approvers": sorted(approvers),
            "can": {
                "approve": up["state"] == "pending" and reference.decide_approval(
                    submitter=up["submitter"], approver=self.user, is_admin=self._is_admin(), approvers=self._approvers(up)
                ).allowed,
                "retract": reference.decide_retract(
                    submitter=up["submitter"], actor=self.user, is_admin=self._is_admin(), state=up["state"]
                ).allowed,
            },
            "current": self.server.current.get(up["key"]) == up["upload_id"],
        }  # fmt: skip
