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

"""The shared spool tier on an Artifactory generic repository.

Layout: <repo>/<prefix>/<writer>/<token>.jsonl, with a sibling <name>.lock holding the claimer's
owner id and time. The claim is last-writer-wins plus a read-back (see `claim`); it is not
atomic. A lock whose owner never finishes is removed once its lease has run out.

The token and URL never reach a log line; errors name the HTTP status only.
"""

from __future__ import annotations

import calendar
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request

from .spool import CLAIM

LOCK = ".lock"

DEFAULT_PREFIX = "clickhouse-spool"


class ArtifactorySpool:
    def __init__(
        self,
        token: str,
        *,
        base: str,
        repo: str,
        prefix: str = DEFAULT_PREFIX,
        timeout: float = 30.0,
        opener=None,
        clock=time.time,
        sleep=time.sleep,
        settle_s: float = 1.0,
        lease_s: float = 300.0,
    ):
        if not token:
            raise ValueError("an Artifactory token is required")
        if not base or not repo:
            raise ValueError("an Artifactory URL and repository are required")
        self._token = token
        self.base, self.repo, self.prefix = base.rstrip("/"), repo, prefix.strip("/")
        self.timeout, self._open, self.clock = timeout, opener, clock
        self.sleep, self.settle_s, self.lease_s = sleep, settle_s, lease_s

    @classmethod
    def from_env(cls, env=None, **kw):
        env = os.environ if env is None else env
        token = (
            env.get("CH_WRITE_ARTIFACTORY_TOKEN") or env.get("ARTIFACTORY_TOKEN") or ""
        )
        base = env.get("CH_WRITE_ARTIFACTORY_URL", "")
        repo = env.get("CH_WRITE_ARTIFACTORY_REPO", "")
        if not (token and base and repo):
            return None
        return cls(
            token,
            base=base,
            repo=repo,
            prefix=env.get("CH_WRITE_ARTIFACTORY_PREFIX", DEFAULT_PREFIX),
            **kw,
        )

    # -- plumbing ---------------------------------------------------------------

    def _req(self, method, url, data=None, ok=(200, 201, 204)):
        req = urllib.request.Request(url, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self._token}")
        try:
            opener = self._open or urllib.request.urlopen
            with opener(
                req, timeout=self.timeout, context=ssl.create_default_context()
            ) as r:
                body = r.read()
                if r.status not in ok:
                    raise OSError(f"artifactory {method}: HTTP {r.status}")
                return r.status, body
        except urllib.error.HTTPError as e:
            return e.code, b""
        except (urllib.error.URLError, TimeoutError) as e:
            raise OSError(f"artifactory {method}: {type(e).__name__}") from None

    def _path(self, name: str) -> str:
        return f"{self.repo}/{self.prefix}/{urllib.parse.quote(name)}"

    # -- the Shared protocol ------------------------------------------------------

    def put(self, name: str, data: bytes) -> None:
        status, _ = self._req("PUT", f"{self.base}/{self._path(name)}", data)
        if status not in (200, 201):
            raise OSError(f"artifactory PUT: HTTP {status}")

    def list(self) -> list[tuple[str, float]]:
        url = f"{self.base}/api/storage/{self.repo}/{self.prefix}?list&deep=1&listFolders=0"
        status, body = self._req("GET", url)
        if status == 404:
            return []
        if status != 200:
            raise OSError(f"artifactory LIST: HTTP {status}")
        now, out = self.clock(), []
        for f in json.loads(body).get("files", []):
            uri = f["uri"].lstrip("/")
            try:
                ts = calendar.timegm(
                    time.strptime(f["lastModified"][:19], "%Y-%m-%dT%H:%M:%S")
                )
            except (KeyError, ValueError):
                ts = now
            out.append((uri, max(0.0, now - ts)))
        return sorted(out)

    def claim(self, name: str, owner: str) -> str | None:
        """Take a lock on `name`; None when another writer holds it.

        Artifactory has no compare-and-swap. Measured on the live service, MOVE lets several
        parallel claimers win (1 to 4 of 6) and `If-None-Match: *` is ignored (all 6 win). What
        did give one winner every time is last-writer-wins: write your owner id to a lock file,
        wait, read it back, and win only if your id survived. A second claimer that slips past
        still cannot double a row on a Replicated table (same token). On a plain table a double
        claim can resend a refused-before-write batch twice: the owner ids are unique per
        process, which keeps that to a true same-instant race.
        """
        if CLAIM in name or LOCK in name:
            return None
        lock = f"{name}{LOCK}"
        stamp = int(self.clock())
        mine = f"{owner}@{stamp}".encode()
        existing = self._lock_owner(lock)
        if existing is not None and stamp - existing[1] < self.lease_s:
            return None
        self.put(lock, mine)
        self.sleep(self.settle_s)
        got = self._lock_owner(lock)
        if got is None or got[0] != owner:
            return None
        return name

    def get(self, name: str) -> bytes:
        status, body = self._req("GET", f"{self.base}/{self._path(name)}")
        if status != 200:
            raise OSError(f"artifactory GET: HTTP {status}")
        return body

    def delete(self, name: str) -> None:
        self._req("DELETE", f"{self.base}/{self._path(name)}")
        self._req("DELETE", f"{self.base}/{self._path(name + LOCK)}")

    def _lock_owner(self, lock: str):
        status, body = self._req("GET", f"{self.base}/{self._path(lock)}")
        if status != 200:
            return None
        try:
            who, _, ts = body.decode().partition("@")
            return who, int(ts)
        except ValueError:
            return None

    def release(self, name: str) -> None:
        self._req("DELETE", f"{self.base}/{self._path(name + LOCK)}")

    def reap(self, lease_s: float) -> int:
        """Remove locks whose lease ran out (their owner died); the item is claimable again."""
        n = 0
        for name, _age in self.list():
            if not name.endswith(LOCK):
                continue
            got = self._lock_owner(name)
            if got is None or self.clock() - got[1] >= lease_s:
                self._req("DELETE", f"{self.base}/{self._path(name)}")
                n += 1
        return n
