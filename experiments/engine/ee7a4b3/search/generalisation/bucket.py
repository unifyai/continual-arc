"""The artifacts bucket from a machine that holds no credential of its own, in the standard library alone (beside
gcs_download.py and gcs_upload.py, whose helpers it uses), so that it runs on a bare machine before anything is set up.

  get       stdin: an hour's storage token of a research account on its first line (continual-arc-baselines'
            scripts/secret.sh --node-token, sent over ssh), then a manifest, one input a line: SOURCE PATH [names].
            SOURCE is an object, or a prefix ending in / whose objects land under PATH by their names below it; PATH
            is under ~ unless absolute. With "names", each object of the prefix becomes an empty file of its name (a
            round's captured games, which the loops count by name alone). A file already there with the object's MD5
            is kept and the rest download side by side, each checked against its MD5. Prints one JSON line and exits
            1 naming every input that is missing or failed.
  session   on the laptop, the token on stdin: OBJECT's resumable upload session, printed. Whoever holds it may
            upload that one object, once, within a week, with no credential (Cloud Storage's resumable uploads), so a
            stage hours long hands its products to the bucket when it ends, long after the token it was given lapsed.
  put       put URL FILE: FILE to the session URL in pieces of 64 MiB, each retried, resumed from what the bucket
            already holds.

Usage: { bash secret.sh --node-token; cat manifest; } | ssh HOST python3 ~/gen/tools/bucket.py get
       bash secret.sh --node-token | python3 bucket.py session generalisation/NAME/train.tgz
       python3 bucket.py put "$URL" products.tgz
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import http.client
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from gcs_download import download, listing
from gcs_upload import API

BUCKET = "ARTIFACTS_BUCKET"
PIECE = 64 << 20   # a multiple of 256 KiB, as resumable uploads require of every piece but the last
WORKERS = 16


def described(token: str, name: str) -> tuple[str, int] | None:
    """An object's MD5 (hex) and size, or None when the bucket has no such object."""
    url = f"{API}/storage/v1/b/{BUCKET}/o/{urllib.parse.quote(name, safe='')}?fields=md5Hash,size"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}),
                                    timeout=60) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    return base64.b64decode(d["md5Hash"]).hex(), int(d["size"])


def planned(token: str, manifest: list[str]) -> tuple[list[tuple[str, str, str]], list[Path], list[str]]:
    """The manifest's downloads (object, MD5, file), the empty files its "names" inputs ask for, and its inputs the
    bucket does not hold. Each input is looked up on its own, side by side: a cell's manifest names hundreds of
    rounds' folders."""
    def one(line: str) -> tuple[list[tuple[str, str, str]], list[Path], list[str]]:
        source, path, *flag = line.split()
        dest = Path(path).expanduser() if path.startswith(("/", "~")) else Path.home() / path
        if not source.endswith("/"):
            found = described(token, source)
            return ([(source, found[0], str(dest))], [], []) if found else ([], [], [source])
        objects = listing(token, BUCKET, source)
        if flag == ["names"]:
            return [], [dest / rel for rel, _, _ in objects], [] if objects else [source]
        return [(source + rel, md5, str(dest / rel)) for rel, md5, _ in objects], [], [] if objects else [source]

    downloads, empty, missing = [], [], []
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for d, e, m in pool.map(one, manifest):
            downloads += d
            empty += e
            missing += m
    return downloads, empty, missing


def get() -> int:
    token = sys.stdin.readline().strip()
    manifest = [line for line in sys.stdin.read().splitlines() if line.strip() and not line.startswith("#")]
    downloads, empty, missing = planned(token, manifest)
    for path in empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    def one(item: tuple[str, str, str]) -> str:
        err = ""
        for attempt in range(3):
            try:
                return download(token, BUCKET, *item)
            except Exception as e:  # noqa: BLE001  (retried, then reported)
                err = f"{type(e).__name__}: {e}"
                time.sleep(5 * (attempt + 1))
        return "failed: " + err

    counts = {"inputs": len(manifest), "files": len(downloads), "downloaded": 0, "same": 0, "names": len(empty),
              "missing": len(missing), "failed": 0}
    failed = []
    with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
        for (name, _, _), outcome in zip(downloads, pool.map(one, downloads)):
            if outcome in ("downloaded", "same"):
                counts[outcome] += 1
            else:
                counts["failed"] += 1
                failed.append(f"{name}: {outcome}")
    print(json.dumps(counts))
    for line in [f"{m}: not in the bucket" for m in missing] + failed[:20]:
        print(line, file=sys.stderr)
    return 1 if missing or failed else 0


def new_session(token: str, name: str) -> str:
    """A resumable upload session for object NAME: its URL, a week's capability to upload that object once."""
    url = f"{API}/upload/storage/v1/b/{BUCKET}/o?uploadType=resumable&name={urllib.parse.quote(name, safe='')}"
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={
        "Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.headers["Location"]


def _put(url: str, body: bytes, content_range: str) -> tuple[int, str]:
    """One request to the session: its status and, on 308, the Range the bucket holds. http.client, since urllib takes
    a 308 for a redirect it refuses to follow for a PUT."""
    u = urllib.parse.urlsplit(url)
    conn = http.client.HTTPSConnection(u.netloc, timeout=600)
    try:
        conn.request("PUT", f"{u.path}?{u.query}", body=body,
                     headers={"Content-Range": content_range, "Content-Length": str(len(body))})
        r = conn.getresponse()
        r.read()
        return r.status, r.getheader("Range") or ""
    finally:
        conn.close()


def held(url: str, size: int) -> int | None:
    """How many bytes of the upload the bucket holds, or None once the object is complete."""
    status, rng = _put(url, b"", f"bytes */{size}")
    if status in (200, 201):
        return None
    if status != 308:
        raise RuntimeError(f"the upload session answered {status}: it has lapsed or was used for another file")
    return int(rng.rsplit("-", 1)[1]) + 1 if rng else 0


def put(url: str, path: str) -> int:
    size = os.path.getsize(path)
    done = held(url, size)
    with open(path, "rb") as fp:
        while done is not None:
            fp.seek(done)
            body = fp.read(PIECE)
            end = done + len(body) - 1
            for attempt in range(5):
                try:
                    status, _ = _put(url, body, f"bytes {done}-{end}/{size}" if body else f"bytes */{size}")
                    if status not in (200, 201, 308):
                        raise RuntimeError(f"the upload session answered {status}")
                    break
                except (OSError, http.client.HTTPException, RuntimeError):
                    if attempt == 4:
                        raise
                    time.sleep(10 * (attempt + 1))
            done = held(url, size)
    print(json.dumps({"uploaded": path, "bytes": size}))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("get")
    s = sub.add_parser("session")
    s.add_argument("object")
    u = sub.add_parser("put")
    u.add_argument("url")
    u.add_argument("file")
    a = p.parse_args()
    if a.cmd == "get":
        return get()
    if a.cmd == "session":
        print(new_session(sys.stdin.readline().strip(), a.object))
        return 0
    return put(a.url, a.file)


if __name__ == "__main__":
    sys.exit(main())
