"""Upload a folder from this machine straight to the bucket, file by file, each upload checked against the file's MD5.

Runs on a GPU node with the system Python (no Google packages): whoever holds the bucket's credentials sends a
short-lived access token on stdin (paper_backup.sh with GCS=1 does, from its TOKEN_CMD), so the node holds no lasting
credential, and the node sends the bytes. The objects under
the destination are listed once, and each object records the version of the file it was uploaded from (``src``: its
size and inode change time, which every write moves and nothing can set back). A file whose version the bucket already
holds is skipped without being read; any other is read, and skipped when its MD5 matches the object's, which then
records its version; the rest are uploaded. So a rerun reads and sends only what changed, whatever the folder's size.
The per-user Python environments and caches a play leaves behind (kernel-venv/, home/, .cache/, ...), the copies a
server was given (live/, adapter.vllm/), a training phase's step checkpoints while it runs (checkpoints/: two or three
saves of 3.7 GB, gone at its end) and each user's session-artifacts/ are left out;
--keep NAME uploads one of those folders after all (paper_backup.sh keeps session-artifacts, prime-agent's own record
of its sessions). A file gone by the time it is read, such as an old adapter a run prunes as it goes, is counted as
gone, not failed. With --witness FILE, a file that only grows while the folder holds the same run (a run's progress),
the destination holds an earlier run when its copy of FILE is not where this one starts, such as a run started over
under the same name: every object there moves first to the destination plus --aside, kept whole, so that two runs never
mix under one prefix.

Usage: <token> | python3 gcs_upload.py --src DIR --bucket BUCKET --dest PREFIX [--exclude NAME ...]
       [--exclude-glob PATTERN ...] [--keep NAME ...] [--witness FILE --aside SUFFIX] [--workers 32]
Prints one JSON line (files, uploaded, same, gone, failed, bytes, aside: where an earlier run went, or null) and exits 1
if any file failed.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fnmatch
import hashlib
import itertools
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

PRUNE = {"kernel-venv", "session-artifacts", "node_modules", ".cache", "__pycache__", "home", "live", "adapter.vllm",
         "checkpoints"}
API = "https://storage.googleapis.com"
# The largest file read whole into memory (upload's docstring says why); bigger ones are sent a block at a time.
WHOLE = 64 << 20


def walk(src: str, exclude: set[str], exclude_globs: "tuple[str, ...]" = (), prune: "set[str]" = PRUNE):
    """Every file under ``src`` (relative paths), the ``prune`` folders and ``exclude`` (top-level names) left out.

    ``exclude_globs`` leaves out any name matching one of them at *any* depth (``exclude`` is top level only), which
    is how a caller skips a big file that sits one level down, e.g. ``opd_scores.json*`` under ``rounds/rNNNN/``."""
    def skipped(name: str) -> bool:
        return any(fnmatch.fnmatch(name, g) for g in exclude_globs)

    for root, dirs, files in os.walk(src):
        rel_root = os.path.relpath(root, src)
        dirs[:] = [d for d in dirs
                   if d not in prune and not (rel_root == "." and d in exclude) and not skipped(d)]
        for f in files:
            if rel_root == "." and f in exclude:
                continue
            if skipped(f):
                continue
            yield os.path.normpath(os.path.join(rel_root, f))


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        for block in iter(lambda: fp.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def version(st: os.stat_result) -> str:
    """A file's version as an object records it: any write changes it, and nothing can put it back."""
    return f"{st.st_size}:{st.st_ctime_ns}"


def listing(token: str, bucket: str, prefix: str) -> dict[str, dict]:
    """Every object under ``prefix``, by name: its MD5 (None for a composite object) and the file version it records."""
    found: dict[str, dict] = {}
    page = None
    while True:
        query = {"prefix": prefix, "fields": "items(name,md5Hash,metadata),nextPageToken", "maxResults": "1000"}
        if page:
            query["pageToken"] = page
        req = urllib.request.Request(f"{API}/storage/v1/b/{bucket}/o?{urllib.parse.urlencode(query)}",
                                     headers={"Authorization": f"Bearer {token}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        for o in d.get("items", []):
            found[o["name"]] = {"md5": base64.b64decode(o["md5Hash"]).hex() if "md5Hash" in o else None,
                                "src": (o.get("metadata") or {}).get("src")}
        page = d.get("nextPageToken")
        if not page:
            return found


def record(token: str, bucket: str, name: str, src: str) -> None:
    """Note on an object already holding a file's bytes the version they came from, by rewriting the object onto itself
    with it, in the bucket, without the bytes: a metadata PATCH needs the devstorage.full_control scope, which the
    devstorage.read_write token a node is sent does not carry (403 "Provided scope(s) are not authorized")."""
    rewrite(token, bucket, name, name, {"metadata": {"src": src}})


def chunks(path: str, size: int):
    """The file's first ``size`` bytes, a block at a time."""
    with open(path, "rb") as fp:
        while size > 0:
            block = fp.read(min(1 << 20, size))
            if not block:
                raise RuntimeError(f"{path} shrank while it was sent")
            size -= len(block)
            yield block


def send(token: str, bucket: str, name: str, parts, size: int, md5: str, src: str | None) -> None:
    """One multipart upload of ``size`` bytes, given as ``parts``, the object's metadata with them, checked against
    ``md5`` once it is in: a file changed since it was hashed fails the check and is tried again."""
    boundary = uuid.uuid4().hex
    meta = {"name": name, **({"metadata": {"src": src}} if src else {})}
    head = (f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n{json.dumps(meta)}\r\n"
            f"--{boundary}\r\nContent-Type: application/octet-stream\r\n\r\n").encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(f"{API}/upload/storage/v1/b/{bucket}/o?uploadType=multipart&fields=md5Hash",
                                 data=itertools.chain((head,), parts, (tail,)), method="POST",
                                 headers={"Authorization": f"Bearer {token}",
                                          "Content-Type": f"multipart/related; boundary={boundary}",
                                          "Content-Length": str(len(head) + size + len(tail))})
    with urllib.request.urlopen(req, timeout=900) as r:
        got = base64.b64decode(json.load(r)["md5Hash"]).hex()
    if got != md5:
        raise RuntimeError(f"{name}: md5 {got} in the bucket, {md5} here")


def fetch(token: str, bucket: str, name: str) -> bytes:
    req = urllib.request.Request(f"{API}/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}?alt=media",
                                 headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def rewrite(token: str, bucket: str, name: str, to: str, resource: dict | None = None) -> None:
    """Copy an object to ``to`` in the same bucket, its metadata and all, or with ``resource``'s (an object resource,
    such as ``{"metadata": {...}}``) in their place, calling again with the rewrite token until the copy is done."""
    url = (f"{API}/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}"
           f"/rewriteTo/b/{bucket}/o/{urllib.parse.quote(to, safe='')}?fields=done,rewriteToken")
    step = ""
    while True:
        req = urllib.request.Request(url + (f"&rewriteToken={step}" if step else ""),
                                     data=json.dumps(resource or {}).encode(), method="POST",
                                     headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
        if d["done"]:
            return
        step = d["rewriteToken"]


def move(token: str, bucket: str, name: str, to: str) -> None:
    """Rewrite an object to ``to`` in the same bucket (a copy within the bucket, metadata and all), then delete it."""
    rewrite(token, bucket, name, to)
    req = urllib.request.Request(f"{API}/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}", method="DELETE",
                                 headers={"Authorization": f"Bearer {token}"})
    urllib.request.urlopen(req, timeout=60).close()


def retried(fn, *args):
    """``fn(*args)``, tried again on a dropped connection or a refusal, then fatal."""
    for attempt in range(3):
        try:
            return fn(*args)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def set_aside(token: str, bucket: str, dest: str, remote: dict, witness: str, aside: str, workers: int) -> None:
    """Move every object under ``dest`` to ``aside``, the witness last, so that a move cut short leaves it to be found
    again and finished."""
    names = sorted(remote, key=lambda n: n == f"{dest}/{witness}")
    others, last = names[:-1], names[-1]
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda n: retried(move, token, bucket, n, aside + n[len(dest):]), others))
    retried(move, token, bucket, last, aside + last[len(dest):])


def upload(token: str, bucket: str, name: str, path: str, remote: dict | None) -> tuple[str, int]:
    """'same' (already there), 'uploaded' (and checked) or 'gone' (no longer on disk), with the bytes sent; or raises.

    A file up to WHOLE is read once and sent from memory, so the bytes sent are the bytes hashed however often it is
    rewritten; a bigger one, an adapter or a round's captured calls, written once, is hashed and then sent a block at a
    time, so that no file is ever held whole: an optimizer's state is gigabytes, and a gateway has a few."""
    try:
        before = os.stat(path)
        if remote and remote["src"] == version(before):
            return "same", 0
        if before.st_size <= WHOLE:
            with open(path, "rb") as fp:
                data = fp.read()
            local, size, parts = hashlib.md5(data).hexdigest(), len(data), (data,)
        else:
            local, size = md5_of(path), before.st_size
            parts = chunks(path, size)
        after = os.stat(path)
        # the version is recorded only for a file that held still while it was read, so one changed meanwhile is
        # read again next time
        src = version(after) if version(after) == version(before) else None
        if remote and remote["md5"] == local:
            if src:
                record(token, bucket, name, src)
            return "same", 0
        send(token, bucket, name, parts, size, local, src)
    except FileNotFoundError:
        return "gone", 0
    return "uploaded", size


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", required=True)
    p.add_argument("--bucket", required=True)
    p.add_argument("--dest", required=True, help="the object prefix, e.g. runs/gpu-nodes/NAME/PLAY")
    p.add_argument("--exclude", action="append", default=[], help="a top-level file or folder of --src to leave out")
    p.add_argument("--exclude-glob", action="append", default=[],
                   help="a name to leave out at any depth, as a glob, e.g. 'opd_scores.json*'")
    p.add_argument("--keep", action="append", default=[], choices=sorted(PRUNE),
                   help="a folder left out by default to upload after all, e.g. session-artifacts")
    p.add_argument("--witness", help="a file of --src that only grows while it holds the same run, such as progress")
    p.add_argument("--aside", help="with --witness: what an earlier run's prefix gets, e.g. .replaced-20261019T0100Z")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    if bool(a.witness) != bool(a.aside):
        p.error("--witness and --aside go together")
    token = sys.stdin.readline().strip()
    src = os.path.expanduser(a.src)
    dest = a.dest.rstrip("/")
    paths = list(walk(src, set(a.exclude), tuple(a.exclude_glob), PRUNE - set(a.keep)))
    remote = retried(listing, token, a.bucket, dest + "/")
    aside = None
    if a.witness and f"{dest}/{a.witness}" in remote:
        theirs = retried(fetch, token, a.bucket, f"{dest}/{a.witness}")
        mine = open(os.path.join(src, a.witness), "rb").read() if os.path.isfile(os.path.join(src, a.witness)) else b""
        if not mine.startswith(theirs):
            aside = dest + a.aside
            set_aside(token, a.bucket, dest, remote, a.witness, aside, a.workers)
            remote = {}
    counts = {"files": len(paths), "uploaded": 0, "same": 0, "gone": 0, "failed": 0, "bytes": 0, "aside": aside}
    failed = []

    def one(rel: str) -> tuple[str, str, int]:
        name = f"{dest}/{rel}"
        for attempt in range(3):
            try:
                return (rel, *upload(token, a.bucket, name, os.path.join(src, rel), remote.get(name)))
            except Exception as e:  # noqa: BLE001  (retried, then counted as failed)
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        return rel, "failed: " + err, 0

    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        for rel, outcome, sent in pool.map(one, paths):
            if outcome in ("uploaded", "same", "gone"):
                counts[outcome] += 1
                counts["bytes"] += sent
            else:
                counts["failed"] += 1
                failed.append(f"{rel}: {outcome}")
    print(json.dumps(counts))
    for line in failed[:20]:
        print(line, file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
