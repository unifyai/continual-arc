"""Download a folder from the bucket straight to this machine, file by file, each download checked against the object's
MD5: gcs_upload.py the other way.

Runs on a GPU node with the system Python (no Google packages): the laptop signs, by sending an hour's token of a
research account on stdin (continual-arc-baselines' secret.sh --node-token), and the node fetches the bytes, so a
play's games never pass through the laptop. A file already here with the object's MD5 is skipped, so a rerun only
fetches what is missing.

Usage: <token> | python3 gcs_download.py --bucket BUCKET --src PREFIX --dest DIR [--exclude NAME ...] [--workers 32]
Prints one JSON line (files, downloaded, same, failed, bytes) and exits 1 if any file failed or none was found.
"""
from __future__ import annotations
import argparse
import base64
import concurrent.futures
import json
import os
import sys
import time
import urllib.parse
import urllib.request

from gcs_upload import API, md5_of


def listing(token: str, bucket: str, prefix: str) -> list[tuple[str, str, int]]:
    """Every object under ``prefix/``: (name relative to it, MD5 hex, size)."""
    prefix = prefix.rstrip("/") + "/"
    out, page = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,md5Hash,size),nextPageToken", "maxResults": "1000"}
        if page:
            q["pageToken"] = page
        url = f"{API}/storage/v1/b/{bucket}/o?{urllib.parse.urlencode(q)}"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}), timeout=60) as r:
            d = json.load(r)
        for it in d.get("items", []):
            if not it["name"].endswith("/"):
                out.append((it["name"][len(prefix):], base64.b64decode(it["md5Hash"]).hex(), int(it["size"])))
        page = d.get("nextPageToken")
        if not page:
            return out


def download(token: str, bucket: str, name: str, md5: str, path: str) -> str:
    """'same' (already here), 'downloaded' (and checked), or raises."""
    if os.path.exists(path) and md5_of(path) == md5:
        return "same"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"{API}/storage/v1/b/{bucket}/o/{urllib.parse.quote(name, safe='')}?alt=media"
    tmp = path + ".part"
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"}), timeout=900) as r, \
            open(tmp, "wb") as fp:
        while block := r.read(1 << 20):
            fp.write(block)
    got = md5_of(tmp)
    if got != md5:
        os.remove(tmp)
        raise RuntimeError(f"{name}: md5 {got} here, {md5} in the bucket")
    os.replace(tmp, path)
    return "downloaded"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bucket", required=True)
    p.add_argument("--src", required=True, help="the object prefix, e.g. results/runs/gpu-nodes/NAME/PLAY")
    p.add_argument("--dest", required=True)
    p.add_argument("--exclude", action="append", default=[], help="a top-level file or folder under --src to leave out")
    p.add_argument("--workers", type=int, default=32)
    a = p.parse_args()
    token = sys.stdin.readline().strip()
    dest = os.path.expanduser(a.dest)
    objs = [o for o in listing(token, a.bucket, a.src) if o[0].split("/")[0] not in set(a.exclude)]
    counts = {"files": len(objs), "downloaded": 0, "same": 0, "failed": 0, "bytes": 0}
    failed = []

    def one(obj: tuple[str, str, int]) -> tuple[tuple[str, str, int], str]:
        rel, md5, _ = obj
        for attempt in range(3):
            try:
                return obj, download(token, a.bucket, f"{a.src.rstrip('/')}/{rel}", md5, os.path.join(dest, rel))
            except Exception as e:  # noqa: BLE001  (retried, then counted as failed)
                err = f"{type(e).__name__}: {e}"
                time.sleep(2 * (attempt + 1))
        return obj, "failed: " + err

    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        for (rel, _, size), outcome in pool.map(one, objs):
            if outcome in ("downloaded", "same"):
                counts[outcome] += 1
                if outcome == "downloaded":
                    counts["bytes"] += size
            else:
                counts["failed"] += 1
                failed.append(f"{rel}: {outcome}")
    print(json.dumps(counts))
    for line in failed[:20]:
        print(line, file=sys.stderr)
    sys.exit(1 if failed or not objs else 0)


if __name__ == "__main__":
    main()
