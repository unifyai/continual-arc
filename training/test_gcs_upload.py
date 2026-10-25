"""gcs_upload.py against a fake bucket (the JSON API's list, multipart upload and rewrite, served on localhost, with no
metadata PATCH, which the read-write token a node is sent may not make): the first run uploads every file with the
version it came from; a rerun with nothing changed sends nothing and records nothing; a changed file alone is sent
again; an object an older uploader left without a version, or a file rewritten with the same bytes, is checked by MD5
and given its version by a rewrite onto itself, never sent again; the pruned folders stay out unless --keep names one; a
destination never reads its neighbour's objects (dagger-u1 against dagger-u10); a file gone before it is read counts as
gone; an upload the bucket stores wrong fails the run; a file past WHOLE is sent a block at a time; and with --witness,
a destination holding an earlier run (its progress not where this run's starts) is moved aside whole, the witness last,
while one holding this run's earlier progress is left in place."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

import gcs_upload

BUCKET = "b"


class Bucket:
    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        self.calls: list[str] = []
        self.corrupt = False
        self.page = 2  # objects a listing page holds, small so that paging is exercised

    def put(self, name: str, data: bytes, src: str | None = None) -> None:
        self.objects[name] = {"data": data, "metadata": {"src": src} if src else {}}


def serve(bucket: Bucket):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def reply(self, doc: dict) -> None:
            body = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def body(self) -> bytes:
            return self.rfile.read(int(self.headers["Content-Length"]))

        def do_GET(self) -> None:
            url = urllib.parse.urlparse(self.path)
            q = urllib.parse.parse_qs(url.query)
            if "/o/" in url.path:  # one object's bytes
                name = urllib.parse.unquote(url.path.split("/o/", 1)[1])
                bucket.calls.append("fetch " + name)
                data = bucket.objects[name]["data"]
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            bucket.calls.append("list")
            names = sorted(n for n in bucket.objects if n.startswith(q["prefix"][0]))
            start = int(q.get("pageToken", ["0"])[0])
            items = [{"name": n, "md5Hash": base64.b64encode(hashlib.md5(bucket.objects[n]["data"]).digest()).decode(),
                      **({"metadata": bucket.objects[n]["metadata"]} if bucket.objects[n]["metadata"] else {})}
                     for n in names[start:start + bucket.page]]
            more = start + bucket.page < len(names)
            self.reply({"items": items, **({"nextPageToken": str(start + bucket.page)} if more else {})})

        def do_POST(self) -> None:
            url = urllib.parse.urlparse(self.path)
            if "/rewriteTo/" in url.path:  # a copy within the bucket
                src, dst = (urllib.parse.unquote(part.split("/o/", 1)[1]) for part in url.path.split("/rewriteTo/"))
                resource = json.loads(self.body())
                bucket.calls.append(f"copy {src} {dst}")
                bucket.objects[dst] = {**bucket.objects[src],
                                       **({"metadata": resource["metadata"]} if "metadata" in resource else {})}
                self.reply({"done": True})
                return
            boundary = self.headers["Content-Type"].split("boundary=")[1]
            parts = self.body().split(f"--{boundary}".encode())
            meta = json.loads(parts[1].split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n"))
            data = parts[2].split(b"\r\n\r\n", 1)[1][:-2]
            bucket.calls.append("upload " + meta["name"])
            bucket.put(meta["name"], data, meta.get("metadata", {}).get("src"))
            stored = b"not what was sent" if bucket.corrupt else data
            self.reply({"md5Hash": base64.b64encode(hashlib.md5(stored).digest()).decode()})

        def do_DELETE(self) -> None:
            name = urllib.parse.unquote(urllib.parse.urlparse(self.path).path.split("/o/", 1)[1])
            bucket.calls.append("delete " + name)
            del bucket.objects[name]
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def bucket(monkeypatch):
    b = Bucket()
    server = serve(b)
    monkeypatch.setattr(gcs_upload, "API", f"http://127.0.0.1:{server.server_port}")
    yield b
    server.shutdown()


def run(monkeypatch, capsys, src: Path, dest: str, *extra: str) -> tuple[int, dict]:
    monkeypatch.setattr(sys, "argv", ["gcs_upload.py", "--src", str(src), "--bucket", BUCKET, "--dest", dest,
                                      "--workers", "4", *extra])
    monkeypatch.setattr(sys, "stdin", io.StringIO("token\n"))
    with pytest.raises(SystemExit) as done:
        gcs_upload.main()
    return done.value.code, json.loads(capsys.readouterr().out.strip().splitlines()[-1])


@pytest.fixture
def run_dir(tmp_path):
    r = tmp_path / "dagger-u1"
    (r / "play/rounds/r0000/captured").mkdir(parents=True)
    (r / "play/records.jsonl").write_text('{"round": 0}\n')
    (r / "play/rounds/r0000/captured/u000.json.gz").write_bytes(b"call" * 1000)
    (r / "train/p0000/adapter").mkdir(parents=True)
    (r / "train/p0000/adapter/optimizer.pt").write_bytes(b"\0" * 5000)
    (r / "train/p0000/adapter.vllm").mkdir()
    (r / "train/p0000/adapter.vllm/model.bin").write_bytes(b"served copy")
    (r / "train/p0001/checkpoints/c0001").mkdir(parents=True)   # a phase in flight's step checkpoint
    (r / "train/p0001/checkpoints/c0001/optimizer.pt").write_bytes(b"\0" * 7000)
    (r / "train/p0001/checkpoints/state.json").write_text("{}")
    state = r / "play/users/u000/system/state"
    (state / "session-artifacts/s1").mkdir(parents=True)
    (state / "session-artifacts/s1/kernel-state.json").write_text("{}")
    (state / "agent/kernel-venv").mkdir(parents=True)
    (state / "agent/kernel-venv/python").write_text("venv")
    return r


def test_first_run_uploads_every_file_with_its_version(bucket, monkeypatch, capsys, run_dir):
    code, counts = run(monkeypatch, capsys, run_dir, "results/runs/gpu-nodes/n/dagger-u1")
    assert code == 0
    assert counts == {"files": 3, "uploaded": 3, "same": 0, "gone": 0, "failed": 0, "bytes": 13 + 4000 + 5000,
                      "aside": None}
    obj = bucket.objects["results/runs/gpu-nodes/n/dagger-u1/train/p0000/adapter/optimizer.pt"]
    assert obj["data"] == b"\0" * 5000
    assert obj["metadata"]["src"] == gcs_upload.version((run_dir / "train/p0000/adapter/optimizer.pt").stat())
    assert not [n for n in bucket.objects if "adapter.vllm" in n or "kernel-venv" in n or "session-artifacts" in n
                or "checkpoints" in n]


def test_a_rerun_with_nothing_changed_sends_and_records_nothing(bucket, monkeypatch, capsys, run_dir):
    run(monkeypatch, capsys, run_dir, "d")
    bucket.calls.clear()
    code, counts = run(monkeypatch, capsys, run_dir, "d")
    assert code == 0 and counts["same"] == 3 and counts["uploaded"] == 0
    assert all(c == "list" for c in bucket.calls)


def test_only_a_changed_file_is_sent_again(bucket, monkeypatch, capsys, run_dir):
    run(monkeypatch, capsys, run_dir, "d")
    bucket.calls.clear()
    with open(run_dir / "play/records.jsonl", "a") as fp:
        fp.write('{"round": 1}\n')
    code, counts = run(monkeypatch, capsys, run_dir, "d")
    assert code == 0 and counts["uploaded"] == 1 and counts["same"] == 2
    assert [c for c in bucket.calls if c != "list"] == ["upload d/play/records.jsonl"]
    assert bucket.objects["d/play/records.jsonl"]["data"].count(b"round") == 2


def test_an_object_with_no_version_is_checked_by_md5_once_and_given_it(bucket, monkeypatch, capsys, run_dir):
    for rel in ("play/records.jsonl", "play/rounds/r0000/captured/u000.json.gz", "train/p0000/adapter/optimizer.pt"):
        bucket.put(f"d/{rel}", (run_dir / rel).read_bytes())  # as the uploader before versions left them
    code, counts = run(monkeypatch, capsys, run_dir, "d")
    assert code == 0 and counts["same"] == 3 and counts["uploaded"] == 0
    assert sorted(c for c in bucket.calls if c != "list") == sorted(f"copy d/{rel} d/{rel}" for rel in (
        "play/records.jsonl", "play/rounds/r0000/captured/u000.json.gz", "train/p0000/adapter/optimizer.pt"))
    assert bucket.objects["d/play/records.jsonl"]["metadata"]["src"] == gcs_upload.version(
        (run_dir / "play/records.jsonl").stat())
    bucket.calls.clear()
    run(monkeypatch, capsys, run_dir, "d")
    assert all(c == "list" for c in bucket.calls)


def test_a_file_rewritten_with_the_same_bytes_is_recorded_in_the_bucket_never_sent(bucket, monkeypatch, capsys,
                                                                                  run_dir):
    """A run rewrites some files with the bytes they held (a relaunch's config.json, a one-byte journal): the hourly
    backup finds a new version, reads the file, and has the object take the version by a rewrite onto itself, which the
    read-write token a node holds may make, as it may not a metadata PATCH."""
    run(monkeypatch, capsys, run_dir, "d")
    bucket.calls.clear()
    path = run_dir / "play/records.jsonl"
    path.write_text(path.read_text())
    code, counts = run(monkeypatch, capsys, run_dir, "d")
    assert code == 0 and counts["same"] == 3 and counts["uploaded"] == 0 and counts["failed"] == 0
    assert [c for c in bucket.calls if c != "list"] == ["copy d/play/records.jsonl d/play/records.jsonl"]
    assert bucket.objects["d/play/records.jsonl"]["metadata"]["src"] == gcs_upload.version(path.stat())


def test_keep_uploads_a_folder_left_out_by_default(bucket, monkeypatch, capsys, run_dir):
    code, counts = run(monkeypatch, capsys, run_dir, "d", "--keep", "session-artifacts")
    assert code == 0 and counts["files"] == 4
    assert "d/play/users/u000/system/state/session-artifacts/s1/kernel-state.json" in bucket.objects
    assert not [n for n in bucket.objects if "kernel-venv" in n]


def test_a_destination_never_reads_its_neighbours_objects(bucket, monkeypatch, capsys, run_dir):
    bucket.put("d10/play/records.jsonl", b"another run's records")
    code, counts = run(monkeypatch, capsys, run_dir, "d1")
    assert code == 0 and counts["uploaded"] == 3
    assert bucket.objects["d10/play/records.jsonl"]["data"] == b"another run's records"


def test_a_file_past_whole_is_sent_a_block_at_a_time_and_checked(bucket, monkeypatch, capsys, run_dir):
    monkeypatch.setattr(gcs_upload, "WHOLE", 1000)  # the optimizer (5,000 bytes) and the captured calls (4,000) stream
    code, counts = run(monkeypatch, capsys, run_dir, "d")
    assert code == 0 and counts["uploaded"] == 3
    assert bucket.objects["d/train/p0000/adapter/optimizer.pt"]["data"] == b"\0" * 5000
    assert bucket.objects["d/play/rounds/r0000/captured/u000.json.gz"]["data"] == b"call" * 1000


def test_a_file_gone_before_it_is_read_counts_as_gone(run_dir):
    assert gcs_upload.upload("token", BUCKET, "d/x", str(run_dir / "pruned-since"), None) == ("gone", 0)


def test_an_upload_the_bucket_stores_wrong_fails_the_run(bucket, monkeypatch, capsys, run_dir):
    bucket.corrupt = True
    monkeypatch.setattr(gcs_upload.time, "sleep", lambda s: None)
    code, counts = run(monkeypatch, capsys, run_dir, "d")
    assert code == 1 and counts["failed"] == 3


def test_an_earlier_run_under_the_same_name_is_moved_aside_whole(bucket, monkeypatch, capsys, run_dir):
    (run_dir / "progress").write_text("10-18 03:47 started\n")
    run(monkeypatch, capsys, run_dir, "d")
    # the run goes on: its progress grows, so the destination holds this very run and nothing moves
    with open(run_dir / "progress", "a") as fp:
        fp.write("10-18 04:00 round 0 done\n")
    bucket.calls.clear()
    code, counts = run(monkeypatch, capsys, run_dir, "d", "--witness", "progress", "--aside", ".replaced-x")
    assert code == 0 and counts["aside"] is None and counts["uploaded"] == 1
    assert not [c for c in bucket.calls if c.startswith(("copy", "delete"))]
    # started over under the same name, with fewer files than before
    (run_dir / "progress").write_text("10-19 01:00 started\n")
    (run_dir / "train/p0000/adapter/optimizer.pt").unlink()
    before = {n: o["data"] for n, o in bucket.objects.items()}
    bucket.calls.clear()
    code, counts = run(monkeypatch, capsys, run_dir, "d", "--witness", "progress", "--aside", ".replaced-x")
    assert code == 0 and counts["aside"] == "d.replaced-x" and counts["uploaded"] == 3
    assert {n.replace("d.replaced-x/", "d/"): o["data"] for n, o in bucket.objects.items()
            if n.startswith("d.replaced-x/")} == before  # the earlier run whole, as it was
    assert "d/train/p0000/adapter/optimizer.pt" not in bucket.objects  # and none of it left among the new run's
    assert bucket.objects["d/progress"]["data"] == b"10-19 01:00 started\n"
    moves = [c for c in bucket.calls if c.startswith("copy")]
    assert moves[-1] == "copy d/progress d.replaced-x/progress"  # the witness last


def test_witness_and_aside_go_together(monkeypatch, capsys, run_dir):
    monkeypatch.setattr(sys, "argv", ["gcs_upload.py", "--src", str(run_dir), "--bucket", BUCKET, "--dest", "d",
                                      "--witness", "progress"])
    with pytest.raises(SystemExit) as done:
        gcs_upload.main()
    assert done.value.code == 2
