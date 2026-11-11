"""QETE with the dump and QETE with the letter (the author, 8 Nov 2026): two rows that put the teacher's knowledge into
Qwen3.5-9B's context, so that retrieval cannot fail, to test whether in-context learning from the teacher memorizes or
generalizes any better than QETE, which leaves it to the student to find what the teacher kept. Both play on QETE's
rails (glm_harness.py ``--teach``): the guided student replays rounds 100 to 199 from the guided teacher's state, GLM-5.3
Flash's hpl (REF_ROWS_NODE_B/glm-hpl-u1-turns), and plays the frozen tests four times each from its final state. This
module is the teacher's side, what the student is given, made once before any play:

  the dump    the author: "every single note GLM has written, and we just dump it into the system prompt for Qwen, saying,
              Here's everything that has been saved from now. Use any of it that might be useful". The student at
              stream round r + 1 is given :data:`SENTENCE`, a line on how the files are written (:data:`FORMAT`), then
              every file GLM's state held at its commit of round r (:func:`saved`), each under its path, oldest change
              first; the tests' students get round 199's. No selection and no summary: the one change is that each grid
              in a JSON file is written as one line of digits per row (:func:`render`), the same content in about half
              the tokens (:func:`unrender` reads it back).
  the letter  the author: "asking GLM to save what it thinks will be useful for learning". After each round r, GLM-5.3 Flash,
              given the same files (:func:`letter_prompt`: the fixed opening and the files first, the ask last, so that
              each call reads the opening and the files older than the round before's from the provider's cache), writes
              a letter to its student, told to teach it whatever will help most on future tasks, including tasks it has
              never seen; it may not decline, and it may run to the room the student's context leaves it (:data:`ROOM`).
              The student at round r + 1 is given the letter written after round r, exactly as GLM wrote it, and the
              tests' students the letter written after round 199. A letter is written before its student sees the
              instance, so it cannot carry that instance's answer.

What a file is, decided from GLM's state itself (:func:`kept`): the state at each commit holds prime-agent's store
(``agent/harness/harness_state.json``), what GLM wrote in its working folder (``workspace/``: programs, kept pairs,
memos, scratch), one memo it wrote at the state's top (``memory_task-48898d06.json``, from round 72), and the harness's
own machinery: the rest of ``agent/`` (the adapter's models.json, settings.json and auth.json, prime-agent's daemon
journals, logs and session leases, the kernel's Python environment), ``home/`` (uv's cache, prime-agent's supervisor
records and its link to agent/), ``sessions/`` (each conversation's transcript, which is prime-agent's record of the
play, not something GLM saved) and ``session-artifacts/`` (each session's kernel state and copy of the store). A file
is the store, or any file outside those four folders but Python's bytecode caches (``__pycache__``), which the
interpreter writes from the .py beside them. Over rounds 0 to 199 nothing else appears.

Every text is kept at GLM's own paths (its state names itself by glm_harness.SPELT); each student's copy is relocated to
its own folder by baselines' ``frozen.relocate``, exactly as its state is (glm_harness.taught_text), so the paths in it
name the student's own files. The play machine has no tokenizer, so each text's tokens, as prime-agent appends it
("\\n\\n" and the text), are counted here with Qwen3.5-9B's own (:data:`TOKENIZER`) for every user of every play a run
will make (``count``), and a play refuses a text it was not given the count of.

  dumps    --source SRC --out DIR --tokenizer TOK [--first 99 --last 199]: DIR/dump/rNNN.md, DIR/material.jsonl (what
           each round holds) and DIR/dump/sizes.jsonl
  letters  --source SRC --out DIR --tokenizer TOK [--first 99 --last 199] [--par N]: DIR/letter/rNNN.md, every call whole
           in DIR/letters.jsonl and its stream in DIR/calls/, and the letters' sizes in DIR/letter/sizes.jsonl; the key
           is OPENROUTER_API_KEY, and every request field but the messages and the reply cap is the teacher's own, its
           driver's overrides (the model, its reasoning effort and its route)
  count    --texts DIR/KIND --tokenizer TOK --run RUN --name NAME (--first F | --test TEST) [--users N]: each user's
           text as that play writes it (RUN as the play machine spells it), its tokens into DIR/KIND/counts.json

SRC is GLM's play folder (glm_harness.SOURCES), of which this reads ``users/u000/system/state.git`` and
``state-log.jsonl`` and, for the letters, ``driver.json``, the teacher's own request settings: a run's ``source/play``
(glm_harness.py's docstring) holds the same history but no driver.json. TOK is Qwen3.5-9B's tokenizer.json at
:data:`TOKENIZER`'s revision. Run where continual_arc_baselines imports, with ``tokenizers`` installed (the material's
laptop venv, not the play machine's).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import glm_harness

SENTENCE = "Here's everything that has been saved up to now. Use any of it that might be useful."
FORMAT = ('Each file below follows a line "==> path <==". Each grid in them is written here as one line of digits per '
          'row, between "grid HxW [" (H rows of W digits) and "]"; the files themselves hold it as JSON, a list of '
          'rows of integers.')
LETTER_OPENING = ("You have been playing a stream of ARC tasks in prime-agent's harness, under the rules and guidance "
                  "you were given, keeping what you chose to keep. Here is everything you have saved so far. " + FORMAT)
# the window Qwen3.5-9B is served with on the route's providers (OpenRouter's endpoints on 8 Nov 2026: Together and
# DeepInfra, 262,144 each), and the room a text has in it beside QETE's own budget, paper.CONTEXT_WINDOW: the
# compaction point, the headroom for the newest observation and the reply cap (49,152 + 16,384 + 65,536)
SERVED_WINDOW = 262_144
ROOM = SERVED_WINDOW - 131_072
LETTER_ASK = ("Write a letter to your student. Your student is Qwen3.5-9B, a much smaller model than you. It will play "
              "the stream's next tasks in the same harness, under the same rules and guidance, from its own copy of "
              "every file above, at the same paths, and later it will meet tasks it has never seen. Your letter is at "
              "the end of its system prompt in every task it plays, exactly as you write it. Teach it whatever will "
              "help it most on future tasks, including tasks it has never seen. You may not decline, and the letter "
              f"may be as long as you judge useful, up to {ROOM:,} tokens, the room the student's context has for it. "
              "Reply with the letter alone.")
STORE = "agent/harness/harness_state.json"
HARNESS = ("agent/", "home/", "sessions/", "session-artifacts/")
HEAD = re.compile(r"^==> (.+) <==$", re.M)
GRID = re.compile(r"grid (\d+)x(\d+) \[\n((?:[0-9]+\n)*)\]")
TOKENIZER = {"repo": "Qwen/Qwen3.5-9B", "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "file": "tokenizer.json",
             "sha256": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"}
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"
TRIES = 3          # a letter's calls: an empty or refused reply is asked again, at most this often in all
# the seconds between a letter's call beginning its reply and the next round's call starting: Z.AI serves the blocks of a
# prompt it has read to a different prompt only some seconds after reading them and keeps them for less than a minute
# (measured on 8 Nov 2026, the probes in the material's run: a call 3 s after another's read none of their shared
# prefix, one 7 s after read it all but the last 64-token block, and one 45 s after read none), while an identical
# prompt reads its cache minutes later, and a call carrying the ``user`` field read none of five calls before it
LAG = 10
HTTP_TRIES = 6     # a call's attempts on a 429, a 5xx or a dropped connection, a minute apart at most


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


# -- grids ------------------------------------------------------------------------------------------------------------

def is_grid(x: Any) -> bool:
    """A grid: one or more rows of the same length, one or more cells each, every cell an integer 0 to 9."""
    return (isinstance(x, list) and len(x) > 0 and all(isinstance(row, list) and len(row) > 0 for row in x)
            and len({len(row) for row in x}) == 1 and all(type(v) is int and 0 <= v <= 9 for row in x for v in row))


def grids(x: Any) -> int:
    if is_grid(x):
        return 1
    if isinstance(x, list):
        return sum(map(grids, x))
    if isinstance(x, dict):
        return sum(map(grids, x.values()))
    return 0


def render(x: Any) -> str:
    """JSON with every grid written as ``grid HxW [``, a line of digits per row, ``]``. A JSON string never holds a raw
    newline, so the lines of digits are the only ones, and :func:`unrender` reads the text back."""
    if is_grid(x):
        return f"grid {len(x)}x{len(x[0])} [\n" + "".join("".join(map(str, row)) + "\n" for row in x) + "]"
    if isinstance(x, list):
        return "[" + ", ".join(map(render, x)) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {render(v)}" for k, v in x.items()) + "}"
    return json.dumps(x, ensure_ascii=False)


def unrender(text: str) -> Any:
    def back(m: re.Match) -> str:
        rows = m.group(3).splitlines()
        if len(rows) != int(m.group(1)) or any(len(row) != int(m.group(2)) for row in rows):
            raise ValueError(f"a grid that is not {m.group(1)}x{m.group(2)}: {m.group(0)[:80]}")
        return json.dumps([[int(c) for c in row] for row in rows])
    return json.loads(GRID.sub(back, text))


def written(path: str, data: bytes) -> tuple[str, int]:
    """A saved file as the texts write it, and how many grids that rewrote: a JSON file that holds a grid with each
    grid as lines of digits (:func:`render`), any other file as it is stored."""
    text = data.decode("utf-8")
    if path.endswith(".json"):
        try:
            obj = json.loads(text)
        except ValueError:   # a .json file GLM left malformed is written as it is
            return text, 0
        n = grids(obj)
        if n:
            return render(obj), n
    return text, 0


# -- GLM's state ------------------------------------------------------------------------------------------------------

def kept(path: str) -> bool:
    """Whether a path of GLM's state is a file the texts carry (the module's docstring)."""
    if path == STORE:
        return True
    return not path.startswith(HARNESS) and "__pycache__/" not in path and not path.endswith(".pyc")


@dataclass(frozen=True)
class Saved:
    path: str     # relative to the state's root
    blob: str
    since: int    # the round since whose commit this path has held this content
    text: str     # as the texts write it
    grids: int


class Teacher:
    """GLM's run as the seeding reads it (``RUN/source/play``): its state history and each round's commit."""

    def __init__(self, source: Path):
        system = Path(source) / "users" / "u000" / "system"
        self.repo = system / "state.git"
        self.commits = {row["round"]: row["commit"] for row in map(json.loads, glm_harness.state_log(system))}
        self._data: dict[str, bytes] = {}
        self._written: dict[str, tuple[str, int]] = {}

    def git(self, *args: str) -> bytes:
        return subprocess.run(["git", f"--git-dir={self.repo}", *args], check=True, capture_output=True).stdout

    def files(self, round_: int) -> list[tuple[str, str]]:
        """The (path, blob) of every file the texts carry in the commit of ``round_``; refuses anything but a regular
        file among them."""
        out = []
        for entry in self.git("ls-tree", "-r", "-z", "--full-tree", self.commits[round_]).split(b"\0"):
            if not entry:
                continue
            meta, path = entry.decode().split("\t", 1)
            mode, _, blob = meta.split()
            if kept(path):
                if mode not in ("100644", "100755"):
                    raise SystemExit(f"round {round_}: {path} is not a regular file (mode {mode})")
                out.append((path, blob))
        return out

    def data(self, blob: str) -> bytes:
        if blob not in self._data:
            self._data[blob] = self.git("cat-file", "blob", blob)
        return self._data[blob]

    def written(self, path: str, blob: str) -> tuple[str, int]:
        key = f"{path.endswith('.json')}:{blob}"
        if key not in self._written:
            self._written[key] = written(path, self.data(blob))
        return self._written[key]

    def saved(self, first: int, last: int) -> dict[int, list[Saved]]:
        """Each round's files from ``first`` to ``last``, oldest change first (the round whose commit first held a
        file's content, then its path), so that a round's texts begin as the round before's did up to its first file
        changed since."""
        out: dict[int, list[Saved]] = {}
        since: dict[str, tuple[str, int]] = {}
        for r in range(last + 1):
            files = self.files(r)
            since = {path: since[path] if since.get(path, ("", 0))[0] == blob else (blob, r) for path, blob in files}
            if r >= first:
                out[r] = sorted((Saved(path, blob, since[path][1], *self.written(path, blob)) for path, blob in files),
                                key=lambda f: (f.since, f.path))
        return out


def files_text(root: str, files: list[Saved]) -> str:
    """Each file under a line naming its path, ``root`` (the state's folder as GLM's state names it) and its own."""
    for f in files:
        if HEAD.search(f.text):
            raise SystemExit(f"{f.path} holds a line the texts head a file with: it could not be read back")
    return "\n\n".join(f"==> {root}/{f.path} <==\n{f.text}" for f in files)


def files_of(text: str) -> dict[str, str]:
    """The files a text holds, by the path each heads it with, as :func:`files_text` wrote them."""
    heads = list(HEAD.finditer(text))
    return {m.group(1): text[m.end() + 1:heads[i + 1].start() - 2 if i + 1 < len(heads) else len(text)]
            for i, m in enumerate(heads)}


def dump(root: str, files: list[Saved]) -> str:
    return f"{SENTENCE}\n\n{FORMAT}\n\n{files_text(root, files)}"


def letter_prompt(root: str, files: list[Saved]) -> str:
    """The letter's one message: the opening every call shares, then the round's files, oldest change first, then the
    ask, so that a call shares with the round before's everything up to the first file changed since."""
    return f"{LETTER_OPENING}\n\n{files_text(root, files)}\n\n{LETTER_ASK}"


def state_root(source_name: str) -> str:
    """The folder GLM's state names itself by: its driver's system folder's ``state/`` (glm_harness.SPELT)."""
    return glm_harness.spelt(source_name) + "/state"


# -- tokens -----------------------------------------------------------------------------------------------------------

def tokenizer(path: Path) -> Any:
    from tokenizers import Tokenizer
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != TOKENIZER["sha256"]:
        raise SystemExit(f"{path} is not {TOKENIZER['repo']}'s tokenizer.json at {TOKENIZER['revision']} ({digest})")
    return Tokenizer.from_file(str(path))


def appended(tok: Any, texts: list[str]) -> list[int]:
    """Each text's tokens as prime-agent appends it to the system prompt: a blank line, then the text."""
    return [len(e.ids) for e in tok.encode_batch(["\n\n" + t for t in texts], add_special_tokens=False)]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# -- the commands -----------------------------------------------------------------------------------------------------

def letter_body(source: Path) -> dict[str, Any]:
    """A letter's request but its message: the teacher's own request fields, its driver's overrides, with the reply cap
    raised to :data:`ROOM`, the room the student has for the letter, and usage accounting on, so that each response
    carries its cost and its cached tokens."""
    driver = json.loads((Path(source) / "driver.json").read_text())
    return {**driver["overrides"], "max_completion_tokens": ROOM, "usage": {"include": True}, "stream": True,
            "stream_options": {"include_usage": True}}


def cmd_dumps(a: argparse.Namespace) -> None:
    """Each round's dump into ``out/dump/rNNN.md``, refused unless it reads back as the round's files; what each round
    holds into ``out/material.jsonl``; and each dump's size, beside the same files as stored, in
    ``out/dump/sizes.jsonl``."""
    out = Path(a.out)
    (out / "dump").mkdir(parents=True, exist_ok=True)
    teacher, root, tok = Teacher(Path(a.source)), state_root(a.source_name), tokenizer(Path(a.tokenizer))
    saved = teacher.saved(a.first, a.last)
    texts = {r: dump(root, files) for r, files in saved.items()}
    for r, text in texts.items():
        if {path[len(root) + 1:]: t for path, t in files_of(text).items()} != {f.path: f.text for f in saved[r]}:
            raise SystemExit(f"round {r}: the dump does not read back as its files")
        for f in saved[r]:
            if f.grids and unrender(f.text) != json.loads(teacher.data(f.blob)):
                raise SystemExit(f"round {r}: {f.path} as written does not read back as the file")
        (out / "dump" / f"r{r:03d}.md").write_text(text)
    as_stored = {r: dump(root, [Saved(f.path, f.blob, f.since, teacher.data(f.blob).decode(), 0) for f in files])
                 for r, files in saved.items()}
    counts = dict(zip(texts, appended(tok, list(texts.values()))))
    stored = dict(zip(as_stored, appended(tok, list(as_stored.values()))))
    with (out / "material.jsonl").open("w") as fp:
        for r, files in saved.items():
            fp.write(json.dumps({"round": r, "commit": teacher.commits[r], "files": [
                {"path": f.path, "blob": f.blob, "since": f.since, "chars": len(f.text), "grids": f.grids}
                for f in files]}) + "\n")
    with (out / "dump" / "sizes.jsonl").open("w") as fp:
        for r, files in saved.items():
            fp.write(json.dumps({"round": r, "files": len(files), "grids": sum(f.grids for f in files),
                                 "chars": len(texts[r]), "tokens": counts[r], "chars_as_stored": len(as_stored[r]),
                                 "tokens_as_stored": stored[r], "sha256": sha(texts[r])}) + "\n")
    print(json.dumps({"dumps": len(texts), "tokens": {r: counts[r] for r in (a.first, a.last)},
                      "as_stored": {r: stored[r] for r in (a.first, a.last)}}), flush=True)


def stream_call(body: dict[str, Any], key: str, record: dict[str, Any], started: threading.Event,
                sse: Path) -> dict[str, Any]:
    """One streamed call: the reply assembled from its chunks (content, reasoning, finish, usage with cost and cached
    tokens, provider), every line of each attempt's stream kept in ``sse``.N.gz; ``started`` is set once the reply has
    begun, the provider having read the prompt in. A 429, a 5xx, a dropped connection or a stream that ends in an
    error is tried again, at most :data:`HTTP_TRIES` times in all, each attempt in ``record``."""
    import httpx
    for attempt in range(1, HTTP_TRIES + 1):
        reply: dict[str, Any] = {"content": "", "reasoning": "", "finish_reason": None, "native_finish_reason": None,
                                 "refusal": None, "usage": None, "id": None, "provider": None, "model": None}
        error = None
        with gzip.open(sse.with_name(f"{sse.name}.{attempt}.gz"), "wt") as fp:
            try:
                with httpx.stream("POST", OPENROUTER, json=body, timeout=httpx.Timeout(1800.0, connect=60.0),
                                  headers={"Authorization": f"Bearer {key}"}) as response:
                    if response.status_code != 200:
                        error = response.read().decode("utf-8", "replace")[:2000]
                        fp.write(error)
                        if response.status_code != 429 and response.status_code < 500:
                            raise SystemExit(f"OpenRouter answered {response.status_code}: {error[:300]}")
                    else:
                        for line in response.iter_lines():
                            fp.write(line + "\n")
                            if not line.startswith("data: ") or line == "data: [DONE]":
                                continue
                            chunk = json.loads(line[6:])
                            if "error" in chunk:
                                error = json.dumps(chunk["error"])[:2000]
                            reply.update({k: chunk[k] for k in ("id", "provider", "model") if chunk.get(k)})
                            reply["usage"] = chunk.get("usage") or reply["usage"]
                            for choice in chunk.get("choices") or []:
                                delta = choice.get("delta") or {}
                                reply["content"] += delta.get("content") or ""
                                reply["reasoning"] += delta.get("reasoning") or ""
                                if delta.get("refusal"):
                                    reply["refusal"] = (reply["refusal"] or "") + delta["refusal"]
                                if delta.get("content") or delta.get("reasoning"):
                                    started.set()
                                for k in ("finish_reason", "native_finish_reason"):
                                    reply[k] = choice.get(k) or reply[k]
                        if reply["finish_reason"] == "error" and error is None:
                            error = "the stream ended with finish_reason error"
                    status = response.status_code
            except httpx.TransportError as dropped:   # the connection dropped, or the stream broke off
                status, error = None, repr(dropped)[:2000]
        record["attempts"].append({"attempt": attempt, "status": status, "error": error, "at": now()})
        if error is None:
            return reply
        time.sleep(min(60, 15 * attempt))
    raise SystemExit(f"no reply in {HTTP_TRIES} attempts: {record['attempts'][-1]}")


def cmd_letters(a: argparse.Namespace) -> None:
    """Each round's letter (the module's docstring), the rounds chained so that every call reads from the provider's
    cache what it shares with the round before's: round r's call starts :data:`LAG` seconds after round r - 1's reply
    has begun, the provider having read its prompt in, at most ``par`` at once. An empty or refused reply is asked again, at most
    :data:`TRIES` calls in all, each kept; a reply cut at the reply cap is kept as far as it got and marked so, since
    another call would be cut at the same cap; a letter longer than :data:`ROOM` stops everything, since the student
    could not be given it whole. A round whose letter is written already is not asked again."""
    import os
    key = os.environ["OPENROUTER_API_KEY"]
    teacher_body = letter_body(Path(a.source))
    out = Path(a.out)
    for sub in ("letter", "calls"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    teacher, root, tok = Teacher(Path(a.source)), state_root(a.source_name), tokenizer(Path(a.tokenizer))
    saved = teacher.saved(a.first, a.last)
    rounds = [r for r in saved if not (out / "letter" / f"r{r:03d}.md").exists()]
    lock = threading.Lock()

    def write(round_: int, gate: threading.Event) -> dict[str, Any]:
        body = {**teacher_body, "messages": [{"role": "user", "content": letter_prompt(root, saved[round_])}]}
        try:
            for n in range(1, TRIES + 1):
                record = {"round": round_, "try": n, "commit": teacher.commits[round_], "started": now(),
                          "attempts": [], "request": body}
                reply = stream_call(body, key, record, gate, out / "calls" / f"r{round_:03d}-t{n}.sse")
                outcome = ("refused" if reply["refusal"] else "empty" if not reply["content"].strip()
                           else "cut at the reply cap" if reply["finish_reason"] == "length" else "kept")
                record.update(ended=now(), response=reply, outcome=outcome)
                with lock, (out / "letters.jsonl").open("a") as fp:
                    fp.write(json.dumps(record) + "\n")
                print(json.dumps({"round": round_, "try": n, "outcome": outcome, "chars": len(reply["content"]),
                                  "usage": reply["usage"]}), flush=True)
                if outcome in ("kept", "cut at the reply cap"):
                    (out / "letter" / f"r{round_:03d}.md").write_text(reply["content"])
                    return record
            raise SystemExit(f"round {round_}: no letter in {TRIES} calls")
        finally:
            gate.set()   # the next round starts even when this one failed, so that the chain never waits forever

    gates = {r: threading.Event() for r in rounds}
    with ThreadPoolExecutor(a.par) as pool:
        futures = []
        for i, r in enumerate(rounds):
            if i:
                gates[rounds[i - 1]].wait()   # the round before has begun its reply: its prompt is read
                time.sleep(LAG)
            futures.append(pool.submit(write, r, gates[r]))
        records = [f.result() for f in futures]
    letters = {rec["round"]: (out / "letter" / f"r{rec['round']:03d}.md").read_bytes().decode("utf-8")
               for rec in records}
    counts = dict(zip(letters, appended(tok, list(letters.values()))))
    with (out / "letter" / "sizes.jsonl").open("a") as fp:
        for rec in records:
            r, usage = rec["round"], rec["response"]["usage"] or {}
            fp.write(json.dumps({"round": r, "try": rec["try"], "outcome": rec["outcome"], "chars": len(letters[r]),
                                 "tokens": counts[r], "sha256": sha(letters[r]),
                                 "prompt_tokens": usage.get("prompt_tokens"),
                                 "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                                 "completion_tokens": usage.get("completion_tokens"),
                                 "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                                     "reasoning_tokens"),
                                 "cost": usage.get("cost"), "provider": rec["response"]["provider"]}) + "\n")
    over = {r: n for r, n in counts.items() if n > ROOM}
    if over:
        raise SystemExit(f"letters longer than the student's room of {ROOM:,} tokens: {over}")


def cmd_count(a: argparse.Namespace) -> None:
    """Each user's text as the play ``a`` names writes it (glm_harness.taught_text, into a scratch folder in place of
    the user's own), its tokens as prime-agent appends it added to ``texts/counts.json`` by the sha256 of its bytes,
    which is how the play finds them."""
    from continual_arc_baselines import frozen as forks
    texts, tok = Path(a.texts), tokenizer(Path(a.tokenizer))
    path = texts / "counts.json"
    counts = json.loads(path.read_text()) if path.exists() else {"tokenizer": TOKENIZER, "texts": {}}
    play_dir = Path(a.run) / "frozen" / a.name / "play"
    starts = glm_harness.starts_of(a.first, a.test, a.users, wiped=False)
    if None in starts:
        raise SystemExit(f"{a.name}: a user forked from nothing has no text (a taught row plays from GLM's states)")
    with tempfile.TemporaryDirectory() as scratch:
        bodies = [glm_harness.taught_text(texts, start, glm_harness.spelt(a.source), forks.user_system(play_dir, user),
                                          Path(scratch) / f"u{user:03d}").read_bytes()
                  for user, start in enumerate(starts)]
    decoded = [body.decode("utf-8") for body in bodies]
    tokens = appended(tok, decoded)
    for body, text, n in zip(bodies, decoded, tokens):
        counts["texts"][hashlib.sha256(body).hexdigest()] = {"tokens": n, "chars": len(text)}
    path.write_text(json.dumps(counts, indent=1))
    print(json.dumps({"play": str(play_dir), "users": len(bodies), "tokens": [min(tokens), max(tokens)]}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("dumps", cmd_dumps), ("letters", cmd_letters)):
        s = sub.add_parser(name)
        s.add_argument("--source", required=True, help="a run's source/play: GLM's state history and state log")
        s.add_argument("--source-name", default="hpl", choices=glm_harness.SOURCES)
        s.add_argument("--out", required=True)
        s.add_argument("--tokenizer", required=True)
        s.add_argument("--first", type=int, default=99)
        s.add_argument("--last", type=int, default=glm_harness.ROUNDS - 1)
        if name == "letters":
            s.add_argument("--par", type=int, default=24)
        s.set_defaults(fn=fn)
    s = sub.add_parser("count")
    s.add_argument("--texts", required=True, help="a folder of rNNN.md, the texts at GLM's paths, and its counts.json")
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--run", required=True, help="the run's folder as the play machine spells it")
    s.add_argument("--source", default="hpl", choices=glm_harness.SOURCES)
    s.add_argument("--name", required=True, help="the play's name, RUN/frozen/NAME")
    which = s.add_mutually_exclusive_group(required=True)
    which.add_argument("--first", type=int)
    which.add_argument("--test", choices=glm_harness.TESTS)
    s.add_argument("--users", type=int, default=glm_harness.PART)
    s.set_defaults(fn=cmd_count)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
