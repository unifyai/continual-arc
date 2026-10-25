"""The oracle's hint for one moment of a game (the pure-OPSD overfit experiment), and the working rules the trained
rows' oracle labels under (:func:`oracle_rules`).

The teacher is the frozen base model reading, as the opening of its own reasoning, c1, the task's card, and then a
hint for the exact moment it is at. An oracle (GLM-5.3) writes the hint from the card, c1, the game so far as the
player sees it (:func:`game_view`) and a state line counted by code over the game's earlier calls (:class:`GameState`),
so the counts survive prime-agent's compaction. The hint is written before the reply, from the call's request and the
game's earlier calls only: the teacher never reads the reply it scores, so this is plain on-policy distillation, not
DAgger. The same :class:`Oracle` makes the hints for the teacher's own plays (the gate) and for the student's training
calls, so the teacher measured is the teacher trained against.

Every hint is checked in code (:func:`check_hint`): its length, no program written out, the player's own voice ("I",
never "you"), and no grid that the game has not printed (the student cannot know such a grid, so it must not learn to
type one) or that is not the card's output for the test input. A hint that fails is asked for once more with the
problem named; prime-agent's own calls (its reviews and summaries) get no hint.

The instruction puts c1 first: after "Decide now" or a rejected reply the hint makes no tool call (c1's rule 2), so
the teacher never reads a hint that contradicts c1 (the first sample hints told the player to run code there).

Two places use it, with the same text, so the teacher's ids are the same token for token:

- in play (loop.py play ``--oracle-card``): :func:`hint_proxy`, every user's proxy (one game each) asking the oracle
  before each of the player's calls and showing c1, the card and the hint, as the reply's reasoning prefill (placement
  thought) or as a notes block at the end of the last message (placement message);
- in training (``thoughts``): a student play's calls, each with the card and the hint as its own thought or note,
  which train.py's teacher reads after the run's context (c1), in its reasoning or in the same notes block.

Placement message is the one to use: with c1, the card and the hint in its own reasoning, the teacher's thinking is
already full of code and action lines, and the student, distilled toward it, learned to write code and action lines
in its thinking and to end its turn there (0.02 to 3.5 such replies a game over two training blocks).

The oracle is reached through an OpenAI-compatible endpoint: the relay (expert_relay.py), which holds the key, on the
same machine or behind an SSH tunnel.

DAgger's and OPSD's labeller (realtime.py) uses only :func:`oracle_rules` from here: the working rules its oracle is
told, c1 with or without the program library's procedure (:func:`library_rules`).

Subcommands (with continual_arc, continual_arc_baselines and ask_sd importable and expert_relay.py running:
``python training/oracle_hint.py ...``):
  sample PLAY:FILE:CALL...  the oracle's hint at recorded moments of captured plays (JSON lines), for reading first
  thoughts PLAY...          up to --max-calls of the plays' player calls, picked at random, each with the card and
                            the oracle's hint as its thought or (--placement message) note; the hints in hints.jsonl
  renote SRC                a hint set's thoughts as notes: the same hints, read in the notes block
  report PLAY...            each play's solves (by task), costs and its hints' outcomes, as JSON
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import expert  # noqa: E402
import overfit  # noqa: E402
import render  # noqa: E402

MODEL = "z-ai/glm-5.3"
BASE_URL = "http://localhost:18200/v1"
EFFORT = "low"
MAX_TOKENS = 16_384
TIMEOUT_S = 420.0
MAX_WORDS = 80            # the hint's words outside code blocks (the instruction asks for this; the check allows more)
MAX_WORDS_CHECKED = 120
MAX_CODE_LINES = 20       # a longer code block writes the program out
TOOL_CLIP = 1500          # a code result's characters shown, JSON grid lines always kept whole

HARNESS_NOTES = ("[python-state]", "[python-skills-unavailable]", "[auto-refinement]")
DIGEST = "[harness-digest]"
SUMMARY = "[compaction-summary]"
DEMO_PAIR = re.compile(r"Demo pair (\d+) input \(\d+x\d+\):")
USED = re.compile(r"Wrong attempts used: (\d+)/(\d+)\. Demo requests used: (\d+)/(\d+)\.")
REJECTED = re.compile(r"I could not find a valid action in your reply \(([^)]*)\)")
GRID_JSON = re.compile(r"\[\s*\[[\d,\s\[\]]*\]\s*\]")
CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.S)
HINT = re.compile(r"<hint>(.*?)</hint>", re.S)
PRINTED = "PRINTED_GRID"   # the hint's stand-in for the grid a code result printed, filled in by fill_printed
RETRIES = 2                # an endpoint error (a tunnel resets a connection now and then: 7% of calls)
YOU = re.compile(r"\byou(?:r|rs|rself)?\b", re.I)

INSTRUCTION = """\
You coach a small model that is playing one game of Continual-ARC, a grid-puzzle game. You write one short hint for \
its next reply. The model reads your hint as its own thought, at the start of its reasoning, so write it in the \
first person, as the player thinking.

You get:
- THE CARD: the puzzle's rule and a transform(grid) program. The program is verified: it gives the right output on \
every copy of this puzzle. The player reads the card too.
- THE PLAYING RULES: how the game reads moves, and the code that loads every grid. The player reads these too.
- THE GAME SO FAR: what the player has seen. Grid rows are left out and long code results are shortened. Older \
parts may be replaced by a summary.
- THE STATE: counts made by code over the whole game. Trust it over the summary.

How the game works: the game reads a move only from the last line of a reply's plain text: {"action": \
"request_demos"}, {"action": "submit", "grid": [[...]]} or {"action": "finish"}. Tool calls only run code in an \
ipython sandbox; a tool named submit or request_demos does nothing. A reply that ends without an action line, or ends \
on a plan, is rejected, and a few rejected replies in a row end the game charged in full. Each demo request and each \
wrong submission costs 1.

The right path for this puzzle: request one demo first. Then, in one ipython call, load every grid with the loader \
from the rules, define the card's transform, check it on every demo pair, and print \
json.dumps(transform(G['Test input'])). Then submit exactly the printed grid. After a CORRECT verdict, finish.

The playing rules come first. After a message that says "Decide now" or "could not find a valid action", the reply \
makes no tool call: it ends at once with the submit line, if a code result printed the card transform's output for \
the test input and that grid was not judged wrong; otherwise it ends with {"action": "request_demos"}.

Rules for the hint:
1. Say only this reply's move and what it ends with. Do not plan later moves: the next reply gets its own hint.
2. If the last reply went wrong (rejected, a wrong submission, a failed code run, a tool named submit), say why in \
one short sentence.
3. Name code; do not copy it. The player has the card's transform and the rules' loader. If the player's own earlier \
code is close to right, give only the lines to change, in one short code block. Never write out a whole program.
4. Never write a grid. To submit the grid a code result printed, end with {"action": "submit", "grid": PRINTED_GRID} \
exactly like that: the printed grid is filled in for you.
5. End the hint with what the reply must end with: one ipython call, or the exact action line.
6. Write as the player thinking: "I", "my". Never "you".
7. At most {max_words} words, plus any code lines. No headings, no lists.

Reply with the hint only, between <hint> and </hint>.""".replace("{max_words}", str(MAX_WORDS))


def text_of(message: dict) -> str:
    return render.text_of(message.get("content"))


def newest(messages: list[dict]) -> list[dict]:
    """The messages after the player's last reply: what the game and the harness answered to it."""
    last = max((j for j, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    return messages[last + 1:]


def player_call(messages: list[dict]) -> bool:
    """Whether a call is the player's own move: not prime-agent's /refine pass, review or summary (expert.agent_call).
    Only these calls are counted into a :class:`GameState` and get a hint, in play and in training alike."""
    return expert.agent_call(messages)


# -- what the oracle reads -----------------------------------------------------------------------------------------

def tool_text(text: str) -> str:
    """A code result shortened to TOOL_CLIP characters, every line that holds a JSON grid kept whole (a grid the hint
    may tell the player to submit must reach the oracle exactly)."""
    grids = [line for line in text.splitlines() if GRID_JSON.search(line)]
    rest = "\n".join(line for line in text.splitlines() if line not in grids)
    rest = render.clip(render.squeeze_grids(rest), TOOL_CLIP)
    return "\n".join([rest, *grids]).strip()


def game_view(messages: list[dict]) -> str:
    """The game so far as the oracle reads it: the call's messages without the system prompt, the player's reasoning
    or the harness's boilerplate; the summary that replaced older turns kept; grid rows collapsed in the game's
    messages; code results shortened."""
    out = []
    for m in messages:
        role, text = m.get("role"), text_of(m)
        if role == "system":
            continue
        if role == "user":
            if text.startswith(DIGEST):
                if SUMMARY in text:
                    summary = text.split(SUMMARY, 1)[1].strip()
                    out.append("EARLIER GAME (a summary written when the history was shortened):\n" + summary)
                continue
            if text.startswith(HARNESS_NOTES):
                out.append("HARNESS NOTE: " + render.clip(text, 300))
                continue
            out.append("GAME:\n" + render.squeeze_grids(text).strip())
        elif role == "assistant":
            parts = [text.strip()] if text.strip() else []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                try:
                    code = json.loads(fn.get("arguments") or "{}").get("code")
                except (ValueError, AttributeError):
                    code = None
                body = code if isinstance(code, str) else fn.get("arguments") or ""
                parts.append(f"[tool call: {fn.get('name')}]\n```\n{render.clip(body, 3000)}\n```")
            out.append("PLAYER:\n" + ("\n".join(parts) if parts else "(an empty reply)"))
        elif role == "tool":
            out.append("CODE RESULT:\n" + tool_text(text))
    return "\n\n".join(out)


@dataclass
class GameState:
    """Counts over a game's calls so far, fed each call's messages in order (:meth:`see`), so they hold after
    prime-agent's compaction drops older messages (the scripted expert, reading only the call's messages, asked for a
    demo already received at 38% of calls)."""

    calls: int = 0
    demo_pairs: set[int] = field(default_factory=set)
    used: tuple[int, int, int, int] | None = None   # wrong, wrong cap, demos, demo cap: the game's latest count
    rejected: int = 0
    test_input: list[list[int]] | None = None        # the latest the game showed, kept when compaction drops it

    def see(self, messages: list[dict]) -> None:
        from overfit_expert import latest_test_input
        self.test_input = latest_test_input(messages) or self.test_input
        for m in messages:
            if m.get("role") == "user":
                text = text_of(m)
                self.demo_pairs |= {int(k) for k in DEMO_PAIR.findall(text)}
                found = USED.findall(text)
                if found:
                    self.used = tuple(int(x) for x in found[-1])
        if any(REJECTED.search(text_of(m)) for m in newest(messages)):
            self.rejected += 1
        self.calls += 1

    def line(self, messages: list[dict]) -> str:
        """The state at the call whose messages were the last seen."""
        new = "\n".join(text_of(m) for m in newest(messages) if m.get("role") != "tool")
        pairs = ", ".join(str(k) for k in sorted(self.demo_pairs)) or "none"
        parts = [f"This is reply {self.calls} of the game.", f"Demo pairs received so far: {pairs}."]
        if self.used:
            w, wc, d, dc = self.used
            parts.append(f"Wrong submissions: {w} of {wc}. Demo requests used: {d} of {dc}.")
        parts.append(f"Replies the game rejected so far: {self.rejected}.")
        printed = any(GRID_JSON.search(text_of(m)) for m in messages if m.get("role") == "tool")
        parts.append("A code result above prints a grid as JSON." if printed
                     else "No code result above prints a grid as JSON.")
        news = []
        rej = REJECTED.search(new)
        if rej:
            news.append(f"the game rejected the last reply ({rej[1]})")
        if "Decide now" in new:
            news.append('the game says "Decide now"')
        if rej or "Decide now" in new:
            news[-1] += ", so this reply must make no tool call"
        if "Your submission was INCORRECT" in new:
            news.append("the last submission was judged INCORRECT")
        if "Your submission was CORRECT" in new or "The instance is over" in new:
            news.append("the instance is over, so the only move left is finish")
        if "Demo request REFUSED" in new:
            news.append("the last demo request was refused at the cap")
        if news:
            parts.append("Newest: " + "; ".join(news) + ".")
        return " ".join(parts)


def oracle_messages(card: str, c1: str, messages: list[dict], state: str, problem: str | None = None) -> list[dict]:
    """The oracle's request for one moment: the instruction, then the card, c1, the game so far and the state; with
    ``problem``, the reason the previous hint for this moment was refused."""
    user = (f"THE CARD:\n{card.strip()}\n\nTHE PLAYING RULES:\n{c1.strip()}\n\nTHE GAME SO FAR:\n"
            f"{game_view(messages)}\n\nTHE STATE:\n{state}\n\nWrite the hint for the player's next reply.")
    if problem:
        user += f"\n\nYour previous hint for this moment was refused: {problem}. Write it again without that problem."
    return [{"role": "system", "content": INSTRUCTION}, {"role": "user", "content": user}]


def library_rules(c1: str) -> str:
    """c1, then the paper's program-library rules (``PROGRAM_LIBRARY_RULES``, the HPL ablation's own text) with the
    library in ``programs``: what DAgger's and OPSD's labels are written under by default (:func:`oracle_rules`)."""
    from continual_arc_baselines.systems.protocol import PROGRAM_LIBRARY_RULES
    return c1.strip() + "\n\n" + PROGRAM_LIBRARY_RULES.format(programs="programs")


def oracle_rules(c1: str, rules: str) -> str:
    """The working rules DAgger's and OPSD's oracle labels every call under (realtime.py's labeller, as forward.py
    label's guidelines), by realtime.sh's ORACLE_RULES: ``library``, c1 and the program library's procedure
    (:func:`library_rules`), which the student is never told, so the procedure reaches its weights through the labels
    alone (the nine final runs); or ``c1``, c1 alone, the oracle not told the procedure either, so that the procedure
    is the one difference (the three rebuttal runs: whether training teaches what to keep when nobody is told it)."""
    return library_rules(c1) if rules == "library" else c1.strip()


# -- the hint's checks ---------------------------------------------------------------------------------------------

def printed_grids(messages: list[dict], code_only: bool = False) -> list[list[list[int]]]:
    """Every grid the game has printed: JSON grids in code results, and grids written as rows (the game's own, or a code
    result's); with ``code_only``, only the code results'."""
    out = []
    for m in messages:
        if code_only and m.get("role") != "tool":
            continue
        text = text_of(m)
        if m.get("role") == "tool":
            for found in GRID_JSON.findall(text):
                try:
                    out.append(json.loads(found))
                except ValueError:
                    continue
        lines, run = text.splitlines(), []
        for line in lines + [""]:
            if render.GRID_ROW.match(line.strip()):
                run.append([int(x) for x in line.split()])
            elif run:
                out.append(run)
                run = []
    return out


def card_output(card: str, test_input: list[list[int]] | None) -> list[list[int]] | None:
    """The card's program run on the test input (None when either is missing)."""
    from overfit_expert import run_program
    blocks = CODE_BLOCK.findall(card)
    return run_program(blocks[0], test_input) if blocks and test_input else None


def check_hint(hint: str, messages: list[dict], expected: list[list[int]] | None = None) -> list[str]:
    """The hint's problems (none: it may be used): too long, a program written out, "you" for the player, a grid the
    game has not printed, or one that is not ``expected`` (the card's output for the test input, when known)."""
    problems = []
    blocks = CODE_BLOCK.findall(hint)
    prose = CODE_BLOCK.sub(" ", hint)
    words = len(GRID_JSON.sub(" ", prose).split())   # a grid's cells are no words
    if not hint.strip():
        problems.append("the hint is empty")
    if words > MAX_WORDS_CHECKED:
        problems.append(f"it has {words} words outside code, more than {MAX_WORDS_CHECKED}")
    if any(len(b.strip().splitlines()) > MAX_CODE_LINES for b in blocks):
        problems.append(f"a code block has more than {MAX_CODE_LINES} lines: name the code instead of copying it")
    if YOU.search(re.sub(r'"[^"\n]*"', " ", prose)):
        problems.append('it speaks to the player as "you": write it as the player\'s own thought, with "I" and "my"')
    shown = printed_grids(messages)
    for found in GRID_JSON.findall(hint):
        try:
            grid = json.loads(found)
        except ValueError:
            problems.append("it holds a grid that is not valid JSON")
            continue
        if grid not in shown:
            problems.append(f"it holds a grid that the game has not printed: write {PRINTED} instead of copying a grid")
        elif expected is not None and grid != expected:
            problems.append("it holds a printed grid that is not the card transform's output for the test input")
    return problems


def fill_printed(hint: str, messages: list[dict], expected: list[list[int]] | None) -> tuple[str, list[str]]:
    """The hint with :data:`PRINTED` replaced by the grid a code result printed, exactly (the oracle never copies a
    grid: its copies of 20-30 wide grids dropped cells), and the problem when there is none to fill in: the
    printed grid must be the card's output for the test input."""
    if PRINTED not in hint:
        return hint, []
    if expected is None or expected not in printed_grids(messages, code_only=True):
        return hint, [f"it submits {PRINTED}, but no code result above printed the card transform's output for the "
                      "test input: run the code first, or request a demo"]
    return hint.replace(PRINTED, json.dumps(expected)), []


# -- the oracle ----------------------------------------------------------------------------------------------------

def http_post(base_url: str, timeout_s: float = TIMEOUT_S, retries: int = RETRIES,
              wait_s: float = 2.0) -> Callable[[dict], dict]:
    """A chat-completion poster for ``base_url`` (the relay), retried ``retries`` times on a transport error or a
    server error."""
    import httpx
    client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=timeout_s, write=60.0, pool=30.0))
    url = base_url.rstrip("/") + "/chat/completions"

    def post(body: dict) -> dict:
        for attempt in range(retries + 1):
            try:
                resp = client.post(url, json=body)
                if resp.status_code < 500 or attempt == retries:
                    resp.raise_for_status()
                    return resp.json()
            except httpx.TransportError:
                if attempt == retries:
                    raise
            time.sleep(wait_s * (attempt + 1))
        raise AssertionError("unreachable")
    return post


class Oracle:
    """Hints for moments of one task's games. ``post`` sends a chat-completion body and returns the endpoint's JSON
    (the relay by default; a fake in tests)."""

    def __init__(self, post: Callable[[dict], dict] | None = None, *, model: str = MODEL, effort: str | None = EFFORT,
                 max_tokens: int = MAX_TOKENS) -> None:
        self.post = post or http_post(BASE_URL)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens

    def ask(self, request: list[dict]) -> tuple[str, dict]:
        body: dict[str, Any] = {"model": self.model, "messages": request, "max_tokens": self.max_tokens,
                                "stream": False, "usage": {"include": True}}
        if self.effort:
            body["reasoning"] = {"effort": self.effort}
        answer = self.post(body)
        message = ((answer.get("choices") or [{}])[0]).get("message") or {}
        return message.get("content") or "", answer.get("usage") or {}

    def hint(self, card: str, c1: str, messages: list[dict], game: GameState) -> dict[str, Any]:
        """The hint for the moment after ``messages`` (``game`` has seen the game's calls up to this one), with what
        it cost: asked once more if the checks refuse it. ``hint`` is None for prime-agent's own calls and for a hint
        refused twice."""
        state = game.line(messages)
        if not player_call(messages):
            return {"hint": None, "state": state, "tries": [], "seconds": 0.0, "skipped": "a prime-agent call"}
        record: dict[str, Any] = {"hint": None, "state": state, "tries": [], "seconds": 0.0}
        t0, problem = time.time(), None
        expected = card_output(card, game.test_input)
        for _ in range(2):
            content, usage = self.ask(oracle_messages(card, c1, messages, state, problem))
            found = HINT.search(content)
            raw = (found[1] if found else content).strip()
            hint, problems = fill_printed(raw, messages, expected)
            problems = problems or check_hint(hint, messages, expected)
            record["tries"].append({"hint": hint, "problems": problems, "usage": usage})
            if not problems:
                record["hint"] = hint
                break
            problem = "; ".join(problems)
        record["seconds"] = round(time.time() - t0, 1)
        return record


def teacher_thought(card: str, hint: str | None) -> str:
    """The teacher's per-call thought after the run's context (c1): the card, then the hint for this moment."""
    return card.strip() + ("\n\n" + hint.strip() if hint else "")


def play_note(c1: str, card: str, hint: str | None) -> str:
    """The notes a hinted play shows in placement message, inside loop.context_block at the end of the last message: c1,
    the card and the hint. train.py's teacher reads the same block for a call with :func:`teacher_thought` as its note
    and c1 as the run's context (test_oracle_hint checks the ids)."""
    return c1.strip() + "\n\n" + teacher_thought(card, hint)


def play_prefill(c1: str, card: str, hint: str | None) -> str:
    """The reasoning prefill a hinted play sends: c1, the card and the hint, as loop.thought_text joins a context.
    Tokenised by the server's template it gives the ids train.py's teacher reads for a call with
    :func:`teacher_thought` as its thought and c1 as the run's context (test_oracle_hint checks this)."""
    import loop
    return loop.thought_text(c1.strip() + "\n\n" + teacher_thought(card, hint))


def safe_hint(oracle: Oracle, card: str, c1: str, messages: list[dict], game: GameState) -> dict[str, Any]:
    """:meth:`Oracle.hint`, with an endpoint that fails giving no hint (the teacher then reads c1 and the card) rather
    than failing the call."""
    try:
        return oracle.hint(card, c1, messages, game)
    except Exception as e:  # noqa: BLE001
        return {"hint": None, "state": game.line(messages), "tries": [], "seconds": 0.0,
                "error": f"{type(e).__name__}: {str(e)[:300]}"}


def hint_proxy(base: type, c1: str, card: str, oracle: Oracle, log: Path | None, placement: str = "thought") -> type:
    """The hinted play's proxy class: the collective learner builds one proxy a user, and each user plays one game, so
    each proxy keeps that game's :class:`GameState`. At each of the player's calls it counts the call in, asks the
    oracle for the hint (from the messages as the harness sent them), and shows c1, the card and the hint: with
    ``placement`` thought as the reply's reasoning prefill (:func:`play_prefill`), with message as a notes block at the
    end of the last message (:func:`play_note`); prime-agent's own calls get c1 and the card alone. The seconds spent on the hint are credited back to the turn's budget, and the proxy's tracker
    is touched while the oracle thinks, as a served call's is (the oracle is not the player's time). Every call gets a
    random seed of its own: the loop's vLLM serves unseeded requests from one random stream, so two plays of
    the same games would otherwise start alike. Each hint's record goes to ``log`` (JSON lines)."""
    from continual_arc_baselines.proxy import KEEPALIVE_S
    log_lock = threading.Lock()

    class HintProxy(base):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.game = GameState()
            self._game_lock = threading.Lock()

        def prepare(self, headers, body, path):
            out, body, parsed = super().prepare(headers, body, path)
            if parsed is None or not path.rstrip("/").endswith("/chat/completions") or not parsed.get("messages"):
                return out, body, parsed
            messages, hint = parsed["messages"], None
            if player_call(messages):
                done, t0 = threading.Event(), time.time()

                def keepalive() -> None:
                    while not done.wait(KEEPALIVE_S):
                        self.tracker.touch()

                threading.Thread(target=keepalive, daemon=True, name="hint-keepalive").start()
                try:
                    with self._game_lock:
                        self.game.see(messages)
                        record = safe_hint(oracle, card, c1, messages, self.game)
                        record["call"] = self.game.calls
                finally:
                    done.set()
                self.tracker.credit(time.time() - t0)
                hint = record.get("hint")
                if log is not None:
                    with log_lock, log.open("a") as fp:
                        fp.write(json.dumps({"t": round(time.time(), 1), **record}) + "\n")
            if placement == "message":
                import loop
                parsed = loop.inject(parsed, loop.context_block(play_note(c1, card, hint)))
            else:
                parsed["chat_template_kwargs"] = {**(parsed.get("chat_template_kwargs") or {}),
                                                  "reasoning_prefill": play_prefill(c1, card, hint)}
            parsed.setdefault("seed", random.getrandbits(31))
            return out, json.dumps(parsed).encode(), parsed

    return HintProxy


# -- sample --------------------------------------------------------------------------------------------------------

def cmd_sample(a: argparse.Namespace) -> None:
    c1 = Path(a.c1).read_text()
    oracle = Oracle(http_post(a.base_url), model=a.model, effort=a.effort or None)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for moment in a.moments:
        play, name, k = moment.rsplit(":", 2)
        doc = dict(overfit.read_docs(Path(play)))[name]
        calls = overfit.agent_calls(doc)
        state = GameState()
        for c in calls[: int(k) + 1]:
            state.see(c["request"]["messages"])
        messages = calls[int(k)]["request"]["messages"]
        card = Path(a.cards_dir, f"card-{doc['task_id']}.md").read_text()
        record = oracle.hint(card, c1, messages, state)
        record.update(moment=moment, task=doc["task_id"], calls=len(calls),
                      view_chars=len(game_view(messages)))
        with out.open("a") as fp:
            fp.write(json.dumps(record) + "\n")
        print(json.dumps({k: record[k] for k in ("moment", "task", "seconds", "hint")}), flush=True)


def moments(runs: list[str]) -> tuple[list[tuple[str, str, dict]], list[tuple[int, dict, GameState]]]:
    """The plays' episodes, and each of their player calls with the game's state at it (the episode's index, the call,
    a copy of the state), every episode's agent calls fed to its state in order as the live proxy feeds them."""
    docs, out = [], []
    for run in runs:
        for name, doc in overfit.read_docs(Path(run)):
            game = GameState()
            for c in overfit.agent_calls(doc):
                if player_call(c["request"]["messages"]):
                    game.see(c["request"]["messages"])
                    out.append((len(docs), c, copy.deepcopy(game)))
            docs.append((Path(run).name, name, doc))
    return docs, out


def cmd_thoughts(a: argparse.Namespace) -> None:
    c1 = Path(a.c1).read_text()
    cards = {t: Path(a.cards_dir, f"card-{t}.md").read_text() for t in overfit.TASKS}
    oracle = Oracle(http_post(a.base_url), model=a.model, effort=a.effort or None)
    docs, found = moments(a.runs)
    picked = random.Random(a.seed).sample(found, min(a.max_calls, len(found))) if a.max_calls else found
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def one(m: tuple[int, dict, GameState]) -> dict[str, Any]:
        k, c, game = m
        return safe_hint(oracle, cards[docs[k][2]["task_id"]], c1, c["request"]["messages"], game)

    with ThreadPoolExecutor(a.workers) as pool:
        records = list(pool.map(one, picked))
    chosen: dict[int, dict[int, str]] = collections.defaultdict(dict)   # episode -> id(call) -> its thought
    stats: collections.Counter = collections.Counter(calls=len(found), picked=len(picked))
    with (out / "hints.jsonl").open("w") as fp:
        for (k, c, _), record in zip(picked, records):
            chosen[k][id(c)] = teacher_thought(cards[docs[k][2]["task_id"]], record.get("hint"))
            stats["hinted" if record.get("hint") else "no_hint"] += 1
            stats["oracle_errors"] += int("error" in record)
            fp.write(json.dumps({"run": docs[k][0], "file": docs[k][1], "call": c["call_id"], **record}) + "\n")
    for k, thoughts in chosen.items():
        run, name, doc = docs[k]
        # the thought opens the teacher's reasoning (placement thought); the note joins c1 in its notes (message)
        field = "note" if a.placement == "message" else "thought"
        doc = {**doc, "calls": [{**c, field: thoughts[id(c)]} for c in doc["calls"] if id(c) in thoughts]}
        overfit.write_doc(out, f"u-{run}-{name}", doc)
    (out / "thoughts.json").write_text(json.dumps(dict(stats), indent=1))
    print(json.dumps(dict(stats)))


def cmd_renote(a: argparse.Namespace) -> None:
    """A hint set made for placement thought, as one for placement message: each call's thought becomes its note, so
    train.py's teacher reads the same card and hint after c1 in its notes block."""
    src, out = Path(a.src), Path(a.out)
    n = 0
    for name, doc in overfit.read_docs(src):
        calls = []
        for c in doc["calls"]:
            c = dict(c)
            if "thought" in c:
                c["note"] = c.pop("thought")
                n += 1
            calls.append(c)
        overfit.write_doc(out, name, {**doc, "calls": calls})
    for f in ("hints.jsonl", "thoughts.json"):
        if (src / f).exists():
            (out / f).write_bytes((src / f).read_bytes())
    print(json.dumps({"out": str(out), "notes": n}))


def play_report(run_dir: Path) -> dict[str, Any]:
    """A play's solves (over all, and by task), its costs, and its hints' outcomes (hints.jsonl, when hinted)."""
    recs = [json.loads(line) for line in open(run_dir / "records.jsonl")]
    by_task = collections.Counter(r["task_id"] for r in recs if r["solved"])
    out: dict[str, Any] = {
        "play": run_dir.name, "solved": sum(bool(r["solved"]) for r in recs), "games": len(recs),
        "by_task": {t: by_task.get(t, 0) for t in sorted({r["task_id"] for r in recs})},
        "demo_requests": sum(r["demo_requests"] for r in recs), "wrong_attempts": sum(r["wrong_attempts"] for r in recs),
        "rejected_replies": sum((r.get("resources") or {}).get("system_parse_failures") or 0 for r in recs),
        "budget_stops": __import__("loop").budget_stops(run_dir)}
    log = run_dir / "hints.jsonl"
    if log.exists():
        hints = [json.loads(line) for line in open(log)]
        cost = sum((t.get("usage") or {}).get("cost") or 0 for h in hints for t in h.get("tries") or [])
        out["hints"] = {"calls": len(hints), "hinted": sum(bool(h.get("hint")) for h in hints),
                        "asked_twice": sum(len(h.get("tries") or []) > 1 for h in hints),
                        "refused_twice": sum(len(h.get("tries") or []) == 2 and not h.get("hint") for h in hints),
                        "errors": sum("error" in h for h in hints), "cost_usd": round(cost, 2),
                        "mean_seconds": round(sum(h.get("seconds") or 0 for h in hints) / max(1, len(hints)), 1)}
    return out


def cmd_report(a: argparse.Namespace) -> None:
    rows = [play_report(Path(r)) for r in a.runs]
    print(json.dumps({"plays": rows, "solved": sum(r["solved"] for r in rows), "games": sum(r["games"] for r in rows)}))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def oracle_args(s: argparse.ArgumentParser) -> None:
        s.add_argument("--c1", default=str(HERE / "rules" / "c1.md"))
        s.add_argument("--cards-dir", required=True, help="the tasks' cards, card-TASK.md (overfit.py cards writes them)")
        s.add_argument("--base-url", default=BASE_URL)
        s.add_argument("--model", default=MODEL)
        s.add_argument("--effort", default=EFFORT)

    s = sub.add_parser("sample")
    s.add_argument("moments", nargs="+", help="PLAY_DIR:FILE:CALL, CALL the agent call's index in the game")
    oracle_args(s)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_sample)
    s = sub.add_parser("thoughts")
    s.add_argument("runs", nargs="+")
    oracle_args(s)
    s.add_argument("--max-calls", type=int, default=1600, help="0: every player call (a block trains on 25 x 64)")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=16)
    s.add_argument("--placement", choices=["thought", "message"], default="thought",
                   help="where train.py's teacher reads each call's card and hint: its reasoning, or its notes")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_thoughts)
    s = sub.add_parser("renote", help="a hint set's thoughts as notes (the same text, read in the notes block)")
    s.add_argument("src")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_renote)
    s = sub.add_parser("report")
    s.add_argument("runs", nargs="+")
    s.set_defaults(fn=cmd_report)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
