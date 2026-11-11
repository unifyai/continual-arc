"""The oracle's hint for one moment of a game (9 Oct, the pure-OPSD overfit).

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
the teacher never reads a hint that contradicts c1 (9 Oct: the first sample hints told the player to run code there).

Two places use it, with the same text, so the teacher's ids are the same token for token:

- in play (loop.py play ``--oracle-card``): :func:`hint_proxy`, every user's proxy (one game each) asking the oracle
  before each of the player's calls and showing c1, the card and the hint, as the reply's reasoning prefill (placement
  thought) or as a notes block at the end of the last message (placement message);
- in training (``thoughts``): a student play's calls, each with the card and the hint as its own thought or note,
  which train.py's teacher reads after the run's context (c1), in its reasoning or in the same notes block.

Placement message is the one to use: with c1, the card and the hint in its own reasoning, the teacher's thinking is
already full of code and action lines, and the student, distilled toward it, learned to write code and action lines
in its thinking and to end its turn there (9 Oct, Step 2 blocks 1-2: 0.02 to 3.5 such replies a game).

The oracle is reached through an OpenAI-compatible endpoint: the laptop's relay (expert_relay.py), on the node through
an SSH tunnel, so no key is ever on the node.

The OPSD row (RL-10) is a different use of the same machinery, card-free: ``watch`` hints a round's own games, one
oracle (GLM-5.3-flash, medium effort) call per player call, straight from the node (AGENTS.md's dated exception),
no card, every grid shown in full, and no card-output check (:func:`oracle_messages_no_card`,
:func:`fill_printed_no_card`); the hint is the only thing the student's teacher reads (train.build_items' notes, an
empty run context), never c1 itself.

Subcommands (on the laptop, from the folder that holds this repo (as loop/) and its sibling repos: PYTHONPATH=continual-arc/src:continual-arc-baselines/src:loop
ask-sd/.venv/bin/python loop/oracle_hint.py ..., with expert_relay.py running; ``thoughts``, ``report`` and ``watch``
on the node too, from ~/prime-rl with uv run):
  sample PLAY:FILE:CALL...  the oracle's hint at recorded moments of captured plays (JSON lines), for reading first
  thoughts PLAY...          up to --max-calls of the plays' player calls, picked at random, each with the card and
                            the oracle's hint as its thought or (--placement message) note; the hints in hints.jsonl
  renote SRC                a hint set's thoughts as notes: the same hints, read in the notes block
  report PLAY...            each play's solves (by task), costs and its hints' outcomes, as JSON
  watch ROUND_DIR            the OPSD row's hint step (paper.sh): hints ROUND_DIR's games as they land, card-free,
                            straight from OpenRouter; writes ROUND_DIR/hints.json once every game is done
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import gzip
import hashlib
import json
import os
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
TASK_ID = re.compile(r"Task id: (\S+)")   # continual_arc_baselines.systems.protocol.render_observation's own line
GRID_JSON = re.compile(r"\[\s*\[[\d,\s\[\]]*\]\s*\]")
CODE_BLOCK = re.compile(r"```[^\n]*\n(.*?)```", re.S)
HINT = re.compile(r"<hint>(.*?)</hint>", re.S)
PRINTED = "PRINTED_GRID"   # the hint's stand-in for the grid a code result printed, filled in by fill_printed
RETRIES = 2                # an endpoint error (the tunnel resets a connection now and then: 9 Oct, 7% of calls)
YOU = re.compile(r"\byou(?:r|rs|rself)?\b", re.I)

# the OPSD row's hint step (RL-10): the oracle is GLM-5.3-flash at medium effort (expert.py's own flash setting),
# reached straight from the node (never through the relay), the key bootstrap already writes to the node's own
# continual-arc-baselines/.env (AGENTS.md's dated exception, this row's hint step only; never logged or printed)
FLASH_MODEL = "z-ai/glm-5.3-flash"
FLASH_EFFORT = "medium"
OPENROUTER_URL = "https://openrouter.ai/api/v1"
ENV_KEY_FILE = Path.home() / "continual-arc-baselines" / ".env"
ENV_KEY_NAME = "OPENROUTER_API_KEY"
HINT_WORKERS = 64            # games in flight at once, so up to this many oracle requests at once
NO_HINT_MAX_FRAC = 0.05      # the round fails once more than this share of its player calls got no usable hint
HINT_ROUND_BUDGET_USD = 40.0
HINT_RUN_BUDGET_USD = 500.0

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

# The OPSD row's oracle (RL-10): card-free, so it must work the puzzle's rule out from the demo pairs itself
# rather than read it off a card -- it gets every grid in full (:func:`game_view`'s own ``squeeze=False``) instead
# of squeezed, and no card-output check (:func:`check_hint`'s ``expected`` stays None; :func:`fill_printed_no_card`
# fills PRINTED_GRID with the latest grid a code result printed, nothing to verify it against). It also gets the
# paper's program-library rules beside c1 (continual_arc_baselines.systems.protocol.PROGRAM_LIBRARY_RULES, the HPL
# ablation of 29 Sep), so it can coach the habit into the player's hints -- the player never reads these rules
# itself (its harness is not row HPL): the habit must be learned into its weights, not told to it directly.
INSTRUCTION_NO_CARD = """\
You coach a small model playing one game of Continual-ARC. Write one short hint for its next reply, in the first \
person, as the player thinking ("I", never "you").
You get THE PLAYING RULES (which also describe a program library the player should keep, under THE TASK'S NAME), \
THE GAME SO FAR (every grid in full) and THE STATE (counts made by code). Work out the puzzle's rule from the demo \
pairs yourself.
Rules: say only this reply's move; if the last reply went wrong, say why in one sentence; you may state the rule \
in words and give at most a few lines of code, never a whole program; on the puzzle's first reply, coach checking \
for the task's saved program and running it on the test input before anything else; after a verdict, coach saving \
or fixing the program under the task's own name; end with exactly what the reply must end with; at most \
{max_words} words. Reply between <hint> and </hint>.
Two rules the hint is checked against, and refused if it breaks them:
1. Never write a grid, or any row or list of grid numbers: not the answer you worked out, not a demo grid, not a \
corrected or partial one. Say what changes in words ("each 3 moves down one row") and let the player's own code \
compute the grid. To submit, end with {"action": "submit", "grid": PRINTED_GRID} only when a code result above \
already printed that grid as JSON; otherwise the move is to run the code that prints it.
2. Write only as the player's own thought, with "I" and "my". Never address anyone: no "you", "your" or "yourself".\
""".replace("{max_words}", str(MAX_WORDS))
# the card-free oracle's tries per moment: one more than the card path's, since without the card it is asked to
# work the rule out itself and refused more often (17 Oct, round 1: 5.3% of player calls refused twice)
HINT_TRIES_NO_CARD = 3


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

def tool_text(text: str, squeeze: bool = True) -> str:
    """A code result shortened to TOOL_CLIP characters, every line that holds a JSON grid kept whole (a grid the hint
    may tell the player to submit must reach the oracle exactly). With ``squeeze`` False (the OPSD row's card-free
    oracle), the rest is kept in full too, grid rows it may itself hold left uncollapsed."""
    grids = [line for line in text.splitlines() if GRID_JSON.search(line)]
    rest = "\n".join(line for line in text.splitlines() if line not in grids)
    rest = render.clip(render.squeeze_grids(rest) if squeeze else rest, TOOL_CLIP)
    return "\n".join([rest, *grids]).strip()


def game_view(messages: list[dict], squeeze: bool = True) -> str:
    """The game so far as the oracle reads it: the call's messages without the system prompt, the player's reasoning
    or the harness's boilerplate; the summary that replaced older turns kept; grid rows collapsed in the game's
    messages; code results shortened. With ``squeeze`` False (the OPSD row's card-free oracle: it must work the
    puzzle's rule out from the demo pairs itself, so every grid stays in full), grid rows are never collapsed."""
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
            out.append("GAME:\n" + (render.squeeze_grids(text) if squeeze else text).strip())
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
            out.append("CODE RESULT:\n" + tool_text(text, squeeze))
    return "\n\n".join(out)


@dataclass
class GameState:
    """Counts over a game's calls so far, fed each call's messages in order (:meth:`see`), so they hold after
    prime-agent's compaction drops older messages (8 Oct: the scripted expert, reading only the call's messages, asked
    for a demo already received at 38% of calls)."""

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


def game_task_alias(messages: list[dict]) -> str | None:
    """The task's alias as the player sees it (never the raw task id: the runner's name for the task, by default
    ``task-<8 hex>``, not a codename -- the regex does not care which), read off
    ``continual_arc_baselines.systems.protocol.render_observation``'s own "Task id: <alias>" line, shown on every
    fresh instance's first message, every row alike -- so the OPSD oracle can name the player's program-library
    file the same way the player itself would. ``None`` when the game has no such line: a call built for a test,
    or a late call whose compaction has dropped the instance's first message, where the hint simply cannot name
    the file (acceptable: the player has the same gap)."""
    for m in messages:
        found = TASK_ID.search(text_of(m))
        if found:
            return found[1]
    return None


def library_rules(c1: str) -> str:
    """THE PLAYING RULES the card-free oracle reads: c1, then the paper's program-library rules
    (``PROGRAM_LIBRARY_RULES``, the HPL ablation's own text) with the library in ``programs``. The OPSD row's hints
    and the DAgger row's labels (realtime.py's labeller, as forward.py label's working rules) are written under this
    one text, so the two rows differ only in what the oracle writes: a hint the teacher reads, or the reply itself."""
    from continual_arc_baselines.systems.protocol import PROGRAM_LIBRARY_RULES
    return c1.strip() + "\n\n" + PROGRAM_LIBRARY_RULES.format(programs="programs")


def oracle_messages_no_card(c1: str, messages: list[dict], state: str, problem: str | None = None) -> list[dict]:
    """:func:`oracle_messages`'s card-free counterpart (the OPSD row): no THE CARD section, the game shown with
    every grid in full (:func:`game_view`'s own ``squeeze=False``) instead of squeezed -- the oracle must work the
    puzzle's rule out from the demo pairs itself, not read it off a card -- and, beside c1, the paper's
    program-library rules (``PROGRAM_LIBRARY_RULES``) and the task's own alias (:func:`game_task_alias`), so the
    oracle can coach the habit by its right file name. Only the oracle reads any of this: the player's own harness
    is not row HPL, so the hint is the only way the habit can reach it.

    The relative ``programs`` (row HPL's own ``programs_dir`` gives an absolute ``<workspace>/programs`` "so that
    no harness has to guess its cwd") works only because prime-agent's ipython sandbox is given ``cwd=workspace``;
    that is load-bearing here, not merely convenient."""
    rules = library_rules(c1)
    alias = game_task_alias(messages)
    task_line = f"\n\nTHE TASK'S NAME (for the program library file): {alias}" if alias else ""
    user = (f"THE PLAYING RULES:\n{rules}\n\nTHE GAME SO FAR:\n{game_view(messages, squeeze=False)}\n\n"
            f"THE STATE:\n{state}{task_line}\n\nWrite the hint for the player's next reply.")
    if problem:
        user += f"\n\nYour previous hint for this moment was refused: {problem}. Write it again without that problem."
    return [{"role": "system", "content": INSTRUCTION_NO_CARD}, {"role": "user", "content": user}]


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


def last_printed_json_grid(messages: list[dict]) -> list[list[int]] | None:
    """The last JSON grid a code result printed (:data:`GRID_JSON`, scanned tool message by tool message, in
    order): unlike :func:`printed_grids`, which appends a message's JSON grids before its row-formatted ones
    regardless of their order in the text, this never returns a row-parsed block that happened to share the
    message with a later JSON grid (an input grid the model echoed for inspection, say) -- the one grid
    :func:`fill_printed_no_card` may trust as "the grid this reply should submit". ``None`` when no tool message
    has printed one."""
    grid = None
    for m in messages:
        if m.get("role") != "tool":
            continue
        for found in GRID_JSON.findall(text_of(m)):
            try:
                grid = json.loads(found)
            except ValueError:
                continue
    return grid


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
    grid: 9 Oct, its copies of 20-30 wide grids dropped cells), and the problem when there is none to fill in: the
    printed grid must be the card's output for the test input."""
    if PRINTED not in hint:
        return hint, []
    if expected is None or expected not in printed_grids(messages, code_only=True):
        return hint, [f"it submits {PRINTED}, but no code result above printed the card transform's output for the "
                      "test input: run the code first, or request a demo"]
    return hint.replace(PRINTED, json.dumps(expected)), []


def fill_printed_no_card(hint: str, messages: list[dict]) -> tuple[str, list[str]]:
    """:func:`fill_printed`'s card-free counterpart (the OPSD row): no card to check a printed grid against, so
    :data:`PRINTED` is filled with the last JSON grid a code result printed (:func:`last_printed_json_grid`, never
    a row-parsed block), trusted as the player must trust it too without a card; the problem when no code result
    has printed one yet."""
    if PRINTED not in hint:
        return hint, []
    grid = last_printed_json_grid(messages)
    if grid is None:
        return hint, [f"it submits {PRINTED}, but no code result above printed a grid: run the code first, or "
                      "request a demo"]
    return hint.replace(PRINTED, json.dumps(grid)), []


# -- the oracle ----------------------------------------------------------------------------------------------------

def _poster(client: Any, url: str, retries: int, wait_s: float) -> Callable[[dict], dict]:
    """A chat-completion poster for ``url`` on ``client``, retried ``retries`` times on a transport error or a
    server error; shared by :func:`http_post` (the relay) and :func:`openrouter_post` (the OPSD hint step's
    direct call, AGENTS.md's dated exception)."""
    import httpx

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


def http_post(base_url: str, timeout_s: float = TIMEOUT_S, retries: int = RETRIES,
              wait_s: float = 2.0) -> Callable[[dict], dict]:
    """A chat-completion poster for ``base_url`` (the relay), retried ``retries`` times on a transport error or a
    server error."""
    import httpx
    client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=timeout_s, write=60.0, pool=30.0))
    return _poster(client, base_url.rstrip("/") + "/chat/completions", retries, wait_s)


def read_env_key(path: Path = ENV_KEY_FILE, name: str = ENV_KEY_NAME) -> str:
    """``name``'s value from the node's own ``.env`` (bootstrap's own write, ``continual_arc_baselines/scripts/
    secret.sh --node-env``, mode 600): the OPSD hint step's only use of a key on a node, under AGENTS.md's dated
    exception. Whatever calls this must never log, print or commit the value it returns."""
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    raise SystemExit(f"no {name} in {path}")


def openrouter_post(key_path: Path = ENV_KEY_FILE, timeout_s: float = TIMEOUT_S, retries: int = RETRIES,
                    wait_s: float = 2.0) -> Callable[[dict], dict]:
    """A chat-completion poster straight to OpenRouter (the OPSD hint step only; AGENTS.md's dated exception to "No
    API key ever goes to a node"), the key read once from ``key_path`` and sent as a header, never logged, printed
    or returned by anything here."""
    import httpx
    client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=timeout_s, write=60.0, pool=30.0),
                          headers={"Authorization": f"Bearer {read_env_key(key_path)}"})
    return _poster(client, OPENROUTER_URL + "/chat/completions", retries, wait_s)


class BudgetExceeded(Exception):
    """The OPSD hint step's spending guard (:class:`Budget`) tripped: a round's or a run's own dollar limit."""


def _read_total(path: Path) -> float:
    try:
        return json.loads(path.read_text()).get("total_usd", 0.0)
    except (FileNotFoundError, ValueError):
        return 0.0


def _write_total(path: Path, total: float) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"total_usd": round(total, 4)}))
    tmp.rename(path)


class Budget:
    """The OPSD hint step's OpenRouter spending guard, since its key has no credit cap of its own: this round's own
    total and the run's total across every round so far, each persisted at its own path (``round_total_path``,
    ``run_total_path``) so a resumed round or run keeps what was already spent -- a round that already tripped its
    own cap must not get a fresh allowance on every retry. :attr:`reason` is set the moment either total is over its
    limit, including immediately at construction when a resumed total is already there; :meth:`add` never raises
    (a tripped guard must still record what the in-flight answer that triggered it, or returned just after, really
    cost) -- :func:`cost_tracked` is what refuses a *new* request once :attr:`reason` is set."""

    def __init__(self, round_total_path: Path, run_total_path: Path, round_limit: float, run_limit: float) -> None:
        self.round_total_path, self.run_total_path = round_total_path, run_total_path
        self.round_limit, self.run_limit = round_limit, run_limit
        self.round_total = _read_total(round_total_path)
        self._run_total = _read_total(run_total_path)
        self._lock = threading.Lock()
        self.reason: str | None = None
        self._check()

    def _check(self) -> None:
        if self.round_total > self.round_limit:
            self.reason = (f"this round's hints have spent ${self.round_total:.2f}, over the "
                          f"${self.round_limit:.2f} round budget")
        elif self._run_total > self.run_limit:
            self.reason = (f"this run's hints have spent ${self._run_total:.2f}, over the "
                          f"${self.run_limit:.2f} run budget")

    def add(self, cost: float) -> None:
        with self._lock:
            self.round_total += cost
            self._run_total += cost
            _write_total(self.round_total_path, self.round_total)
            _write_total(self.run_total_path, self._run_total)
            if not self.reason:
                self._check()

    def abandon(self, reason: str) -> None:
        """Sets :attr:`reason` directly, for a stop that is not a spending limit (the OPSD watcher's own
        ``--parent-pid`` check): reuses the one gate :func:`cost_tracked` already enforces before a *new*
        request, rather than adding a second gate beside it that every caller would have to remember to check
        too. A no-op once a reason -- spending or otherwise -- is already set."""
        with self._lock:
            if not self.reason:
                self.reason = reason


def cost_tracked(post: Callable[[dict], dict], budget: Budget) -> Callable[[dict], dict]:
    """``post`` wrapped to add every answer's dollar cost to ``budget`` (:meth:`Budget.add`, which always records
    it even once over) and to refuse a *new* request outright once ``budget`` is already over (so a tripped limit
    costs nothing further)."""
    def wrapped(body: dict) -> dict:
        if budget.reason:
            raise BudgetExceeded(budget.reason)
        answer = post(body)
        budget.add((answer.get("usage") or {}).get("cost") or 0.0)
        return answer
    return wrapped


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

    def hint(self, card: str | None, c1: str, messages: list[dict], game: GameState) -> dict[str, Any]:
        """The hint for the moment after ``messages`` (``game`` has seen the game's calls up to this one), with what
        it cost: asked once more if the checks refuse it. ``hint`` is None for prime-agent's own calls and for a hint
        refused on every try (twice with a card, :data:`HINT_TRIES_NO_CARD` times without). With ``card`` None (the OPSD row), delegates to :meth:`hint_at` -- no THE CARD section, every
        grid shown in full and no card-output check -- the oracle must work the puzzle's rule out from the demo
        pairs itself."""
        state = game.line(messages)
        if not player_call(messages):
            return {"hint": None, "state": state, "tries": [], "seconds": 0.0, "skipped": "a prime-agent call"}
        if card is None:
            return self.hint_at(c1, messages, state)
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

    def hint_at(self, c1: str, messages: list[dict], state: str) -> dict[str, Any]:
        """:meth:`hint`'s card-free body (``card`` always None), for a ``state`` line computed ahead of time
        (:func:`game_moments`): the OPSD hint watcher walks a whole game's :class:`GameState` in one cheap,
        sequential, no-network pass first, so the oracle calls that follow can run concurrently across games
        instead of one game at a time, which :meth:`hint` itself -- needing a live :class:`GameState` to call
        :meth:`GameState.line` on -- cannot do. Assumes the caller has already confirmed this is a player call."""
        record: dict[str, Any] = {"hint": None, "state": state, "tries": [], "seconds": 0.0}
        t0, problem = time.time(), None
        for _ in range(HINT_TRIES_NO_CARD):
            content, usage = self.ask(oracle_messages_no_card(c1, messages, state, problem))
            found = HINT.search(content)
            raw = (found[1] if found else content).strip()
            hint, problems = fill_printed_no_card(raw, messages)
            problems = problems or check_hint(hint, messages)
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


def safe_hint(oracle: Oracle, card: str | None, c1: str, messages: list[dict], game: GameState) -> dict[str, Any]:
    """:meth:`Oracle.hint`, with an endpoint that fails giving no hint (the teacher then reads c1 and the card) rather
    than failing the call."""
    try:
        return oracle.hint(card, c1, messages, game)
    except Exception as e:  # noqa: BLE001
        return {"hint": None, "state": game.line(messages), "tries": [], "seconds": 0.0,
                "error": f"{type(e).__name__}: {str(e)[:300]}"}


def safe_hint_at(oracle: Oracle, c1: str, messages: list[dict], state: str) -> dict[str, Any]:
    """:meth:`Oracle.hint_at`, with an endpoint that fails (including :class:`BudgetExceeded`, once the spending
    guard has tripped) giving no hint rather than failing the call."""
    try:
        return oracle.hint_at(c1, messages, state)
    except Exception as e:  # noqa: BLE001
        return {"hint": None, "state": state, "tries": [], "seconds": 0.0,
                "error": f"{type(e).__name__}: {str(e)[:300]}"}


def hint_proxy(base: type, c1: str, card: str, oracle: Oracle, log: Path | None, placement: str = "thought") -> type:
    """The hinted play's proxy class: the collective learner builds one proxy a user, and each user plays one game, so
    each proxy keeps that game's :class:`GameState`. At each of the player's calls it counts the call in, asks the
    oracle for the hint (from the messages as the harness sent them), and shows c1, the card and the hint: with
    ``placement`` thought as the reply's reasoning prefill (:func:`play_prefill`), with message as a notes block at the
    end of the last message (:func:`play_note`); prime-agent's own calls get c1 and the card alone. The seconds spent on the hint are credited back to the turn's budget, and the proxy's tracker
    is touched while the oracle thinks, as a served call's is (the oracle is not the player's time). Every call gets a
    random seed of its own: the loop's vLLM serves unseeded requests from one random stream (7 Oct), so two plays of
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


# -- OPSD's hint step: a round's games hinted as they land ---------------------------------------------------------

USER_FILE = re.compile(r"^u(\d+)\.json\.gz$")
GAME_LOAD_WORKERS = 8   # games held parsed in memory at once; a captured uNNN.json.gz parses to ~30x its gz size,
                        # so --workers=64 games in memory at once would be tens of GB -- the oracle requests
                        # themselves still run up to --workers at once, across however many games are loaded


def game_progress_path(round_dir: Path, user: int) -> Path:
    """Where one game's hints are cached, keyed by the captured file's own fingerprint (:func:`game_fingerprint`),
    so a crash redoes only the games still unfinished, and a play retry that rewrites a game (a new instance under
    the same ``uNNN.json.gz`` name) redoes exactly that game, never serving its old hints to the new one."""
    return round_dir / "hints" / f"u{user:03d}.json"


def game_moments(doc: dict[str, Any]) -> list[tuple[int, list[dict], str]]:
    """One game's player calls as the oracle needs them: for each, its position in ``doc["calls"]``
    (:func:`train.load_calls`'s own ``i``), its messages, and the :class:`GameState` line it faced -- computed in
    one cheap, sequential, no-network pass (prime-agent's own calls, as :func:`player_call` gates, get none) so
    the oracle calls for every moment this returns can then run concurrently instead of one game, or one call,
    at a time."""
    game = GameState()
    out = []
    for i, c in enumerate(doc["calls"]):
        if not overfit.chat_call(c) or render.is_compaction(c):
            continue
        messages = c["request"]["messages"]
        if not player_call(messages):
            continue
        game.see(messages)
        out.append((i, messages, game.line(messages)))
    return out


def _stat_fingerprint(stat: os.stat_result) -> tuple[int, int]:
    """``stat``'s mtime and size alone, cheap enough to compare on every poll for every game: a caller already
    holding one ``os.stat()`` (from deciding whether this file needs a fresh look at all) reads this and
    :func:`game_fingerprint`'s hash -- the expensive part -- off the very same call, rather than asking the
    filesystem for the same file's metadata twice, a moment apart, for no reason."""
    return stat.st_mtime_ns, stat.st_size


def game_fingerprint(stat: os.stat_result, data: bytes) -> dict[str, Any]:
    """A game's identity from ``stat`` (the caller's own, not asked for again here) and a short hash of ``data``
    (its gzipped bytes, which the caller already read to parse it). A play retry inside a round
    (``continual_arc.runner.run(resume=True)`` replays the whole round for every user) rewrites the same
    ``uNNN.json.gz`` name with a different game; the mtime alone already changes on any such rewrite (an atomic
    ``tmp.rename``, never an in-place edit), and the hash is the belt-and-suspenders check against a coincidence
    (a filesystem with coarse mtime resolution, or two games landing at the same size)."""
    return {"mtime_ns": stat.st_mtime_ns, "size": stat.st_size, "hash": hashlib.sha256(data).hexdigest()[:16]}


def captured_games(round_dir: Path) -> list[tuple[int, Path]]:
    """``round_dir/captured``'s games, as (user, path), in file order."""
    out = []
    for f in sorted(round_dir.glob("captured/u*.json.gz")):
        m = USER_FILE.match(f.name)
        if m:
            out.append((int(m[1]), f))
    return out


def stale_games(hints_doc: dict[str, Any], round_dir: Path) -> list[str]:
    """Why ``hints_doc`` (a ``hints.json``) is no longer trustworthy against ``round_dir/captured``'s games right
    now, if at all: ``hints.json`` means "this round's hints are complete and authoritative", which stops being
    true the moment a game it covered is rewritten (a play retry inside the round) or a game ``captured`` now
    holds was never covered at all (a watcher that stopped early). Checked against every captured game's own
    fingerprint (:func:`game_fingerprint`), not merely its presence. Empty when ``hints_doc`` still covers
    everything ``captured`` holds, exactly as it was when each game was hinted."""
    games = hints_doc.get("games") or {}
    problems = []
    for user, f in captured_games(round_dir):
        recorded = games.get(str(user))
        if recorded is None:
            problems.append(f"user {user} ({f.name}) has no record in hints.json: it covers fewer games than "
                            "round_dir/captured holds")
            continue
        stat = f.stat()
        current = game_fingerprint(stat, f.read_bytes())
        if recorded != current:
            problems.append(f"user {user} ({f.name})'s fingerprint has moved since hints.json was written "
                            "(a play retry rewrote it)")
    return problems


def parent_alive(pid: int | None) -> bool:
    """Whether ``pid`` (paper.sh's own ``$$``, ``--parent-pid``) is still a process: this watcher is launched
    ``setsid``-detached (so a crashed or killed ``paper.sh`` leaves no SIGHUP to catch), so it must poll for its
    parent's death itself rather than trust being told -- the ticket's "no detached poller may outlive the round".
    ``True`` when ``pid`` is None (no parent given, e.g. a direct CLI call or a test)."""
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True   # exists, just not ours to signal: still alive


def cmd_watch(a: argparse.Namespace) -> None:
    """Hint ``round_dir``'s games (the OPSD row): watches ``round_dir/captured/`` for games as play writes them
    (each ``uNNN.json.gz`` lands by an atomic rename, so its mere existence means it is complete) and hints each
    one, up to :data:`GAME_LOAD_WORKERS` parsed in memory at once, their player calls' oracle requests fanned out
    across up to ``--workers`` at once (:func:`game_moments`); caches every finished game's hints under
    ``round_dir/hints/``, fingerprinted (:func:`game_fingerprint`) so a play retry that rewrites a game's captured
    file is re-hinted rather than served stale hints for a different instance. With ``--until``, keeps polling
    past whatever is there now until that marker file exists *and* one more full poll after it finds nothing new
    (so a game whose file lands in the same instant as the marker is never missed); without ``--until``, processes
    whatever is there once and stops. If ``round_dir/hints.json`` already exists, it is only trusted once
    :func:`stale_games` finds nothing wrong with it against ``captured`` right now; a stale one (a play retry
    rewrote a game it covered, or it never covered everything ``captured`` holds) is moved aside and this runs
    as if it had never existed -- redoing, through the same per-game cache, only the games that actually need it.
    Writes a fresh ``hints.json`` (this round's resume marker, now recording every game it covers and each one's
    own fingerprint, so a later reader can tell it apart from a stale one the same way) once every game is done,
    unless more than :data:`NO_HINT_MAX_FRAC` of the round's player calls got no usable hint, or :class:`Budget`
    ran over -- either fails the stage (non-zero exit) without writing it."""
    round_dir = Path(a.round_dir).expanduser()
    out = round_dir / "hints.json"
    if out.exists():
        problems = stale_games(json.loads(out.read_text()), round_dir)
        if not problems:
            print(json.dumps({"skipped": "hints.json already exists and still covers round_dir/captured"}))
            return
        stale_path = out.with_name(f"{out.name}.stale-{int(time.time())}")
        out.rename(stale_path)
        print(json.dumps({"moved_aside": str(stale_path), "problems": problems}))
    c1 = Path(a.c1).read_text()
    (round_dir / "hints").mkdir(parents=True, exist_ok=True)
    run_total_path = Path(a.run_budget_file).expanduser() if a.run_budget_file else round_dir / "hints-cost.json"
    budget = Budget(round_dir / "hints-round-cost.json", run_total_path, a.round_budget_usd, a.run_budget_usd)
    oracle = Oracle(cost_tracked(openrouter_post(), budget), model=a.model, effort=a.effort or None)
    done_marker = Path(a.until).expanduser() if a.until else None
    hint_pool = ThreadPoolExecutor(a.workers)
    t0 = time.time()

    def one(user: int, path: Path, fingerprint: dict[str, Any]) -> dict[int, dict[str, Any]]:
        progress = game_progress_path(round_dir, user)
        if progress.exists():
            cached = json.loads(progress.read_text())
            if cached.get("fingerprint") == fingerprint:
                return {int(k): v for k, v in cached["records"].items()}
        with gzip.open(path, "rt") as fp:
            doc = json.load(fp)
        moments = game_moments(doc)
        del doc
        futures = [(i, hint_pool.submit(safe_hint_at, oracle, c1, messages, state)) for i, messages, state in moments]
        records = {i: fut.result() for i, fut in futures}
        # a budget-cut (or orphaned) game is never cached (a later, un-stopped run redoes it whole); nor is a call
        # whose file has since been rewritten and resubmitted while this one was still in flight -- the newer
        # call's own write must be the one that lands, or a slow stale hinting of a replaced game could overwrite
        # it after
        cheap = (fingerprint["mtime_ns"], fingerprint["size"])
        if not budget.reason and known.get(user) == cheap:
            tmp = progress.with_name(progress.name + ".tmp")
            tmp.write_text(json.dumps({"fingerprint": fingerprint, "records": records}))
            tmp.rename(progress)
        return records

    known: dict[int, tuple[int, int]] = {}   # user -> its last-submitted _stat_fingerprint
    fingerprints: dict[int, dict[str, Any]] = {}   # user -> its last-submitted full fingerprint, for hints.json
    submitted: dict[int, concurrent.futures.Future] = {}
    orphaned = False
    try:
        with ThreadPoolExecutor(GAME_LOAD_WORKERS) as game_pool:
            while True:
                done = done_marker is None or done_marker.exists()   # read before the glob: see the module docstring
                new = False
                for user, f in captured_games(round_dir):
                    stat = f.stat()
                    cheap = _stat_fingerprint(stat)
                    if known.get(user) == cheap:
                        continue
                    known[user] = cheap
                    fp = game_fingerprint(stat, f.read_bytes())
                    fingerprints[user] = fp
                    submitted[user] = game_pool.submit(one, user, f, fp)
                    new = True
                if budget.reason:
                    break
                if not parent_alive(a.parent_pid):
                    orphaned = True
                    # gate every not-yet-sent oracle request the same way a tripped Budget already does (never
                    # drain the whole queue just because paper.sh is gone and no one is left to read hints.json)
                    budget.abandon("the watcher's paper.sh parent is gone")
                    break
                if not new and done and all(fut.done() for fut in submitted.values()):
                    break
                time.sleep(a.poll_s)
            concurrent.futures.wait(submitted.values())
    finally:
        hint_pool.shutdown(wait=True)

    seconds = time.time() - t0
    if orphaned:
        print(json.dumps({"failed": "paper.sh (--parent-pid) is gone", "seconds": round(seconds, 1)}))
        raise SystemExit("the hint step stopped: its paper.sh parent is gone, so no one is left to use hints.json")
    if budget.reason:
        print(json.dumps({"failed": budget.reason, "seconds": round(seconds, 1)}))
        raise SystemExit(f"the hint step failed: {budget.reason}")

    stats: collections.Counter = collections.Counter()
    items, no_hint = [], []
    for user, fut in sorted(submitted.items()):
        for i, rec in fut.result().items():
            stats["player_calls"] += 1
            tries = rec.get("tries") or []
            stats["glm_calls"] += len(tries)
            stats["prompt_tokens"] += sum((t.get("usage") or {}).get("prompt_tokens") or 0 for t in tries)
            stats["completion_tokens"] += sum((t.get("usage") or {}).get("completion_tokens") or 0 for t in tries)
            stats["errors"] += int("error" in rec)
            stats["asked_twice"] += len(tries) > 1
            if rec.get("hint"):
                items.append({"user": user, "i": int(i), "hint": rec["hint"]})
            else:
                no_hint.append({"user": user, "i": int(i)})
    stats["no_hint"] = len(no_hint)
    no_hint_frac = stats["no_hint"] / stats["player_calls"] if stats["player_calls"] else 0.0
    doc_out = {"round_dir": str(round_dir), "model": a.model, "effort": a.effort, "items": items,
              "no_hint": no_hint, "games": {str(user): fp for user, fp in fingerprints.items()},
              "stats": {**stats, "hinted": len(items), "cost_usd": round(budget.round_total, 2),
                        "no_hint_frac": round(no_hint_frac, 4)},
              "seconds": round(seconds, 1), "minutes": round(seconds / 60, 1)}
    print(json.dumps({k: v for k, v in doc_out.items() if k not in ("items", "no_hint", "games")}
                     | {"items": len(items), "no_hint": len(no_hint), "games": len(fingerprints)}))
    max_frac = getattr(a, "no_hint_max_frac", NO_HINT_MAX_FRAC)
    if no_hint_frac > max_frac:
        raise SystemExit(f"{stats['no_hint']}/{stats['player_calls']} player calls ({no_hint_frac:.1%}) have no "
                         f"usable hint, over {max_frac:.0%}: refusing to write hints.json")
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc_out))
    tmp.rename(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def oracle_args(s: argparse.ArgumentParser) -> None:
        s.add_argument("--c1", default=str(HERE / "runs-dagger1/c1.md"))
        s.add_argument("--cards-dir", default=str(HERE / "runs-overfit1/cards"))
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
    s = sub.add_parser("watch", help="the OPSD row's hint step: hint round_dir's games as they land, writing "
                                     "round_dir/hints.json once every one is done (resumable per game)")
    s.add_argument("round_dir")
    s.add_argument("--c1", required=True)
    s.add_argument("--model", default=FLASH_MODEL)
    s.add_argument("--effort", default=FLASH_EFFORT)
    s.add_argument("--workers", type=int, default=HINT_WORKERS)
    s.add_argument("--until", default=None,
                   help="a marker file: keep polling for new games until it exists and a full poll finds none; "
                        "omit to process whatever is in round_dir/captured/ once and stop")
    s.add_argument("--poll-s", type=float, default=10.0, dest="poll_s")
    s.add_argument("--round-budget-usd", type=float, default=HINT_ROUND_BUDGET_USD, dest="round_budget_usd")
    s.add_argument("--run-budget-usd", type=float, default=HINT_RUN_BUDGET_USD, dest="run_budget_usd")
    s.add_argument("--run-budget-file", default=None, dest="run_budget_file",
                   help="where the run's cumulative spend persists across rounds (default: round_dir/hints-cost.json; "
                        "paper.sh points every round of a run at the same file, under the run's own directory)")
    s.add_argument("--no-hint-max-frac", type=float, default=NO_HINT_MAX_FRAC, dest="no_hint_max_frac",
                   help="refuse to write hints.json once more than this share of the round's player calls has no "
                        "usable hint (default %(default)s). A call with no hint is dropped from training, never "
                        "trained against an unhinted teacher, so this guards against a broken oracle, not wrong data")
    s.add_argument("--parent-pid", type=int, default=None, dest="parent_pid",
                   help="paper.sh's own $$: this watcher is setsid-detached, so it polls for this pid's death "
                        "itself and stops rather than outlive a paper.sh that was killed before it could say so")
    s.set_defaults(fn=cmd_watch)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
