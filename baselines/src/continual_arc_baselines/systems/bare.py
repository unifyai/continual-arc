"""The bare model as a system: the harness protocol with no harness, the model alone with only the two actions,
request a demonstration or submit.

Each observation, rendered exactly as every harness receives it (:mod:`continual_arc_baselines.systems.base`), goes to
the model as the next user message of one conversation per instance, and its reply is parsed for the action as a
harness's is, with the same re-prompts and give-up. There is nothing else: no tools, no code, no memory, no files, one
model call per message. The instance's closing verdict is not sent: it exists for a harness's storage step, a bare
model stores nothing, and at a high reasoning effort it would cost a whole reasoning call.

Calls go to the run's proxy (``ARC_LLM_BASE_URL``), streamed, so a reply that reasons for half an hour is never silent
long enough to meet the proxy's read timeout. Earlier replies stay in the conversation as their text, without their
reasoning, as chat templates render a past turn.

The two floors of row F are built on it: :class:`BareModelLearner`, the stateless floor, is the model itself, choosing
whatever actions it wants rather than following a policy of ours, and :class:`BareDemoCacheLearner`, the demo-cache
floor, is the same model with a cache of the pairs it bought for each task, shown again at no cost when the task
returns; on an ARC-AGI-3 game it keeps the demonstrations it bought, the solver's plays of other levels of the game.
:class:`BareProgramLibraryLearner` is the program library, row PL (named M in the run records): on the grid tracks its
program is ``transform(grid)``, and on an ARC-AGI-3 game a world model, the game's rules as code
(:mod:`continual_arc_baselines.systems.world_model`). All of them take ``end="finish"`` in the grid, so their rules are
the harness rows' (protocol.RULES_FINISH); the verdict is still not sent, but for the program library's ask for a
game's program after an instance. At several users the cache and the programs are the team's, as any learner's state is
(:mod:`continual_arc_baselines.shared`): each user's copy is read as the round begins (``load_state``), written as it
ends (``save_state``), and merged into the one state with the other users' between them.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from continual_arc.arc3.env import (
    ActFeedback,
    Arc3LearnerObservation,
    DemonstrationFeedback,
    ResetFeedback,
)
from continual_arc.arc3.instances import Act, Demonstration
from continual_arc.env import (
    Action,
    DemosFeedback,
    GiveUp,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)
from continual_arc.grids import InvalidGrid, normalize_grid
from continual_arc.tasks import Pair

from continual_arc_baselines.proxy import PLACEHOLDER_KEY
from continual_arc_baselines.systems import arc3_protocol, protocol, world_model
from continual_arc_baselines.systems.base import SystemLearner, SystemUnavailable
from continual_arc_baselines.systems.protocol import objects_from_end, render_grid, shape
from continual_arc_baselines.systems.sandbox import (
    SandboxSpec,
    darwin_user_dirs,
    python_runtime_root,
    resolve_backend,
    wrap_command,
)
from continual_arc_baselines.tracking import instance_ended


class BareModelLearner(SystemLearner):
    """The model alone behind the harness protocol. ``max_tokens`` caps each reply; ``timeout_s`` is the longest
    silence tolerated inside one streamed reply."""

    name = "bare"

    def __init__(self, model: str, *, max_tokens: int = 65_536, timeout_s: float = 3600.0,
                 context_window: int | None = None, **base: Any):
        super().__init__(**base)
        self.model, self.max_tokens, self.timeout_s = model, max_tokens, timeout_s
        self.context_window = context_window  # what a harness is told about its model; a bare model is told nothing
        self.messages: list[dict[str, str]] = []
        self._client: httpx.Client | None = None
        self.stats["truncations"] = 0

    def start(self) -> None:
        self._client = httpx.Client(timeout=httpx.Timeout(connect=30.0, read=self.timeout_s, write=60.0, pool=30.0))
        self.messages = []

    def stop(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        self.messages = []

    def new_session(self) -> None:
        self.messages = []
        self._fresh_session = True

    def wipe_state(self) -> None:
        self.messages = []

    def _close_instance(self, obs: Any) -> None:
        self.transcript.take()  # the verdict, with nothing to store it in

    def ask(self, message: str, *, reprompt: bool = False) -> str:
        assert self._client is not None
        if self.messages and self.messages[-1]["role"] == "user":
            # the last message got no reply (empty_reply="reprompt"): it and this one are one turn, as a conversation
            # holds no empty assistant message
            self.messages[-1]["content"] += "\n\n" + message
        else:
            self.messages.append({"role": "user", "content": message})
        url = self.environ["ARC_LLM_BASE_URL"].rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {self.environ.get('ARC_LLM_API_KEY', PLACEHOLDER_KEY)}"}
        body = {"model": self.model, "messages": self.messages, "max_tokens": self.max_tokens, "stream": True}
        parts: list[str] = []
        finish = None
        failure = None
        try:
            with self._client.stream("POST", url, json=body, headers=headers) as response:
                if response.status_code >= 400:
                    failure = f"HTTP {response.status_code}: {response.read()[:300]!r}"
                else:
                    for line in response.iter_lines():
                        if not line.startswith("data:"):
                            continue  # blank lines and keep-alive comments
                        data = line[5:].strip()
                        if data == "[DONE]":
                            continue  # read on to the stream's end, so the connection goes back to the pool
                        chunk = json.loads(data)
                        if chunk.get("error"):
                            failure = f"error in the stream: {json.dumps(chunk['error'])[:300]}"
                            break
                        for choice in chunk.get("choices") or []:
                            parts.append((choice.get("delta") or {}).get("content") or "")
                            finish = choice.get("finish_reason") or finish
        except (httpx.TimeoutException, httpx.ConnectError) as error:
            # silent for timeout_s, or the proxy is gone: the system is down, whatever empty_reply says
            raise SystemUnavailable(f"{type(error).__name__}: {error}") from error
        except (httpx.HTTPError, ValueError) as error:
            failure = f"{type(error).__name__}: {error}"
        self.last_turn_outcome = {"finish_reason": finish, **({"error": failure} if failure else {})}
        if failure is not None:
            if self.empty_reply == "restart":
                raise SystemUnavailable(failure)
            parts = []   # a model call that failed is a turn with no text, re-prompted like one
        reply = "".join(parts)
        if finish == "length":
            self.stats["truncations"] += 1
        if reply:
            self.messages.append({"role": "assistant", "content": reply})
        return reply

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system": self.name, "model": self.model, "max_tokens": self.max_tokens,
                "timeout_s": self.timeout_s}


class BareDemoCacheLearner(BareModelLearner):
    """The demo-cache floor: the bare model, plus a cache of the demonstration pairs it has bought for each
    task, up to ``cache_cap`` a task (the per-instance demo cap). When a task returns, the first message of the
    instance shows its cached pairs again at no cost, before the instance; the model may still request more, and they
    join the cache. On an ARC-AGI-3 game the cache keeps its demonstrations, the solver's plays of other levels, shown
    again as they were bought. Nothing else is kept: no answers, no verdicts, no conversation."""

    name = "bare-demo-cache"
    # The cache, and the program library's programs, live in memory; they reach the state directory only for the
    # commits of a logged state and a team's merge (save_state), so a state directory put back would not put them back.
    state_in_files = False

    def __init__(self, model: str, *, cache_cap: int = 8, **kwargs: Any):
        super().__init__(model, **kwargs)
        self.cache_cap = cache_cap
        self.cache: dict[str, list] = {}

    def cached_pairs(self, task_id: str) -> list:
        """The pairs shown again when ``task_id`` returns."""
        return self.cache.get(task_id, [])

    def keep_pairs(self, task_id: str, pairs: list) -> None:
        """Pairs just bought for ``task_id``, kept up to the cap."""
        kept = self.cache.setdefault(task_id, [])
        kept.extend(pairs[: max(self.cache_cap - len(kept), 0)])

    def instance_preface(self, obs: LearnerObservation | Arc3LearnerObservation) -> str:
        pairs = self.cached_pairs(obs.task_id)
        if not pairs:
            return ""
        if isinstance(obs, Arc3LearnerObservation):
            shown = "\n\n".join(arc3_protocol.render_demonstration(d, k) for k, d in enumerate(pairs, 1))
            return f"Demonstrations of game {obs.task_id} that you received before, shown again at no cost:\n\n{shown}"
        shown = "\n\n".join(f"Earlier pair {k} input ({shape(p.input)}):\n{render_grid(p.input)}\n"
                             f"Earlier pair {k} output ({shape(p.output)}):\n{render_grid(p.output)}"
                             for k, p in enumerate(pairs, 1))
        return f"Demonstration pairs of task {obs.task_id} that you received before, shown again at no cost:\n\n{shown}"

    def observe(self, obs: LearnerObservation, action: Any, feedback: Any) -> None:
        if isinstance(feedback, DemosFeedback) and not feedback.refused and feedback.pairs:
            self.keep_pairs(obs.task_id, list(feedback.pairs))
        if isinstance(feedback, DemonstrationFeedback) and not feedback.refused:
            self.keep_pairs(obs.task_id, [feedback.demonstration])
        super().observe(obs, action, feedback)

    def wipe_state(self) -> None:
        super().wipe_state()
        self.cache = {}
        (self.state_dir / "cache.json").unlink(missing_ok=True)

    def save_state(self) -> None:
        """The cache, kept in memory while the run plays, as ``cache.json`` for the round's commit."""
        (self.state_dir / "cache.json").write_text(json.dumps({task: [p.to_json() for p in pairs]
                                                               for task, pairs in self.cache.items()}))

    def load_state(self) -> None:
        """The cache ``cache.json`` holds, none without it, each task's pairs up to the cap: a team's cache, merged from
        its users' purchases in user order, can hold more of a task's pairs than one instance could buy."""
        path = self.state_dir / "cache.json"
        kept = json.loads(path.read_text()) if path.exists() else {}
        self.cache = {task: [Demonstration.from_json(p) if "steps" in p else Pair.from_json(p)
                             for p in pairs[: self.cache_cap]] for task, pairs in kept.items()}

    def resources(self) -> dict[str, float]:
        out = super().resources()
        out["state_bytes"] = float(len(json.dumps({k: [p.to_json() for p in v] for k, v in self.cache.items()})))
        return out

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system": self.name, "cache_cap": self.cache_cap}


PROGRAM_NOTE = """\
Here every submission is a program instead of a grid. To submit, write a Python function transform(grid) that
returns the output grid for an input of this task, in one ```python block using only the standard library, and end
your reply with {"action": "submit"}. Your program is run on the test input and its output is submitted. Before that
it is checked, at no cost, on every demonstration pair you have for this task, and a failure is reported to you;
after 3 failed checks in an instance, a program's output is submitted without a check. A program that reproduces all
the pairs you have is kept for the task: when the task returns, it is run on the new test input first and its output
is submitted."""

# Row PL0, a stateless program library (M0 in the run records): every submission a program, checked as in PL, with the
# demo cache and the kept programs wiped before every instance (stateless=True), so its rules drop the one sentence
# about keeping.
PROGRAM_NOTE_STATELESS = PROGRAM_NOTE[:PROGRAM_NOTE.index(" A program that reproduces all")]

PROGRAM_REPROMPT = ('I could not find a valid action in your reply ({reason}). Reply with your program in one '
                    '```python block and {{"action": "submit"}} on the last line, or with {{"action": "request_demos"}}.')

# Row PL on an ARC-AGI-3 game: the program is the game's rules, a world model (systems/world_model.py), and the model may
# also play as the demo-cache floor does. {max_checks} and {max_programs} are the learner's.
GAME_PROGRAM_NOTE = """\
Here you may also answer with the game's rules as a program. In one ```python block, using only the standard library,
define step(board, action), the board after an action, and won(board), whether a board shows the level completed. A
board is a list of 64 rows of 64 colors 0-15, top row first, as boards are shown here; an action is [id], or [6, x, y]
for a click. To carry what a board does not show, also define init(board), your state at a level's first board, and
render(state), the board a state shows; step and won then take that state. Where your program cannot tell the board,
such as after an action whose outcome it does not model, render may return None (or step, if your states are boards):
that board is not compared, nothing after it in that level is checked, and the search never goes through it, but it
is never a win, so the check fails where the level completed. You may define lost(state), whether the game is over;
clicks(state), the [x, y] clicks worth trying (by default one cell of each one-colored patch of the board); and
key(state), a value equal for states that are the same. End your reply with {{"action": "model"}}.
Your program is checked at no cost: it is run from the first board of every level of this game you have seen, a
demonstration's or your own, through every action played on it, and must give every board the game showed and say won
exactly where the level completed (and lost, if you define it, where the game ended). A failed check is reported to
you; after {max_checks} failed checks in an instance, a program is used without a check. The benchmark then searches
your program for the shortest sequence of actions that wins from the board now, and plays it, stopping as soon as the
game shows a board your program did not predict, which is reported to you. You may send at most {max_programs}
programs per instance. A program that passes its check is kept for the game: when the game returns, its plan is played
first, before you are asked."""

# Row PL0 on a game: the same, wiped before every instance, so its rules drop the one sentence about keeping.
GAME_PROGRAM_NOTE_STATELESS = GAME_PROGRAM_NOTE[:GAME_PROGRAM_NOTE.index(" A program that passes its check is kept")]

GAME_REPROMPT = ('I could not find a valid action in your reply ({reason}). Reply with a program in one ```python block '
                 'and {{"action": "model"}} on the last line, or with an action of the game.')

# After an instance of a game, when no kept program reproduces every transition the game has shown.
GAME_PROGRAM_ASK = ("The instance is over. Before the next one, write this game's rules as a program, as the rules "
                    "describe, from every board of it you have seen: it is checked at no cost on every transition of the "
                    "game you have seen, and kept for the game if it passes. Reply with the program in one ```python "
                    'block and {"action": "model"} on the last line.')

PROGRAM_ACTIONS = frozenset({"model", "program", "world_model", "submit_model", "submit_program"})

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_PY_FENCE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def extract_program(text: str) -> str | None:
    """The program in a reply: the ``program`` string of the last JSON object, else the last fenced Python block, else
    the whole text when it defines ``transform``."""
    for candidate in [m.group(1) for m in reversed(list(_JSON_FENCE.finditer(text)))] + objects_from_end(text):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("program"), str) and "def transform" in obj["program"]:
            return obj["program"]
    m = _PY_FENCE.search(text)
    if m:
        return m.group(1)
    return text if "def transform" in text else None


def run_sandboxed(script: Path, stdin: str, *, sandbox: str, timeout_s: float, workdir: Path) -> tuple[str, str | None]:
    """``python -I -S script`` fed ``stdin``, under the harnesses' sandbox, with no network and no access to the benchmark,
    reading and writing ``workdir`` alone: (its last line of output, None), or ("", what went wrong)."""
    backend = resolve_backend(sandbox)
    python = Path(sys.executable)
    reads = [python.parent.parent]
    runtime = python_runtime_root(python)
    if runtime:
        reads.append(runtime)
    spec = SandboxSpec(read_paths=reads + [workdir], write_paths=[workdir, *darwin_user_dirs()],
                       tmp_dir=workdir, home=workdir)
    argv = wrap_command(backend, [str(python), "-I", "-S", str(script)], spec, workdir / "sandbox.sb")
    try:
        out = subprocess.run(argv, input=stdin, capture_output=True, text=True, timeout=timeout_s, check=False,
                             cwd=workdir, env={"HOME": str(workdir), "PATH": "/usr/bin:/bin"})
    except subprocess.TimeoutExpired:
        return "", "timeout"
    if out.returncode != 0:
        return "", (out.stderr or "").strip()[-400:] or f"exit {out.returncode}"
    lines = out.stdout.strip().splitlines()
    return (lines[-1] if lines else ""), None


def run_transform(code: str, grid, *, sandbox: str, timeout_s: float, workdir: Path) -> tuple[Any, str | None]:
    """Run a model-written ``transform(grid)`` on one grid in a subprocess under the harnesses' sandbox (run_sandboxed).
    Returns (grid, None), or (None, what went wrong)."""
    script = workdir / "program.py"
    script.write_text(code + textwrap.dedent('''

        if __name__ == "__main__":
            import json, sys
            _g = json.loads(sys.stdin.read())
            print(json.dumps(transform(_g)))
    '''))
    line, error = run_sandboxed(script, json.dumps(grid), sandbox=sandbox, timeout_s=timeout_s, workdir=workdir)
    if error is not None:
        return None, error
    try:
        return normalize_grid(json.loads(line)), None
    except (ValueError, InvalidGrid) as error:
        return None, f"bad output: {error}"


def extract_world_model(text: str) -> str | None:
    """The game program in a reply: the last fenced Python block that defines ``step``, else the ``program`` string of
    the last JSON object that does."""
    blocks = [m.group(1) for m in _PY_FENCE.finditer(text) if "def step" in m.group(1)]
    if blocks:
        return blocks[-1]
    for candidate in objects_from_end(text):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("program"), str) and "def step" in obj["program"]:
            return obj["program"]
    return None


def asks_for_model(text: str) -> bool:
    """Whether a reply's action, its last JSON object with an action, is the program action (PROGRAM_ACTIONS)."""
    for candidate in objects_from_end(text):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(obj, dict) and "action" in obj:
            return str(obj["action"]).strip().lower() in PROGRAM_ACTIONS
    return False


WON = "won"
"""What a plan predicts of its last action: the level completes, so no board after it is compared."""


@dataclass
class _Level:
    """What the program library keeps while it plays one instance of a game."""

    key: tuple
    recorder: world_model.Recorder = field(default_factory=world_model.Recorder)
    bought: list = field(default_factory=list)       # this instance's demonstrations, as trajectories
    plan: list = field(default_factory=list)         # the planned actions still to play
    boards: list = field(default_factory=list)       # the board each of them is predicted to leave, WON for the last
    expected: Any = None                             # what the planned action just played should have left
    last: Act | None = None                          # that action
    played: int = 0                                  # the plan's actions played
    total: int = 0                                   # the plan's length
    checks: int = 0                                  # failed checks in the instance
    programs: int = 0                                # programs sent in the instance
    asked: bool = False                              # whether the model has been asked in the instance
    kept_note: str = ""                              # what became of the kept program at the instance's start
    shown: set = field(default_factory=set)          # the trajectories the conversation has shown, by name


class BareProgramLibraryLearner(BareDemoCacheLearner):
    """The program library, row PL (M in the run records): the bare demo-cache floor, with one requirement, that every
    submission is a program, one for each task; the model decides every demonstration request. The model writes
    ``transform(grid)`` in a Python block and ends with the submit action; the
    program runs on the test input in the harnesses' sandbox and its output is submitted, and a grid alone is re-prompted.
    Before that, the program is checked at no cost on every pair the model has for the task (the cached pairs shown at
    the start of the instance, then the pairs bought in it), and a failure goes back to it; after ``max_checks`` failed
    checks in an instance a program's output is submitted without a check, so no instance can loop. A program that
    reproduces every pair is kept for the task, the latest replacing the one before. When the task returns, the kept
    program runs on the new test input first and its output is submitted before the model is asked (no model call); if
    that is wrong, the model sees the program, the cached pairs and the failure. The rules are the harness rows' with
    :data:`PROGRAM_NOTE`; no verdict turn.

    On an ARC-AGI-3 game the program is the game's rules, a world model (:mod:`continual_arc_baselines.systems.world_model`),
    and the model may also play, reset and request demonstrations as the demo-cache floor does, since probing the
    actions its demonstrations did not show is how a game's rules are learned. A program comes with the action
    ``{"action": "model"}``; it is checked at no cost on every trajectory the game has shown, the demonstrations' and
    the learner's own play, earlier instances' included, and a failure goes back to the model; after ``max_checks``
    failed checks in an instance a program is used without a check, and at most ``max_programs`` are taken an
    instance. A program used is searched for the shortest sequence that wins from the board now (at most ``max_states``
    states expanded, ``search_s`` seconds), and the plan is played with no model call, every board compared with the
    one the program predicted: the first that differs stops the plan and goes to the model. A program that passes its
    check is kept for the game. When the game returns, the kept program is searched from the level's first board and
    its plan played before the model is asked; if the model is asked, it sees the program and what became of it. After
    an instance in which the model was asked, if no kept program reproduces every transition the game has shown, the
    model is asked for one (:data:`GAME_PROGRAM_ASK`), checked as above, up to ``max_checks`` tries, at no cost to the
    budget: a model that plays without writing a program would otherwise keep nothing. The rules are the harness rows'
    with :data:`GAME_PROGRAM_NOTE`. A stateless learner (row PL0) keeps nothing between instances and is not asked
    after one."""

    name = "bare-program-library"

    def __init__(self, model: str, *, max_checks: int = 3, program_timeout_s: float = 10.0, sandbox: str = "auto",
                 max_programs: int = 6, max_states: int = world_model.MAX_STATES,
                 search_s: float = world_model.SECONDS, **kwargs: Any):
        kwargs.setdefault("end", "finish")
        super().__init__(model, **kwargs)
        self.max_checks = max_checks
        self.program_timeout_s = program_timeout_s
        self.sandbox = sandbox
        self.max_programs = max_programs
        self.max_states = max_states
        self.search_s = search_s
        self.library: dict[str, str] = {}
        self.trajectories: dict[str, list[dict]] = {}   # every trajectory each game has shown, by game
        self._programs = Path(tempfile.mkdtemp(prefix="arc-bare-programs-"))
        self._instance: tuple | None = None
        self._earlier: list = []          # the cached pairs shown at the start of the instance
        self._bought: list = []           # the pairs bought in the instance
        self._checks = 0
        self._kept_run: tuple[str, str] | None = None   # (what happened, program) for the kept program's first run
        self._kept_pending = False
        self._level: _Level | None = None   # the game instance in play
        self.stats.update({"programs_kept": 0, "checks_failed": 0, "program_errors": 0, "grid_refusals": 0,
                           "library_runs": 0, "library_hits": 0, "library_misses": 0, "programs": 0, "plans": 0,
                           "plan_actions": 0, "surprises": 0, "no_plans": 0, "program_asks": 0})

    def run(self, program: str, grid: Any) -> tuple[Any, str | None]:
        return run_transform(program, grid, sandbox=self.sandbox, timeout_s=self.program_timeout_s,
                             workdir=self._programs)

    # -- what the model is shown ------------------------------------------------

    def instance_preface(self, obs: LearnerObservation) -> str:
        parts = []
        if self._earlier:
            shown = "\n\n".join(f"Earlier pair {k} input ({shape(p.input)}):\n{render_grid(p.input)}\n"
                                 f"Earlier pair {k} output ({shape(p.output)}):\n{render_grid(p.output)}"
                                 for k, p in enumerate(self._earlier, 1))
            parts.append(f"Demonstration pairs of task {obs.task_id} that you received before, shown again at no cost:\n\n"
                         f"{shown}")
        if self._kept_run is not None:
            what, program = self._kept_run
            parts.append(f"The program kept for task {obs.task_id} from an earlier instance {what}:\n"
                         f"```python\n{program.rstrip()}\n```")
        return "\n\n".join(parts)

    def _known_pairs(self) -> list[tuple[str, Any]]:
        return ([(f"earlier pair {k}", p) for k, p in enumerate(self._earlier, 1)]
                + [(f"demo pair {k}", p) for k, p in enumerate(self._bought, 1)])

    def _check(self, program: str) -> str | None:
        for name, pair in self._known_pairs():
            got, error = self.run(program, pair.input)
            if error is not None:
                return f"Your program was checked on {name} at no cost and failed ({error})."
            if got != pair.output:
                return (f"Your program was checked on {name} at no cost and its output is wrong. It gave\n"
                        f"{render_grid(got)}\nwhere the pair's output is\n{render_grid(pair.output)}")
        return None

    def _interpret(self, obs: LearnerObservation, reply: str) -> tuple[str, Any]:
        """('action', an action), ('check', a failed check to send back) or ('unusable', why)."""
        program = extract_program(reply)
        rest = _PY_FENCE.sub("", reply) if program else reply
        try:
            parsed: Any = protocol.parse_action(rest, finish=True)
            reason = ""
        except protocol.ParseError as error:
            parsed, reason = None, str(error)
        if isinstance(parsed, RequestDemos):
            return "action", parsed
        if isinstance(parsed, protocol.Finish):
            self.stats["finish_refusals"] += 1
            return "unusable", "finish is refused until the instance is over"
        if program is None or "def transform" not in program:
            if isinstance(parsed, Submit):
                self.stats["grid_refusals"] += 1
                return "unusable", "a submission here is a program in a ```python block, not a grid"
            return "unusable", reason or "no program and no demo request"
        problem = self._check(program)
        if problem is not None and self._checks < self.max_checks:
            self._checks += 1
            self.stats["checks_failed"] += 1
            return "check", (f"{problem}\nFailed checks in this instance: {self._checks} of {self.max_checks}. Reply "
                             'with a corrected program and {"action": "submit"}, or with {"action": "request_demos"}.')
        if problem is None:
            self.library[obs.task_id] = program
            self.stats["programs_kept"] += 1
        got, error = self.run(program, obs.input)
        if error is not None:
            self.stats["program_errors"] += 1
            if self._checks < self.max_checks:
                self._checks += 1
                return "check", (f"Your program failed on the test input ({error}), so nothing was submitted. Failed "
                                 f"checks in this instance: {self._checks} of {self.max_checks}.")
            return "action", Submit([[10]])   # no output to submit: charged like a wrong answer
        return "action", Submit(got)

    # -- the loop ---------------------------------------------------------------

    def act(self, obs: LearnerObservation | Arc3LearnerObservation) -> Action:
        if isinstance(obs, Arc3LearnerObservation):
            return self._act_game(obs)
        self._begin_instance(obs)   # row PL0's kept programs and demo cache are wiped here, before every instance
        if self._instance != (obs.task_id, obs.step):
            self._instance, self._checks, self._kept_run, self._bought = (obs.task_id, obs.step), 0, None, []
            self._earlier = list(self.cache.get(obs.task_id, []))
            program = self.library.get(obs.task_id)
            if program is not None:
                self.stats["library_runs"] += 1
                got, error = self.run(program, obs.input)
                if error is None:
                    self._kept_run = ("was run on this test input first, and its output was submitted", program)
                    self._kept_pending = True
                    return Submit(got)
                self._kept_run = (f"failed on this test input ({error}), so nothing was submitted", program)
        pending = self.transcript.take()

        def observation(fresh: bool) -> str:
            preface = self.instance_preface(obs) if fresh else ""
            return protocol.render_observation(obs, pending=pending, demo_cost=self.demo_cost,
                                               max_cost=self.max_cost, first_in_session=fresh, finish=True,
                                               rules_note=PROGRAM_NOTE_STATELESS if self.stateless else PROGRAM_NOTE,
                                               preface=preface)

        try:
            reply = self._ask_resilient(observation, kind="observation", step=obs.step, task=obs.task_id)
            unusable = 0
            while True:
                kind, value = self._interpret(obs, reply)
                if kind == "action":
                    return value
                if kind == "unusable":
                    self.stats["parse_failures"] += 1
                    self.note(f"step {obs.step}: unusable reply ({value}); tail: {reply[-300:]!r}")
                    if unusable == self.max_reprompts:
                        self.note(f"step {obs.step}: giving up after {unusable + 1} unusable replies")
                        return GiveUp()
                    unusable += 1
                    self.stats["reprompts"] += 1
                    message = PROGRAM_REPROMPT.format(reason=value)
                else:
                    message = value

                def follow_up(fresh: bool, message: str = message) -> str:
                    return observation(True) + "\n\n" + message if fresh else message

                reply = self._ask_resilient(follow_up, reprompt=True, kind="check" if kind == "check" else "reprompt",
                                            step=obs.step, task=obs.task_id)
        except SystemUnavailable as error:
            self.stats["forced_give_ups"] += 1
            self.note(f"step {obs.step}: system unavailable after {self.max_restarts} restart(s) "
                      f"({error}); this action is a give-up")
        return GiveUp()

    def observe(self, obs: LearnerObservation | Arc3LearnerObservation, action: Any, feedback: Any) -> None:
        if isinstance(obs, Arc3LearnerObservation):
            self._observe_game(obs, action, feedback)
        if isinstance(feedback, DemosFeedback) and not feedback.refused and feedback.pairs:
            self._bought.extend(feedback.pairs)
        if self._kept_pending and isinstance(feedback, SubmitFeedback):
            self._kept_pending = False
            self.stats["library_hits" if feedback.correct else "library_misses"] += 1
        super().observe(obs, action, feedback)

    # -- a game ------------------------------------------------------------------

    def _serve(self, request: dict) -> dict:
        """One request of :func:`world_model.serve`, in the sandbox: its answer, or ``{"error": ...}``."""
        script = self._programs / "world_model.py"
        if not script.exists():
            script.write_text(Path(world_model.__file__).read_text())
        line, error = run_sandboxed(script, json.dumps(request), sandbox=self.sandbox, timeout_s=self.search_s + 60,
                                    workdir=self._programs)
        if error is not None:
            return {"error": error}
        try:
            return json.loads(line)
        except ValueError:
            return {"error": "the check gave no answer"}

    def _game_trajectories(self, game: str, level: _Level) -> list[dict]:
        """Every trajectory the game has shown: earlier instances', then this instance's demonstrations and play."""
        return self.trajectories.get(game, []) + level.bought + level.recorder.trajectories

    def _check_game(self, game: str, level: _Level, program: str) -> str | None:
        """None when ``program`` reproduces every trajectory the game has shown, else what the model is told."""
        trajectories = self._game_trajectories(game, level)
        answer = self._serve({"op": "check", "program": program, "trajectories": trajectories,
                              "seconds": self.search_s})
        if "error" in answer:
            self.stats["program_errors"] += 1
            return f"Your program could not be checked ({answer['error']})."
        failure = answer["failure"]
        if failure is None:
            return None
        message = answer["message"]
        index = failure.get("trajectory")
        if index is not None and trajectories[index]["name"] not in level.shown:
            # a level this conversation never showed: its first board and the actions before the failure
            trajectory = trajectories[index]
            before = [a for s in trajectory["steps"][:failure["step"] or 0] for a in s["actions"]]
            message += (f"\nThat level's first board:\n{world_model.render_board(trajectory['start'])}"
                        + (f"\nThe actions before it: {', '.join(world_model.describe_action(a) for a in before)}."
                           if before else ""))
        return message

    def _search(self, program: str, history: dict, obs: Arc3LearnerObservation) -> dict:
        answer = self._serve({"op": "plan", "program": program, "history": history,
                              "available": list(obs.available_actions), "max_states": self.max_states,
                              "seconds": self.search_s})
        if "error" in answer:
            self.stats["program_errors"] += 1
            return {"found": False, "reason": "error", "error": answer["error"]}
        return answer

    def _why_no_plan(self, found: dict) -> str:
        return {"exhausted": "it reached every state it could without a win",
                "bound": f"it expanded {self.max_states:,} states without a win",
                "time": f"it ran out of its {self.search_s:g} seconds",
                }.get(found.get("reason"), f"it could not be run ({found.get('error', found.get('reason'))})")

    def _start_plan(self, level: _Level, found: dict, board: list) -> None:
        level.plan = [Act(*a) for a in found["plan"]]
        level.boards = []
        for changes in found["changes"]:
            board = world_model.apply(board, changes)
            level.boards.append(board)
        level.boards.append(WON)
        level.played, level.total = 0, len(level.plan)
        self.stats["plans"] += 1

    def _next_planned(self, level: _Level) -> Act:
        level.last = level.plan.pop(0)
        level.expected = level.boards.pop(0)
        level.played += 1
        self.stats["plan_actions"] += 1
        return level.last

    def _surprise(self, level: _Level, obs: Arc3LearnerObservation) -> str | None:
        """None when the planned action just played left the board its program predicted, else what the model is told."""
        played = (f"after its action {level.played} of {level.total} "
                  f"(action {arc3_protocol.describe(level.last)})")
        if level.expected is WON:
            return f"The plan of your program stopped {played}: your program predicted it would complete the level, and the game went on."
        if obs.state == "GAME_OVER":
            return f"The plan of your program stopped {played}: the game ended, which your program did not predict."
        board = obs.frames[-1]
        wrong = [(x, y) for y in range(world_model.SIZE) for x in range(world_model.SIZE)
                 if board[y][x] != level.expected[y][x]]
        if not wrong:
            return None
        return (f"The plan of your program stopped {played}: the game showed a board your program did not predict. In "
                f"the {len(wrong)} cells where they differ, the game shows\n"
                f"{world_model.render_changes([(x, y, board[y][x]) for x, y in wrong])}\nand your program predicted\n"
                f"{world_model.render_changes([(x, y, level.expected[y][x]) for x, y in wrong])}")

    def _open_level(self, obs: Arc3LearnerObservation) -> _Level:
        """The instance's level, made at its first observation, when the kept program, if any, is searched from the
        level's first board."""
        level = _Level((obs.task_id, obs.step))
        self._level = level
        program = self.library.get(obs.task_id)
        if program is None:
            return level
        self.stats["library_runs"] += 1
        found = self._search(program, {"start": obs.frames[-1], "steps": []}, obs)
        if found.get("found"):
            self._start_plan(level, found, obs.frames[-1])
            level.kept_note = ("was searched from this level's first board and its plan played first, until the game "
                               "showed what the program did not predict (the feedback above)")
        else:
            self.stats["no_plans"] += 1
            level.kept_note = (f"was searched from this level's first board and found no sequence of actions that wins: "
                               f"{self._why_no_plan(found)}, so nothing was played")
        return level

    def _game_preface(self, obs: Arc3LearnerObservation, level: _Level) -> str:
        """The first message's preface: the cached demonstrations, named earlier demonstrations, and the kept program
        with what became of it."""
        parts = []
        demos = self.cached_pairs(obs.task_id)
        if demos:
            shown = "\n\n".join("Earlier d" + arc3_protocol.render_demonstration(d, k)[1:]
                                  for k, d in enumerate(demos, 1))
            parts.append(f"Demonstrations of game {obs.task_id} that you received before, shown again at no cost:\n\n"
                         f"{shown}")
            level.shown.update(f"earlier demonstration {k}" for k in range(1, len(demos) + 1))
        program = self.library.get(obs.task_id)
        if program is not None and level.kept_note:
            parts.append(f"The program kept for game {obs.task_id} from an earlier instance {level.kept_note}:\n"
                         f"```python\n{program.rstrip()}\n```")
        return "\n\n".join(parts)

    def _interpret_game(self, obs: Arc3LearnerObservation, level: _Level, reply: str) -> tuple[str, Any]:
        """('action', an action), ('plan', None) when a program's plan starts, ('check', what to tell the model) or
        ('unusable', why)."""
        program = extract_world_model(reply)
        rest = _PY_FENCE.sub("", reply) if program else reply
        if asks_for_model(rest):
            if program is None:
                return "unusable", "a program is one ```python block that defines step and won"
            if level.programs >= self.max_programs:
                return "unusable", (f"the {self.max_programs} programs of this instance are spent: play with act, "
                                    "reset or request_demo")
            level.programs += 1
            self.stats["programs"] += 1
            failure = self._check_game(obs.task_id, level, program)
            if failure is not None and level.checks < self.max_checks:
                level.checks += 1
                self.stats["checks_failed"] += 1
                return "check", (f"{failure}\nFailed checks in this instance: {level.checks} of {self.max_checks}. "
                                 'Reply with a corrected program and {"action": "model"}, or with an action of the game.')
            if failure is None:
                self.library[obs.task_id] = program
                self.stats["programs_kept"] += 1
            history = level.recorder.current or {"start": obs.frames[-1], "steps": []}
            found = self._search(program, history, obs)
            if found.get("found"):
                self._start_plan(level, found, obs.frames[-1])
                return "plan", None
            self.stats["no_plans"] += 1
            head = ("Your program passes its check" if failure is None
                    else "Your program was used without a check, since the checks of this instance are spent")
            return "check", (f"{head}, but its search found no sequence of actions that wins from the board now: "
                             f"{self._why_no_plan(found)}. Reply with a corrected program and "
                             '{"action": "model"}, or with an action of the game.')
        try:
            parsed: Any = arc3_protocol.parse_action(rest, finish=True)
        except protocol.ParseError as error:
            return "unusable", str(error)
        if isinstance(parsed, protocol.Finish):
            self.stats["finish_refusals"] += 1
            return "unusable", "finish is refused until the instance is over"
        if isinstance(parsed, list):   # a sequence: the rest is played without another message
            first, self._queue = parsed[0], parsed[1:]
            return "action", first
        return "action", parsed

    def _act_game(self, obs: Arc3LearnerObservation) -> Any:
        self._begin_instance(obs)   # row PL0's programs, trajectories and cache are wiped here, before every instance
        proto = self.protocol_for(obs)
        level = self._level
        if level is None or level.key != (obs.task_id, obs.step):
            level = self._open_level(obs)
        level.recorder.board(obs.frames[-1])
        if self._queue:   # the rest of a sequence the model sent
            self.stats["batched_actions"] += 1
            return self._queue.pop(0)
        if level.expected is not None:   # a plan in play: did the game do what the program predicted?
            surprise = self._surprise(level, obs)
            level.expected = None
            if surprise is None and level.plan:
                return self._next_planned(level)
            if surprise is not None:
                self.stats["surprises"] += 1
                self.transcript.add_text(surprise)
            level.plan, level.boards = [], []
        elif level.plan:   # the kept program's plan, from the level's first board
            return self._next_planned(level)
        return self._ask_game(obs, level, proto)

    def _ask_game(self, obs: Arc3LearnerObservation, level: _Level, proto: Any) -> Any:
        level.asked = True
        pending = self.transcript.take()
        note = (GAME_PROGRAM_NOTE_STATELESS if self.stateless else GAME_PROGRAM_NOTE).format(
            max_checks=self.max_checks, max_programs=self.max_programs)

        def observation(fresh: bool) -> str:
            preface = self._game_preface(obs, level) if fresh else ""
            if fresh:
                level.shown.update(t["name"] for t in level.bought + level.recorder.trajectories)
            return proto.render_observation(obs, pending=pending, first_in_session=fresh, finish=True, rules_note=note,
                                            preface=preface)

        try:
            reply = self._ask_resilient(observation, kind="observation", step=obs.step, task=obs.task_id)
            unusable = 0
            while True:
                level.shown.update(t["name"] for t in level.bought + level.recorder.trajectories)
                kind, value = self._interpret_game(obs, level, reply)
                if kind == "action":
                    return value
                if kind == "plan":
                    return self._next_planned(level)
                if kind == "unusable":
                    self.stats["parse_failures"] += 1
                    self.note(f"step {obs.step}: unusable reply ({value}); tail: {reply[-300:]!r}")
                    if unusable == self.max_reprompts:
                        self.note(f"step {obs.step}: giving up after {unusable + 1} unusable replies")
                        return GiveUp()
                    unusable += 1
                    self.stats["reprompts"] += 1
                    message = GAME_REPROMPT.format(reason=value)
                else:
                    message = value

                def follow_up(fresh: bool, message: str = message) -> str:
                    return observation(True) + "\n\n" + message if fresh else message

                reply = self._ask_resilient(follow_up, reprompt=True, kind="check" if kind == "check" else "reprompt",
                                            step=obs.step, task=obs.task_id)
        except SystemUnavailable as error:
            self.stats["forced_give_ups"] += 1
            self.note(f"step {obs.step}: system unavailable after {self.max_restarts} restart(s) "
                      f"({error}); this action is a give-up")
        return GiveUp()

    def _observe_game(self, obs: Arc3LearnerObservation, action: Any, feedback: Any) -> None:
        level = self._level
        if level is None:
            return
        if isinstance(feedback, ActFeedback) and feedback.valid and not feedback.ignored:
            level.recorder.act(action.to_json(), won=feedback.level_completed,
                               over=feedback.state == "GAME_OVER" and not feedback.level_completed)
        elif isinstance(feedback, ResetFeedback):
            level.recorder.reset()
            level.plan, level.boards, level.expected = [], [], None
        elif isinstance(feedback, DemonstrationFeedback) and not feedback.refused:
            level.bought.append(world_model.trajectory_of_demonstration(
                feedback.demonstration.to_json(), f"demonstration {feedback.demo_requests_used}"))
        if instance_ended(feedback) and not level.asked and isinstance(feedback, ActFeedback) and feedback.level_completed:
            self.stats["library_hits"] += 1   # won by the kept program's plan, with no model call

    def _close_instance(self, obs: Any) -> None:
        if not isinstance(obs, Arc3LearnerObservation) or self._level is None:
            super()._close_instance(obs)
            return
        verdict = self.transcript.take()
        level = self._level
        try:
            if level.asked and not self.stateless and self._started:
                self._ask_for_program(obs, level, verdict)
        finally:
            self._keep_level(obs.task_id, level)

    def _ask_for_program(self, obs: Arc3LearnerObservation, level: _Level, verdict: list[str]) -> None:
        """After an instance, a program for the game when no kept one reproduces every transition it has shown
        (:data:`GAME_PROGRAM_ASK`), checked at no cost, up to ``max_checks`` tries."""
        kept = self.library.get(obs.task_id)
        if kept is not None and self._check_game(obs.task_id, level, kept) is None:
            return
        self.stats["program_asks"] += 1
        message = "\n".join([*verdict, "", GAME_PROGRAM_ASK]) if verdict else GAME_PROGRAM_ASK
        try:
            for attempt in range(self.max_checks):
                started = time.monotonic()
                self.last_turn_outcome = {}
                reply = self.ask(message, reprompt=attempt > 0)
                self.log_turn("program_ask" if attempt == 0 else "program_retry", message, reply,
                              seconds=round(time.monotonic() - started, 2), step=obs.step, task=obs.task_id)
                program = extract_world_model(reply)
                if program is None:
                    message = ("I could not find a program in your reply: write it in one ```python block that "
                               'defines step and won, with {"action": "model"} on the last line.')
                    continue
                failure = self._check_game(obs.task_id, level, program)
                if failure is None:
                    self.library[obs.task_id] = program
                    self.stats["programs_kept"] += 1
                    return
                self.stats["checks_failed"] += 1
                message = f"{failure}\nTries left: {self.max_checks - attempt - 1}."
        except SystemUnavailable as error:
            self.note(f"step {obs.step}: the ask for a program failed ({error}); restarting the system")
            self.close()
            self.stats["restarts"] += 1

    def _keep_level(self, game: str, level: _Level) -> None:
        """The instance's demonstrations and play join the game's trajectories, named as later conversations call
        them."""
        kept = self.trajectories.setdefault(game, [])
        demos = sum(1 for t in kept if t["name"].startswith("earlier demonstration"))
        plays = sum(1 for t in kept if t["name"].startswith("your play of an earlier level"))
        for trajectory in level.bought:
            demos += 1
            kept.append({**trajectory, "name": f"earlier demonstration {demos}"})
        for trajectory in level.recorder.trajectories:
            plays += 1
            kept.append({**trajectory, "name": f"your play of an earlier level of this game ({plays})"})

    # -- state ---------------------------------------------------------------------

    def wipe_state(self) -> None:
        super().wipe_state()
        self.library = {}
        self.trajectories = {}
        (self.state_dir / "library.json").unlink(missing_ok=True)
        (self.state_dir / "trajectories.json").unlink(missing_ok=True)

    def save_state(self) -> None:
        """The cache, then the kept programs as ``library.json`` and the games' trajectories as ``trajectories.json``,
        for the round's commit."""
        super().save_state()
        (self.state_dir / "library.json").write_text(json.dumps(self.library))
        (self.state_dir / "trajectories.json").write_text(json.dumps(self.trajectories))

    def load_state(self) -> None:
        super().load_state()
        path = self.state_dir / "library.json"
        self.library = json.loads(path.read_text()) if path.exists() else {}
        path = self.state_dir / "trajectories.json"
        self.trajectories = json.loads(path.read_text()) if path.exists() else {}

    def resources(self) -> dict[str, float]:
        out = super().resources()
        out["state_bytes"] += float(sum(len(p) for p in self.library.values())
                                    + sum(len(json.dumps(t)) for t in self.trajectories.values()))
        return out

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "system": self.name, "max_checks": self.max_checks,
                "program_timeout_s": self.program_timeout_s, "sandbox": self.sandbox,
                "max_programs": self.max_programs, "max_states": self.max_states, "search_s": self.search_s}
