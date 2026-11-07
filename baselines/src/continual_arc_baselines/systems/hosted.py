"""A submitted Python learner, run whole inside the sandbox.

The verified tier of Continual-ARC reruns a submission's code on a private stream of new rules (the benchmark's README,
Submitting a result). ``arc-baselines run --learner pkg:Class`` would build that code in this process, beside the
environment and its task registry, which hold the private rules, and the tracking proxy, which holds the submitter's
OpenRouter key. A submitted learner, any adapter or harness it brings included, is untrusted code, so
:class:`HostedLearner` keeps all of it in a sandbox of its own and forwards the benchmark's calls to it:

* the learner runs in :mod:`learner_host` (standard library and the public ``continual_arc`` only), started with
  the submission's own interpreter and speaking JSON lines;
* it reads its virtual environment, its checkout, the Python runtime the environment was made from and the operating
  system's programs and libraries, nothing else;
* it writes a private state directory under the attempt, kept for the whole run (it is the thing that learns, and
  its working directory), and its own HOME and TMPDIR;
* it reaches the network only through the tracking proxy (on Linux a private network namespace and the bridge of
  :func:`start_bridge`; on macOS Seatbelt allows the proxy's loopback port alone), with every base-URL variable
  pointing at the proxy and every key variable a placeholder, so it never holds a key.

A learner plays a frozen test one instance at a time from the end of its stream (:meth:`HostedLearner.freeze`): its files
are set aside as the stream left them, and before every instance its host is ended and a copy put back, so nothing it keeps
in its files or its processes reaches the next instance, whatever its code. So what a learner carries from its stream to a
test is what it keeps as files in its working directory: it is built again before every test instance.

Installing a submission runs its build code, so :func:`install_submission` (``arc-baselines install-submission``)
installs it under the sandbox too, with the new virtual environment and a cache of its own writable and the network only
through a filter to the package index and the code host (:mod:`continual_arc_baselines.systems.connect_filter`).

Everything those trees hold is the submission's to write, so nothing on this side of the sandbox believes it: no path
written there decides what the sandbox binds, a link there is never followed and nothing is read from or written to a
file there. What the install records, the interpreter's runtime among them, lies beside the environment, where the build
cannot reach (:func:`record_of`).

:class:`ServedLearner` is what any learner in another process shares: the protocol's calls over JSON lines, each reply
bounded, since the far side is untrusted. :class:`~continual_arc_baselines.systems.remote.RemoteLearner` serves the same
learner from a machine of its own.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from continual_arc.arc3.env import (
    ActFeedback,
    Arc3LearnerObservation,
    DemonstrationFeedback,
    ResetFeedback,
    ResetLevel,
)
from continual_arc.arc3.instances import Act
from continual_arc.env import (
    DemosFeedback,
    GiveUp,
    GiveUpFeedback,
    LearnerObservation,
    RequestDemos,
    Submit,
    SubmitFeedback,
)
from continual_arc.learner import BaseLearner

from continual_arc_baselines.frozen import HELD, delete, put_back
from continual_arc_baselines.proxy import PLACEHOLDER_KEY
from continual_arc_baselines.systems.connect_filter import ConnectFilter
from continual_arc_baselines.systems.sandbox import (
    CONFINED,
    SandboxSpec,
    darwin_user_dirs,
    default_mask_paths,
    install_root,
    package_manager_roots,
    proxy_port,
    resolve_backend,
    start_bridge,
    wrap_command,
)

log = logging.getLogger(__name__)

HOST = Path(__file__).with_name("learner_host.py")
PUBLIC_BENCHMARK = next(url for name, _, url in (r.partition(" @ ") for r in metadata.requires("continual-arc-baselines"))
                        if name.split("[")[0] == "continual-arc")
"""The benchmark the baselines themselves depend on (their pyproject's URL), installed for a submission that does not
bring ``continual-arc`` itself."""
MAX_REPLY = 16 * 2**20
"""The longest line a learner may send, in characters: its replies are grids and counters, and an unbounded line would
exhaust this process's memory."""
MAX_STARTUP_LINES = 64
"""How many lines a learner's start may print on the channel before its hello: what an interpreter's start-up says."""
MAX_RESOURCES = 64
"""How many counters a learner's ``resources`` may report, each of which goes in every record."""
INHERITED_ENV = ("LANG", "LC_ALL", "LC_CTYPE", "TERM", "TZ")
BASE_URL_VARIABLES = ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENROUTER_BASE_URL", "ARC_LLM_BASE_URL")
KEY_VARIABLES = ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ARC_LLM_API_KEY")
TMP_PREFIX = "arc-hl-"
WRITABLE = ("state", "home")
"""What the learner can write that outlasts its host, under the attempt's system folder: everything a frozen test puts
back (its temporary directory is new each time the host starts). A frozen learner's own wait in ``HELD`` beside them
(:meth:`HostedLearner.freeze`)."""


class HostedLearnerError(RuntimeError):
    """The submitted learner raised, broke the protocol, stopped answering or exited: the run stops here."""


def _inherited() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k in INHERITED_ENV or k.startswith("LC_")}


def _tail(path: Path, chars: int = 4000) -> str:
    try:
        return path.read_text(errors="replace")[-chars:]
    except OSError:
        return ""


class ServedLearner(BaseLearner):
    """A learner in another process, its calls and replies JSON lines over that process's stdin and stdout, as
    :mod:`learner_host` speaks them, one reply bounded by :data:`MAX_REPLY`.

    One user at a time (``concurrent_users`` stays False): the runner makes one call at a time. ``timeout_s`` bounds
    each call and ``startup_timeout_s`` the import and construction; past either the run stops."""

    concurrent_users = False

    def __init__(self, timeout_s: float = 3600.0, startup_timeout_s: float = 600.0, close_timeout_s: float = 60.0):
        self.timeout_s = timeout_s
        self.startup_timeout_s = startup_timeout_s
        self.close_timeout_s = close_timeout_s
        self.workdir: Path | None = None
        self.tracker: Any = None
        self.hello: dict[str, Any] = {}
        self._round: int | None = None   # the round the runner has begun and not yet ended
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self._log = None

    @property
    def log_path(self) -> Path:
        """The host's stderr: the learner's prints and logs, and any traceback."""
        return self.workdir / "learner.log"

    def _serve_from(self, argv: list[str], **popen: Any) -> None:
        """Start the far side and wait for its hello."""
        self._replies = queue.Queue(maxsize=8)  # a flood of lines waits in its pipe, not in this process's memory
        self._proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log, text=True,
                                      encoding="utf-8", errors="replace", bufsize=1, start_new_session=True, **popen)
        threading.Thread(target=self._pump, args=(self._proc, self._replies), daemon=True,
                         name="served-learner-out").start()
        self.hello = self._receive("start", self.startup_timeout_s, startup=True)

    @staticmethod
    def _pump(proc: subprocess.Popen, replies: queue.Queue) -> None:
        assert proc.stdout is not None
        while line := proc.stdout.readline(MAX_REPLY):
            replies.put(line)
        replies.put(None)

    def _note(self, message: str) -> None:
        """Into learner.log beside the host's own output, marked as the adapter's."""
        log.info(message)
        if self._log is not None:
            self._log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} [hosted] {message}\n")
            self._log.flush()

    def _fail(self, message: str) -> HostedLearnerError:
        self._kill()
        tail = _tail(self.log_path) if self.workdir else ""
        return HostedLearnerError(f"{message}\n--- the end of {self.log_path} ---\n{tail}" if tail else message)

    def _receive(self, what: str, timeout_s: float, startup: bool = False) -> Any:
        deadline = time.monotonic() + timeout_s
        noise = 0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._fail(f"the learner did not answer {what} within {timeout_s:g}s")
            try:
                line = self._replies.get(timeout=remaining)
            except queue.Empty:
                continue
            if line is None:
                try:
                    code = self._proc.wait(timeout=5) if self._proc is not None else None
                except subprocess.TimeoutExpired:
                    code = None  # it closed the channel and lives on
                raise self._fail(f"the learner's host exited (code {code}) during {what}")
            if len(line) >= MAX_REPLY:
                raise self._fail(f"the learner sent a line of {MAX_REPLY} characters or more during {what}")
            try:
                reply = json.loads(line)
            except (ValueError, RecursionError):
                reply = None
            if not isinstance(reply, dict) or "ok" not in reply:
                noise += 1
                if startup and noise <= MAX_STARTUP_LINES:  # what the interpreter's start-up (a .pth file) printed
                    self._note(f"before the host started: {line.rstrip()[:500]}")
                    continue
                raise self._fail(f"the learner's host broke the protocol during {what}: {line[:300]!r}")
            if not reply["ok"]:
                raise HostedLearnerError(f"the learner raised in {what}: {reply.get('error')}\n"
                                         f"{reply.get('traceback', '')}")
            return reply.get("result")

    def _call(self, op: str, timeout_s: float | None = None, **fields: Any) -> Any:
        if self._proc is None:
            raise HostedLearnerError(f"the learner is not running ({op})")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(json.dumps({"op": op, **fields}, default=str) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise self._fail(f"the learner's host is gone ({op}: {error})") from error
        return self._receive(op, self.timeout_s if timeout_s is None else timeout_s)

    def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:  # the whole session: the host and anything the learner started
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait(timeout=30)

    # -- the learner interface -----------------------------------------------

    def begin_round(self, n_instances: int) -> None:
        self._round = int(n_instances)
        self._call("begin_round", n=self._round)

    def end_round(self) -> None:
        self._call("end_round")
        self._round = None

    def act(self, obs: LearnerObservation | Arc3LearnerObservation):
        return decode_action(self._call("act", obs=encode_observation(obs)), obs)

    def observe(self, obs: LearnerObservation | Arc3LearnerObservation, action, feedback) -> None:
        self._call("observe", obs=encode_observation(obs), action=encode_action(action),
                   feedback=encode_feedback(feedback))

    def resources(self) -> dict[str, float]:
        """The learner's own counters, none under the proxy's names, then the proxy's: the model calls are measured, not
        reported."""
        reported = self._call("resources")
        if not isinstance(reported, dict):
            raise self._fail(f"resources() returned {reported!r}, not a dict")
        numbers = [(str(k), float(v)) for k, v in reported.items()
                   if isinstance(v, int | float) and math.isfinite(v) and not str(k).startswith("llm_")]
        out = dict(numbers[:MAX_RESOURCES])
        usage = getattr(self.tracker, "usage", None)
        if usage is not None:
            out.update(usage.as_resources())
        return out


class HostedLearner(ServedLearner):
    """A submission's learner, ``package.module:Class``, run under the sandbox with the interpreter ``python`` of the
    virtual environment it was installed into, ``code`` its checkout (read only), ``runtime`` the paths the interpreter's
    Python comes from when it is not one of the system's, as :func:`install_submission` found them.

    ``reset`` is forwarded like every other call: wiping its own state when asked is the learner's part of the
    benchmark's contract, and the state directory is kept for the whole run. Once the learner is frozen (:meth:`freeze`),
    a reset puts its files back as its stream left them instead, and starts it again.
    """

    def __init__(self, python: str, learner: str, code: str | None = None, runtime: list[str] | None = None,
                 sandbox: str = "auto", timeout_s: float = 3600.0, startup_timeout_s: float = 600.0,
                 close_timeout_s: float = 60.0):
        super().__init__(timeout_s, startup_timeout_s, close_timeout_s)
        self.python = Path(python).expanduser().absolute()  # the venv's own path, not the interpreter it links to
        self.learner = learner
        self.code = Path(code).expanduser().absolute() if code else None
        self.runtime = [Path(path).expanduser().absolute() for path in runtime or ()]
        self.sandbox = sandbox
        self.base_url: str | None = None
        self.frozen: Path | None = None  # the learner's files as its stream left them, put back before every instance
        self._backend: str | None = None
        self._tmp_dir: Path | None = None
        self._bridge: subprocess.Popen | None = None

    # -- paths ---------------------------------------------------------------

    @property
    def venv(self) -> Path:
        return self.python.parent.parent

    @property
    def state_dir(self) -> Path:
        """The learner's persistent memory and working directory, kept for the whole run (while it is frozen, a copy
        of it as the stream left it)."""
        return self.workdir / "state"

    @property
    def home_dir(self) -> Path:
        return self.workdir / "home"

    # -- the sandbox ---------------------------------------------------------

    def child_env(self) -> dict[str, str]:
        """An allowlist: nothing of this process's environment but the locale, so no key and no path leaks in."""
        env = _inherited()
        env.update({
            "HOME": str(self.home_dir),
            "TMPDIR": str(self._tmp_dir),
            "PATH": f"{self.venv / 'bin'}:/usr/bin:/bin",
            "VIRTUAL_ENV": str(self.venv),
            "ARC_STATE_DIR": str(self.state_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            CONFINED: "1",
        })
        if self.base_url:
            env.update(dict.fromkeys(BASE_URL_VARIABLES, self.base_url))
            env.update(dict.fromkeys(KEY_VARIABLES, PLACEHOLDER_KEY))
        return env

    def sandbox_spec(self) -> SandboxSpec:
        reads = [self.workdir / HOST.name, self.workdir / "netbridge.py", *self.runtime]
        for path in self.runtime:
            reads += package_manager_roots(path)
        port = proxy_port(self.base_url) if self.base_url else None
        temp_dirs = darwin_user_dirs()
        return SandboxSpec(read_paths=reads, write_paths=[self.state_dir, *temp_dirs], tmp_dir=self._tmp_dir,
                           home=self.home_dir, mask_paths=default_mask_paths(),
                           opaque_paths=[self.venv, *([self.code] if self.code else [])],
                           loopback_ports=[port] if port else [], unix_socket_dirs=temp_dirs)

    # -- the host process ----------------------------------------------------

    def attach(self, workdir: Path, environ: dict[str, str], tracker: Any = None, frozen: bool = False) -> None:
        """Start the host: the learner is imported and built before the run begins, so a submission that cannot
        start fails at once. Of ``environ`` (the proxy's variables) only the proxy's address is used. Files a frozen
        test stopped hard left aside (a run taken up again with ``--resume``) get their place back first, so the
        learner goes on from the state its stream left, unless it is ``frozen``, a host started anew in the middle of a
        frozen test (:mod:`~continual_arc_baselines.systems.remote`): then it starts from a copy of them."""
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.base_url = environ.get("ARC_LLM_BASE_URL") or environ.get("OPENAI_BASE_URL")
        self.tracker = tracker
        self._log = self.log_path.open("a")
        try:
            if (self.workdir / HELD).is_dir():
                self.frozen = self.workdir / HELD
                if frozen:
                    self._start_frozen()
                    return
                self._give_back()
                self._note("its stream's files, which a frozen test stopped hard left aside, have their place back")
            self._start()
        except BaseException:
            self.close()  # the run never starts, so nothing else ends the host, its bridge and its directory
            raise

    def _start(self) -> None:
        if not self.python.exists():
            raise HostedLearnerError(f"no interpreter at {self.python}: install the submission first "
                                     "(arc-baselines install-submission)")
        self._backend = resolve_backend(self.sandbox)
        self._tmp_dir = Path(tempfile.mkdtemp(prefix=TMP_PREFIX, dir="/tmp" if os.path.isdir("/tmp") else None))
        for path in (self.state_dir, self.home_dir):
            path.mkdir(parents=True, exist_ok=True)
        host = self.workdir / HOST.name
        shutil.copyfile(HOST, host)
        inner = [str(self.python), "-I", str(host), self.learner, *(["--code", str(self.code)] if self.code else [])]
        if self._backend == "bwrap":
            inner, self._bridge = start_bridge(inner, proxy_port(self.base_url) if self.base_url else None,
                                               self._tmp_dir / "proxy.sock", self.workdir / "netbridge.py")
        argv = wrap_command(self._backend, inner, self.sandbox_spec(), self.workdir / "sandbox.sb")
        self._serve_from(argv, cwd=self.state_dir, env=self.child_env())
        if self.hello.get("concurrent_users"):
            self._note("the learner says it serves users concurrently; it is served one call at a time")
        self._note(f"started {self.hello.get('learner')} (python {self.hello.get('python')}, continual_arc "
                   f"{self.hello.get('continual_arc')}, sandbox {self._backend})")

    def _kill(self) -> None:
        super()._kill()
        bridge, self._bridge = self._bridge, None
        if bridge is not None:
            bridge.terminate()
            bridge.wait(timeout=30)

    def _end_host(self) -> None:
        """Ask the learner to close, then end the host's whole session, its bridge and its temporary directory. Never
        raises."""
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._call("close", timeout_s=self.close_timeout_s)
                self._proc.wait(timeout=self.close_timeout_s)
        except (HostedLearnerError, subprocess.TimeoutExpired) as error:
            self._note(f"did not close cleanly: {error}")
        finally:
            self._kill()
            if self._tmp_dir is not None:
                shutil.rmtree(self._tmp_dir, ignore_errors=True)
                self._tmp_dir = None

    # -- a frozen test ---------------------------------------------------------

    def freeze(self) -> None:
        """Play every instance from here on from the learner as it is now, at the end of its stream. The host is ended,
        the learner's state directory and HOME are set aside whole (:data:`HELD`), and a copy of each is put in its
        place, at the same path, before the host starts again and builds the learner from them. Every :meth:`reset`
        does the same in place of the learner's own. Under bubblewrap nothing the learner keeps in its files or its
        processes reaches the next instance, whatever its code: everything it could write is put back, and every
        process it started ends with the sandbox's pid namespace. Seatbelt (macOS) also lets it write the user's
        temporary folders, which are not put back, and a process that leaves the host's session outlives it. What a
        learner keeps only in its own process it loses here, and a new host builds it with what its files hold."""
        self._end_host()
        held = self.workdir / HELD
        held.mkdir()
        for name in WRITABLE:
            (self.workdir / name).rename(held / name)
        self.frozen = held
        self._note("frozen: every instance from here on starts from a copy of the files its stream left")
        self._start_frozen()

    def _start_frozen(self) -> None:
        for name in WRITABLE:
            put_back(self.frozen / name, self.workdir / name)
        self._start()
        if self._round is not None:
            # the runner begins a round before it resets for the round's instance, so the new learner joins it here
            self._call("begin_round", n=self._round)

    def _give_back(self) -> None:
        """The stream's files back at their own paths from where :meth:`freeze` set them aside, the copies deleted. A
        folder that is not there was never set aside (a run stopped between the two), so its own is the stream's."""
        for name in WRITABLE:
            if not (self.frozen / name).exists():
                continue
            if (self.workdir / name).exists():
                delete(self.workdir / name)
            (self.frozen / name).rename(self.workdir / name)
        self.frozen.rmdir()
        self.frozen = None

    def reset(self) -> None:
        if self.frozen is None:
            self._call("reset")
            return
        self._end_host()
        self._start_frozen()

    def close(self) -> None:
        """End the host, whose errors are noted rather than raised, since this runs while the run unwinds and an error
        from it would hide the one that ended the run; then, after a frozen test, give the stream's files their place
        back."""
        try:
            self._end_host()
            if self.frozen is not None:
                self._give_back()
        finally:
            self._close_log()

    def abandon(self) -> None:
        """The connection that served this host has gone. Mid frozen test the arena is about to reboot and the stream's
        files stay aside for the next instance, so the host is only ended; otherwise this is :meth:`close`."""
        if self.frozen is None:
            self.close()
            return
        try:
            self._kill()
        finally:
            self._close_log()

    def _close_log(self) -> None:
        if self._log is not None:
            self._log.close()
            self._log = None

    def describe(self) -> dict[str, Any]:
        record = record_of(self.venv)
        return {"system": "hosted", "learner": self.learner, "python": str(self.python),
                "code": str(self.code) if self.code else None, "sandbox": self._backend or self.sandbox,
                "timeout_s": self.timeout_s, "host": dict(self.hello),
                "submission": json.loads(record.read_text()) if record.is_file() else None}


# -- the wire format ---------------------------------------------------------
#
# Every message carries its class's name under ``type`` beside its fields, as learner_host decodes it. An observation is
# its track's own (``LearnerObservation`` of a grid, ``Arc3LearnerObservation`` of an ARC-AGI-3 game), and a learner
# answers it with one of its track's actions: a grid takes ``Submit``, a game ``Act`` and ``ResetLevel``, and both take
# ``RequestDemos`` and ``GiveUp``.

GRID_ACTIONS = ("Submit", "RequestDemos", "GiveUp")
GAME_ACTIONS = ("Act", "ResetLevel", "RequestDemos", "GiveUp")
BARE_ACTIONS = {cls.__name__: cls for cls in (RequestDemos, GiveUp, ResetLevel)}
"""The actions that carry no field."""
FEEDBACK_TYPES = (SubmitFeedback, DemosFeedback, GiveUpFeedback, ActFeedback, ResetFeedback, DemonstrationFeedback)


def encode_observation(obs: LearnerObservation | Arc3LearnerObservation) -> dict:
    return {"type": type(obs).__name__, **obs.to_json()}


def encode_action(action) -> dict:
    """An action the environment took, which the learner observes."""
    if isinstance(action, Submit):
        return {"type": "Submit", "grid": action.grid}
    if isinstance(action, Act):
        return {"type": "Act", "action": action.action, "x": action.x, "y": action.y}
    if isinstance(action, tuple(BARE_ACTIONS.values())):
        return {"type": type(action).__name__}
    raise TypeError(f"unknown action {action!r}")


def _integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def decode_action(data: Any, obs: LearnerObservation | Arc3LearnerObservation):
    """The learner's action on ``obs``: one of its track's, of which nothing but the type and its own fields is read. A
    Submit's grid is passed on as it came, since the environment counts one that is not a grid as an invalid submission.
    An Act's action is an integer and each of its coordinates an integer or null, since the environment counts any
    integer it cannot play as an invalid action, and anything else would stop it."""
    game = isinstance(obs, Arc3LearnerObservation)
    kinds = GAME_ACTIONS if game else GRID_ACTIONS
    kind = data.get("type") if isinstance(data, dict) else None
    if kind in kinds:
        if kind == "Submit":
            return Submit(grid=data.get("grid"))
        if kind != "Act":
            return BARE_ACTIONS[kind]()
        action, x, y = (data.get(name) for name in ("action", "x", "y"))
        if _integer(action) and all(v is None or _integer(v) for v in (x, y)):
            return Act(action, x, y)
    raise HostedLearnerError(f"act returned {repr(data)[:300]}, not one of the actions on {'a game' if game else 'a grid'} "
                             f"({', '.join(kinds)}) with its fields")


def encode_feedback(feedback) -> dict:
    """What the environment said of an action: a demonstration of a game as its own JSON (``Demonstration.to_json``)."""
    if not isinstance(feedback, FEEDBACK_TYPES):
        raise TypeError(f"unknown feedback {feedback!r}")
    out = {"type": type(feedback).__name__, **dataclasses.asdict(feedback)}
    if isinstance(feedback, DemonstrationFeedback) and feedback.demonstration is not None:
        out["demonstration"] = feedback.demonstration.to_json()
    return out


# -- installing a submission -------------------------------------------------


def _site_packages(venv: Path) -> list[Path]:
    return sorted(venv.glob("lib/python*/site-packages"))


def brings_benchmark(venv: Path) -> bool:
    """Whether the environment has a ``continual-arc`` of its own, read from its files: running its interpreter
    outside the sandbox would run the submission's start-up code (``.pth`` files)."""
    return any(any(sp.glob("continual_arc-*.dist-info")) for sp in _site_packages(venv))


def _empty_dir(path: Path, what: str) -> None:
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"the {what} {path} exists and is not empty: install into a new directory")
    path.mkdir(parents=True, exist_ok=True)


def record_of(venv: Path) -> Path:
    """What :func:`install_submission` installed, beside the virtual environment, out of its build code's reach."""
    return venv.with_name(venv.name + ".submission.json")


def install_submission(repo: str, commit: str, venv: str | Path, *, checkout: str | Path | None = None,
                       cache: str | Path | None = None, python: str = "3.12", sandbox: str = "auto",
                       benchmark: str = PUBLIC_BENCHMARK, read_paths: list[str | Path] = (),
                       https_proxy: str | None = None, timeout_s: float = 3600.0) -> dict[str, Any]:
    """Make a new virtual environment at ``venv`` with uv and install ``git+<repo>@<commit>`` into it, then the
    public benchmark (``benchmark``) when the submission did not bring ``continual-arc`` itself, and clone the
    repository at ``commit`` into ``checkout`` when one is given (the run's ``code``).

    Installing runs the package's build code, so every step runs under the sandbox: it reads the operating system, uv, the
    interpreter (``uv python find --system <python>``) and ``read_paths`` (a local repository, for tests), and writes only
    ``venv``, ``checkout`` and a cache of the submission's own (``<venv>-cache`` by default, holding uv's cache and HOME), so one
    submission's build cannot plant anything in another's. Its network is a bridge to the HTTPS proxy at ``https_proxy``,
    a :class:`~continual_arc_baselines.systems.connect_filter.ConnectFilter` that the caller holds, or one started here.
    Returns the learner kwargs' paths and what went in. This process's own files, the install's log and :func:`record_of`,
    lie beside the environment in folders the sandbox does not show, since a link the build plants in anything it can
    write would send this process's writes wherever it names."""
    backend = resolve_backend(sandbox)
    venv = Path(venv).expanduser().absolute()
    cache = Path(cache).expanduser().absolute() if cache else venv.with_name(venv.name + "-cache")
    checkout = Path(checkout).expanduser().absolute() if checkout else None
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is not on PATH")
    found = subprocess.run([uv, "python", "find", "--system", python], capture_output=True, text=True, check=False)
    if found.returncode != 0 or not found.stdout.strip():
        raise RuntimeError(f"no Python {python} for the submission ({found.stderr.strip()}); "
                           f"`uv python install {python}` first")
    interpreter = Path(found.stdout.strip())
    _empty_dir(venv, "virtual environment")
    if checkout:
        _empty_dir(checkout, "checkout")
    home = cache / "home"
    for path in (cache / "uv", home):
        path.mkdir(parents=True, exist_ok=True)
    own = venv.with_name(venv.name + ".install")  # this process's: the log, the sandbox's profile and the bridge's script
    own.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="arc-in-", dir="/tmp" if os.path.isdir("/tmp") else None))

    reads: list[Path] = [Path(uv).resolve(), interpreter, *[Path(p).expanduser().absolute() for p in read_paths]]
    root = install_root(str(interpreter))
    if root:
        reads.append(root)
    for path in list(reads):
        reads += package_manager_roots(path)
    temp_dirs = darwin_user_dirs()
    # HOME is named only in the environment, since it lies in what the build can write
    spec = SandboxSpec(read_paths=reads, write_paths=[venv, cache, *([checkout] if checkout else []), *temp_dirs],
                       tmp_dir=tmp, mask_paths=default_mask_paths(), unix_socket_dirs=temp_dirs)
    transcript = own / "install.log"
    env = {**_inherited(), "PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(home), "TMPDIR": str(tmp),
           "UV_CACHE_DIR": str(cache / "uv"), "UV_PYTHON_DOWNLOADS": "never", "UV_NO_CONFIG": "1",
           "GIT_TERMINAL_PROMPT": "0", CONFINED: "1"}

    def note(line: str) -> None:
        with transcript.open("a") as fh:
            fh.write(f"[filter] {line}\n")

    def step(argv: list[str], proxy: str) -> None:
        inner, bridge = list(argv), None
        if backend == "bwrap":
            inner, bridge = start_bridge(argv, proxy_port(proxy), tmp / "proxy.sock", own / "netbridge.py")
        wrapped = wrap_command(backend, inner, spec, own / "install.sb", share_net=backend != "bwrap")
        proxied = {name: proxy for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy")}
        try:
            out = subprocess.run(wrapped, env={**env, **proxied}, cwd=cache, capture_output=True, text=True,
                                 errors="replace", timeout=timeout_s, check=False)
        finally:
            if bridge is not None:
                bridge.terminate()
                bridge.wait(timeout=30)
        with transcript.open("a") as fh:
            fh.write(f"$ {' '.join(argv)}\n{out.stdout}{out.stderr}[exit {out.returncode}]\n")
        if out.returncode != 0:
            raise RuntimeError(f"{' '.join(argv)} failed (exit {out.returncode}):\n{(out.stderr or out.stdout)[-3000:]}")

    target = venv / "bin" / "python"
    requirement = f"git+{repo}@{commit}"

    def install(proxy: str) -> bool:
        step([uv, "venv", "--quiet", "--no-project", "--python", str(interpreter), str(venv)], proxy)
        step([uv, "pip", "install", "--python", str(target), requirement], proxy)
        brought = brings_benchmark(venv)
        if not brought:
            step([uv, "pip", "install", "--python", str(target), benchmark], proxy)
        if checkout:
            step(["git", "clone", "--quiet", repo, str(checkout)], proxy)
            step(["git", "-C", str(checkout), "checkout", "--quiet", "--detach", commit], proxy)
        return brought

    try:
        if https_proxy:
            brought = install(https_proxy)
        else:
            with ConnectFilter(note=note) as local:
                brought = install(local.url)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    record = {"repo": repo, "commit": commit, "requirement": requirement,
              "benchmark": "the submission's own" if brought else benchmark, "python": str(interpreter),
              "runtime": [str(root), str(interpreter)] if root else [str(interpreter)], "sandbox": backend,
              "installed_at": datetime.now(UTC).isoformat(timespec="seconds")}
    record_of(venv).write_text(json.dumps(record, indent=1) + "\n")
    return {"python": str(target), "code": str(checkout) if checkout else None, "venv": str(venv),
            "cache": str(cache), **{k: record[k] for k in ("runtime", "repo", "commit", "benchmark")}}
