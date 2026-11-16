"""Stand-ins the tests share: a tokenizer whose chat template shows everything it is given, a played game as a
learner hands it to the instance hook, with its run's records, a frozen test's users as the benchmark's runner serves
them, and a node's clone of the scripts."""
from __future__ import annotations

import gzip
import json
import subprocess
import types
from pathlib import Path


class CharTokenizer:
    """One id per character. The chat template writes each message's role and content (a string as it is, anything
    else as JSON), its reasoning and tool calls, then the tools and any template switches, so a change to anything the
    template is given changes the ids."""

    eos_token_id = 3   # never rendered

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False, **kwargs):
        assert tokenize is False
        lines = []
        for m in messages:
            content = m.get("content")
            lines.append(f"<{m['role']}> " + (content if isinstance(content, str) else json.dumps(content)))
            if m.get("reasoning_content") is not None:
                lines.append("<think> " + m["reasoning_content"])
            for tc in m.get("tool_calls") or []:
                lines.append("<call> " + json.dumps(tc["function"], sort_keys=True))
        if tools:
            lines.append("<tools> " + json.dumps(tools, sort_keys=True))
        if kwargs:
            lines.append("<kwargs> " + json.dumps(kwargs, sort_keys=True))
        if add_generation_prompt:
            lines.append("<assistant>")
        return "\n".join(lines) + "\n"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


TOK = CharTokenizer()


def call(call_id, prompt, reply, finish="stop", started=0.0, messages=None):
    """A captured call as the learner keeps it: the request sent and the server's answer with its token ids."""
    return types.SimpleNamespace(call_id=call_id, turn=0, phase="play", status="done", finish_reason=finish,
                                 started=started,
                                 request={"messages": messages or [{"role": "user", "content": "go"}]},
                                 response={"prompt_token_ids": prompt, "choices": [
                                     {"token_ids": reply, "finish_reason": finish}]})


def trace(user, round_, task_id, solved, seed, calls=None):
    """One instance's game as the learner hands it to its instance hook."""
    return types.SimpleNamespace(user=user, slot=0, round=round_, task_id=task_id, task_label=task_id, phase="stream",
                                 solved=solved, turns=1, calls=calls or [call(0, [1000 + user, round_], [10 + seed])])


RECORD_DEFAULTS = {"step": 0, "phase": "stream", "source": "fake", "exposure_index": 0, "gap": None,
                   "instance_seed": 0, "reset_before": False, "wrong_attempts": 0, "invalid_submissions": 0,
                   "demo_requests": 0, "gave_up": False}


def write_records(path: Path, entries) -> None:
    """A records.jsonl with every field the benchmark's records need, a line per (user, round, task_id, solved,
    cost)."""
    lines = [json.dumps({**RECORD_DEFAULTS, "user": u, "round": r, "task_id": t, "solved": s, "cost": c})
             for u, r, t, s, c in entries]
    path.write_text("\n".join(lines) + "\n")


def read_game(path: Path) -> dict:
    with gzip.open(path, "rt") as fp:
        return json.load(fp)


def write_game(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fp:
        json.dump(doc, fp)


def flat_game(t) -> dict:
    """``t`` as a captured game: the shape the instance hook saves."""
    return {"user": t.user, "slot": t.slot, "round": t.round, "task_id": t.task_id, "task_label": t.task_label,
            "phase": t.phase, "solved": t.solved, "turns": t.turns,
            "calls": [{"call_id": c.call_id, "turn": c.turn, "phase": c.phase, "status": c.status,
                       "finish_reason": c.finish_reason, "request": c.request, "response": c.response}
                      for c in t.calls]}


class TurnTokenizer(CharTokenizer):
    """A chat template laid out as a model's is: the tools and switches first, then each message as a turn, an assistant
    turn's reasoning in a think block and its end of turn (id 3) after it; the generation prompt opens an assistant
    turn and its thinking, so a reply rendered after it continues the prompt's text."""

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False, **kwargs):
        assert tokenize is False
        text = ""
        if tools:
            text += "<tools>" + json.dumps(tools, sort_keys=True) + "\n"
        if kwargs:
            text += "<kwargs>" + json.dumps(kwargs, sort_keys=True) + "\n"
        for m in messages:
            content = m.get("content")
            content = content if isinstance(content, str) else json.dumps(content)
            if m["role"] == "assistant":
                calls = "".join("<call>" + json.dumps(tc["function"], sort_keys=True) for tc in m.get("tool_calls") or [])
                text += f"<assistant>\n<think>\n{m.get('reasoning_content') or ''}\n</think>\n{content}{calls}\x03\n"
            else:
                text += f"<{m['role']}>\n{content}\n"
        if add_generation_prompt:
            text += "<assistant>\n<think>\n"
        return text


def checkout(root: Path) -> None:
    """``root``, which holds the package's scripts and source (``training/``) as a node's clone does, a git checkout of
    them: every run a script launches keeps its code's commit."""
    git = ["git", "-C", str(root), "-c", "user.email=t@t", "-c", "user.name=t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "training"], check=True)
    subprocess.run([*git, "commit", "-qm", "the node's code"], check=True)


class ForkedUsers:
    """A frozen test's users as the runner serves them at once: each instance a demonstration, then the answer it
    shows, and as it ends a note in the user's state of what it played; users in ``fail`` raise at their first action
    once every other user's instance has ended, as a play the node's stop cuts short."""

    concurrent_users = True

    def __init__(self, play_dir: Path, users: int, fail=()) -> None:
        import threading
        self.play_dir, self.fail, self.users = play_dir, set(fail), users
        self.demos, self.calls, self.ended, self.acted = {}, {}, 0, set()
        self.lock, self.others = threading.Lock(), threading.Event()

    def begin_round(self, n):
        pass

    def end_round(self):
        pass

    def reset(self):
        pass

    def act(self, obs):
        from continual_arc.env import RequestDemos, Submit
        with self.lock:
            self.calls[obs.user] = self.calls.get(obs.user, 0) + 1
            self.acted.add(obs.user)
        if obs.user in self.fail:
            self.others.wait(30)
            raise RuntimeError("the node stopped")
        d = self.demos.get(obs.user)
        if d is None:
            return RequestDemos()
        shift = (d.output[0][0] - d.input[0][0]) % 10
        return Submit([[(c + shift) % 10 for c in row] for row in obs.input])

    def observe(self, obs, action, feedback):
        from continual_arc.env import DemosFeedback
        from continual_arc_baselines.tracking import instance_ended
        if isinstance(feedback, DemosFeedback):
            self.demos[obs.user] = feedback.pairs[0]
        if instance_ended(feedback):
            state = self.play_dir / "users" / f"u{obs.user:03d}" / "system" / "state"
            (state / "played").write_text(obs.task_id)
            with self.lock:
                self.ended += 1
                if self.ended == self.users - len(self.fail):
                    self.others.set()

    def user_resources(self, user):
        return {"calls": float(self.calls.get(user, 0))}

    def resources(self):
        return {"calls": float(sum(self.calls.values()))}
