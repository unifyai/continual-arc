"""The player (continual_arc_training.play): the schedule every row plays, cut a round at a time; a round's games saved
apart from every other round's; and play_round against the baselines' real learners, with a stand-in harness and model
server."""
from __future__ import annotations

import inspect
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from continual_arc_baselines.systems.base import SystemLearner
from fakes import call, read_game, trace

from continual_arc_training import play

# -- the schedule ------------------------------------------------------------------------------------------------


def test_a_cut_is_a_literal_prefix_of_the_whole_schedule():
    full = play.full_schedule()
    assert full.users == play.SCHEDULE_USERS == 100
    cut2 = play.schedule_cut(1)   # rounds 0 and 1
    assert cut2.rounds()[-1] == 1 and cut2.slots == full.slots[:len(cut2)]   # a true slice, not a shorter rebuild
    cut1 = play.schedule_cut(0)
    assert cut1.rounds()[-1] == 0 and cut1.slots == cut2.slots[:len(cut1)]   # cuts nest


def test_the_schedule_is_arc1_small_served_to_that_many_users():
    from continual_arc.configs import CANONICAL_USERS

    assert play.served().users == 100 and play.served().name == "arc1-small"
    eight = play.served(8)
    assert eight.users == 8 and eight.name == "arc1-small" and 8 in CANONICAL_USERS


def test_a_smaller_user_count_gives_exactly_a_larger_ones_first_users_round_by_round():
    """Every field a user's own stream or epilogue order depends on (the seed and the user's own index, the dataset) is
    unaffected by the number of users, so runs at 1, 2 and 4 users play the first users of one schedule."""
    full, small = play.full_schedule(), play.full_schedule(users=4)
    assert small.users == 4

    def by_round(schedule):
        out: dict[int, list] = {}
        for slot, r in zip(schedule.slots, schedule.rounds(), strict=True):
            out.setdefault(r, []).append(slot)
        return out

    big, little = by_round(full), by_round(small)
    assert len(little) == len(big)
    assert all(slots == big[r][:4] for r, slots in little.items())
    cut = play.schedule_cut(2, users=4)
    assert cut.users == 4 and cut.rounds()[-1] == 2 and cut.slots == small.slots[:len(cut)]


def test_a_cut_runs_on_past_the_stream_into_the_epilogue():
    """Rounds 0-199 are the stream and 200-299 the epilogue, in which every user meets each of the 100 held-out tasks
    once."""
    full = play.full_schedule(users=2)
    rounds = full.rounds()
    assert rounds[-1] == 299
    phase = {r: s.phase for s, r in zip(full.slots, rounds, strict=True)}
    assert {phase[r] for r in range(200)} == {"stream"} and {phase[r] for r in range(200, 300)} == {"epilogue"}
    cut = play.schedule_cut(200, users=2)
    assert cut.slots == full.slots[:len(cut)] and [s.phase for s in cut.slots[-3:]] == ["stream", "epilogue",
                                                                                           "epilogue"]
    for u in range(2):
        held = [s.task_id for s in full.slots if s.user == u and s.phase == "epilogue"]
        assert len(held) == len(set(held)) == 100


def test_the_epilogue_alone_is_the_held_out_tasks_for_its_users():
    from continual_arc.configs import epilogue_for_users

    cut = play.epilogue_cut(0, epilogue_users=2, users=4)
    assert cut.slots == epilogue_for_users(play.served(4), play.SEED, 2).slots[:len(cut)]
    assert cut.rounds()[-1] == 0 and all(s.phase == "epilogue" for s in cut.slots)


def test_the_wide_epilogue_deals_the_epilogue_alones_instances_width_at_a_time():
    """Transfer's references meet users 0-3's 400 held-out instances 20 to a round: every instance of the 4-user
    epilogue alone once, in its own order, so their round count is 20, not 100."""
    narrow = play.epilogue_cut(99, epilogue_users=4, users=4)
    wide = play.wide_epilogue_cut(19, width=20, epilogue_users=4, users=4)
    assert wide.users == 20 and wide.rounds()[-1] == 19
    assert [(s.task_id, s.instance_seed, s.phase) for s in wide.slots] == \
        [(s.task_id, s.instance_seed, s.phase) for s in narrow.slots]
    assert [s.user for s in wide.slots[:21]] == list(range(20)) + [0] and not any(s.reset_before for s in wide.slots)
    assert play.wide_epilogue_cut(0, width=20, epilogue_users=4, users=4).slots == wide.slots[:20]
    with pytest.raises(ValueError, match="whole rounds"):
        play.wide_epilogue_cut(0, width=30, epilogue_users=4, users=4)


def test_one_reference_play_over_users_0_to_3_serves_every_run_at_1_2_and_4_users():
    """Transfer matches a run's epilogue to its references by task and instance seed. User u's held-out instances are
    the same in a run at any number of users, in the epilogue alone and in the benchmark's fresh reference schedule, so
    H0 and the bare model, each played once over users 0-3, serve the 1-, 2- and 4-user runs alike."""
    from continual_arc.configs import build_schedule

    def held(slots):
        out: dict[int, list] = {}
        for s in slots:
            if s.phase == "epilogue":
                out.setdefault(s.user, []).append((s.task_id, s.instance_seed))
        return out

    reference = held(play.epilogue_cut(99, epilogue_users=4, users=4).slots)
    assert sorted(reference) == [0, 1, 2, 3] and all(len(v) == 100 for v in reference.values())
    assert held(build_schedule(play.served(4), "fresh", play.SEED).slots) == reference
    for users in (1, 2, 4):
        assert held(play.full_schedule(users=users).slots) == {u: reference[u] for u in range(users)}


# -- a round's games ---------------------------------------------------------------------------------------------

def test_a_round_dir_is_four_digits_wide_and_distinct_per_round(tmp_path):
    assert play.round_dir(tmp_path, 0).name == "r0000" and play.round_dir(tmp_path, 9999).name == "r9999"
    assert play.round_dir(tmp_path, 1) == tmp_path / "rounds" / "r0001"


def test_each_rounds_games_are_saved_apart_in_the_order_their_calls_ended(tmp_path):
    """Every user plays every round under the same user id: a game saved by user id alone would have round 1's save
    overwrite round 0's."""
    hook = play.save_instance_hook(tmp_path)
    first = trace(0, 0, "A", True, seed=1, calls=[call("b", [1], [2], started=2.0), call("a", [1], [3], started=1.0)])
    hook(first)
    hook(trace(0, 1, "A", False, seed=2))
    r0 = read_game(play.round_dir(tmp_path, 0) / "captured" / "u000.json.gz")
    r1 = read_game(play.round_dir(tmp_path, 1) / "captured" / "u000.json.gz")
    assert (r0["solved"], r1["solved"]) == (True, False) and r0["task_id"] == "A" and r0["round"] == 0
    assert [c["call_id"] for c in r0["calls"]] == ["b", "a"]
    assert set(r0["calls"][0]) == set(play.CALL_FIELDS) and r1["calls"][0]["response"]["choices"][0]["token_ids"] == [12]
    assert not list((play.round_dir(tmp_path, 0) / "captured").glob("*.tmp"))


def test_a_value_json_has_no_form_for_is_saved_as_text_and_a_save_that_fails_costs_the_round_nothing_else(
        tmp_path, caplog):
    hook = play.save_instance_hook(tmp_path)
    odd = trace(0, 0, "A", True, seed=1)
    odd.calls[0].status = ...   # a harness's turn record can hold values JSON has no form for
    hook(odd)
    assert read_game(play.round_dir(tmp_path, 0) / "captured" / "u000.json.gz")["calls"][0]["status"] == "Ellipsis"
    blocked = play.round_dir(tmp_path, 0) / "captured" / "u001.json.gz"
    (blocked / "in-the-way").mkdir(parents=True)   # the save's rename cannot replace a folder that holds something
    hook(trace(1, 0, "A", True, seed=1))
    hook(trace(2, 0, "A", True, seed=1))
    assert "could not save instance of user 1, round 0" in caplog.text
    assert (play.round_dir(tmp_path, 0) / "captured" / "u002.json.gz").is_file()


# -- play_round against the baselines' learners -------------------------------------------------------------------

_SYSTEM_LEARNER_PARAMS = frozenset(inspect.signature(SystemLearner.__init__).parameters) - {"self"}


class _StubSystem(SystemLearner):
    """A SystemLearner that answers every observation with unparseable text, so the learner's own reprompt loop ends
    every instance in a forced give-up with no model or harness process, and that writes a file tagged ``round_tag``
    into its state on every call, so a wipe (or none) shows. It takes PrimeAgentLearner's own constructor arguments and
    passes on those SystemLearner names."""
    name = "stub"
    round_tag = 0

    def __init__(self, **kwargs):
        super().__init__(**{k: v for k, v in kwargs.items() if k in _SYSTEM_LEARNER_PARAMS})

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def new_session(self) -> None:
        pass

    def ask(self, message, *, reprompt=False):
        state = self.workdir / "state"
        state.mkdir(parents=True, exist_ok=True)
        (state / f"round{self.round_tag}.md").write_text("memory")
        return "I don't know what to do."

    def wipe_state(self) -> None:
        shutil.rmtree(self.workdir / "state", ignore_errors=True)


def _schedule(users=2, rounds=3, task_id="fake000"):
    """A collective schedule over the benchmark's data-free FakeSource: one instance a user a round, all one task."""
    from continual_arc.schedule import Schedule, Slot

    slots = [Slot(task_id=task_id, instance_seed=r * 1000 + u, phase="stream", user=u)
             for r in range(rounds) for u in range(users)]
    return Schedule(name="wipe-test", seed=0, slots=slots, sources={task_id: "fake"}, users=users)


def _registry(n_tasks=1):
    from continual_arc.tasks import TaskRegistry
    from continual_arc.testing import FakeSource
    return TaskRegistry({"fake": FakeSource(n_tasks=n_tasks)})


def _stream_then_epilogue(users=2):
    """Shaped as the real schedule ends: two stream rounds over fake000, then two epilogue rounds in which each user
    meets the held-out fake001 and fake002 once each, in an order of its own."""
    from continual_arc.schedule import Schedule, Slot

    slots = [Slot(task_id="fake000", instance_seed=r * 1000 + u, phase="stream", user=u)
             for r in range(2) for u in range(users)]
    slots += [Slot(task_id=f"fake00{1 + (r + u) % 2}", instance_seed=r * 1000 + u, phase="epilogue", user=u)
              for r in range(2, 4) for u in range(users)]
    return Schedule(name="epilogue-test", seed=0, slots=slots, sources={f"fake00{i}": "fake" for i in range(3)},
                    users=users)


def _state_files(run_dir, user):
    state = run_dir / "users" / f"u{user:03d}" / "system" / "state"
    return sorted(p.name for p in state.iterdir()) if state.is_dir() else []


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", _StubSystem)
    return _StubSystem


@pytest.mark.parametrize("row", ["h0", "dagger", "pg"])
def test_h0_is_wiped_before_every_round_and_a_trained_row_keeps_its_memory(tmp_path, stub, row):
    """A call plays one instance a user, so the harness's own in-process wipe never fires across calls: three calls
    over one run folder, each a new learner as a new process would build, only the state on disk carried between
    them."""
    full, registry, run_dir = _schedule(), _registry(), tmp_path / row
    for r in range(3):
        stub.round_tag = r
        play.play_round(row, run_dir, full.first_rounds(r + 1), base_url="http://127.0.0.1:1/v1", served_name="base",
                        registry=registry)
        want = [f"round{r}.md"] if row == "h0" else [f"round{i}.md" for i in range(r + 1)]
        assert all(_state_files(run_dir, u) == want for u in range(2))
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    assert [r["round"] for r in records] == [0, 0, 1, 1, 2, 2]


def test_a_trained_row_plays_on_into_the_epilogue_with_its_memory_kept(tmp_path, stub):
    full, registry, run_dir = _stream_then_epilogue(), _registry(3), tmp_path / "dagger"
    for r in range(4):
        stub.round_tag = r
        play.play_round("dagger", run_dir, full.first_rounds(r + 1), base_url="http://127.0.0.1:1/v1",
                        served_name="base", registry=registry)
        assert all(_state_files(run_dir, u) == [f"round{i}.md" for i in range(r + 1)] for u in range(2))
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in records] == ["stream"] * 4 + ["epilogue"] * 4
    assert [r["task_id"] for r in records[4:]] == ["fake001", "fake002", "fake002", "fake001"]
    for r in (2, 3):
        assert sorted(f.name for f in (play.round_dir(run_dir, r) / "captured").iterdir()) == ["u000.json.gz",
                                                                                            "u001.json.gz"]


def test_a_round_already_recorded_plays_nothing_and_meta_json_follows_the_newest_cut(tmp_path, stub):
    """The runner reloads meta.json whole on a resumed call, freezing its protocol block at the first cut, while
    summary.json's is recomputed every time: play_round writes the fresh one back."""
    full, registry, run_dir = _schedule(users=1, rounds=2), _registry(), tmp_path / "h0"
    for n in (1, 2):
        play.play_round("h0", run_dir, full.first_rounds(n), base_url="http://127.0.0.1:1/v1", served_name="base",
                        registry=registry)
    meta = json.loads((run_dir / "meta.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert meta["protocol"]["deviations"] == summary["protocol"]["deviations"]
    assert "stream cut at round 2" in " ".join(meta["protocol"]["deviations"])
    assert play.round_recorded(run_dir, full.first_rounds(2))
    assert not play.round_recorded(run_dir, _schedule(users=1, rounds=3))


def test_a_harness_row_plays_under_the_fixed_harness_and_records_its_settings(tmp_path, monkeypatch):
    captured: list[dict] = []

    class Capturing(_StubSystem):
        def __init__(self, **kwargs):
            captured.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr("continual_arc_baselines.systems.prime_agent.PrimeAgentLearner", Capturing)
    schedule, registry = _schedule(users=1, rounds=1), _registry()
    play.play_round("h0", tmp_path / "h0", schedule, base_url="http://127.0.0.1:1/v1", served_name="r0",
                    registry=registry)
    play.play_round("opsd", tmp_path / "opsd", schedule, base_url="http://127.0.0.1:1/v1", served_name="r0",
                    registry=registry)
    h0, opsd = captured
    assert h0["stateless"] is True and opsd["stateless"] is False
    for kw in (h0, opsd):
        assert {k: kw[k] for k in play.HARNESS} == play.HARNESS and {k: kw[k] for k in play.BLOCK} == play.BLOCK
        assert (kw["model"], kw["context_window"], kw["max_tokens"]) == (play.MODEL, play.CONTEXT_WINDOW,
                                                                         play.MAX_TOKENS)
    driver = json.loads((tmp_path / "h0" / "driver.json").read_text())
    assert driver == {"row": "h0", "served": "r0", "stateless": True, "model": play.MODEL, "harness": play.HARNESS,
                      "harness_budget": {"name": "block", **play.BLOCK}, "sampling": play.SAMPLING}
    with pytest.raises(ValueError, match="row must be one of"):
        play.play_round("h", tmp_path / "h", schedule, base_url="http://127.0.0.1:1/v1", served_name="r0")


class _StreamingModel:
    """A model server that streams every reply as vLLM does for ``"stream": true`` (the reply in one delta, then its
    finish), the same unparseable text each time, and keeps every request body it was sent."""

    def __init__(self, reply: str = "I am not sure what the rule is.") -> None:
        model, self.bodies, lock = self, [], threading.Lock()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args) -> None:
                pass

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                with lock:
                    model.bodies.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for delta, finish in (({"role": "assistant", "content": reply}, None), ({}, "stop")):
                    chunk = {"id": "c", "object": "chat.completion.chunk", "model": body["model"],
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def stop(self) -> None:
        self.server.shutdown()


def test_the_bare_row_is_the_bare_model_at_the_floor_settings_on_the_served_model(tmp_path):
    """Transfer's floor reference: the baselines' bare model at the floors' settings, asking the served base model at
    the rows' sampling and thinking with the rows' reply cap, every instance a conversation of its own and nothing kept
    between them."""
    model = _StreamingModel()
    schedule = _stream_then_epilogue()
    schedule.slots = [s for s in schedule.slots if s.phase == "epilogue"]
    run_dir = tmp_path / "bare"
    try:
        play.play_round("bare", run_dir, schedule, base_url=model.url, served_name=play.MODEL, registry=_registry(3))
    finally:
        model.stop()
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    assert len(records) == 4 and {r["phase"] for r in records} == {"epilogue"} and not any(r["solved"] for r in records)
    assert model.bodies
    for body in model.bodies:
        assert body["model"] == play.MODEL and body["stream"] is True and body["max_tokens"] == play.MAX_TOKENS
        assert {k: body[k] for k in play.SAMPLING} == play.SAMPLING
        assert body["chat_template_kwargs"] == play.TEMPLATE_KWARGS and "return_token_ids" not in body
    assert sum(len(b["messages"]) == 1 for b in model.bodies) == 4   # each instance starts a conversation afresh
    driver = json.loads((run_dir / "driver.json").read_text())
    assert driver["row"] == "bare" and driver["floor"] == play.FLOOR and driver["max_tokens"] == play.MAX_TOKENS
    assert "harness" not in driver
