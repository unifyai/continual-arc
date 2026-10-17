from continual_arc.addons import (
    PHASE_ADDON_REVIEW,
    BaseAddon,
    BasePlugin,
    EpisodeRequest,
    PluginResponse,
)
from continual_arc.env import ContinualArcEnv, EnvConfig, GiveUp, Submit
from continual_arc.runner import run
from continual_arc.schedule import RANDOM, sequential_params, stream_schedule
from continual_arc.tasks import Pair
from continual_arc.testing import EchoLearner, ShiftMemoryLearner


class InlineSource:
    name = "inline"

    def task_ids(self):
        return ["inline-task"]

    def test_instance(self, task_id, instance_seed):
        del task_id, instance_seed
        return Pair(input=[[1]], output=[[2]])

    def demos(self, task_id, instance_seed, request_index, n, exclude):
        del task_id, instance_seed, request_index, n, exclude
        return [Pair(input=[[1]], output=[[2]])]


class OneReview(BaseAddon):
    name = "one_review"

    def __init__(self):
        self.seen = 0

    def on_instance_end(self, record):
        if record.episode_type == "benchmark" and self.seen == 0:
            self.seen += 1
            return [
                EpisodeRequest(
                    task_id=record.task_id,
                    source=record.source,
                    instance_seed=999,
                    addon=self.name,
                    metadata={"parent": record.step},
                )
            ]
        return ()


class FullPlugin(BasePlugin):
    name = "full_plugin"

    def __init__(self):
        self.hooks = []

    def on_episode_start(self, observation, context):
        self.hooks.append(("episode_start", observation.episode_type, context.benchmark_clock))
        return PluginResponse(metadata={"seen_by_plugin": True})

    def on_observation(self, observation, context):
        self.hooks.append(("observation", observation.episode_type, context.benchmark_clock))

    def on_action(self, observation, action, context):
        self.hooks.append(("action", type(action).__name__, context.task_id))
        return PluginResponse(action=GiveUp())

    def on_feedback(self, observation, action, feedback, context):
        self.hooks.append(("feedback", type(feedback).__name__, context.task_id))

    def on_episode_end(self, record, context):
        self.hooks.append(("episode_end", record.episode_type, context.benchmark_clock))
        context.runtime.metric("plugin/test", 1, record.step)

    def on_run_end(self, records, context):
        self.hooks.append(("run_end", len(records), context.benchmark_clock))

    def summary(self):
        return {"hook_count": len(self.hooks)}


class PartialPlugin:
    name = "partial_plugin"

    def on_action(self, observation, action, context):
        del observation, action, context
        return PluginResponse(action=GiveUp())


def test_addon_episode_is_inserted_without_changing_primary_metadata(fake_registry, fake_source):
    schedule = stream_schedule(fake_source.task_ids()[:2], 4, seed=0, source=fake_source.name,
                               params=sequential_params(2), mode="sequential")
    result = run(
        ShiftMemoryLearner(),
        schedule,
        config=EnvConfig(attempt_cap=3),
        registry=fake_registry,
        addon=OneReview(),
    )

    assert len(result.records) == 5
    primary = [r for r in result.records if r.episode_type == "benchmark"]
    reviews = [r for r in result.records if r.episode_type == "addon"]
    assert len(primary) == len(schedule) == 4
    assert len(reviews) == 1
    assert reviews[0].phase == PHASE_ADDON_REVIEW
    assert reviews[0].addon == "one_review"
    assert reviews[0].metadata == {"parent": 0}
    assert [r.exposure_index for r in primary] == [0, 1, 0, 1]
    assert result.summary["benchmark"]["n"] == 4
    assert result.summary["addons"]["n"] == 1
    assert result.summary["addon"]["name"] == "one_review"


def test_env_reset_removes_inserted_addon_episodes(fake_registry, fake_source):
    schedule = stream_schedule(fake_source.task_ids()[:1], 1, seed=0, source=fake_source.name, params=RANDOM, mode="random")
    from continual_arc.env import ContinualArcEnv

    env = ContinualArcEnv(schedule, registry=fake_registry)
    env.reset()
    env.insert_next([
        EpisodeRequest(task_id=fake_source.task_ids()[0], source=fake_source.name, instance_seed=1)
    ])
    assert env.n_steps == 2
    env.reset()
    assert env.n_steps == 1


def test_inserted_episode_preserves_dynamic_request_source(fake_registry, fake_source):
    schedule = stream_schedule(fake_source.task_ids()[:1], 1, seed=0, source=fake_source.name, params=RANDOM, mode="random")
    inline = InlineSource()
    fake_registry.add_source(inline)
    env = ContinualArcEnv(schedule, registry=fake_registry)
    env.reset()
    env.insert_next([
        EpisodeRequest(
            task_id="inline-task",
            source=inline.name,
            instance_seed=3,
            plugin="generated",
            episode_type="addon",
        )
    ])
    result = env.step(Submit([[2]]))

    assert result.record is not None
    assert result.record.source == "inline"
    assert result.record.plugin == "generated"


def test_full_plugin_lifecycle_can_transform_and_record(fake_registry, fake_source):
    schedule = stream_schedule(fake_source.task_ids()[:1], 1, seed=0, source=fake_source.name, params=RANDOM, mode="random")
    plugin = FullPlugin()
    result = run(EchoLearner(), schedule, registry=fake_registry, plugins=[plugin])

    assert result.records[0].gave_up
    assert result.records[0].metadata["seen_by_plugin"] is True
    assert [hook[0] for hook in plugin.hooks] == [
        "episode_start", "observation", "action", "feedback", "episode_end", "run_end"
    ]
    assert result.summary["plugins"]["names"] == ["full_plugin"]
    assert result.summary["plugins"]["plugins"]["full_plugin"]["hook_count"] == 6
    assert result.summary["plugins"]["runtime_metrics"] == [
        {"name": "plugin/test", "value": 1.0, "step": 0}
    ]


def test_plugin_can_implement_only_the_hooks_it_needs(fake_registry, fake_source):
    schedule = stream_schedule(fake_source.task_ids()[:1], 1, seed=0, source=fake_source.name, params=RANDOM, mode="random")
    result = run(EchoLearner(), schedule, registry=fake_registry, plugins=[PartialPlugin()])

    assert result.records[0].gave_up
    assert result.summary["plugins"]["names"] == ["partial_plugin"]
