"""Composable run-time plug-ins for Continual-ARC.

Plug-ins sit around the learner and the benchmark environment. They can observe
the run, change the learner-facing observation, change an action before it reaches
the environment, change the feedback delivered to the learner, and enqueue
auxiliary episodes. The environment still owns raw task data, scoring, and raw
feedback. This keeps interventions explicit in the run record.

The older ``Addon`` interface remains supported. ``BaseAddon`` adapts it to the
new plug-in manager so existing spaced-review implementations continue to work.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from continual_arc.env import Action, EnvConfig, Feedback, InstanceRecord, Observation
from continual_arc.schedule import Schedule

PHASE_ADDON_REVIEW = "addon_review"


@dataclass(frozen=True)
class EpisodeRequest:
    """Request one auxiliary task instance.

    ``source`` must be registered in the environment's :class:`TaskRegistry`.
    ``instance_seed`` is part of the reproducibility contract: a plug-in should
    derive it deterministically from the run seed and its own state.
    """

    task_id: str
    source: str
    instance_seed: int
    phase: str = PHASE_ADDON_REVIEW
    reset_before: bool = False
    addon: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    plugin: str | None = None
    episode_type: str = "addon"

    @property
    def owner(self) -> str | None:
        """The plug-in that requested this episode, with legacy support."""
        return self.plugin or self.addon

    def to_json(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source": self.source,
            "instance_seed": self.instance_seed,
            "phase": self.phase,
            "reset_before": self.reset_before,
            "addon": self.addon,
            "metadata": self.metadata,
            "plugin": self.plugin,
            "episode_type": self.episode_type,
        }


@dataclass(frozen=True)
class PluginResponse:
    """Optional changes returned by a plug-in hook.

    ``None`` means no override. Requests are queued until the current episode
    ends, so a plug-in cannot interrupt the environment halfway through an
    instance. Metadata is diagnostic only and never affects scoring.
    """

    observation: Observation | None = None
    action: Action | None = None
    feedback: Feedback | None = None
    episode_requests: tuple[EpisodeRequest, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    stop_after_episode: bool = False


@dataclass(frozen=True)
class PluginContext:
    """Read-only state supplied to a plug-in hook."""

    run_id: str
    schedule: Schedule
    config: EnvConfig
    step: int | None
    n_steps: int
    task_id: str | None
    phase: str | None
    episode_type: str | None
    benchmark_clock: int
    runtime: PluginRuntime


class PluginRuntime:
    """Run-scoped output API available to plug-ins.

    Metrics are kept in the run summary. Events are written as JSONL when the
    runner has an output directory. Artifact registration records a path for
    downstream evidence handling; it does not copy or delete files.
    """

    def __init__(self, events_path: str | Path | None = None) -> None:
        self.events_path = Path(events_path) if events_path else None
        self.metrics: list[dict[str, Any]] = []
        self.artifacts: list[dict[str, str | None]] = []

    def event(self, kind: str, payload: dict[str, Any]) -> None:
        if self.events_path is None:
            return
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with self.events_path.open("a") as stream:
            stream.write(json.dumps({"kind": kind, "payload": payload}) + "\n")

    def metric(self, name: str, value: float, step: int | None = None) -> None:
        self.metrics.append({"name": name, "value": float(value), "step": step})

    def artifact(self, path: str | Path, name: str | None = None) -> None:
        self.artifacts.append({"path": str(Path(path)), "name": name})


@runtime_checkable
class Plugin(Protocol):
    """Full plug-in protocol.

    Hooks may return ``None`` or :class:`PluginResponse`. The runner calls them
    in priority order and passes each hook the output of the previous plug-in.
    """

    name: str
    priority: int

    def on_run_start(self, context: PluginContext) -> PluginResponse | None: ...

    def on_episode_start(
        self, observation: Observation, context: PluginContext
    ) -> PluginResponse | None: ...

    def on_observation(
        self, observation: Observation, context: PluginContext
    ) -> PluginResponse | None: ...

    def on_action(
        self, observation: Observation, action: Action, context: PluginContext
    ) -> PluginResponse | None: ...

    def on_feedback(
        self,
        observation: Observation,
        action: Action,
        feedback: Feedback,
        context: PluginContext,
    ) -> PluginResponse | None: ...

    def on_episode_end(
        self, record: InstanceRecord, context: PluginContext
    ) -> PluginResponse | None: ...

    def on_run_end(self, records: list[InstanceRecord], context: PluginContext) -> None: ...

    def summary(self) -> dict[str, Any]: ...


class BasePlugin:
    """Convenience base class with no-op hooks."""

    name = "plugin"
    priority = 0

    def on_run_start(self, context: PluginContext) -> PluginResponse | None:
        del context
        return None

    def on_episode_start(self, observation: Observation, context: PluginContext) -> PluginResponse | None:
        del observation, context
        return None

    def on_observation(self, observation: Observation, context: PluginContext) -> PluginResponse | None:
        del observation, context
        return None

    def on_action(self, observation: Observation, action: Action, context: PluginContext) -> PluginResponse | None:
        del observation, action, context
        return None

    def on_feedback(
        self, observation: Observation, action: Action, feedback: Feedback, context: PluginContext
    ) -> PluginResponse | None:
        del observation, action, feedback, context
        return None

    def on_episode_end(self, record: InstanceRecord, context: PluginContext) -> PluginResponse | None:
        del record, context
        return None

    def on_run_end(self, records: list[InstanceRecord], context: PluginContext) -> None:
        del records, context

    def summary(self) -> dict[str, Any]:
        return {}


@runtime_checkable
class Addon(Protocol):
    """Legacy episode-completion add-on protocol."""

    name: str

    def on_run_start(self, schedule: Schedule, config: EnvConfig) -> None: ...

    def on_instance_end(self, record: InstanceRecord) -> Iterable[EpisodeRequest]: ...

    def on_run_end(self, records: list[InstanceRecord]) -> None: ...

    def summary(self) -> dict[str, Any]: ...


class BaseAddon:
    """Legacy base class retained as a compatibility adapter."""

    interface_version = 1
    name = "addon"

    def on_run_start(self, schedule: Schedule, config: EnvConfig) -> None:
        del schedule, config

    def on_instance_end(self, record: InstanceRecord) -> Iterable[EpisodeRequest]:
        del record
        return ()

    def on_run_end(self, records: list[InstanceRecord]) -> None:
        del records

    def summary(self) -> dict[str, Any]:
        return {}


class PluginManager:
    """Compose full plug-ins and adapt legacy add-ons."""

    def __init__(
        self,
        plugins: Sequence[Plugin | Addon],
        schedule: Schedule,
        config: EnvConfig,
        run_id: str,
        runtime: PluginRuntime,
    ) -> None:
        self.plugins = sorted(list(plugins), key=lambda p: (getattr(p, "priority", 0), p.name))
        names = [p.name for p in self.plugins]
        if len(names) != len(set(names)):
            raise ValueError(f"plugin names must be unique, got {names!r}")
        self.schedule = schedule
        self.config = config
        self.run_id = run_id
        self.runtime = runtime
        self._pending_requests: list[EpisodeRequest] = []
        self._record_metadata: dict[str, Any] = {}
        self._benchmark_clock = 0
        self.stop_requested = False

    def _legacy(self, plugin: Plugin | Addon) -> bool:
        return (
            getattr(plugin, "interface_version", 2) == 1
            or (hasattr(plugin, "on_instance_end") and not hasattr(plugin, "on_episode_end"))
        )

    def context(
        self,
        *,
        step: int | None = None,
        n_steps: int | None = None,
        task_id: str | None = None,
        phase: str | None = None,
        episode_type: str | None = None,
    ) -> PluginContext:
        return PluginContext(
            run_id=self.run_id,
            schedule=self.schedule,
            config=self.config,
            step=step,
            n_steps=n_steps if n_steps is not None else len(self.schedule),
            task_id=task_id,
            phase=phase,
            episode_type=episode_type,
            benchmark_clock=self._benchmark_clock,
            runtime=self.runtime,
        )

    @staticmethod
    def _response(value: PluginResponse | None) -> PluginResponse:
        if value is None:
            return PluginResponse()
        if not isinstance(value, PluginResponse):
            raise TypeError(f"plugin hook must return PluginResponse or None, got {value!r}")
        return value

    def _collect(self, plugin: Plugin | Addon, response: PluginResponse) -> None:
        self._pending_requests.extend(
            replace(request, plugin=request.plugin or plugin.name)
            for request in response.episode_requests
        )
        self._record_metadata.update(response.metadata)
        self.stop_requested = self.stop_requested or response.stop_after_episode

    def on_run_start(self) -> None:
        context = self.context()
        for plugin in self.plugins:
            if self._legacy(plugin):
                plugin.on_run_start(self.schedule, self.config)
            else:
                hook = getattr(plugin, "on_run_start", None)
                if hook is not None:
                    self._collect(plugin, self._response(hook(context)))

    def on_episode_start(self, observation: Observation) -> Observation:
        current = observation
        context = self.context(
            step=current.step,
            n_steps=current.n_steps,
            task_id=current.task_id,
            phase=current.phase,
            episode_type=current.episode_type,
        )
        for plugin in self.plugins:
            if self._legacy(plugin):
                continue
            hook = getattr(plugin, "on_episode_start", None)
            if hook is None:
                continue
            response = self._response(hook(current, context))
            self._collect(plugin, response)
            if response.observation is not None:
                current = response.observation
        return self.on_observation(current)

    def on_observation(self, observation: Observation) -> Observation:
        current = observation
        context = self.context(
            step=current.step,
            n_steps=current.n_steps,
            task_id=current.task_id,
            phase=current.phase,
            episode_type=current.episode_type,
        )
        for plugin in self.plugins:
            if self._legacy(plugin):
                continue
            hook = getattr(plugin, "on_observation", None)
            if hook is None:
                continue
            response = self._response(hook(current, context))
            self._collect(plugin, response)
            if response.observation is not None:
                current = response.observation
        return current

    def on_action(self, observation: Observation, action: Action) -> Action:
        current = action
        context = self.context(
            step=observation.step,
            n_steps=observation.n_steps,
            task_id=observation.task_id,
            phase=observation.phase,
            episode_type=observation.episode_type,
        )
        for plugin in self.plugins:
            if self._legacy(plugin):
                continue
            hook = getattr(plugin, "on_action", None)
            if hook is None:
                continue
            response = self._response(hook(observation, current, context))
            self._collect(plugin, response)
            if response.action is not None:
                current = response.action
        return current

    def on_feedback(self, observation: Observation, action: Action, feedback: Feedback) -> Feedback:
        current = feedback
        context = self.context(
            step=observation.step,
            n_steps=observation.n_steps,
            task_id=observation.task_id,
            phase=observation.phase,
            episode_type=observation.episode_type,
        )
        for plugin in self.plugins:
            if self._legacy(plugin):
                continue
            hook = getattr(plugin, "on_feedback", None)
            if hook is None:
                continue
            response = self._response(hook(observation, action, current, context))
            self._collect(plugin, response)
            if response.feedback is not None:
                current = response.feedback
        return current

    def on_episode_end(self, record: InstanceRecord) -> None:
        context = self.context(
            step=record.step,
            n_steps=len(self.schedule),
            task_id=record.task_id,
            phase=record.phase,
            episode_type=record.episode_type,
        )
        for plugin in self.plugins:
            if self._legacy(plugin):
                requests = plugin.on_instance_end(record)
                self._pending_requests.extend(
                    replace(request, addon=request.addon or plugin.name)
                    for request in requests
                )
            else:
                hook = getattr(plugin, "on_episode_end", None)
                if hook is not None:
                    self._collect(plugin, self._response(hook(record, context)))
        if self._record_metadata:
            record.metadata.update(self._record_metadata)
            self._record_metadata = {}
        if record.episode_type == "benchmark":
            self._benchmark_clock += 1

    def take_episode_requests(self) -> list[EpisodeRequest]:
        requests = self._pending_requests
        self._pending_requests = []
        return requests

    def on_run_end(self, records: list[InstanceRecord]) -> None:
        context = self.context(n_steps=len(self.schedule), episode_type=None)
        for plugin in self.plugins:
            if self._legacy(plugin):
                plugin.on_run_end(records)
            else:
                hook = getattr(plugin, "on_run_end", None)
                if hook is not None:
                    hook(records, context)

    def summary(self) -> dict[str, Any]:
        return {
            "names": [plugin.name for plugin in self.plugins],
            "plugins": {
                plugin.name: (getattr(plugin, "summary", lambda: {})() or {})
                for plugin in self.plugins
            },
            "runtime_metrics": list(self.runtime.metrics),
            "artifacts": list(self.runtime.artifacts),
        }
