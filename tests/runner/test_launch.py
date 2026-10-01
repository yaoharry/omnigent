"""Focused tests for registry-driven spawn-environment launch dispatch."""

from __future__ import annotations

from typing import Any

import pytest

from omnigent.harness_plugins import model_env_keys, spawn_env_builders
from omnigent.runner import launch
from omnigent.spec.types import AgentSpec, ExecutorSpec, LLMConfig


def _spec(harness: str) -> AgentSpec:
    return AgentSpec(
        spec_version=1,
        name="launch-test",
        executor=ExecutorSpec(type="omnigent", config={"harness": harness}),
    )


def test_builtin_sdk_spawn_builders_are_registry_rows() -> None:
    builders = spawn_env_builders()
    # ACP CLI rows need row/session context and intentionally stay on the
    # explicit branch; every model-env harness uses the common builder shape.
    for harness in model_env_keys():
        if harness == "acp":
            continue
        assert harness in builders
        assert builders[harness].startswith("omnigent.harnesses.config.spawn_env:")


def test_launch_resolves_a_registry_builder_with_common_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, Any] = {}

    def builder(spec: object, *, cwd: object, workdir: object) -> dict[str, str]:
        seen.update(spec=spec, cwd=cwd, workdir=workdir)
        return {"HARNESS_FAKE_MODEL": "default"}

    monkeypatch.setattr(launch, "spawn_env_builders", lambda: {"fake": "fake:builder"})
    monkeypatch.setattr(launch, "load_object", lambda _path: builder)
    monkeypatch.setattr(launch, "_HARNESS_MODEL_ENV_KEY", {"fake": "HARNESS_FAKE_MODEL"})

    result = launch.build_spawn_env_from_spec(_spec("fake"), "fake", model_override="picked")

    assert result == {"HARNESS_FAKE_MODEL": "picked"}
    assert seen["spec"].name == "launch-test"
    assert seen["cwd"] is None
    assert seen["workdir"] is None


def test_launch_leaves_native_environment_to_native_dispatch() -> None:
    assert launch.build_spawn_env_from_spec(_spec("claude-native"), "claude-native") is None


@pytest.mark.asyncio
async def test_workflow_compaction_compatibility_adapter_injects_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deprecated workflow entry point still resolves explicit services."""
    from omnigent.runtime import workflow

    stores = (object(), object(), object())
    client = object()
    runner_client = object()
    captured: dict[str, Any] = {}

    async def _fake_compact(**kwargs: Any) -> object:
        captured.update(kwargs)
        return "result"

    monkeypatch.setattr(workflow, "_compact_conversation_now", _fake_compact)
    monkeypatch.setattr(
        workflow,
        "_load_initial_history",
        lambda _store, _conversation_id: workflow.LoadedHistory(items=[object()]),
    )
    monkeypatch.setattr(workflow, "get_conversation_store", lambda: stores[0])
    monkeypatch.setattr(workflow, "get_file_store", lambda: stores[1])
    monkeypatch.setattr(workflow, "get_artifact_store", lambda: stores[2])
    monkeypatch.setattr(workflow, "_get_llm_client", lambda: client)
    monkeypatch.setattr(workflow, "_get_runner_client_for_compaction", lambda _id: runner_client)

    result = await workflow.compact_conversation_now(
        task_id="task-1",
        conversation_id="conv-1",
        spec=_spec("fake"),
        llm_config=LLMConfig("gpt-test"),
    )

    assert result == "result"
    services = captured["services"]
    assert services.conversation_store is stores[0]
    assert services.file_store is stores[1]
    assert services.artifact_store is stores[2]
    assert services.llm_client is client
    assert services.runner_client is runner_client


def test_workflow_prepare_messages_compatibility_adapter_injects_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from omnigent.runtime import workflow

    stores = (object(), object())
    captured: dict[str, Any] = {}

    def _fake_prepare(*args: Any, **kwargs: Any) -> tuple[str, list[dict[str, Any]], int]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return "system", [], 0

    monkeypatch.setattr(workflow, "_prepare_messages_explicit", _fake_prepare)
    monkeypatch.setattr(workflow, "get_file_store", lambda: stores[0])
    monkeypatch.setattr(workflow, "get_artifact_store", lambda: stores[1])

    result = workflow._prepare_messages(
        _spec("fake"),
        LLMConfig("gpt-test"),
        [],
        None,
        [],
        object(),
        None,
    )

    assert result == ("system", [], 0)
    assert captured["kwargs"]["file_store"] is stores[0]
    assert captured["kwargs"]["artifact_store"] is stores[1]
