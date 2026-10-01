"""Tests for the explicit runtime compaction service boundary."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from omnigent.entities import CompactionData
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.runtime import compaction_service
from omnigent.runtime.compaction import CompactionResult, SummaryMetadata
from omnigent.runtime.compaction_service import (
    CompactionServices,
    compact_conversation_now,
)
from omnigent.runtime.history import LoadedHistory
from omnigent.spec import AgentSpec
from omnigent.spec.types import ExecutorSpec, LLMConfig


class _Store:
    """Small store fake that records compaction appends and deduplicates them."""

    def __init__(self) -> None:
        self.appended: list[Any] = []

    def list_items(self, _conversation_id: str, **kwargs: Any) -> Any:
        if kwargs.get("type") == "compaction" and self.appended:
            return SimpleNamespace(data=[self.appended[0]])
        return SimpleNamespace(data=[])

    def append(self, _conversation_id: str, items: list[Any]) -> None:
        self.appended.extend(items)


def _spec() -> AgentSpec:
    return AgentSpec(
        spec_version=1,
        name="compaction-test",
        skills_filter="none",
        executor=ExecutorSpec(config={"harness": "claude-sdk"}),
    )


def _services(store: _Store, llm_client: object, runner_client: object) -> CompactionServices:
    return CompactionServices(
        conversation_store=store,  # type: ignore[arg-type]
        file_store=object(),  # type: ignore[arg-type]
        artifact_store=object(),  # type: ignore[arg-type]
        llm_client=llm_client,  # type: ignore[arg-type]
        runner_client=runner_client,
    )


@pytest.mark.asyncio
async def test_empty_history_uses_supplied_services_only(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _Store()
    client = object()
    runner = object()
    services = _services(store, client, runner)

    def fail_global_lookup() -> object:
        raise AssertionError("compaction service must not resolve global runtime services")

    monkeypatch.setattr("omnigent.runtime.get_services", fail_global_lookup)
    monkeypatch.setattr(
        compaction_service,
        "load_initial_history",
        lambda *_args: LoadedHistory(items=[]),
    )

    result = await compact_conversation_now(
        task_id="task-empty",
        conversation_id="conv-empty",
        spec=_spec(),
        llm_config=LLMConfig(model="openai/gpt-5"),
        services=services,
    )

    assert result == CompactionResult(messages=[], summary_metadata=None)
    assert store.appended == []


@pytest.mark.asyncio
async def test_summary_persists_once_and_preserves_injected_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _Store()
    client = object()
    runner = object()
    services = _services(store, client, runner)
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        compaction_service,
        "load_initial_history",
        lambda *_args: LoadedHistory(items=[cast(Any, object())]),
    )

    def fake_prepare_messages(*args: Any, **kwargs: Any) -> tuple[str, list[dict[str, Any]], int]:
        captured["prepare_args"] = args
        captured["prepare_kwargs"] = kwargs
        return "instructions", [{"role": "user", "content": "history"}], 7

    monkeypatch.setattr(compaction_service, "prepare_messages", fake_prepare_messages)
    summary = SummaryMetadata(
        text="summary",
        last_item_id="item-1",
        model="databricks/test-model",
        token_count=2,
    )

    async def fake_compact(*args: Any, **kwargs: Any) -> CompactionResult:
        captured["compact_args"] = args
        captured["compact_kwargs"] = kwargs
        return CompactionResult(
            messages=[{"role": "assistant", "content": "summary"}], summary_metadata=summary
        )

    monkeypatch.setattr(compaction_service, "compact", fake_compact)

    config = LLMConfig(model="claude-test", extra={"temperature": 0.2})
    for _ in range(2):
        await compact_conversation_now(
            task_id="task-summary",
            conversation_id="conv-summary",
            spec=_spec(),
            llm_config=config,
            services=services,
            model_override="databricks/test-model",
            preserve_recent_window=3,
        )

    prepared_config = captured["prepare_args"][1]
    assert prepared_config.model == "databricks/test-model"
    assert prepared_config.extra["model_override"] == "databricks/test-model"
    assert captured["prepare_kwargs"]["file_store"] is services.file_store
    assert captured["prepare_kwargs"]["artifact_store"] is services.artifact_store
    compact_kwargs = captured["compact_kwargs"]
    assert compact_kwargs["llm_client"] is client
    assert compact_kwargs["runner_client"] is runner
    assert compact_kwargs["model"] == "databricks/test-model"
    assert compact_kwargs["config"].recent_window == 2
    assert len(store.appended) == 1
    assert isinstance(store.appended[0].data, CompactionData)
    assert store.appended[0].data.summary == "summary"


@pytest.mark.asyncio
async def test_missing_summary_raises_without_persisting(monkeypatch: pytest.MonkeyPatch) -> None:
    store = _Store()
    services = _services(store, object(), object())
    monkeypatch.setattr(
        compaction_service,
        "load_initial_history",
        lambda *_args: LoadedHistory(items=[cast(Any, object())]),
    )
    monkeypatch.setattr(
        compaction_service,
        "prepare_messages",
        lambda *_args, **_kwargs: ("instructions", [{"role": "user"}], 1),
    )

    async def no_summary(*_args: Any, **_kwargs: Any) -> CompactionResult:
        return CompactionResult(messages=[], summary_metadata=None)

    monkeypatch.setattr(compaction_service, "compact", no_summary)

    with pytest.raises(OmnigentError) as exc_info:
        await compact_conversation_now(
            task_id="task-fail",
            conversation_id="conv-fail",
            spec=_spec(),
            llm_config=LLMConfig(model="openai/gpt-5"),
            services=services,
        )

    assert exc_info.value.code == ErrorCode.INTERNAL_ERROR
    assert store.appended == []
