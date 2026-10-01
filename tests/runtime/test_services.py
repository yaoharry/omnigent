"""Tests for the process-scoped runtime services boundary."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import pytest

import omnigent.runtime.services as services_module
from omnigent.runtime import (
    get_agent_cache,
    get_caps,
    get_conversation_store,
    get_services,
)
from omnigent.runtime.caps import RuntimeCaps
from omnigent.runtime.services import RuntimeServices


def test_runtime_services_have_independent_mutable_defaults() -> None:
    first = RuntimeServices()
    second = RuntimeServices()

    assert first.caps is not second.caps
    assert first._dispatch_capabilities is not second._dispatch_capabilities
    assert first._tool_manager_var is not second._tool_manager_var


def test_get_services_constructs_one_object_under_concurrent_first_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_type = services_module.RuntimeServices
    created: list[RuntimeServices] = []

    def construct() -> RuntimeServices:
        # Keep the lock held long enough for the other workers to contend.
        time.sleep(0.01)
        value = original_type()
        created.append(value)
        return value

    monkeypatch.setattr(services_module, "_services", None)
    monkeypatch.setattr(services_module, "RuntimeServices", construct)
    with ThreadPoolExecutor(max_workers=16) as pool:
        values = list(pool.map(lambda _index: services_module.get_services(), range(32)))

    assert len(created) == 1
    assert all(value is created[0] for value in values)


def test_initialize_replaces_startup_services_only() -> None:
    services = RuntimeServices()
    old_terminal_registry = services.terminal_registry
    runner_router = cast(Any, object())
    resource_registry = cast(Any, object())
    services.runner_router = runner_router
    services.resource_registry = resource_registry

    conversation_store = object()
    agent_store = object()
    agent_cache = object()
    caps = RuntimeCaps(execution_timeout=123)
    services.initialize(
        conversation_store=conversation_store,  # type: ignore[arg-type]
        agent_store=agent_store,  # type: ignore[arg-type]
        agent_cache=agent_cache,  # type: ignore[arg-type]
        caps=caps,
    )

    assert services.conversation_store is conversation_store
    assert services.agent_store is agent_store
    assert services.agent_cache is agent_cache
    assert services.caps is caps
    assert services.terminal_registry is not old_terminal_registry
    assert services.runner_router is runner_router
    assert services.resource_registry is resource_registry


def test_public_getters_read_the_canonical_services_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation_store = object()
    agent_cache = object()
    services = get_services()
    monkeypatch.setattr(services, "conversation_store", conversation_store)
    monkeypatch.setattr(services, "agent_cache", agent_cache)
    monkeypatch.setattr(services, "caps", RuntimeCaps(execution_timeout=456))
    assert get_conversation_store() is conversation_store
    assert get_agent_cache() is agent_cache
    assert get_caps() is services.caps
    assert services.caps.execution_timeout == 456
