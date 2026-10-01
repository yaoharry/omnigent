"""Focused tests for the server resource lifecycle owner."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from omnigent.runtime import get_services
from omnigent.runtime.services import RuntimeServices
from omnigent.server.lifecycle import ServerLifecycle


class _FakeRouter:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def aclose(self) -> None:
        self._events.append("router")


class _FakeTerminalRegistry:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def shutdown(self) -> None:
        self._events.append("terminal")


class _FakeTitleCoordinator:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def shutdown(self) -> None:
        self._events.append("title")


class _FakeMcpPool:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def shutdown_all(self) -> None:
        self._events.append("mcp")


class _FailingMcpPool(_FakeMcpPool):
    async def shutdown_all(self) -> None:
        self._events.append("mcp")
        raise RuntimeError("mcp shutdown failed")


class _FakeScheduler:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def start(self) -> None:
        self._events.append("scheduler_start")

    def stop(self) -> None:
        self._events.append("scheduler_stop")


class _FailingReaper:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def start(self) -> None:
        self._events.append("reaper_start")
        raise RuntimeError("reaper start failed")

    async def shutdown(self) -> None:
        self._events.append("reaper_shutdown")


def _lifecycle(
    events: list[str],
    *,
    initialize: Any,
    scheduler_factory: Any = None,
    reaper_factory: Any = None,
    services: Any = None,
    mcp_pool: Any = None,
) -> ServerLifecycle:
    services = get_services() if services is None else services
    terminal_registry = _FakeTerminalRegistry(events)
    services.terminal_registry = terminal_registry  # type: ignore[assignment]
    router = _FakeRouter(events)

    async def cancel_managed_launches() -> None:
        events.append("managed")

    return ServerLifecycle(
        app=SimpleNamespace(state=SimpleNamespace()),  # type: ignore[arg-type]
        runner_router=router,  # type: ignore[arg-type]
        background_title_coordinator=_FakeTitleCoordinator(events),  # type: ignore[arg-type]
        mcp_pool=mcp_pool or _FakeMcpPool(events),  # type: ignore[arg-type]
        server_metrics=object(),  # type: ignore[arg-type]
        server_metrics_otel=object(),  # type: ignore[arg-type]
        server_config={},
        install_notifier=lambda: lambda: events.append("notifier"),
        make_ws_factory=lambda: "ws",
        make_direct_attach_resolver=lambda: "direct",
        initialize=initialize,
        cancel_managed_launches=cancel_managed_launches,
        publish_metrics=lambda _metrics, _otel: _record_async(events, "metrics"),
        scheduler_factory=scheduler_factory,
        reaper_factory=reaper_factory,
        runtime_services=services,
    )


async def _record_async(events: list[str], name: str) -> None:
    events.append(name)
    await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_startup_failure_closes_everything_acquired_before_yield() -> None:
    """A startup error must unwind all earlier bindings and resources."""
    events: list[str] = []

    async def fail_startup() -> None:
        raise RuntimeError("startup failed")

    services = get_services()
    previous = {
        name: getattr(services, name)
        for name in (
            "terminal_registry",
            "runner_router",
            "runner_ws_factory",
            "runner_direct_attach_resolver",
        )
    }
    lifecycle = _lifecycle(events, initialize=fail_startup)
    try:
        with pytest.raises(RuntimeError, match="startup failed"):
            async with lifecycle.lifespan():
                pytest.fail("startup should not reach the lifespan body")

        assert {"notifier", "title", "router", "terminal", "mcp"} <= set(events)
        assert services.runner_router is previous["runner_router"]
        assert services.runner_ws_factory is previous["runner_ws_factory"]
        assert services.runner_direct_attach_resolver is previous["runner_direct_attach_resolver"]
    finally:
        for name, value in previous.items():
            setattr(services, name, value)


@pytest.mark.asyncio
async def test_lifecycle_can_be_entered_twice_and_releases_each_binding() -> None:
    """Repeated ASGI lifespan contexts do not retain a prior binding."""
    events: list[str] = []
    services = get_services()
    previous_terminal = services.terminal_registry
    runs = 0

    async def initialize() -> None:
        nonlocal runs
        runs += 1

    async def publish(_metrics: object, _otel: object) -> None:
        await asyncio.Event().wait()

    lifecycle = _lifecycle(events, initialize=initialize)
    lifecycle.publish_metrics = publish  # type: ignore[assignment]
    try:
        async with lifecycle.lifespan():
            assert services.runner_router is lifecycle.runner_router
        first_cleanup_count = len(events)
        first_cleanup = events[:first_cleanup_count]
        first_order = [
            first_cleanup.index(name) for name in ("managed", "title", "router", "terminal", "mcp")
        ]
        assert first_order == sorted(first_order)

        async with lifecycle.lifespan():
            assert services.runner_router is lifecycle.runner_router
        assert runs == 2
        assert len(events) > first_cleanup_count
        second_cleanup = events[first_cleanup_count:]
        second_order = [
            second_cleanup.index(name)
            for name in ("managed", "title", "router", "terminal", "mcp")
        ]
        assert second_order == sorted(second_order)
    finally:
        services.terminal_registry = previous_terminal


@pytest.mark.asyncio
async def test_supplied_runtime_services_own_terminal_without_global_lookup() -> None:
    """An embedded lifecycle uses its supplied services object."""
    events: list[str] = []
    global_services = get_services()
    previous_global_terminal = global_services.terminal_registry
    supplied = RuntimeServices()
    lifecycle = _lifecycle(events, initialize=lambda: _noop(), services=supplied)
    global_services.terminal_registry = None
    try:
        async with lifecycle.lifespan():
            assert supplied.terminal_registry is not None
        assert "terminal" in events
    finally:
        global_services.terminal_registry = previous_global_terminal


@pytest.mark.asyncio
async def test_rebinding_does_not_skip_owned_cleanup_or_overwrite_new_binding() -> None:
    """Cleanup closes captured resources while preserving a later owner."""
    events: list[str] = []
    supplied = RuntimeServices()
    replacement_router = object()
    replacement_terminal = _FakeTerminalRegistry(events)
    lifecycle = _lifecycle(events, initialize=lambda: _noop(), services=supplied)
    try:
        async with lifecycle.lifespan():
            supplied.runner_router = replacement_router  # type: ignore[assignment]
            supplied.terminal_registry = replacement_terminal  # type: ignore[assignment]
        assert "terminal" in events
        assert supplied.runner_router is replacement_router
        assert replacement_terminal is supplied.terminal_registry
    finally:
        supplied.terminal_registry = None
        supplied.runner_router = None


@pytest.mark.asyncio
async def test_shutdown_exception_does_not_skip_remaining_cleanup() -> None:
    """AsyncExitStack drains every cleanup callback after one failure."""
    events: list[str] = []
    lifecycle = _lifecycle(
        events,
        initialize=lambda: _noop(),
        mcp_pool=_FailingMcpPool(events),
        services=RuntimeServices(),
    )
    with pytest.raises(RuntimeError, match="mcp shutdown failed"):
        async with lifecycle.lifespan():
            pass
    assert {"router", "terminal", "title", "notifier", "managed", "mcp"} <= set(events)


@pytest.mark.asyncio
async def test_reaper_start_failure_unwinds_scheduler_and_background_tasks() -> None:
    """A reaper startup error unwinds resources acquired earlier in startup."""
    events: list[str] = []
    scheduler = _FakeScheduler(events)
    reaper = _FailingReaper(events)
    lifecycle = _lifecycle(
        events,
        initialize=lambda: _noop(),
        scheduler_factory=lambda: scheduler,
        reaper_factory=lambda: reaper,
        services=RuntimeServices(),
    )
    with pytest.raises(RuntimeError, match="reaper start failed"):
        async with lifecycle.lifespan():
            pytest.fail("reaper startup should fail before the lifespan body")
    assert {
        "reaper_start",
        "reaper_shutdown",
        "scheduler_start",
        "scheduler_stop",
        "managed",
        "title",
        "notifier",
        "router",
        "terminal",
        "mcp",
    } <= set(events)


async def _noop() -> None:
    return None
