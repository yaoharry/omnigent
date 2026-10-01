"""Lifecycle ownership for the server's process-local resources.

The FastAPI application owns a small set of long-lived resources (runner
routing, terminal state, background tasks, and the MCP pool).  Keeping their
startup and teardown together is important: ASGI can fail while entering a
lifespan, before the application reaches its ``yield``.  ``ServerLifecycle``
uses an :class:`~contextlib.AsyncExitStack` so every resource registers its
cleanup immediately after it is acquired.

Harness subprocesses and session resource registries belong to the runner
process.  This module deliberately does not construct or bind either of
those runner-owned resources.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from omnigent.runtime import get_services
from omnigent.server.managed_sandbox_reaper import ManagedSandboxReaper
from omnigent.server.performance_metrics import (
    ServerMetricsOtelPublisher,
    ServerPerformanceMetrics,
)
from omnigent.server.scheduled import ScheduledTaskScheduler

if TYPE_CHECKING:
    from fastapi import FastAPI

    from omnigent.runner.routing import RunnerRouter
    from omnigent.runtime.services import RuntimeServices
    from omnigent.server.background_session_titles import BackgroundSessionTitleCoordinator
    from omnigent.server.mcp_pool import ServerMcpPool

_logger = logging.getLogger(__name__)

AsyncAction = Callable[[], Awaitable[None]]
SyncAction = Callable[[], None]
NotifierInstaller = Callable[[], SyncAction]
SchedulerFactory = Callable[[], ScheduledTaskScheduler | None]
ReaperFactory = Callable[[], ManagedSandboxReaper | None]
MetricsPublisher = Callable[
    [ServerPerformanceMetrics, ServerMetricsOtelPublisher], Awaitable[None]
]


async def _cancel_task(task: asyncio.Task[None]) -> None:
    """Cancel and drain one background task."""
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError):
        await task


@dataclass(slots=True)
class ServerLifecycle:
    """Own the server resources that must span an ASGI lifespan.

    The callbacks are deliberately narrow seams for the application-specific
    pieces of startup (default-agent seeding, policy registration, and account
    browser opening).  The lifecycle owner remains responsible for all
    long-lived resources and for registering cleanup before an awaited start.
    """

    app: FastAPI
    runner_router: RunnerRouter
    background_title_coordinator: BackgroundSessionTitleCoordinator
    mcp_pool: ServerMcpPool
    server_metrics: ServerPerformanceMetrics
    server_metrics_otel: ServerMetricsOtelPublisher
    server_config: dict[str, Any] | None
    install_notifier: NotifierInstaller
    make_ws_factory: Callable[[], Any]
    make_direct_attach_resolver: Callable[[], Any]
    initialize: AsyncAction
    cancel_managed_launches: AsyncAction
    publish_metrics: MetricsPublisher
    scheduler_factory: SchedulerFactory | None = None
    reaper_factory: ReaperFactory | None = None
    runtime_services: RuntimeServices | None = None

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Start the server resources and reliably tear them down."""
        async with AsyncExitStack() as stack:
            await self._start(stack)
            yield

    async def _start(self, stack: AsyncExitStack) -> None:
        """Acquire resources in dependency order and register their cleanup."""
        # Keep the server's existing thread budget and post-uvicorn logging
        # setup at the front of startup, before any store-backed callback.
        from anyio import to_thread

        to_thread.current_default_thread_limiter().total_tokens = 200

        from omnigent.telemetry import init_client

        init_client(config=self.server_config)
        log_level_name = os.environ.get("OMNIGENT_LOG_LEVEL", "INFO").upper()
        logging.getLogger("omnigent").setLevel(
            getattr(logging, log_level_name, logging.INFO),
        )

        services = self.runtime_services or get_services()
        # Register already-constructed resources before acquiring anything that
        # can fail. AsyncExitStack still runs the remaining callbacks when one
        # cleanup callback raises.
        stack.push_async_callback(self.mcp_pool.shutdown_all)
        terminal_registry = services.terminal_registry
        if terminal_registry is not None:
            stack.push_async_callback(terminal_registry.shutdown)
        stack.push_async_callback(self.runner_router.aclose)
        stack.push_async_callback(self.background_title_coordinator.shutdown)
        stack.push_async_callback(self.cancel_managed_launches)
        if terminal_registry is None:
            raise RuntimeError("runtime not initialized — call init() first")

        self._bind_service(stack, services, "runner_router", self.runner_router)

        uninstall_notifier = self.install_notifier()
        stack.callback(uninstall_notifier)

        self._bind_service(
            stack,
            services,
            "runner_ws_factory",
            self.make_ws_factory(),
        )
        self._bind_service(
            stack,
            services,
            "runner_direct_attach_resolver",
            self.make_direct_attach_resolver(),
        )

        # Application-specific initialization is intentionally before the
        # periodic/background resources. If it fails, the stack above still
        # closes the router, terminal registry, notifier, and MCP pool.
        await self.initialize()

        async def run_metrics() -> None:
            await self.publish_metrics(self.server_metrics, self.server_metrics_otel)

        metrics_task = asyncio.create_task(
            run_metrics(),
            name="server-metrics-publisher",
        )
        stack.push_async_callback(_cancel_task, metrics_task)

        if self.scheduler_factory is not None:
            scheduler = self.scheduler_factory()
            if scheduler is not None:
                self.app.state.scheduled_task_scheduler = scheduler
                # ``stop`` is synchronous and idempotent. Register it before
                # awaiting ``start`` so a failed DB read still clears timers.
                stack.callback(scheduler.stop)
                try:
                    await scheduler.start()
                except Exception as exc:
                    _logger.exception(
                        "scheduled task scheduler failed to start; continuing "
                        "without recurring tasks (%s)",
                        exc,
                    )

        if self.reaper_factory is not None:
            reaper = self.reaper_factory()
            if reaper is not None:
                self.app.state.managed_sandbox_reaper = reaper
                # Register before start: ``start`` may fail after creating its
                # task, and shutdown is safe when no task was created.
                stack.push_async_callback(reaper.shutdown)
                await reaper.start()

    @staticmethod
    def _bind_service(
        stack: AsyncExitStack,
        services: RuntimeServices,
        name: str,
        value: Any,
    ) -> None:
        previous = getattr(services, name)
        setattr(services, name, value)

        def restore() -> None:
            # A later owner may have replaced this process-wide compatibility
            # binding. Do not tear down or overwrite the later owner's value.
            if getattr(services, name) is value:
                setattr(services, name, previous)

        stack.callback(restore)
