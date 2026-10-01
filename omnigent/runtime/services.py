"""Process-scoped services used by the runtime.

The server and runner are separate processes, so these services are local to
the process that owns them. The object is intentionally a small state
container: public getter/setter functions in :mod:`omnigent.runtime` remain
the compatibility boundary while callers migrate away from module attributes.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, cast

from omnigent.runtime.caps import RuntimeCaps

if TYPE_CHECKING:
    from omnigent.runner.resource_registry import SessionResourceRegistry
    from omnigent.runner.routing import RunnerRouter
    from omnigent.runtime.agent_cache import AgentCache
    from omnigent.runtime.harnesses.process_manager import HarnessProcessManager
    from omnigent.stores import (
        AgentStore,
        ArtifactStore,
        ConversationStore,
        FileStore,
    )
    from omnigent.stores.comment_store import CommentStore
    from omnigent.stores.policy_store import PolicyStore
    from omnigent.terminals import TerminalRegistry
    from omnigent.tools import ToolManager
    from omnigent.tools.base import ToolContext


@dataclass
class DispatchCapability:
    """Legacy process-local handle for background tool dispatch.

    The current runner dispatch path does not register these handles. The
    shape remains available through the compatibility API until the 0.18
    removal window so older embedders can migrate explicitly.
    """

    tool_mgr: ToolManager
    tool_ctx: ToolContext
    policy_engine: Any
    conversation_id: str
    root_task_id: str | None
    agent_name: str


@dataclass
class RuntimeServices:
    """Process-scoped runtime dependencies and lifecycle hooks.

    ``init()`` replaces persistent stores, caps, and the terminal registry.
    Runner hooks and compatibility state are deliberately retained until the
    corresponding lifecycle owner clears them, matching the previous module
    state semantics.

    This is not an application-isolation mechanism: a process has one active
    service set. Multi-process isolation comes from the server/runner process
    boundary; per-request state should be passed explicitly.
    """

    conversation_store: ConversationStore | None = None
    agent_store: AgentStore | None = None
    agent_cache: AgentCache | None = None
    file_store: FileStore | None = None
    artifact_store: ArtifactStore | None = None
    comment_store: CommentStore | None = None
    policy_store: PolicyStore | None = None
    caps: RuntimeCaps = field(default_factory=RuntimeCaps)

    # Server-owned registries and runner lifecycle hooks.
    terminal_registry: TerminalRegistry | None = None
    resource_registry: SessionResourceRegistry | None = None
    harness_process_manager: HarnessProcessManager | None = None
    runner_client: Any | None = None
    runner_router: RunnerRouter | None = None
    runner_ws_factory: Any | None = None
    runner_direct_attach_resolver: Any | None = None
    runner_id: str | None = None

    # Compatibility state retained for the old public APIs. No production
    # caller currently registers either value; remove with those APIs in 0.18.
    _tool_manager_var: ContextVar[ToolManager | None] = field(
        default_factory=lambda: cast(
            ContextVar[Any],
            ContextVar("_tool_manager", default=None),
        ),
        repr=False,
    )
    _dispatch_capabilities: dict[str, DispatchCapability] = field(
        default_factory=dict,
        repr=False,
    )

    def initialize(
        self,
        *,
        conversation_store: ConversationStore,
        agent_store: AgentStore,
        agent_cache: AgentCache,
        file_store: FileStore | None = None,
        artifact_store: ArtifactStore | None = None,
        comment_store: CommentStore | None = None,
        policy_store: PolicyStore | None = None,
        caps: RuntimeCaps | None = None,
    ) -> None:
        """Replace startup-owned services while preserving lifecycle hooks."""
        from omnigent.terminals import TerminalRegistry

        self.conversation_store = conversation_store
        self.agent_store = agent_store
        self.agent_cache = agent_cache
        self.file_store = file_store
        self.artifact_store = artifact_store
        self.comment_store = comment_store
        self.policy_store = policy_store
        self.caps = caps if caps is not None else RuntimeCaps()
        self.terminal_registry = TerminalRegistry()


_services: RuntimeServices | None = None
_services_lock = Lock()


def get_services() -> RuntimeServices:
    """Return the process-scoped runtime services object."""
    global _services
    services = _services
    if services is None:
        with _services_lock:
            services = _services
            if services is None:
                services = RuntimeServices()
                _services = services
    return services
