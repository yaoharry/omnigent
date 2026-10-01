"""Public compatibility API for the process-scoped runtime services."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from omnigent.runtime.caps import RuntimeCaps
from omnigent.runtime.services import DispatchCapability, get_services

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


def init(
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
    """Initialize startup-owned stores, caps, and the terminal registry."""
    get_services().initialize(
        conversation_store=conversation_store,
        agent_store=agent_store,
        agent_cache=agent_cache,
        file_store=file_store,
        artifact_store=artifact_store,
        comment_store=comment_store,
        policy_store=policy_store,
        caps=caps,
    )


def get_conversation_store() -> ConversationStore:
    """Return the initialized conversation store."""
    store = get_services().conversation_store
    if store is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return store


def get_agent_store() -> AgentStore:
    """Return the initialized agent store."""
    store = get_services().agent_store
    if store is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return store


def get_file_store() -> FileStore | None:
    """Return the optional file store, or ``None`` when unconfigured."""
    return get_services().file_store


def get_artifact_store() -> ArtifactStore | None:
    """Return the optional artifact store, or ``None`` when unconfigured."""
    return get_services().artifact_store


def get_comment_store() -> CommentStore | None:
    """Return the optional comment store, or ``None`` when unconfigured."""
    return get_services().comment_store


def get_policy_store() -> PolicyStore | None:
    """Return the optional policy store, or ``None`` when unconfigured."""
    return get_services().policy_store


def get_agent_cache() -> AgentCache:
    """Return the initialized agent cache."""
    cache = get_services().agent_cache
    if cache is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return cache


def get_tool_manager() -> ToolManager:
    """Return the legacy workflow ToolManager (deprecated; remove in 0.18)."""
    mgr = get_services()._tool_manager_var.get()
    if mgr is None:
        raise RuntimeError("no ToolManager set for this workflow")
    return mgr


def set_tool_manager(mgr: ToolManager | None) -> None:
    """Set the legacy workflow ToolManager (deprecated; remove in 0.18)."""
    get_services()._tool_manager_var.set(mgr)


def register_dispatch_capability(
    parent_task_id: str,
    capability: DispatchCapability,
) -> None:
    """Register legacy dispatch state (deprecated; remove in 0.18)."""
    get_services()._dispatch_capabilities[parent_task_id] = capability


def get_dispatch_capability(parent_task_id: str) -> DispatchCapability | None:
    """Return legacy dispatch state (deprecated; remove in 0.18)."""
    return get_services()._dispatch_capabilities.get(parent_task_id)


def unregister_dispatch_capability(parent_task_id: str) -> None:
    """Remove legacy dispatch state (deprecated; remove in 0.18)."""
    get_services()._dispatch_capabilities.pop(parent_task_id, None)


def get_caps() -> RuntimeCaps:
    """Return the configured runtime caps."""
    return get_services().caps


def get_terminal_registry() -> TerminalRegistry:
    """Return the initialized server terminal registry."""
    registry = get_services().terminal_registry
    if registry is None:
        raise RuntimeError("runtime not initialized — call init() first")
    return registry


def get_resource_registry() -> SessionResourceRegistry | None:
    """Return the legacy resource registry (deprecated; remove in 0.18)."""
    return get_services().resource_registry


def set_resource_registry(registry: SessionResourceRegistry | None) -> None:
    """Set the legacy resource registry (deprecated; remove in 0.18)."""
    get_services().resource_registry = registry


def set_runner_client(client: Any) -> None:
    """Set the legacy fixed runner client used by focused integrations."""
    get_services().runner_client = client


def get_runner_client() -> Any:
    """Return the legacy fixed runner client, or ``None``."""
    return get_services().runner_client


def set_runner_router(router: RunnerRouter | None) -> None:
    """Set the conversation-aware runner router, or clear it with ``None``."""
    get_services().runner_router = router


def get_runner_router() -> RunnerRouter | None:
    """Return the configured conversation-aware runner router, or ``None``."""
    return get_services().runner_router


def set_runner_ws_factory(factory: Any) -> None:
    """Set the runner WebSocket connection factory, or clear it with ``None``."""
    get_services().runner_ws_factory = factory


def get_runner_ws_factory() -> Any:
    """Return the runner WebSocket connection factory, or ``None``."""
    return get_services().runner_ws_factory


def set_runner_direct_attach_resolver(resolver: Any) -> None:
    """Set the runner direct-attach resolver, or clear it with ``None``."""
    get_services().runner_direct_attach_resolver = resolver


def get_runner_direct_attach_resolver() -> Any:
    """Return the runner direct-attach resolver, or ``None``."""
    return get_services().runner_direct_attach_resolver


def set_runner_id(runner_id: str | None) -> None:
    """Set the legacy process runner id (deprecated; remove in 0.18)."""
    get_services().runner_id = runner_id


def get_runner_id() -> str | None:
    """Return the legacy process runner id (deprecated; remove in 0.18)."""
    return get_services().runner_id


def set_harness_process_manager(manager: HarnessProcessManager | None) -> None:
    """Set the legacy harness process manager (deprecated; remove in 0.18)."""
    get_services().harness_process_manager = manager


def get_harness_process_manager() -> HarnessProcessManager:
    """Return the legacy harness process manager (deprecated; remove in 0.18)."""
    manager = get_services().harness_process_manager
    if manager is None:
        raise RuntimeError(
            "HarnessProcessManager not initialized — Omnigent lifespan startup "
            "must call set_harness_process_manager() before any workflow "
            "dispatches to a non-default harness"
        )
    return manager
