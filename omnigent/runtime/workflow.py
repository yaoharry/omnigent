"""Deprecated compatibility facade for the former workflow module.

Provider and spawn configuration now live under ``omnigent.harnesses.config``;
history/compaction and spec-tree helpers live in their owning modules. This
facade remains for one deprecation window while downstream callers migrate.
It is scheduled for removal in 0.18.
"""

from __future__ import annotations

# This module intentionally re-exports private compatibility names.
# ruff: noqa: F401
import json
from dataclasses import dataclass
from typing import Any

from omnigent.harnesses.config.providers import (
    AgentHarnessType,
    UcodeHarnessConfig,
    _apply_cli_config_databricks_to_pi,
    _apply_provider_family,
    _apply_provider_to_openai_agents,
    _apply_provider_to_pi,
    _catalog_default_model,
    _configure_brokered_codex_with_ucode,
    _inject_ucode_agent_state,
    _legacy_databricks_provider,
    _load_global_auth,
    _optional_provider_family,
    _origin_of,
    _provider_auth_command,
    _provider_harness_name,
    _resolve_bound_launch_model,
    _resolve_catalog_default_model,
    _resolve_provider_for_build,
    _resolve_spec_model,
    _synthesize_codex_api_key_provider,
    _synthesize_databricks_provider,
    configure_agent_harness_with_provider,
    configure_agent_harness_with_ucode,
)
from omnigent.harnesses.config.spawn_env import (
    _add_claude_sdk_skills_env,
    _apply_harness_path_override,
    _build_acp_cli_spawn_env,
    _build_acp_spawn_env,
    _build_antigravity_spawn_env,
    _build_claude_sdk_spawn_env,
    _build_codex_spawn_env,
    _build_copilot_spawn_env,
    _build_cursor_spawn_env,
    _build_goose_spawn_env,
    _build_hermes_spawn_env,
    _build_kimi_spawn_env,
    _build_openai_agents_sdk_spawn_env,
    _build_pi_spawn_env,
    _build_qwen_spawn_env,
    _config_flag_is_true,
    _resolve_retry_policy,
    _serialize_os_env,
    _serialize_retry_policy,
    _set_openai_agents_reasoning_item_id_policy_env,
)
from omnigent.llms import Client as LLMClient
from omnigent.runtime import (
    get_artifact_store,
    get_conversation_store,
    get_file_store,
    get_runner_router,
)
from omnigent.runtime.compaction_service import (
    CompactionResult,
    CompactionServices,
    apply_request_model_override,
    persist_compaction_item,
    route_bare_model_for_compaction,
)
from omnigent.runtime.compaction_service import (
    compact_conversation_now as _compact_conversation_now,
)
from omnigent.runtime.history import (
    LoadedHistory,
    fetch_all_items,
    find_latest_compaction_item,
    load_initial_history,
)
from omnigent.runtime.history import (
    prepare_messages as _prepare_messages_explicit,
)
from omnigent.spec import AgentSpec
from omnigent.spec.tree import find_sub_agent, search_sub_agent_tree
from omnigent.spec.types import LLMConfig

# Historical private names retained as aliases during the deprecation window.
_apply_request_model_override = apply_request_model_override
_find_latest_compaction_item = find_latest_compaction_item
_load_initial_history = load_initial_history
_maybe_persist_compaction_item = persist_compaction_item
_route_bare_model_for_compaction = route_bare_model_for_compaction
_find_spec_by_name = find_sub_agent
_search_sub_agent_tree = search_sub_agent_tree
_LoadedHistory = LoadedHistory

_llm_client: LLMClient | None = None


def _get_llm_client() -> LLMClient:
    """Return the legacy shared client for compatibility callers."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _get_runner_client_for_compaction(conversation_id: str | None) -> Any | None:
    """Resolve the pinned runner client for the deprecated compaction API."""
    if conversation_id is None:
        return None
    router = get_runner_router()
    if router is None:
        return None
    routed = router.client_for_existing_conversation(conversation_id)
    return routed.client if routed else None


def prepare_messages(
    spec: AgentSpec,
    llm_config: LLMConfig,
    history: list[Any],
    instructions: str | None,
    tool_schemas: list[dict[str, Any]],
    compaction_state: Any,
    content_cache: dict[str, str] | None,
    *,
    conversation_id: str | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Resolve process stores before calling the explicit history helper."""
    return _prepare_messages_explicit(
        spec,
        llm_config,
        history,
        instructions,
        tool_schemas,
        compaction_state,
        content_cache,
        conversation_id=conversation_id,
        file_store=get_file_store(),
        artifact_store=get_artifact_store(),
    )


_prepare_messages = prepare_messages


async def compact_conversation_now(
    *,
    task_id: str,
    conversation_id: str,
    spec: AgentSpec,
    llm_config: LLMConfig,
    instructions: str | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    model_override: str | None = None,
    preserve_recent_window: int | None = None,
) -> Any:
    """Compatibility adapter resolving process services for compaction."""
    conversation_store = get_conversation_store()
    loaded_history = _load_initial_history(conversation_store, conversation_id)
    if not loaded_history.items:
        return CompactionResult(messages=[], summary_metadata=None)
    services = CompactionServices(
        conversation_store=conversation_store,
        file_store=get_file_store(),
        artifact_store=get_artifact_store(),
        llm_client=_get_llm_client(),
        runner_client=_get_runner_client_for_compaction(conversation_id),
    )
    return await _compact_conversation_now(
        task_id=task_id,
        conversation_id=conversation_id,
        spec=spec,
        llm_config=llm_config,
        services=services,
        instructions=instructions,
        tool_schemas=tool_schemas,
        model_override=model_override,
        preserve_recent_window=preserve_recent_window,
    )


def _strip_mcp_tool_prefix(name: str) -> str:
    """Strip one ``mcp__<server>__`` prefix from a tool name."""
    if name.startswith("mcp__"):
        parts = name.split("__", 2)
        if len(parts) == 3:
            return parts[2]
    return name


@dataclass
class _AsyncToolHandle:
    """LLM-facing handle returned for an asynchronous tool dispatch."""

    task_id: str
    tool_name: str
    status: str
    message: str

    def to_handle_json(self) -> str:
        """Serialize the handle for the executor tool-result boundary."""
        return json.dumps(
            {
                "task_id": self.task_id,
                "tool_name": self.tool_name,
                "status": self.status,
                "message": self.message,
            }
        )


def _async_handle_message(task_id: str, tool_name: str) -> str:
    """Build the instruction attached to a new asynchronous tool handle."""
    return (
        f"Tool {tool_name!r} dispatched asynchronously. "
        f"The result will be auto-delivered as a system message "
        f"when ready. To abort, call sys_cancel_task with "
        f"task_id={task_id!r}."
    )


__all__ = [
    "CompactionServices",
    "LoadedHistory",
    "UcodeHarnessConfig",
    "apply_request_model_override",
    "compact_conversation_now",
    "configure_agent_harness_with_provider",
    "configure_agent_harness_with_ucode",
    "fetch_all_items",
    "load_initial_history",
    "prepare_messages",
    "route_bare_model_for_compaction",
]
