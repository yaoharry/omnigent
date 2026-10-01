"""Explicit service boundary for conversation compaction."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from omnigent.entities import CompactionData, NewConversationItem
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.llms import Client as LLMClient
from omnigent.runtime.compaction import (
    CompactionResult,
    SummaryMetadata,
    _CompactionState,
    compact,
)
from omnigent.runtime.history import (
    LoadedHistory,
    load_initial_history,
    prepare_messages,
)
from omnigent.spec import AgentSpec
from omnigent.spec.types import CompactionConfig, LLMConfig
from omnigent.stores import ArtifactStore, ConversationStore, FileStore

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompactionServices:
    """Dependencies required by an explicit compaction request."""

    conversation_store: ConversationStore
    file_store: FileStore | None
    artifact_store: ArtifactStore | None
    llm_client: LLMClient
    runner_client: Any | None = None


def apply_request_model_override(
    llm_config: LLMConfig,
    model_override: str | None,
) -> LLMConfig:
    """Return ``llm_config`` with an optional per-request model override."""
    if model_override is None:
        return llm_config
    merged_extra = {**llm_config.extra, "model_override": model_override}
    return LLMConfig(
        model=model_override,
        extra=merged_extra,
        connection=llm_config.connection,
        request_timeout=llm_config.request_timeout,
        retry=llm_config.retry,
    )


def route_bare_model_for_compaction(llm_config: LLMConfig) -> LLMConfig:
    """Prefix bare Databricks and Anthropic model ids for generic compaction."""
    model = llm_config.model
    if "/" in model:
        return llm_config
    if model.startswith("databricks-"):
        return replace(llm_config, model=f"databricks/{model}")
    if model.startswith("claude-"):
        return replace(llm_config, model=f"anthropic/{model}")
    return llm_config


def persist_compaction_item(
    summary: SummaryMetadata,
    task_id: str,
    conversation_id: str,
    conv_store: ConversationStore,
) -> None:
    """Persist a valid summary once, making replay idempotent."""
    if (
        not summary.text
        or not summary.last_item_id
        or summary.last_item_id.startswith("synthetic_")
    ):
        _logger.warning(
            "Skipping compaction persist for task %s: empty summary "
            "or bogus last_item_id (text=%r, last_item_id=%r)",
            task_id,
            summary.text[:80] if summary.text else None,
            summary.last_item_id,
        )
        return
    existing = conv_store.list_items(
        conversation_id,
        type="compaction",
        order="desc",
        limit=1,
    )
    if existing.data and existing.data[0].response_id == task_id:
        return
    conv_store.append(
        conversation_id,
        [
            NewConversationItem(
                type="compaction",
                response_id=task_id,
                data=CompactionData(
                    summary=summary.text,
                    last_item_id=summary.last_item_id,
                    model=summary.model,
                    token_count=summary.token_count,
                ),
            )
        ],
    )


async def compact_conversation_now(
    *,
    task_id: str,
    conversation_id: str,
    spec: AgentSpec,
    llm_config: LLMConfig,
    services: CompactionServices,
    instructions: str | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    model_override: str | None = None,
    preserve_recent_window: int | None = None,
) -> CompactionResult:
    """Run and persist an explicit compaction pass using injected services."""
    conv_store = services.conversation_store
    loaded: LoadedHistory = load_initial_history(conv_store, conversation_id)
    history = loaded.items
    if not history:
        return CompactionResult(messages=[], summary_metadata=None)

    effective_llm_config = route_bare_model_for_compaction(
        apply_request_model_override(llm_config, model_override)
    )
    compaction_config = spec.compaction
    if preserve_recent_window is not None:
        # The compaction helper's boundary is inclusive. Explicit /compact
        # uses zero so the latest completed item can be summarized immediately.
        trigger_threshold = compaction_config.trigger_threshold if compaction_config else 0.8
        compaction_config = CompactionConfig(
            trigger_threshold=trigger_threshold,
            recent_window=max(preserve_recent_window - 1, 0),
        )

    compaction_state = _CompactionState(
        context_window=None,
        last_summary=None,
        config=compaction_config,
        model=effective_llm_config.model,
        connection=effective_llm_config.connection,
        conversation_id=conversation_id,
    )
    _sys_instructions, messages, sys_tokens = prepare_messages(
        spec,
        effective_llm_config,
        history,
        instructions,
        tool_schemas or [],
        compaction_state,
        content_cache={},
        conversation_id=conversation_id,
        file_store=services.file_store,
        artifact_store=services.artifact_store,
    )
    from omnigent.llms.context_window import get_model_context_window

    result = await compact(
        messages,
        history,
        config=compaction_state.config,
        context_window=get_model_context_window(effective_llm_config.model),
        system_token_budget=sys_tokens,
        model=compaction_state.model,
        task_id=task_id,
        llm_client=services.llm_client,
        connection=compaction_state.connection,
        runner_client=services.runner_client,
        force=True,
        fail_on_summary_error=True,
        conversation_id=conversation_id,
    )
    if result.summary_metadata is None:
        raise OmnigentError(
            "Compaction did not produce a persisted summary. The conversation was "
            "left unchanged; check server logs for the summarization failure.",
            code=ErrorCode.INTERNAL_ERROR,
        )
    persist_compaction_item(
        result.summary_metadata,
        task_id,
        conversation_id,
        conv_store,
    )
    return result
