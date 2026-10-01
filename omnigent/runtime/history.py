"""Conversation history and prompt preparation for the runtime.

These helpers are deliberately store-driven. The compatibility functions in
``runtime.workflow`` resolve process services before calling them, while the
new module can be exercised with isolated stores in unit tests.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from omnigent.entities import (
    NON_CONTENT_ITEM_TYPES,
    CompactionData,
    ConversationItem,
)
from omnigent.errors import StaleCursorError, restart_on_stale_cursor
from omnigent.runtime.compaction import (
    _CompactionState,
    compaction_to_history_items,
    count_tokens,
)
from omnigent.runtime.content_resolver import resolve_content_references
from omnigent.runtime.prompt import build_instructions, history_to_input_items
from omnigent.spec import AgentSpec
from omnigent.spec.types import LLMConfig
from omnigent.stores import ArtifactStore, ConversationStore, FileStore

_logger = logging.getLogger(__name__)


def prepare_messages(
    spec: AgentSpec,
    llm_config: LLMConfig,
    history: list[ConversationItem],
    instructions: str | None,
    tool_schemas: list[dict[str, Any]],
    compaction_state: _CompactionState,
    content_cache: dict[str, str] | None,
    *,
    conversation_id: str | None = None,
    file_store: FileStore | None = None,
    artifact_store: ArtifactStore | None = None,
) -> tuple[str, list[dict[str, Any]], int]:
    """Build system instructions and Responses API input items.

    ``file_store`` and ``artifact_store`` are explicit so this function does
    not need to reach into process-global runtime state.
    """
    del llm_config  # Kept in the signature for compatibility with the old helper.
    sys_instructions = build_instructions(spec, instructions, tool_schemas)
    resolved = history
    if file_store is not None and artifact_store is not None:
        resolved = resolve_content_references(
            history,
            file_store,
            artifact_store,
            content_cache,
            session_id=conversation_id,
        )
    messages = history_to_input_items(resolved, preserve_framework_notices=True)
    sys_tokens = count_tokens(
        [{"role": "system", "content": sys_instructions}],
        compaction_state.model,
    )
    return sys_instructions, messages, sys_tokens


@restart_on_stale_cursor
def fetch_all_items(
    conv_store: ConversationStore,
    conversation_id: str,
    after: str | None = None,
) -> list[ConversationItem]:
    """Fetch every conversation item after ``after`` across all pages."""
    all_items: list[ConversationItem] = []
    cursor = after
    while True:
        page = conv_store.list_items(conversation_id, after=cursor)
        all_items.extend(page.data)
        if not page.has_more:
            break
        cursor = page.last_id
    return all_items


def find_latest_compaction_item(
    conv_store: ConversationStore,
    conversation_id: str,
) -> ConversationItem | None:
    """Return the newest persisted compaction item for a conversation."""
    page = conv_store.list_items(
        conversation_id,
        type="compaction",
        order="desc",
        limit=1,
    )
    return page.data[0] if page.data else None


@dataclass(frozen=True)
class LoadedHistory:
    """History items plus the compaction cursor used to load them."""

    items: list[ConversationItem]
    last_compaction_created_at: int | None = None


def load_initial_history(
    conv_store: ConversationStore,
    conversation_id: str,
) -> LoadedHistory:
    """Load the compacted tail, or the complete conversation when un-compacted."""
    compaction_item = find_latest_compaction_item(conv_store, conversation_id)
    if compaction_item is None:
        return LoadedHistory(
            items=[
                item
                for item in fetch_all_items(conv_store, conversation_id)
                if item.type not in NON_CONTENT_ITEM_TYPES
            ]
        )

    assert isinstance(compaction_item.data, CompactionData)
    last_id = compaction_item.data.last_item_id
    if not compaction_item.data.summary or not last_id or last_id.startswith("synthetic_"):
        _logger.warning(
            "Ignoring broken compaction item %s for %s: empty summary or bogus last_item_id=%r",
            compaction_item.id,
            conversation_id,
            last_id,
        )
        return LoadedHistory(
            items=[
                item
                for item in fetch_all_items(conv_store, conversation_id)
                if item.type not in NON_CONTENT_ITEM_TYPES
            ]
        )

    try:
        recent_items = fetch_all_items(
            conv_store,
            conversation_id,
            after=compaction_item.data.last_item_id,
        )
    except StaleCursorError:
        _logger.warning(
            "Compaction anchor %r for %s no longer exists; loading full history",
            last_id,
            conversation_id,
        )
        return LoadedHistory(
            items=[
                item
                for item in fetch_all_items(conv_store, conversation_id)
                if item.type not in NON_CONTENT_ITEM_TYPES
            ]
        )

    content_items = [i for i in recent_items if i.type not in NON_CONTENT_ITEM_TYPES]
    return LoadedHistory(
        items=compaction_to_history_items(compaction_item) + content_items,
        last_compaction_created_at=compaction_item.created_at,
    )
