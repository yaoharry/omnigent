"""Helpers for resolving named agents in an :class:`AgentSpec` tree."""

from __future__ import annotations

from omnigent.spec.types import AgentSpec


def find_sub_agent(spec: AgentSpec, name: str) -> AgentSpec | None:
    """Resolve a named sub-agent, including the synthesized web researcher."""
    found = search_sub_agent_tree(spec, name)
    if found is not None:
        return found

    # The web researcher is synthesized in memory by WebFetchTool and is not
    # serialized into the parent bundle. Keep this import lazy so the spec
    # package does not depend on the tools package during normal parsing.
    from omnigent.tools.builtins.web_fetch import (
        RESEARCHER_NAME,
        reconstruct_researcher_spec,
    )

    if name == RESEARCHER_NAME:
        return reconstruct_researcher_spec(spec)
    return None


def search_sub_agent_tree(spec: AgentSpec, name: str) -> AgentSpec | None:
    """Recursively search ``spec.sub_agents`` for ``name``."""
    for sub_agent in spec.sub_agents:
        if sub_agent.name == name:
            return sub_agent
        found = search_sub_agent_tree(sub_agent, name)
        if found is not None:
            return found
    return None
