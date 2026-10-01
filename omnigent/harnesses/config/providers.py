"""Provider and credential resolution for harness launches.

This module owns provider selection and ucode/config translation. It deliberately
does not import runtime stores, compaction, or workflow orchestration.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from omnigent.cli_invocation import cli_invocation
from omnigent.errors import ErrorCode, OmnigentError
from omnigent.inner.model_egress import (
    UCODE_SIGNER_BINDING_ID,
    registered_model_provider_binding,
)
from omnigent.models.model_catalog import resolve_catalog_model
from omnigent.models.model_resolver import ModelResolutionError
from omnigent.onboarding.databricks_config import get_workspace_url_for_profile
from omnigent.onboarding.detected import effective_config_with_detected
from omnigent.onboarding.provider_config import (
    ANTHROPIC_FAMILY,
    BEDROCK_KIND,
    CHAT_WIRE_API,
    CLI_CONFIG_KIND,
    DATABRICKS_KIND,
    KEY_KIND,
    OPENAI_FAMILY,
    RESPONSES_WIRE_API,
    SUBSCRIPTION_KIND,
    FamilyConfig,
    ProviderEntry,
    default_provider_for_harness,
    first_available_provider,
    harness_family,
    load_config,
    load_providers,
)
from omnigent.onboarding.ucode_state import UcodeAgentState, read_ucode_state
from omnigent.spec import AgentSpec
from omnigent.spec.parser import check_unresolved_env_vars
from omnigent.spec.types import ApiKeyAuth, DatabricksAuth, ProviderAuth
from omnigent.util.env_credentials import expand_envvars_with_omnigent_prefix

_logger = logging.getLogger(__name__)

AgentHarnessType = Literal[
    "claude-sdk", "codex", "pi", "openai-agents-sdk", "antigravity", "kimi", "qwen", "goose"
]


@dataclass(frozen=True)
class UcodeHarnessConfig:
    """Env-var mapping for one harness's ucode agent state.

    :param agent_name: ucode agent key, e.g. ``"claude"`` or ``"codex"``.
    :param model_key: Harness model env var.
    :param base_url_key: Harness gateway base URL env var.
    :param base_url_family: Optional provider key to use when the ucode agent
        entry only has provider-specific base URLs, e.g. ``"claude"``.
    :param base_urls_key: Optional harness gateway base URLs env var for
        agents with multiple provider URLs.
    :param host_key: Harness gateway workspace host env var.
    :param auth_key: Optional harness gateway auth command env var.
    :param refresh_key: Optional harness gateway auth refresh interval env var.
    :param catalog_family: Normalized Databricks catalog family used when
        neither the spec nor ucode state names a model.
    """

    agent_name: str
    model_key: str
    base_url_key: str
    base_url_family: str | None
    base_urls_key: str | None
    host_key: str
    auth_key: str | None
    refresh_key: str | None
    catalog_family: str


_UCODE_HARNESS_CONFIGS: dict[AgentHarnessType, UcodeHarnessConfig] = {
    "claude-sdk": UcodeHarnessConfig(
        agent_name="claude",
        model_key="HARNESS_CLAUDE_SDK_MODEL",
        base_url_key="HARNESS_CLAUDE_SDK_GATEWAY_BASE_URL",
        base_url_family="claude",
        base_urls_key=None,
        host_key="HARNESS_CLAUDE_SDK_GATEWAY_HOST",
        auth_key="HARNESS_CLAUDE_SDK_GATEWAY_AUTH_COMMAND",
        refresh_key="HARNESS_CLAUDE_SDK_GATEWAY_AUTH_REFRESH_INTERVAL_MS",
        catalog_family="claude",
    ),
    "codex": UcodeHarnessConfig(
        agent_name="codex",
        model_key="HARNESS_CODEX_MODEL",
        base_url_key="HARNESS_CODEX_GATEWAY_BASE_URL",
        base_url_family="codex",
        base_urls_key=None,
        host_key="HARNESS_CODEX_GATEWAY_HOST",
        auth_key="HARNESS_CODEX_GATEWAY_AUTH_COMMAND",
        refresh_key="HARNESS_CODEX_GATEWAY_AUTH_REFRESH_INTERVAL_MS",
        catalog_family="openai",
    ),
    "pi": UcodeHarnessConfig(
        agent_name="pi",
        model_key="HARNESS_PI_MODEL",
        base_url_key="HARNESS_PI_GATEWAY_BASE_URL",
        base_url_family="claude",
        base_urls_key="HARNESS_PI_GATEWAY_BASE_URLS",
        host_key="HARNESS_PI_GATEWAY_HOST",
        auth_key="HARNESS_PI_GATEWAY_AUTH_COMMAND",
        refresh_key="HARNESS_PI_GATEWAY_AUTH_REFRESH_INTERVAL_MS",
        catalog_family="claude",
    ),
    "openai-agents-sdk": UcodeHarnessConfig(
        agent_name="codex",
        model_key="HARNESS_OPENAI_AGENTS_MODEL",
        base_url_key="HARNESS_OPENAI_AGENTS_GATEWAY_BASE_URL",
        base_url_family="codex",
        base_urls_key=None,
        host_key="HARNESS_OPENAI_AGENTS_GATEWAY_HOST",
        auth_key="HARNESS_OPENAI_AGENTS_GATEWAY_AUTH_COMMAND",
        refresh_key=None,
        catalog_family="openai",
    ),
    "qwen": UcodeHarnessConfig(
        agent_name="qwen",
        model_key="HARNESS_QWEN_MODEL",
        base_url_key="HARNESS_QWEN_GATEWAY_BASE_URL",
        base_url_family="openai",
        base_urls_key=None,
        host_key="HARNESS_QWEN_GATEWAY_HOST",
        auth_key="HARNESS_QWEN_GATEWAY_AUTH_COMMAND",
        refresh_key=None,
        catalog_family="openai",
    ),
    # NB: ``antigravity`` is intentionally absent. Unlike the gateway
    # harnesses above, the Antigravity SDK authenticates Gemini-natively
    # (API key or Vertex AI) and has no OpenAI-compatible ``base_url``, so it
    # has no ucode gateway entry — ``_build_antigravity_spawn_env`` threads
    # ``HARNESS_ANTIGRAVITY_API_KEY`` / ``_VERTEX`` directly.
}


def configure_agent_harness_with_ucode(
    env: dict[str, str],
    profile: str | None,
    *,
    harness_type: AgentHarnessType,
) -> None:
    """Inject per-harness model, URL, and auth values from ucode state.

    The harness-specific constants live here so callers only declare which
    agent harness they are configuring. ucode's per-agent ``agents`` entries
    are the source of truth for gateway URLs and auth commands.

    :param env: Mutable spawn-env dict, modified in place.
    :param profile: The ``executor.profile`` / provider-config value,
        e.g. ``"oss"``.  ``None`` short-circuits the entire lookup.
    :param harness_type: Canonical harness type, e.g. ``"claude-sdk"``.
    """
    if not profile:
        return
    workspace_url = get_workspace_url_for_profile(profile)
    if workspace_url is None:
        return
    state = read_ucode_state(workspace_url)
    if state is None:
        return
    config = _UCODE_HARNESS_CONFIGS[harness_type]
    agent_state = state.agent(config.agent_name)
    if agent_state is None:
        return
    _inject_ucode_agent_state(
        env,
        agent_state,
        model_key=config.model_key,
        base_url_key=config.base_url_key,
        base_url_family=config.base_url_family,
        base_urls_key=config.base_urls_key,
        host_key=config.host_key,
        auth_key=config.auth_key,
        refresh_key=config.refresh_key,
        workspace_url=state.workspace_host,
    )
    # When ucode caches no model, resolve a Databricks endpoint so the CLI
    # cannot fall back to a direct-provider model the gateway rejects.
    if config.model_key not in env:
        env[config.model_key] = _resolve_catalog_default_model(
            "databricks",
            config.catalog_family,
            context=f"ucode {harness_type!r} gateway",
        )


def _configure_brokered_codex_with_ucode(
    env: dict[str, str],
    spec: AgentSpec,
    provider: ProviderEntry,
) -> None:
    """Bind the supported Databricks Codex route to signer-only authority."""
    profile = provider.profile
    if os.environ.get("HARNESS_CODEX_GATEWAY_AUTH_COMMAND"):
        raise OmnigentError(
            "signer-backed Codex conflicts with HARNESS_CODEX_GATEWAY_AUTH_COMMAND",
            code=ErrorCode.INVALID_INPUT,
        )
    sandbox = spec.os_env.sandbox if spec.os_env is not None else None
    if sandbox is None or sandbox.type == "none":
        raise OmnigentError(
            "signer-backed Codex requires an active os_env sandbox",
            code=ErrorCode.INVALID_INPUT,
        )
    if not spec.model_egress:
        raise OmnigentError(
            "signer-backed Codex requires an explicit model_egress grant",
            code=ErrorCode.INVALID_INPUT,
        )
    if sandbox.egress_rules:
        raise OmnigentError(
            "signer-backed Codex does not support os_env.sandbox.egress_rules; "
            "brokered sessions are model-only",
            code=ErrorCode.INVALID_INPUT,
        )
    if not profile:
        raise OmnigentError(
            "signer-backed Codex requires an explicit Databricks profile",
            code=ErrorCode.INVALID_INPUT,
        )
    workspace_url = get_workspace_url_for_profile(profile)
    state = read_ucode_state(workspace_url) if workspace_url is not None else None
    agent_state = state.agent("codex") if state is not None else None
    endpoint = agent_state.base_url if agent_state is not None else None
    if state is None or endpoint is None:
        raise OmnigentError(
            "signer-backed Codex requires configured ucode Codex state; run `ucode configure`",
            code=ErrorCode.INVALID_INPUT,
        )
    assert agent_state is not None
    try:
        registered_model_provider_binding(
            binding_id=UCODE_SIGNER_BINDING_ID,
            trusted_session_endpoint=endpoint,
            trusted_host=state.workspace_host,
        )
    except ValueError as exc:
        raise OmnigentError(str(exc), code=ErrorCode.INVALID_INPUT) from exc
    if "HARNESS_CODEX_MODEL" not in env:
        env["HARNESS_CODEX_MODEL"] = agent_state.model or _resolve_catalog_default_model(
            "databricks",
            "openai",
            context="ucode 'codex' signer",
        )
    env["HARNESS_CODEX_SIGNER_PROVIDER"] = UCODE_SIGNER_BINDING_ID
    env["HARNESS_CODEX_SIGNER_ENDPOINT"] = endpoint
    env["HARNESS_CODEX_GATEWAY_HOST"] = state.workspace_host
    env["HARNESS_CODEX_DATABRICKS_PROFILE"] = profile
    env["HARNESS_CODEX_MODEL_EGRESS"] = json.dumps(spec.model_egress)


def _inject_ucode_agent_state(
    env: dict[str, str],
    state: UcodeAgentState,
    *,
    model_key: str,
    base_url_key: str,
    base_url_family: str | None,
    base_urls_key: str | None,
    host_key: str,
    auth_key: str | None,
    refresh_key: str | None,
    workspace_url: str,
) -> None:
    """Copy one ucode agent entry into harness env vars.

    :param env: Mutable spawn-env dict, modified in place.
    :param state: Parsed ucode per-agent state.
    :param model_key: Harness model env var, e.g. ``"HARNESS_CODEX_MODEL"``.
    :param base_url_key: Harness gateway base URL env var.
    :param base_url_family: Optional provider key to use when ``state`` has
        provider-specific base URLs instead of a single base URL.
    :param base_urls_key: Optional harness gateway base URLs env var for
        agents with multiple provider URLs.
    :param host_key: Harness gateway workspace host env var.
    :param auth_key: Optional harness gateway auth command env var.
    :param refresh_key: Optional harness gateway auth refresh interval env var.
    :param workspace_url: Workspace URL for token refresh commands.
    """
    if model_key not in env and state.model:
        env[model_key] = state.model
    base_url = state.base_url
    if base_url is None and base_url_family is not None:
        base_url = state.base_urls.get(base_url_family)
    if base_url:
        env[base_url_key] = base_url
    if base_urls_key and state.base_urls:
        env[base_urls_key] = json.dumps(state.base_urls, sort_keys=True)
    env[host_key] = workspace_url
    if auth_key and state.auth_command:
        env[auth_key] = state.auth_command
    if refresh_key and state.auth_refresh_interval_ms is not None:
        env[refresh_key] = str(state.auth_refresh_interval_ms)


# Maps single-family harnesses to the generic-provider family they consume.
# (``pi`` is handled separately — it consumes both families.) The keys are
# the canonical harness names used by the Chunk-1a provider-config layer
# (``omnigent/onboarding/provider_config.py`` ``_HARNESS_FAMILY``);
# ``openai-agents`` (no ``-sdk``) is that layer's name for the
# openai-agents-sdk harness, so :func:`_provider_harness_name` translates.
_PROVIDER_HARNESS_FAMILY: dict[AgentHarnessType, str] = {
    "claude-sdk": ANTHROPIC_FAMILY,
    "codex": OPENAI_FAMILY,
    "openai-agents-sdk": OPENAI_FAMILY,
    # Antigravity is Gemini-native but routes generic-provider traffic over
    # the OpenAI-compatible wire (OpenRouter / LiteLLM / Databricks gateway),
    # so it consumes the ``openai`` family like openai-agents-sdk.
    "antigravity": OPENAI_FAMILY,
    # Qwen Code routes through OpenAI-compatible providers (like Kimi v1).
    "qwen": OPENAI_FAMILY,
}

# Maps harnesses that gate the vendor-neutral gateway transport on a
# ``HARNESS_*_GATEWAY`` truthy flag to that env var name. The flag enables
# the executor's gateway path (base URL + token command + model) regardless
# of which producer fed it — generic providers or the Databricks AI gateway.
# ``openai-agents-sdk`` is absent: its executor takes the API key / base URL
# directly with no such gate (see :func:`_apply_provider_to_openai_agents`).
_HARNESS_GATEWAY_FLAG: dict[AgentHarnessType, str] = {
    "claude-sdk": "HARNESS_CLAUDE_SDK_GATEWAY",
    "codex": "HARNESS_CODEX_GATEWAY",
    "pi": "HARNESS_PI_GATEWAY",
    "qwen": "HARNESS_QWEN_GATEWAY",
}

# Maps a generic-provider family to the key pi uses in its
# ``HARNESS_PI_GATEWAY_BASE_URLS`` JSON object (pi's own family naming).
_PI_FAMILY_KEY: dict[str, str] = {
    ANTHROPIC_FAMILY: "claude",
    OPENAI_FAMILY: "openai",
}

# Per-harness ``HARNESS_*_DATABRICKS_PROFILE`` env var name, used by the
# databricks-kind provider branch (which delegates to the existing ucode
# path). This stays Databricks-named: it is a ``~/.databrickscfg`` profile
# the executor uses for Databricks-specific credential resolution / token
# refresh, not part of the vendor-neutral gateway transport.
# ``openai-agents-sdk`` uses the same var name but has no enable flag.
_HARNESS_DATABRICKS_PROFILE: dict[AgentHarnessType, str] = {
    "claude-sdk": "HARNESS_CLAUDE_SDK_DATABRICKS_PROFILE",
    "codex": "HARNESS_CODEX_DATABRICKS_PROFILE",
    "pi": "HARNESS_PI_DATABRICKS_PROFILE",
    "openai-agents-sdk": "HARNESS_OPENAI_AGENTS_DATABRICKS_PROFILE",
    "qwen": "HARNESS_QWEN_DATABRICKS_PROFILE",
    # NB: no ``antigravity`` — it has no Databricks/gateway path (Gemini-native).
    # NB: no ``kimi`` — upstream kimi has no per-spawn provider override flag,
    # so Omnigent cannot thread a Databricks gateway through. Users configure
    # providers via ``kimi provider add`` in ``~/.kimi-code/config.toml``
    # (Omnigent-side provider injection is a deferred follow-up).
}


def _provider_harness_name(harness_type: AgentHarnessType) -> str:
    """Translate a workflow harness type to the provider-config harness name.

    The Chunk-1a provider-config layer keys harness→family with
    ``"openai-agents"`` (no ``-sdk`` suffix), whereas this module's
    :data:`AgentHarnessType` uses ``"openai-agents-sdk"``. Every other
    harness name matches verbatim, so only that one differs.

    :param harness_type: Canonical workflow harness type, e.g.
        ``"openai-agents-sdk"`` or ``"claude-sdk"``.
    :returns: The provider-config harness name, e.g. ``"openai-agents"`` or
        ``"claude-sdk"``.
    """
    return "openai-agents" if harness_type == "openai-agents-sdk" else harness_type


def _provider_auth_command(family: FamilyConfig) -> str:
    """Return a bearer-token shell command for *family*, failing loud if absent.

    Mirrors the executors' transport contract: the executors' gateway path
    invokes a shell command that prints the bearer token. A static
    ``api_key`` (already resolved to plaintext by
    :meth:`ProviderEntry.family`) becomes ``printf %s <shlex-quoted-key>``;
    a user-supplied dynamic ``auth_command`` passes through verbatim.

    :param family: The resolved provider family (``base_url`` + secret
        expanded by :meth:`ProviderEntry.family`).
    :returns: A shell command that prints the bearer token to stdout, e.g.
        ``"printf %s sk-or-abc"`` or the literal ``auth_command``.
    :raises OmnigentError: If the family carries neither a static
        ``api_key`` nor an ``auth_command`` (should not happen post-parse).
    """
    if family.api_key is not None:
        # printf %s avoids the trailing newline ``echo`` would add and is
        # shell-safe for keys with special characters via shlex.quote.
        return f"printf %s {shlex.quote(family.api_key)}"
    if family.auth_command is not None:
        return family.auth_command
    raise OmnigentError(
        "provider family has no credential (neither 'api_key' nor 'auth_command') "
        "to build a bearer-token command from.",
        code=ErrorCode.INVALID_INPUT,
    )


def _origin_of(base_url: str) -> str:
    """Return the scheme://host[:port] origin of *base_url*.

    The gateway executors expect a ``HARNESS_*_GATEWAY_HOST`` workspace
    origin separate from the full base URL (which carries the API path).

    :param base_url: The endpoint base URL, e.g.
        ``"https://openrouter.ai/api/v1"`` or ``"http://localhost:4000/v1"``.
    :returns: The origin, e.g. ``"https://openrouter.ai"`` or
        ``"http://localhost:4000"``.
    """
    from urllib.parse import urlparse

    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def configure_agent_harness_with_provider(
    env: dict[str, str],
    entry: ProviderEntry,
    *,
    harness_type: AgentHarnessType,
) -> None:
    """Inject per-harness model, URL, and auth from a generic provider.

    The open-source counterpart to :func:`configure_agent_harness_with_ucode`:
    it takes a resolved :class:`ProviderEntry` (from the ``providers:`` block
    of ``~/.omnigent/config.yaml``) and emits the **same** vendor-neutral
    ``HARNESS_*_GATEWAY_*`` env vars the Databricks producer emits (base URL,
    host, a bearer-token command, model default), so the executors' existing
    gateway path handles a
    LiteLLM / OpenRouter / local endpoint with no executor changes. Dispatch
    is on :attr:`ProviderEntry.kind`:

    - ``key`` / ``gateway`` / ``local`` — resolve the harness's family and
      emit the ``HARNESS_*_GATEWAY_*`` env vars (see
      :func:`_apply_provider_family`).
    - ``subscription`` — the native/CLI harness carries its own login; no
      gateway vars. For codex, pin the built-in ``openai`` provider
      (``HARNESS_CODEX_MODEL_PROVIDER``) so a custom default in the user's
      ``~/.codex/config.toml`` cannot shadow the subscription.
    - ``cli-config`` — pin the entry's ``model_provider``
      (``HARNESS_CODEX_MODEL_PROVIDER``); the provider table + credential
      come from the user's ``~/.codex/config.toml``, which the executor
      bridges into the per-session ``CODEX_HOME``. Codex harness only.
    - ``databricks`` — delegate to the existing ucode path keyed on the
      provider's profile, reusing :func:`configure_agent_harness_with_ucode`
      so the ``polly`` / Databricks coding-agent flow is unchanged.
    - ``bedrock`` — rejected (raises): AWS Bedrock mode is wired only into
      the native ``omnigent claude`` launch, not the in-process / gateway
      harnesses.

    :param env: Mutable spawn-env dict, modified in place.
    :param entry: The resolved provider entry to apply.
    :param harness_type: Canonical harness type, e.g. ``"claude-sdk"``.
    :raises OmnigentError: If an inline-family provider lacks the family
        the harness requires, no model can be resolved for it, or the harness
        is ``antigravity`` (which is Gemini-native and has no gateway path).
    """
    if harness_type == "antigravity":
        # The Antigravity SDK authenticates Gemini-natively (a direct API key
        # or Vertex AI) and has no OpenAI-compatible base_url, so it cannot
        # consume a generic provider / gateway. ``_build_antigravity_spawn_env``
        # threads HARNESS_ANTIGRAVITY_API_KEY / _VERTEX directly instead, so
        # this path must never run for it — fail loud rather than emit inert
        # gateway env vars the executor no longer reads.
        raise OmnigentError(
            "The 'antigravity' harness authenticates Gemini-natively (API key "
            "or Vertex AI) and does not support generic providers or gateway "
            "routing. Set executor.auth to an api_key, or executor.config "
            "vertex/project/location, instead of a 'providers:' entry.",
            code=ErrorCode.INVALID_INPUT,
        )
    if entry.kind == BEDROCK_KIND:
        # Bedrock mode is wired only into the native ``omnigent claude`` launch
        # (:func:`omnigent.harnesses.claude_native.main._bedrock_config_for_native_claude`),
        # which sets CLAUDE_CODE_USE_BEDROCK + AWS_BEARER_TOKEN_BEDROCK directly.
        # The in-process / gateway harnesses have no Bedrock path, so emitting
        # the generic ``HARNESS_*_GATEWAY_*`` vars would silently point the
        # harness at the Bedrock endpoint as if it spoke the Anthropic Messages
        # API and fail at request time. Fail loud instead.
        raise OmnigentError(
            f"provider {entry.name!r} (kind 'bedrock') is only supported by the "
            f"native 'omnigent claude' terminal, not the {harness_type!r} harness. "
            "For agents / 'omnigent run', use a 'gateway' provider "
            "(OpenAI/Anthropic-compatible endpoint), or a 'databricks' / 'key' "
            "provider.",
            code=ErrorCode.INVALID_INPUT,
        )
    if entry.kind == SUBSCRIPTION_KIND:
        # A logged-in CLI (claude / codex) carries its own auth; the
        # native/CLI harness reads its own login. Emitting inline-family
        # gateway vars here would point the harness at a non-existent
        # endpoint. (Chunk 1b routes only the inline-family + databricks
        # kinds; subscription routing — toggling the CLI's logged-in model
        # — is a later chunk.)
        if harness_type == "codex":
            # The codex executor symlinks the user's ~/.codex/config.toml
            # into the per-session CODEX_HOME, so a custom default
            # ``model_provider`` there (e.g. isaac's Databricks AI Gateway)
            # would silently hijack a Subscription selection. Pin codex's
            # built-in ``openai`` provider so "Subscription" always means
            # the ChatGPT login — a no-op when the user's config sets no
            # custom default.
            env["HARNESS_CODEX_MODEL_PROVIDER"] = "openai"
        return

    if entry.kind == CLI_CONFIG_KIND:
        # The pi harness consumes both families and can route a cli-config
        # Databricks AI Gateway (the gateway's Anthropic Messages surface is one
        # Pi speaks natively) — the same provider pi-native routes via
        # ``_cli_config_pi_provider``. Translate it into the pi gateway
        # transport rather than failing loud; a non-Databricks cli-config is
        # never selected for pi (see ``default_provider_for_harness``), so it
        # won't reach here.
        if harness_type == "pi":
            _apply_cli_config_databricks_to_pi(env, entry)
            return
        # A custom model provider defined (and authenticated) by the codex
        # CLI's own config.toml: pin it by name; the executor's bridged
        # config.toml carries the provider table + credential. Only the
        # codex harness reads that file — openai-agents-sdk / claude-sdk
        # cannot consume a codex provider table, so fail loud rather than
        # launch them credential-less.
        if harness_type != "codex":
            raise OmnigentError(
                f"provider {entry.name!r} (kind 'cli-config') pins a provider in "
                f"~/.codex/config.toml and can only drive the 'codex' harness, "
                f"not {harness_type!r}. Configure a key/gateway provider for this "
                "harness in ~/.omnigent/config.yaml.",
                code=ErrorCode.INVALID_INPUT,
            )
        # entry.model_provider is required by the cli-config parse branch.
        env["HARNESS_CODEX_MODEL_PROVIDER"] = str(entry.model_provider)
        return

    if entry.kind == DATABRICKS_KIND:
        # A Databricks profile: reuse the existing ucode path so the
        # Databricks coding agent / polly keep working unchanged. The
        # profile name drives model + base URL + auth-command lookup from
        # ~/.databrickscfg + ucode state. This mirrors the legacy
        # DatabricksAuth branch: enable the neutral gateway transport (the
        # Databricks AI gateway is one producer of that transport), record
        # the Databricks profile (Databricks-specific, used by the executor
        # for token refresh), then delegate gateway enrichment to ucode.
        profile = entry.profile
        flag = _HARNESS_GATEWAY_FLAG.get(harness_type)
        if flag is not None:
            env[flag] = "true"
        if profile:
            env[_HARNESS_DATABRICKS_PROFILE[harness_type]] = profile
        configure_agent_harness_with_ucode(env, profile, harness_type=harness_type)
        return

    # Inline-family kinds: key / gateway / local.
    if harness_type == "pi":
        _apply_provider_to_pi(env, entry)
        return
    family_name = _PROVIDER_HARNESS_FAMILY[harness_type]
    family = entry.family(family_name)
    if family is None:
        raise OmnigentError(
            f"provider {entry.name!r} has no {family_name!r} family, required by the "
            f"{harness_type!r} harness. Add a '{family_name}:' block to that provider in "
            f"~/.omnigent/config.yaml.",
            code=ErrorCode.INVALID_INPUT,
        )
    if harness_type == "openai-agents-sdk":
        _apply_provider_to_openai_agents(env, family)
    else:
        _apply_provider_family(env, harness_type, family)


# Maps an omnigent provider family to the bundled catalog provider name
# whose default model serves it. The two happen to share names today
# (``anthropic`` family ⇄ ``anthropic`` catalog, ``openai`` family ⇄
# ``openai`` catalog), but routing through this map keeps the family→catalog
# coupling explicit and one place to change if a family ever fans out to a
# differently-named catalog (e.g. an openai-compatible vendor).
_FAMILY_CATALOG_TARGET: dict[str, tuple[str, str]] = {
    ANTHROPIC_FAMILY: ("anthropic", "claude"),
    OPENAI_FAMILY: ("openai", "openai"),
}


def _resolve_catalog_default_model(
    provider_name: str,
    family: str,
    *,
    context: str,
) -> str:
    """Resolve one live catalog default or raise an actionable input error."""
    try:
        return resolve_catalog_model(provider_name, family=family).model_id
    except ModelResolutionError as exc:
        raise OmnigentError(
            f"No default model resolved for {context}: the {provider_name!r} model "
            f"catalog has no compatible {family!r} entry. Set 'executor.model' in "
            "the agent YAML or a provider 'models.default', or retry when catalog "
            "discovery is available.",
            code=ErrorCode.INVALID_INPUT,
        ) from exc


def _catalog_default_model(family_name: str) -> str:
    """Return the bundled catalog's default model for a provider family.

    Used as the model-resolution fallback for a ``key`` / ``gateway`` /
    ``local`` provider on a KNOWN family (anthropic / openai) when neither
    the spec nor the provider's ``models.default`` names a model: rather than
    fail loud, resolve a sensible vendor model from the catalog. The neutral
    gateway path never falls back to a ``databricks-*`` model. This is a real,
    designed default — see
    :func:`omnigent.onboarding.providers.default_chat_model` for the rule
    (newest general-purpose chat model for that vendor) — not an invented
    one masking missing data: it only applies on a family the catalog knows.

    :param family_name: The omnigent family, ``"anthropic"`` or
        ``"openai"``.
    :returns: The live catalog's preferred model id.
    :raises OmnigentError: If the family is unknown or discovery has no
        compatible model.
    """
    target = _FAMILY_CATALOG_TARGET.get(family_name)
    if target is None:
        raise OmnigentError(
            f"No model catalog is configured for provider family {family_name!r}.",
            code=ErrorCode.INVALID_INPUT,
        )
    provider_name, catalog_family = target
    return _resolve_catalog_default_model(
        provider_name,
        catalog_family,
        context=f"provider family {family_name!r}",
    )


def _apply_provider_family(
    env: dict[str, str],
    harness_type: AgentHarnessType,
    family: FamilyConfig,
) -> None:
    """Apply a provider family to a gateway-style harness (claude-sdk / codex).

    Emits the same vendor-neutral ``HARNESS_*_GATEWAY_*`` env vars the
    Databricks producer emits so the executor's existing gateway path is
    reused unchanged. The model default is only applied when the spec did
    not already set the model env var.

    :param env: Mutable spawn-env dict, modified in place.
    :param harness_type: ``"claude-sdk"`` or ``"codex"``.
    :param family: The resolved provider family for this harness.
    :raises OmnigentError: If no model is resolvable (neither the spec nor
        the family declares one), or if a Codex provider is configured for
        the unsupported Chat Completions wire API.
    """
    if harness_type == "codex" and family.wire_api == CHAT_WIRE_API:
        raise OmnigentError(
            "The 'codex' harness requires an OpenAI Responses API endpoint, but "
            f"provider endpoint {family.base_url!r} is configured with "
            "'wire_api: chat'. Choose a Responses-capable endpoint and set "
            "'wire_api: responses', or use a harness that supports Chat Completions.",
            code=ErrorCode.INVALID_INPUT,
        )

    cfg = _UCODE_HARNESS_CONFIGS[harness_type]
    env[_HARNESS_GATEWAY_FLAG[harness_type]] = "true"
    env[cfg.base_url_key] = family.base_url
    env[cfg.host_key] = _origin_of(family.base_url)
    if cfg.auth_key is not None:
        env[cfg.auth_key] = _provider_auth_command(family)
    # Model precedence: spec model (already in env via _resolve_spec_model) >
    # provider ``models.default`` > catalog family default > fail loud.
    if cfg.model_key not in env and family.default_model:
        env[cfg.model_key] = family.default_model
    if cfg.model_key not in env:
        env[cfg.model_key] = _catalog_default_model(_PROVIDER_HARNESS_FAMILY[harness_type])
    if harness_type == "codex":
        # Codex defaults to the Responses wire API; OpenRouter-style
        # chat-only gateways set wire_api: chat. See codex_harness.py.
        env["HARNESS_CODEX_WIRE_API"] = family.wire_api or RESPONSES_WIRE_API


def _apply_provider_to_openai_agents(env: dict[str, str], family: FamilyConfig) -> None:
    """Apply a provider family to the openai-agents-sdk harness.

    Unlike the gateway harnesses, the OpenAI-Agents executor takes the API
    key and base URL directly (no ``GATEWAY`` enable flag). A static key uses
    ``HARNESS_OPENAI_AGENTS_API_KEY``; a dynamic token command uses the
    auth-command + host pair. ``wire_api`` maps to ``USE_RESPONSES``.

    :param env: Mutable spawn-env dict, modified in place.
    :param family: The resolved ``openai`` provider family.
    :raises OmnigentError: If no model is resolvable (neither the spec nor
        the family declares one).
    """
    env["HARNESS_OPENAI_AGENTS_GATEWAY_BASE_URL"] = family.base_url
    if family.api_key:
        env["HARNESS_OPENAI_AGENTS_API_KEY"] = family.api_key
    else:
        # Dynamic token command: the executor wraps httpx with a shell-command
        # bearer auth, refreshing the token from this command + host.
        env["HARNESS_OPENAI_AGENTS_GATEWAY_AUTH_COMMAND"] = _provider_auth_command(family)
        env["HARNESS_OPENAI_AGENTS_GATEWAY_HOST"] = _origin_of(family.base_url)
    # Model precedence: spec model > provider ``models.default`` > catalog
    # family default > fail loud. The openai-agents harness always consumes
    # the ``openai`` family.
    if "HARNESS_OPENAI_AGENTS_MODEL" not in env and family.default_model:
        env["HARNESS_OPENAI_AGENTS_MODEL"] = family.default_model
    if "HARNESS_OPENAI_AGENTS_MODEL" not in env:
        env["HARNESS_OPENAI_AGENTS_MODEL"] = _catalog_default_model(OPENAI_FAMILY)
    if family.wire_api is not None:
        env["HARNESS_OPENAI_AGENTS_USE_RESPONSES"] = (
            "true" if family.wire_api == RESPONSES_WIRE_API else "false"
        )


def _optional_provider_family(
    entry: ProviderEntry, family_name: str
) -> tuple[FamilyConfig | None, OmnigentError | None]:
    """Attempt to resolve a provider family, returning success or failure info.

    For the ``pi`` harness, which carries a single credential but probes
    both families: a family whose ``$VAR`` is unresolved is treated as
    unavailable rather than fatal, so e.g. a user who only exported
    ``ANTHROPIC_API_KEY`` can still run pi on the anthropic family.

    All credential resolution failures (unresolved ``env:`` var, missing
    keychain secret) raise :class:`OmnigentError` at family-access time
    (structural validation already happened at parse time). They are caught
    here so the other family can be tried; the error is returned to the
    caller so it can be surfaced when no family succeeds.

    Returns a two-tuple ``(family, error)`` so the caller can include the
    original resolution error in its failure message, naming the missing
    variable instead of emitting a generic "no family resolves" message.

    :param entry: The resolved provider entry.
    :param family_name: Family key, e.g. ``"openai"`` or ``"anthropic"``.
    :returns: ``(FamilyConfig, None)`` when the family resolved, or
        ``(None, OmnigentError)`` when resolution failed, or ``(None, None)``
        when the family is simply not configured.
    """
    try:
        family = entry.family(family_name)
        return family, None
    except OmnigentError as exc:
        return None, exc


def _apply_provider_to_pi(env: dict[str, str], entry: ProviderEntry) -> None:
    """Apply a provider to the pi harness, which consumes both families.

    pi reads ``HARNESS_PI_GATEWAY_BASE_URLS`` (a JSON object keyed by pi's
    own family names) and a single auth command. When both families are
    configured with different credentials pi can only carry one — it uses the
    ``anthropic`` family's auth when present, else the ``openai`` family's.
    For a single-key gateway (e.g. a LiteLLM proxy) this is exact.

    A family whose credential env var is unset is skipped (not fatal) so a
    user who exported only one vendor's key can still run pi on that family.
    If neither family resolves, this fails loud, including the original
    credential resolution error(s) so the user knows which env var to set.

    :param env: Mutable spawn-env dict, modified in place.
    :param entry: The resolved provider entry (at least one inline family).
    :raises OmnigentError: If no configured family's credentials resolve,
        or no model can be resolved for the chosen family.
    """
    anthropic, anthropic_err = _optional_provider_family(entry, ANTHROPIC_FAMILY)
    openai, openai_err = _optional_provider_family(entry, OPENAI_FAMILY)
    base_urls: dict[str, str] = {}
    if anthropic is not None:
        base_urls[_PI_FAMILY_KEY[ANTHROPIC_FAMILY]] = anthropic.base_url
    if openai is not None:
        base_urls[_PI_FAMILY_KEY[OPENAI_FAMILY]] = openai.base_url
    if not base_urls:
        # At least one family was configured (the provider passed parse-time
        # validation) but its credential could not be resolved. Surface the
        # original error(s) so the user knows which env var to set, rather
        # than a generic "set the api_key env var" message that omits the name.
        cred_errors = [str(err) for err in (anthropic_err, openai_err) if err is not None]
        detail = (
            f" ({'; '.join(cred_errors)})"
            if cred_errors
            else " — set the api_key env var for its 'anthropic' or 'openai' family in your shell"
        )
        raise OmnigentError(
            f"pi harness: provider {entry.name!r} configures no family whose "
            f"credentials resolve{detail}, then retry.",
            code=ErrorCode.INVALID_INPUT,
        )
    # pi carries a single credential: anthropic's when present, else openai's.
    # The model fallback must match the family that supplied that credential.
    auth_source: FamilyConfig | None
    if anthropic is not None:
        auth_source = anthropic
        auth_family = ANTHROPIC_FAMILY
    else:
        auth_source = openai
        auth_family = OPENAI_FAMILY
    assert auth_source is not None  # base_urls non-empty ⇒ one family resolved
    env[_HARNESS_GATEWAY_FLAG["pi"]] = "true"
    env["HARNESS_PI_GATEWAY_BASE_URLS"] = json.dumps(base_urls, sort_keys=True)
    if openai is not None and openai.wire_api is not None:
        env["HARNESS_PI_GATEWAY_OPENAI_WIRE_API"] = openai.wire_api
    env["HARNESS_PI_GATEWAY_HOST"] = _origin_of(next(iter(base_urls.values())))
    env["HARNESS_PI_GATEWAY_AUTH_COMMAND"] = _provider_auth_command(auth_source)
    # Model precedence: spec model > provider ``models.default`` > catalog
    # family default (of the auth-source family) > fail loud.
    if "HARNESS_PI_MODEL" not in env and auth_source.default_model:
        env["HARNESS_PI_MODEL"] = auth_source.default_model
    if "HARNESS_PI_MODEL" not in env:
        env["HARNESS_PI_MODEL"] = _catalog_default_model(auth_family)


def _apply_cli_config_databricks_to_pi(env: dict[str, str], entry: ProviderEntry) -> None:
    """Apply a cli-config Databricks AI Gateway to the pi (gateway-harness) path.

    The gateway-harness pi launch (``omnigent run`` / agents) and pi-native
    (the terminal) both resolve the same default provider
    (:func:`default_provider_for_harness`), so when that default is a
    ``cli-config`` Databricks AI Gateway, this path must route it rather than
    fail loud. We reuse the pi-native translation
    (:func:`omnigent.harnesses.pi_native.credentials._cli_config_pi_provider`) — which
    reads the codex ``[model_providers.X]`` transport, rewrites the base URL to
    the gateway's Anthropic Messages surface (``/anthropic``) Pi speaks
    natively, and builds the per-request bearer-token ``!command`` apiKey — then
    maps its fields onto the ``HARNESS_PI_GATEWAY_*`` env vars the pi harness
    wrap reads (the same vars :func:`_apply_provider_to_pi` emits).

    :param env: Mutable spawn-env dict, modified in place.
    :param entry: The resolved ``cli-config`` provider entry (a Databricks
        gateway — selection guarantees a non-Databricks cli-config never
        reaches here).
    :raises OmnigentError: If the cli-config entry cannot be translated into a
        Pi gateway provider (its codex table can't be resolved or it is not a
        recognized Databricks AI Gateway) — selection should prevent this, so a
        failure here is a real misconfiguration worth surfacing.
    """
    # Imported lazily: pi_native_credentials is on the runner's session-create
    # hot path and pulls onboarding-only deps; keep this off workflow import.
    from omnigent.harnesses.pi_native.credentials import _cli_config_pi_provider

    # The spec model (if any) is already in HARNESS_PI_MODEL; thread it so the
    # gateway translation honors an explicit override, else its default.
    model_override = env.get("HARNESS_PI_MODEL")
    provider = _cli_config_pi_provider(entry, model=model_override)
    if provider is None:
        raise OmnigentError(
            f"provider {entry.name!r} (kind 'cli-config') was selected for the 'pi' "
            "harness but its codex [model_providers] table could not be resolved as a "
            "Databricks AI Gateway. Check the [model_providers] base_url + auth in "
            "~/.codex/config.toml, or configure a key/gateway provider for pi in "
            "~/.omnigent/config.yaml.",
            code=ErrorCode.INVALID_INPUT,
        )
    # Pi speaks the gateway's Anthropic Messages surface — register it under
    # pi's "claude" family key (mirrors _apply_provider_to_pi's anthropic path).
    base_urls = {_PI_FAMILY_KEY[ANTHROPIC_FAMILY]: provider.base_url}
    env[_HARNESS_GATEWAY_FLAG["pi"]] = "true"
    env["HARNESS_PI_GATEWAY_BASE_URLS"] = json.dumps(base_urls, sort_keys=True)
    env["HARNESS_PI_GATEWAY_HOST"] = _origin_of(provider.base_url)
    # provider.api_key is a "!command" form (Pi's models.json convention); the
    # gateway transport env var wants the bare shell command, so strip the "!".
    env["HARNESS_PI_GATEWAY_AUTH_COMMAND"] = provider.api_key.lstrip("!")
    env["HARNESS_PI_MODEL"] = provider.model


def _synthesize_databricks_provider(profile: str | None) -> ProviderEntry:
    """
    Build an in-memory ``databricks``-kind provider for a legacy credential.

    Legacy Databricks credentials — a spec ``DatabricksAuth`` /
    ``executor.profile``, the global ``auth: {type: databricks}`` block, or a
    ``databricks-`` model name — are folded into the generic provider path by
    wrapping them in a synthesized :class:`ProviderEntry`, so the single
    :func:`configure_agent_harness_with_provider` databricks branch wires the
    gateway transport instead of a per-builder ``else``. Never persisted;
    ``profile`` ``None`` enables the gateway with no pinned profile (the
    executor resolves the default ``~/.databrickscfg``).

    :param profile: The ``~/.databrickscfg`` profile, or ``None``.
    :returns: A ``databricks``-kind :class:`ProviderEntry`.
    """
    return ProviderEntry(name="databricks", kind=DATABRICKS_KIND, profile=profile)


def _legacy_databricks_provider(
    profile: str | None,
    *,
    harness_type: AgentHarnessType,
    for_launch: bool,
) -> ProviderEntry | None:
    """
    Synthesize a databricks provider for a legacy credential, when applicable.

    Returns a synthesized ``databricks`` :class:`ProviderEntry` only for a
    launch (*for_launch*) of a gateway-flag harness (claude-sdk / codex / pi /
    qwen) — the harnesses whose databricks apply branch reproduces the legacy
    ``else`` env exactly. Returns ``None`` otherwise (the ``/model`` readout /
    cost / native paths and the openai-agents harness), so those keep their own
    handling byte-for-byte.

    :param profile: The legacy ``~/.databrickscfg`` profile, or ``None``.
    :param harness_type: Canonical harness type, e.g. ``"codex"``.
    :param for_launch: Whether this resolution feeds an actual spawn.
    :returns: A synthesized databricks provider, or ``None``.
    """
    if for_launch and _HARNESS_GATEWAY_FLAG.get(harness_type) is not None:
        return _synthesize_databricks_provider(profile)
    return None


def _synthesize_codex_api_key_provider(auth: ApiKeyAuth) -> ProviderEntry:
    """Route a resolved inline key through Codex's provider transport.

    :param auth: Spec or global API-key authentication, including its endpoint.
    :returns: An in-memory OpenAI-compatible provider; never persisted.
    """
    return ProviderEntry(
        name="api_key",
        kind=KEY_KIND,
        families={
            OPENAI_FAMILY: FamilyConfig(
                base_url=auth.base_url or "https://api.openai.com/v1",
                # ApiKeyAuth is already resolved; family lookup must not
                # expand literal dollar signs in the secret a second time.
                auth_command=f"printf %s {shlex.quote(auth.api_key)}",
                wire_api=RESPONSES_WIRE_API,
            )
        },
    )


def _resolve_spec_model(spec: AgentSpec) -> str | None:
    """Return the model identifier from the spec's executor block."""
    return spec.executor.model


def _resolve_bound_launch_model(spec: AgentSpec, harness: str) -> str | None:
    """Validate the selected model against this session's inference binding."""
    from omnigent.inference_config import load_runtime_inference_config, resolve_bound_model

    identity = str(spec.executor.config.get("harness") or harness)
    return resolve_bound_model(
        load_runtime_inference_config(load_config()), identity, _resolve_spec_model(spec)
    )


def _resolve_provider_for_build(
    spec: AgentSpec,
    *,
    harness_type: AgentHarnessType,
    for_launch: bool = False,
    actual_harness: str | None = None,
) -> ProviderEntry | None:
    """Resolve the provider that should route *harness_type*, if any.

    The single credential resolver, shared by the spawn-env builders (with
    *for_launch*) and the readout / cost / native paths (without). Precedence,
    most explicit first:

    1. ``spec.executor.auth`` is a :class:`ProviderAuth` → that named provider
       (fails loud when undeclared).
    2. A legacy Databricks credential — ``executor.auth: {type: databricks}``,
       a legacy ``executor.profile``, the global ``auth: {type: databricks}``
       block, or a ``databricks-`` model name — resolves to a *synthesized*
       ``databricks`` provider so the one
       :func:`configure_agent_harness_with_provider` databricks branch wires it
       (no per-builder ``else``). Folded only ``for_launch`` of a gateway-flag
       harness; elsewhere it returns ``None`` so the readout / native /
       openai-agents paths keep their own handling.
    3. An :class:`ApiKeyAuth` (spec or global) → a synthesized key provider
       for a Codex launch; otherwise ``None`` (the claude-sdk / openai-agents
       builders thread the key themselves).
    4. The per-family global default (``providers: … default: true``), then an
       ambient-detected default.
    5. (``for_launch`` only) the first credential that can serve the family even
       though it is not marked default — so a launch credentials the head (e.g.
       Debby's codex head with only a never-defaulted Databricks workspace)
       rather than failing with "Invalid API key". Off for the readout / cost
       paths so they never show a provider the user did not choose.

    :param spec: The agent spec.
    :param harness_type: Canonical workflow harness type, e.g. ``"codex"``.
    :param for_launch: ``True`` for the spawn-env builders (permissive: fold
        legacy Databricks credentials into the provider path and fall back to
        the first available credential). ``False`` (readout / cost / native)
        keeps strict, config-only resolution with no synthesis or fallback.
    :param actual_harness: Preserve a native harness identity when its transport
        reuses an SDK provider adapter.
    :returns: The :class:`ProviderEntry` to route through, or ``None``.
    :raises OmnigentError: If a named :class:`ProviderAuth` references a
        provider absent from the ``providers:`` block.
    """
    from omnigent.inference_config import load_runtime_inference_config, resolve_bound_provider

    explicit_config = load_runtime_inference_config(load_config())
    identity = actual_harness or str(spec.executor.config.get("harness") or harness_type)
    bound = resolve_bound_provider(explicit_config, identity, spec.executor.auth)
    if bound is not None:
        return bound
    harness = _provider_harness_name(harness_type)
    auth = spec.executor.auth
    if isinstance(auth, ProviderAuth):
        # A named provider is resolved against the explicit config merged with
        # ambient detections, so a spec may name a detected provider too.
        providers = load_providers(effective_config_with_detected(explicit_config))
        entry = providers.get(auth.name)
        if entry is None:
            raise OmnigentError(
                f"executor.auth references provider {auth.name!r}, but no such provider is "
                "configured under 'providers:' in ~/.omnigent/config.yaml. "
                f"Run `{cli_invocation()} setup --no-internal-beta` to configure one.",
                code=ErrorCode.INVALID_INPUT,
            )
        return entry
    if isinstance(auth, DatabricksAuth):
        # Spec databricks auth → synthesized provider for a gateway-harness
        # launch, else None so the builder's own DatabricksAuth branch runs.
        return _legacy_databricks_provider(
            auth.profile or None, harness_type=harness_type, for_launch=for_launch
        )
    if auth is not None:
        if isinstance(auth, ApiKeyAuth) and for_launch and harness_type == "codex":
            return _synthesize_codex_api_key_provider(auth)
        # ApiKeyAuth — threaded by the claude-sdk / openai-agents builders.
        return None
    legacy_profile = spec.executor.profile or spec.executor.config.get("profile")
    if legacy_profile:
        # A legacy profile is a Databricks credential and wins over a configured
        # default, exactly as before — folded into the synthesized provider for
        # a gateway-harness launch, else None so the legacy ``else`` runs.
        return _legacy_databricks_provider(
            str(legacy_profile), harness_type=harness_type, for_launch=for_launch
        )

    # No spec auth. Most explicit wins, ambient last, then a launch-only fallback.
    explicit_default = default_provider_for_harness(explicit_config, harness)
    if explicit_default is not None:
        return explicit_default
    global_auth = _load_global_auth()
    if isinstance(global_auth, DatabricksAuth):
        # None (readout / non-gateway) → defer to the builder's global-auth path.
        return _legacy_databricks_provider(
            global_auth.profile or None, harness_type=harness_type, for_launch=for_launch
        )
    if global_auth is not None:
        if isinstance(global_auth, ApiKeyAuth) and for_launch and harness_type == "codex":
            return _synthesize_codex_api_key_provider(global_auth)
        # Global ApiKeyAuth — threaded by the builder's global-auth branch.
        return None
    model = _resolve_spec_model(spec)
    if model is not None and model.startswith(("databricks-", "databricks/")):
        # The model name itself signals Databricks intent (no pinned profile).
        return _legacy_databricks_provider(None, harness_type=harness_type, for_launch=for_launch)
    effective = effective_config_with_detected(explicit_config)
    ambient_default = default_provider_for_harness(effective, harness)
    if ambient_default is not None:
        return ambient_default
    # Launch-only last resort: no default anywhere, but a credential that serves
    # this family is configured (e.g. a Databricks workspace the user added but
    # never set as the default). The runner is the one chokepoint every head
    # (CLI, web UI, or a remote host) funnels through, so this credentials the
    # head on every surface, for any agent. Resolved per spawn — nothing is
    # persisted; the startup creds line names the same provider via
    # :func:`first_available_provider`, so the readout cannot disagree.
    if for_launch:
        family = harness_family(harness)
        if family is not None:
            return first_available_provider(effective, family)
    return None


def _load_global_auth() -> ApiKeyAuth | DatabricksAuth | None:
    """
    Load the ``auth:`` block from ``~/.omnigent/config.yaml``.

    Reads the user-level global config file (respecting
    ``$OMNIGENT_CONFIG_HOME`` for test isolation) and parses the
    optional ``auth:`` mapping into a typed auth dataclass.  Returns
    ``None`` when the file does not exist, the ``auth:`` key is absent,
    or the block is not a recognized shape.

    This provides a user-level auth default: agents that do not declare
    ``executor.auth`` in their own spec inherit credentials from here,
    so the user only configures auth once during ``omnigent setup``
    rather than in every agent YAML.

    :returns: A :class:`ApiKeyAuth` or :class:`DatabricksAuth`, or
        ``None`` when the global config has no ``auth:`` block or the
        file is missing.
    """
    config_home = os.environ.get("OMNIGENT_CONFIG_HOME")
    path = (
        Path(config_home) / "config.yaml"
        if config_home
        else Path.home() / ".omnigent" / "config.yaml"
    )
    if not path.exists():
        return None
    try:
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    except Exception:  # noqa: BLE001 - malformed global auth fails open
        return None
    raw_auth = raw.get("auth")
    if not isinstance(raw_auth, dict):
        return None
    auth_type = str(raw_auth.get("type", ""))
    if auth_type == "api_key":
        api_key = str(raw_auth.get("api_key") or "")
        if not api_key:
            return None
        # Expand $VAR references (the config file may store the literal
        # env-var reference; expand at use-time so the secret never
        # needs to live in the YAML file itself).
        api_key = expand_envvars_with_omnigent_prefix(api_key)
        check_unresolved_env_vars("auth.api_key", api_key)
        raw_base_url = raw_auth.get("base_url")
        base_url: str | None = None
        if raw_base_url:
            base_url = expand_envvars_with_omnigent_prefix(str(raw_base_url))
            check_unresolved_env_vars("auth.base_url", base_url)
        return ApiKeyAuth(api_key=api_key, base_url=base_url)
    if auth_type == "databricks":
        profile_val = str(raw_auth.get("profile") or "")
        return DatabricksAuth(profile=profile_val) if profile_val else None
    return None
