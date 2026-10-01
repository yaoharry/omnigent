"""Spec-to-environment builders for harness subprocesses.

The builders share one stable callable shape:
(spec, *, cwd=None, workdir=None) -> dict[str, str].
Provider resolution lives in omnigent.harnesses.config.providers.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
from pathlib import Path

from omnigent.errors import ErrorCode, OmnigentError
from omnigent.harnesses.config import providers as _providers
from omnigent.inner.datamodel import OSEnvSpec
from omnigent.onboarding.detected import codex_config_provider_dismissed
from omnigent.onboarding.provider_config import (
    DATABRICKS_KIND,
    OPENAI_FAMILY,
    RESPONSES_WIRE_API,
    load_config,
)
from omnigent.spec import AgentSpec
from omnigent.spec.types import ApiKeyAuth, DatabricksAuth, RetryPolicy

_logger = logging.getLogger(__name__)


def _add_claude_sdk_skills_env(
    env: dict[str, str],
    spec: AgentSpec,
    workdir: Path | None,
) -> None:
    """
    Populate the skills-related ``HARNESS_CLAUDE_SDK_*`` env vars.

    Threads ``spec.skills_filter`` (always — the harness wrap
    falls back to ``"all"`` on a missing var, which would
    silently override an explicit ``skills: none``), ``spec.name``,
    and the bundle's extracted on-disk path so the harness wrap
    can wire ``ClaudeAgentOptions.plugins`` for agent-bundled
    skills.

    :param env: Dict mutated in-place with the new keys.
    :param spec: The agent spec.
    :param workdir: The bundle's on-disk path (extracted by the
        agent cache). ``None`` skips the plugin-dir wiring.
    """
    # ``skills_filter`` is JSON-encoded because the value can be
    # ``"all"`` / ``"none"`` / a list of skill names. Always set:
    # the harness wrap's ``_resolve_skills_filter`` falls back to
    # ``"all"`` on a missing var, which would silently override
    # an explicit ``skills: none`` from the spec. (The original
    # regression that motivated this whole bridge.)
    env["HARNESS_CLAUDE_SDK_SKILLS_FILTER"] = json.dumps(spec.skills_filter)
    if spec.name:
        env["HARNESS_CLAUDE_SDK_AGENT_NAME"] = spec.name
    if workdir is not None:
        env["HARNESS_CLAUDE_SDK_BUNDLE_DIR"] = str(workdir)


def _build_claude_sdk_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Build the env-var dict the claude-sdk harness wrap reads.

    Maps spec.executor fields → the ``HARNESS_CLAUDE_SDK_*`` env
    vars defined in ``omnigent/inner/claude_sdk_harness.py``.
    Per the v1 spec-config-flow design (see §Step 5b in the
    design doc), per-spawn env overrides are how Omnigent threads
    per-spec config into the subprocess without polluting
    ``os.environ``.

    :param spec: The agent spec.
    :param workdir: The bundle's on-disk path (extracted by the
        agent cache). Threaded through as
        ``HARNESS_CLAUDE_SDK_BUNDLE_DIR`` so the harness wrap can
        wire the SDK ``--plugin-dir`` for agent-bundled skills.
    :returns: A dict of env-var overrides for
        :meth:`HarnessProcessManager.get_client(env=...)`.
    """
    env: dict[str, str] = {}
    model = _providers._resolve_bound_launch_model(spec, "claude-sdk")
    if model is not None:
        from omnigent.inference_config import binding_for_harness, load_runtime_inference_config

        # Specs may pin the provider-routed spelling ("anthropic/<name>") so
        # generic clients route correctly, but the claude CLI rejects
        # vendor prefixes. A bound gateway owns its literal model namespace.
        env["HARNESS_CLAUDE_SDK_MODEL"] = (
            model
            if binding_for_harness(load_runtime_inference_config(), "claude-sdk") is not None
            else model.removeprefix("anthropic/")
        )
    # Session workspace (the selected working folder), not the bundle workdir.
    # Without this the SDK subprocess inherits the runner's launch cwd — see
    # ``HARNESS_CLAUDE_SDK_CWD`` in ``omnigent/inner/claude_sdk_harness.py``.
    if cwd is not None:
        env["HARNESS_CLAUDE_SDK_CWD"] = str(cwd)

    # ── Auth resolution ────────────────────────────────────────────────
    # Priority (highest first):
    # 0. Generic provider — spec.executor.auth: {type: provider, name: X},
    #    OR (no spec auth) the per-family global default from the
    #    ``providers:`` config block (:func:`_providers._resolve_provider_for_build`).
    #    Routes a LiteLLM / OpenRouter / local / Databricks-profile provider.
    # 1. spec.executor.auth — explicit typed auth in the agent YAML.
    # 2. Legacy spec.executor.profile / executor.config["profile"] (deprecated).
    # 3. Global config ~/.omnigent/config.yaml auth: — only when spec has
    #    no auth at all (same guard as openai-agents to prevent global defaults
    #    from silently overriding YAML-declared legacy profiles).
    # 4. Auto-Databricks: databricks-* model prefix triggers Databricks routing.
    provider = _providers._resolve_provider_for_build(
        spec, harness_type="claude-sdk", for_launch=True
    )
    if provider is not None:
        _providers.configure_agent_harness_with_provider(env, provider, harness_type="claude-sdk")
    else:
        # No provider resolved → the only remaining credential is an ApiKeyAuth
        # (spec ``executor.auth`` or the global ``auth:`` block). The databricks /
        # legacy-profile / databricks-model cases were folded into the
        # synthesized-provider path above, so no profile or ucode wiring remains
        # here. The executor strips ANTHROPIC_API_KEY to force subscription auth
        # inside Claude Code, so the key is threaded via the CLI's apiKeyHelper
        # (a shell command the CLI invokes; shlex.quote keeps it shell-safe).
        auth_from_spec = spec.executor.auth
        if auth_from_spec is None:
            auth_from_spec = _providers._load_global_auth()
        if isinstance(auth_from_spec, ApiKeyAuth) and auth_from_spec.api_key:
            _key_cmd = f"printf %s {shlex.quote(auth_from_spec.api_key)}"
            env["HARNESS_CLAUDE_SDK_API_KEY_HELPER"] = _key_cmd
            if auth_from_spec.base_url:
                env["HARNESS_CLAUDE_SDK_GATEWAY_BASE_URL"] = auth_from_spec.base_url
                env["HARNESS_CLAUDE_SDK_GATEWAY_AUTH_COMMAND"] = _key_cmd
        # Enable the gateway for an ApiKeyAuth ``base_url`` (a custom endpoint) or
        # a ``databricks-`` model; without the flag the executor ignores the base
        # URL and falls through to api.anthropic.com.
        if (isinstance(auth_from_spec, ApiKeyAuth) and bool(auth_from_spec.base_url)) or (
            model is not None and model.startswith(("databricks-", "databricks/"))
        ):
            env["HARNESS_CLAUDE_SDK_GATEWAY"] = "true"
    _add_claude_sdk_skills_env(env, spec, workdir)
    # OS env: enabling this in the inner ClaudeSDKExecutor is
    # what gates the SDK-native ``Bash/Read/Edit/Write/Glob/Grep``
    # tools. The legacy non-AP path enables them by default
    # for omnigent-style specs (the inner CLI auto-creates a
    # ``caller_process`` os_env when ``--os`` is set, and the
    # SDK's bundled CLI exposes the natives unconditionally in
    # some configurations); routing through Omnigent mode without a
    # similar default leaves Omnigent mode users staring at a
    # tool list that's ~80% smaller. Forward the spec's
    # OSEnvSpec verbatim when present; default to
    # ``caller_process + sandbox=none`` otherwise so the parity
    # holds.
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_CLAUDE_SDK_OS_ENV"] = os_env_payload
    # Phase 1f: thread the spec's RetryPolicy through to the
    # claude-sdk subprocess so the inner ``ClaudeSDKExecutor``
    # picks up custom retry budgets (``ANTHROPIC_MAX_RETRIES``,
    # ``ANTHROPIC_REQUEST_TIMEOUT_SECONDS``) instead of falling
    # back to the executor's hard-coded ``RetryPolicy()`` default.
    # Omitted when the policy matches defaults — see
    # :func:`_serialize_retry_policy`.
    retry_payload = _serialize_retry_policy(_resolve_retry_policy(spec))
    if retry_payload is not None:
        env["HARNESS_CLAUDE_SDK_RETRY_POLICY"] = retry_payload
    # Permission mode: controls whether Claude asks for approval before
    # calling native tools. When set to anything other than the default
    # ``"bypassPermissions"``, the SDK's ``can_use_tool`` callback is
    # active and approval requests surface via Omnigent elicitation. Read from
    # Omitted when not set — harness falls back to ``"bypassPermissions"``.
    permission_mode = spec.executor.config.get("permission_mode")
    if permission_mode is not None:
        env["HARNESS_CLAUDE_SDK_PERMISSION_MODE"] = str(permission_mode)
    return env


def _apply_harness_path_override(
    env: dict[str, str],
    harness: str,
) -> None:
    """Thread a config ``harness.<canonical>.command`` into ``OMNIGENT_<NAME>_PATH``.

    The harness wraps read ``OMNIGENT_<NAME>_PATH`` to locate their vendor
    CLI (the headless CLI-subprocess family historically read ``HARNESS_*_PATH``;
    both are honored, ``OMNIGENT_*`` canonical). A user can set that path via
    config (``harness.codex.command: /usr/local/bin/codex``); this threads it
    into the spawn env when the ambient env var isn't already set (ambient
    wins, per the shared ``env > config > default`` precedence). A no-op when
    config has no ``command`` for *harness* or the ambient env var is set.

    :param env: The spawn-env dict being built (mutated in place).
    :param harness: A harness id (canonical or alias), e.g. ``"codex"``.
    """
    from omnigent.harness_aliases import canonicalize_harness
    from omnigent.harness_startup_config import (
        _harness_path_env_var,
        config_harness_path_override,
    )

    path = config_harness_path_override(harness, load_config())
    if path is not None:
        env[_harness_path_env_var(canonicalize_harness(harness) or harness)] = path


def _build_codex_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Build the env-var dict the codex harness wrap reads.

    Maps spec.executor fields → the ``HARNESS_CODEX_*`` env vars
    defined in ``omnigent/inner/codex_harness.py``. Mirrors
    :func:`_build_claude_sdk_spawn_env` — same per-spawn env-var
    pattern from §Step 5a. The codex-specific env vars
    (``HARNESS_CODEX_PATH``, ``HARNESS_CODEX_ENABLE_WEB_SEARCH``,
    ``HARNESS_CODEX_DISABLE_NATIVE_TOOLS``) are not threaded
    through here in v1: the legacy
    :func:`omnigent.inner.executor_factory.create_executor`
    path doesn't surface them either, so AP-side parity is
    preserved by leaving them at the inner executor's defaults.
    Operators who want non-default values set those env vars
    on the Omnigent server directly (they propagate to the subprocess
    through normal env inheritance — the wrap's per-spawn
    overrides only override, they don't filter).

    :param spec: The agent spec.
    :param workdir: The bundle's on-disk path (extracted by the
        agent cache). Threaded through as
        ``HARNESS_CODEX_BUNDLE_DIR`` so the harness wrap's executor
        can also source bundled skills from
        ``<bundle>/skills/<dir>/``.
    :returns: A dict of env-var overrides for
        :meth:`HarnessProcessManager.get_client(env=...)`.
    """
    env: dict[str, str] = {}
    model = _providers._resolve_bound_launch_model(spec, "codex")
    if model is not None:
        env["HARNESS_CODEX_MODEL"] = model

    # Generic-provider branch (slotted ahead of the legacy-profile /
    # databricks-prefix path): a ProviderAuth on the spec, or — when the spec
    # declares no auth — the per-family global default. See
    # :func:`_providers._resolve_provider_for_build`. Otherwise the existing path is
    # unchanged.
    provider = _providers._resolve_provider_for_build(spec, harness_type="codex", for_launch=True)
    if provider is not None:
        if provider.kind == DATABRICKS_KIND:
            _providers._configure_brokered_codex_with_ucode(env, spec, provider)
        else:
            _providers.configure_agent_harness_with_provider(env, provider, harness_type="codex")
    elif codex_config_provider_dismissed(load_config()):
        # No credential resolved. If the user Removed codex's custom
        # ~/.codex/config.toml provider (dismissed), pin the built-in ``openai``
        # provider so the dismissal holds at run time — the executor's bridged
        # config.toml would otherwise still route this launch through that removed
        # default model_provider. (Gateway mode never reaches here: it resolves a
        # provider above, and the executor rejects a double pin.)
        env["HARNESS_CODEX_MODEL_PROVIDER"] = "openai"
    # Skills bridge — same shape as the claude-sdk variant. Always
    # set so the harness wrap doesn't fall back to its ``"all"``
    # default and override an explicit ``skills: none`` spec.
    env["HARNESS_CODEX_SKILLS_FILTER"] = json.dumps(spec.skills_filter)
    if spec.name:
        env["HARNESS_CODEX_AGENT_NAME"] = spec.name
    # Session workspace (the selected working folder), not the bundle workdir.
    # Without this the codex subprocess inherits the runner's launch cwd — see
    # ``HARNESS_CODEX_CWD`` in ``omnigent/inner/codex_harness.py``.
    if cwd is not None:
        env["HARNESS_CODEX_CWD"] = str(cwd)
    if workdir is not None:
        env["HARNESS_CODEX_BUNDLE_DIR"] = str(workdir)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_CODEX_OS_ENV"] = os_env_payload
    # Phase 1f: thread the spec's RetryPolicy through to the
    # codex subprocess. See :func:`_build_claude_sdk_spawn_env`
    # for the rationale.
    retry_payload = _serialize_retry_policy(_resolve_retry_policy(spec))
    if retry_payload is not None:
        env["HARNESS_CODEX_RETRY_POLICY"] = retry_payload
    _apply_harness_path_override(env, "codex")
    return env


def _build_pi_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Build the env-var dict the pi harness wrap reads.

    Maps spec.executor fields → the ``HARNESS_PI_*`` env vars
    defined in ``omnigent/inner/pi_harness.py``. Mirrors
    :func:`_build_claude_sdk_spawn_env` /
    :func:`_build_codex_spawn_env` — same per-spawn env-var
    pattern from §Step 5a.

    :param spec: The agent spec.
    :param cwd: Runtime working directory for the Pi CLI. This is the
        session workspace, not the agent bundle workdir.
    :param workdir: The bundle's on-disk path (extracted by the
        agent cache). Threaded through as ``HARNESS_PI_BUNDLE_DIR``
        so the harness wrap's executor can source bundled skills
        from ``<bundle>/skills/<dir>/``.
    :returns: A dict of env-var overrides for
        :meth:`HarnessProcessManager.get_client(env=...)`.
    """
    env: dict[str, str] = {}
    model = _providers._resolve_bound_launch_model(spec, "pi")
    from omnigent.inference_config import binding_for_harness, load_runtime_inference_config

    if binding_for_harness(load_runtime_inference_config(), "pi") is not None:
        env["HARNESS_PI_PRESERVE_MODEL_IDS"] = "true"
    if model is not None:
        env["HARNESS_PI_MODEL"] = model

    # Generic-provider branch (slotted ahead of the legacy-profile /
    # databricks-prefix path): a ProviderAuth on the spec, or — when the spec
    # declares no auth — the per-family global default. pi consumes both
    # families (see :func:`_apply_provider_to_pi`). Otherwise the existing
    # path is unchanged.
    provider = _providers._resolve_provider_for_build(spec, harness_type="pi", for_launch=True)
    if provider is not None:
        _providers.configure_agent_harness_with_provider(env, provider, harness_type="pi")
        if "HARNESS_PI_PRESERVE_MODEL_IDS" in env and provider.family(OPENAI_FAMILY) is not None:
            env.setdefault("HARNESS_PI_GATEWAY_OPENAI_WIRE_API", RESPONSES_WIRE_API)
    # Skills bridge — same shape as the claude-sdk + codex variants.
    # Always set so the harness wrap doesn't fall back to ``"all"``
    # and override an explicit ``skills: none`` from the spec.
    env["HARNESS_PI_SKILLS_FILTER"] = json.dumps(spec.skills_filter)
    env["HARNESS_PI_CONTEXT_FILES"] = json.dumps(spec.executor.config.get("context_files", True))
    env["HARNESS_PI_SYSTEM_PROMPT_MODE"] = spec.executor.config.get("system_prompt_mode", "append")
    if spec.name:
        env["HARNESS_PI_AGENT_NAME"] = spec.name
    if cwd is not None:
        env["HARNESS_PI_CWD"] = str(cwd)
    if workdir is not None:
        env["HARNESS_PI_BUNDLE_DIR"] = str(workdir)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_PI_OS_ENV"] = os_env_payload
    _apply_harness_path_override(env, "pi")
    return env


def _build_qwen_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Build the env-var dict the qwen harness wrap reads.

    Maps spec.executor fields → the ``HARNESS_QWEN_*`` env vars
    defined in ``omnigent/inner/qwen_harness.py``. Mirrors
    :func:`_build_claude_sdk_spawn_env` /
    :func:`_build_codex_spawn_env`.

    :param spec: The agent spec.
    :param workdir: The bundle's on-disk path (extracted by the agent
        cache). Accepted for signature parity with the other
        ``_build_*_spawn_env`` builders; the qwen wrap does not yet
        consume a bundle dir (no skills bridge — see docs/QWEN_FOLLOWUPS.md).
    :returns: A dict of env-var overrides for
        :meth:`HarnessProcessManager.get_client(env=...)`.
    """
    del workdir
    env: dict[str, str] = {}
    model = _providers._resolve_bound_launch_model(spec, "qwen")
    if model is not None:
        env["HARNESS_QWEN_MODEL"] = model
    # Session workspace (selected working folder). ``None`` lets the qwen
    # harness fall back to OMNIGENT_RUNNER_WORKSPACE — see HARNESS_QWEN_CWD.
    if cwd is not None:
        env["HARNESS_QWEN_CWD"] = str(cwd)

    # Generic-provider branch (slotted ahead of the legacy-profile /
    # databricks-prefix path): a ProviderAuth on the spec, or — when the spec
    # declares no auth — the per-family global default. qwen routes through
    # OpenAI-compatible providers.
    provider = _providers._resolve_provider_for_build(spec, harness_type="qwen", for_launch=True)
    if provider is not None:
        from omnigent.inference_config import binding_for_harness, load_runtime_inference_config

        family = provider.family(OPENAI_FAMILY)
        if (
            binding_for_harness(load_runtime_inference_config(), "qwen") is not None
            and family is not None
            and family.wire_api == RESPONSES_WIRE_API
        ):
            raise ValueError("Qwen's configured gateway must support the chat wire API.")
        _providers.configure_agent_harness_with_provider(env, provider, harness_type="qwen")
    # NB: no skills bridge for qwen yet. Unlike the claude-sdk / codex
    # variants, the qwen wrap (omnigent/inner/qwen_harness.py) and
    # QwenExecutor have no skills concept, so emitting
    # HARNESS_QWEN_SKILLS_FILTER / _AGENT_NAME / _BUNDLE_DIR would set env
    # nothing reads. Wire those through when skills land — see
    # docs/QWEN_FOLLOWUPS.md.
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_QWEN_OS_ENV"] = os_env_payload
    _apply_harness_path_override(env, "qwen")
    return env


def _build_goose_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Build the env-var dict the headless goose harness wrap reads.

    Maps spec.executor fields → the ``HARNESS_GOOSE_*`` env vars defined in
    ``omnigent/inner/goose_harness.py``. Unlike the SDK harnesses, Goose owns its
    own auth via ``goose configure`` (keyring / ``~/.config/goose/config.yaml``),
    so this builder wires **no** provider/gateway credential — it forwards only an
    optional model override and the os_env/sandbox spec. A ``databricks-*`` model
    is dropped (not a valid Goose model id; the provider/model then come from the
    user's Goose config), mirroring how the native CLIs handle gateway ids.

    :param spec: The agent spec.
    :param workdir: The bundle's on-disk path. Accepted for signature parity with
        the other ``_build_*_spawn_env`` builders; the goose wrap consumes no
        bundle dir yet (no skills bridge).
    :returns: A dict of env-var overrides for the harness process spawn.
    """
    del workdir
    env: dict[str, str] = {}
    model = _providers._resolve_spec_model(spec)
    if model is not None and not model.startswith(("databricks-", "databricks/")):
        env["HARNESS_GOOSE_MODEL"] = model
    # Session workspace (selected working folder). ``None`` lets the goose
    # harness fall back to OMNIGENT_RUNNER_WORKSPACE — see HARNESS_GOOSE_CWD.
    if cwd is not None:
        env["HARNESS_GOOSE_CWD"] = str(cwd)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_GOOSE_OS_ENV"] = os_env_payload
    _apply_harness_path_override(env, "goose")
    return env


def _build_acp_cli_spawn_env(
    spec: AgentSpec,
    *,
    harness: str,
    cwd: Path | None = None,
    workdir: Path | None = None,
    session_id: str | None = None,
) -> dict[str, str]:
    """Build the generic-ACP env for one builtin ACP CLI harness (catalog row).

    Rows in :data:`omnigent.acp_cli_harnesses.ACP_CLI_HARNESSES` all run the
    shared ``omnigent/inner/acp_harness.py`` wrap; this maps a row + spec to
    the ``HARNESS_ACP_*`` vars it reads. Like goose/acp, a vendor ACP CLI owns
    its own auth and model, so no provider/gateway credential and no model var
    is wired. The binary resolves via the ``OMNIGENT_<NAME>_PATH`` env
    override, then the config ``harness.<name>.command`` path, then PATH plus
    the common global install dirs.

    :param spec: The agent spec.
    :param harness: The catalog row key, e.g. ``"grok"``.
    :param workdir: Accepted for signature parity with the other builders; the
        ACP wrap consumes no bundle dir.
    :returns: A dict of ``HARNESS_ACP_*`` env-var overrides for the spawn.
    """
    del workdir
    from omnigent._platform import resolve_cli_binary
    from omnigent.acp_cli_harnesses import ACP_CLI_HARNESSES
    from omnigent.harness_startup_config import (
        config_harness_path_override,
        resolve_harness_path,
    )

    row = ACP_CLI_HARNESSES[harness]
    executable = (
        resolve_harness_path(harness)
        or config_harness_path_override(harness, load_config())
        or resolve_cli_binary(row.binary)
        or row.binary
    )
    env = {
        "HARNESS_ACP_COMMAND": shlex.join([executable, *row.args]),
        "HARNESS_ACP_NAME": row.label,
        # Rows whose CLI doesn't yet support session-scoped MCP and ignores
        # session/new mcpServers (e.g. jcode) opt out of advertising the
        # Omnigent MCP server.
        "HARNESS_ACP_OMNIGENT_MCP": "1" if row.omnigent_mcp else "0",
    }
    # Session workspace (selected working folder). ``None`` lets the wrap fall
    # back to OMNIGENT_RUNNER_WORKSPACE — see HARNESS_ACP_CWD.
    if cwd is not None:
        env["HARNESS_ACP_CWD"] = str(cwd)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_ACP_OS_ENV"] = os_env_payload
    # Permission stance for approval cards. Absent leaves the harness wrap on its
    # ``auto`` default (prompt); ``bypassPermissions`` skips the card for a call no
    # policy had an opinion on, so a headless ACP worker doesn't park on a prompt.
    permission_mode = spec.executor.config.get("permission_mode")
    if permission_mode is not None:
        env["HARNESS_ACP_PERMISSION_MODE"] = str(permission_mode)

    # Managed-connect support for jcode: point it at a session-private JCODE_HOME
    # (with a config.toml pinning the gateway provider) + a fresh broker bearer. A
    # no-op when not connected (connect_jcode_gateway_env returns None), and — like the
    # other connect harnesses — suppressed when the spec configures its own API key, so
    # an explicit key is never silently rerouted through the owner's gateway.
    if harness == "jcode":
        from omnigent.host.databricks_credential import api_key_auth_precludes_broker
        from omnigent.host.jcode_databricks import (
            configured_jcode_gateway_env,
            connect_jcode_gateway_env,
        )
        from omnigent.inference_config import (
            binding_for_harness,
            load_runtime_inference_config,
            resolve_bound_model,
            resolve_bound_provider,
        )

        config = load_runtime_inference_config(load_config())
        bound = resolve_bound_provider(config, harness, spec.executor.auth)
        if bound is not None:
            from omnigent.models.model_catalog import ResolvedModelProvider, _resolve_bearer_token

            family = bound.family(OPENAI_FAMILY)
            model = resolve_bound_model(config, harness, _providers._resolve_spec_model(spec))
            binding = binding_for_harness(config, harness)
            if family is None or not model:
                raise ValueError("Jcode requires an OpenAI-compatible provider and default model.")
            if family.wire_api == RESPONSES_WIRE_API:
                raise ValueError("Jcode's configured gateway must support the chat wire API.")
            token = _resolve_bearer_token(
                ResolvedModelProvider(
                    kind=bound.kind,
                    api_key=family.api_key,
                    auth_command=family.auth_command,
                )
            )
            models = binding.model_allowlist if binding is not None else None
            gateway_env = configured_jcode_gateway_env(
                base_url=family.base_url,
                api_key=token,
                model=model,
                models=models or (),
                session_id=session_id,
            )
            env["HARNESS_ACP_MODEL"] = model
            env["HARNESS_ACP_SEND_MODEL"] = "1"
            if models is not None:
                env["HARNESS_ACP_MODEL_LIST"] = ",".join(models)
        else:
            gateway_env = (
                None
                if api_key_auth_precludes_broker(spec)
                else connect_jcode_gateway_env(session_id=session_id)
            )
        if gateway_env is not None:
            env.update(gateway_env)
            # The ACP wrap forwards only passthrough-named vars to the jcode subprocess,
            # so name every managed-connect var (JCODE_DBX_TOKEN / JCODE_HOME /
            # JCODE_RUNTIME_DIR). Preserve any existing value, dedupe, and join.
            existing = env.get("HARNESS_ACP_ENV_PASSTHROUGH", "").split(",")
            names = {n.strip() for n in existing if n.strip()} | set(gateway_env)
            env["HARNESS_ACP_ENV_PASSTHROUGH"] = ",".join(sorted(names))

    return env


def _build_acp_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """Build the env-var dict the generic ACP harness wrap reads.

    Prefers a one-shot agent embedded in ``spec.executor.config``; otherwise
    resolves the picked ``acp:<slug>`` to a user-configured agent in the global
    ``acp:`` block. The selected command + protocol knobs become the
    ``HARNESS_ACP_*`` env vars defined in ``omnigent/inner/acp_harness.py``.

    Like Goose, a generic ACP agent owns its own auth, so this wires **no**
    provider/gateway credential. A ``databricks-*`` model is dropped (not a valid
    third-party model id); the agent's own configured model (or a flag in its
    command) then applies. A bare ``acp`` id uses the first configured agent;
    an explicit unknown name fails instead of launching a different agent.

    :param spec: The agent spec.
    :param workdir: Accepted for signature parity with the other builders; the
        ACP wrap consumes no bundle dir.
    :returns: A dict of ``HARNESS_ACP_*`` env-var overrides for the spawn.
    """
    del workdir
    env: dict[str, str] = {}
    raw_harness = ""
    cfg = getattr(spec.executor, "config", None)
    if isinstance(cfg, dict):
        raw_harness = str(cfg.get("harness") or "")
    slug = raw_harness.split(":", 1)[1] if raw_harness.startswith("acp:") else ""

    # Lazily import the config reader — the hot spawn-env path shouldn't pull in
    # the onboarding/config stack eagerly (mirrors the cursor builder).
    # Also lazy: model_catalog pulls the onboarding provider config eagerly.
    from omnigent.models.model_catalog import (
        _acp_launch_model,
        acp_curated_models,
        validate_acp_model,
    )
    from omnigent.onboarding.acp_auth import (
        AcpAgentEntry,
        acp_agents,
        parse_env_passthrough,
        resolve_acp_agent,
    )

    has_embedded = isinstance(cfg, dict) and "acp_agent" in cfg
    embedded = cfg.get("acp_agent") if isinstance(cfg, dict) else None
    agent: AcpAgentEntry | None = None
    if has_embedded:
        if not isinstance(embedded, dict):
            raise ValueError("executor acp_agent must be a mapping with name and command")
        name = embedded.get("name")
        command = embedded.get("command")
        if not (
            isinstance(name, str) and name.strip() and isinstance(command, str) and command.strip()
        ):
            raise ValueError("executor acp_agent requires non-empty string name and command")
        omnigent_mcp = embedded.get("omnigent_mcp", True)
        if not isinstance(omnigent_mcp, bool):
            raise ValueError("executor acp_agent omnigent_mcp must be a boolean")
        inject_system_prompt = embedded.get("inject_system_prompt", True)
        if not isinstance(inject_system_prompt, bool):
            raise ValueError("executor acp_agent inject_system_prompt must be a boolean")
        session_id_mode = embedded.get("session_id_mode", "server")
        if session_id_mode not in ("server", "client"):
            raise ValueError(
                f"executor acp_agent session_id_mode must be 'server' or 'client', "
                f"got {session_id_mode!r}"
            )
        send_model = embedded.get("send_model", False)
        if not isinstance(send_model, bool):
            raise ValueError("executor acp_agent send_model must be a boolean")
        model = embedded.get("model")
        if model is not None and not isinstance(model, str):
            raise ValueError("executor acp_agent model must be a string or null")
        agent = AcpAgentEntry(
            slug=slug or "agent",
            name=name.strip(),
            command=command.strip(),
            model=model,
            session_id_mode=session_id_mode,
            send_model=send_model,
            omnigent_mcp=omnigent_mcp,
            inject_system_prompt=inject_system_prompt,
            env_passthrough=parse_env_passthrough(embedded.get("env_passthrough")),
        )
    else:
        agent = resolve_acp_agent(slug) if slug else None
        if raw_harness.startswith("acp:") and agent is None:
            raise OmnigentError(
                f"ACP agent {slug!r} is not configured on this runner. "
                "Check the selected host's ACP configuration.",
                code=ErrorCode.INVALID_INPUT,
            )
        if agent is None:
            agents = acp_agents()
            agent = agents[0] if agents else None

    if agent is not None:
        env["HARNESS_ACP_COMMAND"] = agent.command
        env["HARNESS_ACP_NAME"] = agent.name
        env["HARNESS_ACP_SESSION_ID_MODE"] = agent.session_id_mode
        if agent.send_model:
            env["HARNESS_ACP_SEND_MODEL"] = "1"
        env["HARNESS_ACP_OMNIGENT_MCP"] = "1" if agent.omnigent_mcp else "0"
        if not agent.inject_system_prompt:
            env["HARNESS_ACP_INJECT_SYSTEM_PROMPT"] = "0"
        if agent.env_passthrough:
            # Names only; the harness reads each value from its own environment.
            env["HARNESS_ACP_ENV_PASSTHROUGH"] = ",".join(agent.env_passthrough)

        model = _acp_launch_model(spec)
        validate_acp_model(spec, model)
        if model is not None:
            env["HARNESS_ACP_MODEL"] = model
    # else: no agent configured — leave HARNESS_ACP_COMMAND unset so the wrap
    # raises a clear request-time error pointing the user at `omnigent setup`.

    # The approved catalog is independent of the selected model.
    curated = acp_curated_models(spec)
    if curated:
        env["HARNESS_ACP_MODEL_LIST"] = ",".join(curated)

    # Credential vars the operator declared off-limits for generic ACP agents.
    # The vendor CLI activates built-in providers on the mere presence of
    # their credential (any value), flooding its own picker with entries that
    # bypass the deployment's curated set. Names are forwarded (never values);
    # the harness reads each from its own environment before spawning the CLI.
    # Unset means no scrubbing, matching pi-native's OMNIGENT_PI_ENV_UNSET.
    denylist = os.environ.get("OMNIGENT_ACP_ENV_UNSET", "").strip()
    if denylist:
        env["HARNESS_ACP_ENV_UNSET"] = denylist

    # Session workspace (selected working folder). ``None`` lets the acp
    # harness fall back to OMNIGENT_RUNNER_WORKSPACE — see HARNESS_ACP_CWD.
    if cwd is not None:
        env["HARNESS_ACP_CWD"] = str(cwd)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_ACP_OS_ENV"] = os_env_payload
    # Permission stance for approval cards. Absent leaves the harness wrap on its
    # ``auto`` default (prompt); ``bypassPermissions`` skips the card for a call no
    # policy had an opinion on, so a headless ACP worker doesn't park on a prompt.
    permission_mode = spec.executor.config.get("permission_mode")
    if permission_mode is not None:
        env["HARNESS_ACP_PERMISSION_MODE"] = str(permission_mode)
    return env


def _config_flag_is_true(value: object) -> bool:
    """Interpret a free-form executor config value as a boolean flag."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _set_openai_agents_reasoning_item_id_policy_env(
    env: dict[str, str],
    value: object | None,
) -> None:
    """Validate and encode the OpenAI Agents SDK reasoning replay policy."""
    if value is None:
        return
    if not isinstance(value, str) or value not in {"preserve", "omit"}:
        raise ValueError("reasoning_item_id_policy must be 'preserve', 'omit', or unset")
    env["HARNESS_OPENAI_AGENTS_REASONING_ITEM_ID_POLICY"] = value


def _build_openai_agents_sdk_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Build the env-var dict the openai-agents harness wrap reads.

    Maps spec.executor fields → the ``HARNESS_OPENAI_AGENTS_*``
    env vars defined in
    ``omnigent/inner/openai_agents_sdk_harness.py``. Threads
    model + auth + Responses replay settings.

    Auth resolution order (highest priority first):

    1. ``spec.executor.auth`` — explicit typed auth in the agent YAML.
    2. Legacy ``spec.executor.profile`` / ``spec.executor.config["profile"]``
       (**deprecated** — use ``executor.auth: {type: databricks, …}``).
       Both 1 and 2 are spec-level declarations; the spec always wins.
    3. Global config ``~/.omnigent/config.yaml`` ``auth:`` block —
       **only consulted when the spec declares no auth at all** (neither
       new nor legacy style). This prevents the user's global default
       from silently overriding a YAML that uses the old profile field.
    4. Auto-Databricks: ``databricks-`` / ``databricks/`` model prefix
       with no auth → fall back to the SDK ``"DEFAULT"`` profile so
       ambient OPENAI_API_KEY doesn't short-circuit Databricks routing.

    :param spec: The agent spec.
    :returns: A dict of env-var overrides for
        :meth:`HarnessProcessManager.get_client(env=...)`. May
        be empty when no auth is configured and the model name does not
        start with ``databricks-``; in that case the wrap falls back to
        ``OPENAI_BASE_URL`` / ``OPENAI_API_KEY`` env vars and the
        ``use_responses=True`` default.
    """
    del cwd, workdir
    env: dict[str, str] = {}
    model = _providers._resolve_bound_launch_model(spec, "openai-agents-sdk")
    if model is not None:
        env["HARNESS_OPENAI_AGENTS_MODEL"] = model
    _set_openai_agents_reasoning_item_id_policy_env(
        env,
        spec.executor.config.get("reasoning_item_id_policy"),
    )

    # ── Auth resolution ────────────────────────────────────────────────
    # Priority: generic provider → spec.executor.auth → global config auth →
    # legacy profile in config dict → auto-Databricks for databricks-* models.
    #
    # 0. Generic provider — spec.executor.auth: {type: provider, name: X},
    #    OR (no spec auth) the per-family global default. Sets API key /
    #    base URL / model and maps the openai family's wire_api to
    #    USE_RESPONSES. No ucode enrichment (no Databricks profile to look
    #    up), so it returns early. A spec's explicit ``use_responses`` still
    #    wins over the provider's wire_api.
    provider = _providers._resolve_provider_for_build(
        spec, harness_type="openai-agents-sdk", for_launch=True
    )
    if provider is not None:
        _providers.configure_agent_harness_with_provider(
            env, provider, harness_type="openai-agents-sdk"
        )
        use_responses = spec.executor.config.get("use_responses")
        if use_responses is not None:
            env["HARNESS_OPENAI_AGENTS_USE_RESPONSES"] = (
                "true" if _config_flag_is_true(use_responses) else "false"
            )
        return env

    # Global config auth is only consulted when the spec declares NO
    # auth at all — neither the new executor.auth block nor the legacy
    # executor.profile / executor.config["profile"].  A spec that uses
    # the old profile style is still an explicit spec-level auth
    # declaration and must not be silently overridden by the user's
    # global default (which may be a different auth type entirely).
    #
    # ProviderAuth is fully handled by the early-return block above (it
    # resolves a provider or fails loud), so spec.executor.auth is narrowed
    # to ApiKeyAuth / DatabricksAuth / None here.
    spec_auth = spec.executor.auth
    auth: ApiKeyAuth | DatabricksAuth | None = (
        spec_auth if isinstance(spec_auth, (ApiKeyAuth, DatabricksAuth)) else None
    )
    _spec_has_legacy_profile = bool(spec.executor.profile or spec.executor.config.get("profile"))
    if auth is None and not _spec_has_legacy_profile:
        auth = _providers._load_global_auth()

    if isinstance(auth, ApiKeyAuth):
        env["HARNESS_OPENAI_AGENTS_API_KEY"] = auth.api_key
        if auth.base_url:
            env["HARNESS_OPENAI_AGENTS_GATEWAY_BASE_URL"] = auth.base_url
    elif isinstance(auth, DatabricksAuth):
        env["HARNESS_OPENAI_AGENTS_DATABRICKS_PROFILE"] = auth.profile
    else:
        # Legacy path: executor.config["profile"] (deprecated — use executor.auth instead).
        # DEPRECATED: config["profile"] will be removed once all specs migrate to auth:.
        profile = spec.executor.config.get("profile")
        if not profile and model and model.startswith(("databricks-", "databricks/")):
            # databricks- / databricks/ prefix: route to Databricks (avoiding the
            # OPENAI_API_KEY short-circuit) via the SDK's DEFAULT profile. The
            # ambient DATABRICKS_CONFIG_PROFILE env var is deliberately NOT
            # consulted — credentials are controlled by the spec or by
            # `omnigent setup` provider config, never by shell environment.
            profile = "DEFAULT"
        if profile:
            # Single canonical env var: ``DATABRICKS_PROFILE``. No
            # ``GATEWAY=true`` gate (unlike claude-sdk / codex /
            # pi) because :class:`OpenAIAgentsSDKExecutor` takes the
            # profile name directly and resolves credentials itself —
            # a separate truthy gate would be dead surface.
            env["HARNESS_OPENAI_AGENTS_DATABRICKS_PROFILE"] = str(profile)

    # Resolve the effective profile for ucode state lookup (model/base-URL enrichment).
    # For api_key auth there is no profile to look up, so ucode enrichment is skipped.
    ucode_profile: str | None = None
    if isinstance(auth, DatabricksAuth):
        ucode_profile = auth.profile
    elif "HARNESS_OPENAI_AGENTS_DATABRICKS_PROFILE" in env:
        ucode_profile = env["HARNESS_OPENAI_AGENTS_DATABRICKS_PROFILE"]

    use_responses = spec.executor.config.get("use_responses")
    if use_responses is not None:
        env["HARNESS_OPENAI_AGENTS_USE_RESPONSES"] = (
            "true" if _config_flag_is_true(use_responses) else "false"
        )
    _providers.configure_agent_harness_with_ucode(
        env,
        ucode_profile,
        harness_type="openai-agents-sdk",
    )
    return env


def _build_cursor_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Build the ``HARNESS_CURSOR_*`` env-var dict the cursor harness wrap reads.

    Maps spec.executor fields → the ``HARNESS_CURSOR_*`` env vars defined
    in ``omnigent/inner/cursor_harness.py``. Unlike the gateway-backed
    builders (claude-sdk / codex / pi / openai-agents), there is NO gateway or
    Databricks-profile resolution: the Cursor SDK talks only to Cursor's own
    backend (``CURSOR_API_KEY``) and has no custom API base-URL override, so it
    never routes through the Databricks AI gateway. That is also why cursor is
    intentionally absent from :data:`AgentHarnessType` and the gateway/ucode
    dicts above.

    Auth: an explicit ``executor.auth: {type: api_key, api_key: ...}`` is
    forwarded as ``HARNESS_CURSOR_API_KEY`` (the cursor harness passes it to the
    Cursor SDK as its ``api_key``). When the spec declares no auth at all, a
    ``CURSOR_API_KEY`` registered once via ``omnigent setup`` (the dedicated
    ``cursor:`` config block — see :mod:`omnigent.onboarding.cursor_auth`) is
    used instead, so a user need not export it in every shell. With neither, the
    harness falls back to an inherited ``CURSOR_API_KEY`` — a ``DatabricksAuth``
    profile does not apply to cursor and is ignored.

    Model: ``executor.model`` wins. When unset (common for Polly/Debby brain
    overrides), ``cursor.model`` then global ``model`` from config are used so
    the SDK does not fall through to ``auto-smart``.

    :param spec: The agent spec.
    :param workdir: The bundle's on-disk path, threaded as
        ``HARNESS_CURSOR_BUNDLE_DIR``.
    :returns: A dict of env-var overrides for
        :meth:`HarnessProcessManager.get_client(env=...)`.
    """
    env: dict[str, str] = {}
    model = _providers._resolve_spec_model(spec)
    if model is None:
        # Brain-picker sessions (Polly/Debby) often leave executor.model unset;
        # without a fallback the Cursor SDK defaults to auto-smart, which many
        # API keys reject. Prefer cursor.model, then global model.
        cfg = load_config()
        cursor_block = cfg.get("cursor")
        if isinstance(cursor_block, dict):
            cursor_model = cursor_block.get("model")
            if isinstance(cursor_model, str) and cursor_model.strip():
                model = cursor_model.strip()
        if model is None:
            global_model = cfg.get("model")
            if isinstance(global_model, str) and global_model.strip():
                model = global_model.strip()
    if model is not None:
        env["HARNESS_CURSOR_MODEL"] = model
    # Session workspace (the selected working folder), not the bundle workdir.
    # Without this the cursor subprocess inherits the runner's launch cwd — see
    # ``HARNESS_CURSOR_CWD`` in ``omnigent/inner/cursor_harness.py``.
    if cwd is not None:
        env["HARNESS_CURSOR_CWD"] = str(cwd)
    # Auth precedence: an explicit api-key auth on the spec wins; with NO spec
    # auth at all, fall back to a CURSOR_API_KEY registered once via
    # ``omnigent setup`` (the dedicated ``cursor:`` config block), else an
    # ambient CURSOR_API_KEY (an exported key / a host launched with one). A
    # Databricks / provider auth has no cursor equivalent and never silently
    # adopts a stored or ambient cursor key.
    if isinstance(spec.executor.auth, ApiKeyAuth):
        env["HARNESS_CURSOR_API_KEY"] = spec.executor.auth.api_key
    elif spec.executor.auth is None:
        # Imported lazily — the onboarding layer pulls in the secret store /
        # keyring, which the hot spawn-env path shouldn't import eagerly.
        from omnigent.onboarding.cursor_auth import resolve_cursor_api_key

        stored_key = resolve_cursor_api_key()
        if stored_key:
            env["HARNESS_CURSOR_API_KEY"] = stored_key
        else:
            ambient_key = os.environ.get("CURSOR_API_KEY")
            if ambient_key and ambient_key.strip():
                env["HARNESS_CURSOR_API_KEY"] = ambient_key.strip()
    # Always set so the wrap doesn't fall back to ``"all"`` and override an
    # explicit ``skills: none`` from the spec (parity with the peer builders).
    env["HARNESS_CURSOR_SKILLS_FILTER"] = json.dumps(spec.skills_filter)
    if spec.name:
        env["HARNESS_CURSOR_AGENT_NAME"] = spec.name
    if workdir is not None:
        env["HARNESS_CURSOR_BUNDLE_DIR"] = str(workdir)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_CURSOR_OS_ENV"] = os_env_payload
    # Permission stance for native-tool elicitation. Default ``auto`` (set by
    # the harness wrap when unset) skips ApprovalCards so headless / Polly
    # Cursor SDK workers don't stall; an explicit config value overrides.
    permission_mode = spec.executor.config.get("permission_mode")
    if permission_mode is not None:
        env["HARNESS_CURSOR_PERMISSION_MODE"] = str(permission_mode)
    return env


def _build_kimi_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """Build the env-var dict the kimi harness wrap reads.

    Maps ``spec.executor`` fields → the ``HARNESS_KIMI_*`` env vars
    defined in :mod:`omnigent.inner.kimi_harness`.

    The upstream Kimi Code CLI has no per-spawn provider override flag
    (no ``--config-file`` / ``--mcp-config-file``), so this builder
    only threads the model, working directory, and ``os_env`` sandbox
    spec. Provider routing for kimi lives in ``~/.kimi-code/config.toml``
    and is managed out-of-band via ``kimi provider add``. Unlike the
    sibling builders, ``_build_kimi_spawn_env`` never calls
    :func:`_providers.configure_agent_harness_with_provider` (there is no env-var
    surface to translate a provider into), so the rejection of declared
    auth has to live here: a spec that declares an explicit
    provider / Databricks / api_key auth raises directly so the user
    understands why their auth didn't take effect rather than silently
    routing through whatever default kimi already had.

    :param spec: The agent spec.
    :param cwd: Runtime working directory for the kimi subprocess — the
        session workspace (the folder the user launched in), NOT the agent
        bundle dir. Threaded as ``HARNESS_KIMI_CWD`` so kimi's tools operate on
        the user's project rather than the /tmp bundle (upstream kimi has no
        ``--work-dir`` flag, so the subprocess ``cwd=`` is the only lever).
        When unset, the harness wrap falls back to ``OMNIGENT_RUNNER_WORKSPACE``.
        Mirrors :func:`_build_pi_spawn_env`'s ``cwd`` handling.
    :returns: A dict of env-var overrides.
    :raises OmnigentError: If the spec declares ``executor.auth`` —
        upstream kimi has no per-spawn provider override, so the
        declared auth cannot be honored and we fail loud rather than
        launch against an unrelated ambient provider.
    """
    del workdir
    if spec.executor.auth is not None:
        raise OmnigentError(
            "The 'kimi' harness does not support per-invocation provider / "
            "auth injection: upstream kimi has no per-spawn config override "
            "(no ``--config-file`` / ``--mcp-config-file``). Remove "
            "``executor.auth`` from the spec and configure the provider once "
            "via `kimi provider add` in $KIMI_CODE_HOME/config.toml (default "
            "~/.kimi-code/config.toml), then pin the "
            "resulting model id in the agent spec. Omnigent-side provider "
            "injection is a deferred follow-up.",
            code=ErrorCode.INVALID_INPUT,
        )
    env: dict[str, str] = {}
    model = _providers._resolve_spec_model(spec)
    if model is not None:
        env["HARNESS_KIMI_MODEL"] = model
    if cwd is not None:
        env["HARNESS_KIMI_CWD"] = str(cwd)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_KIMI_OS_ENV"] = os_env_payload
    _apply_harness_path_override(env, "kimi")
    return env


def _build_hermes_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """Build the env-var dict the hermes harness wrap reads.

    Maps ``spec.executor`` fields → the ``HARNESS_HERMES_*`` env vars defined
    in :mod:`omnigent.inner.hermes_harness`. Hermes owns its own file-based
    auth (``hermes setup`` / ``hermes model``, credentials under its
    ``HERMES_HOME``), so — like :func:`_build_kimi_spawn_env` — this threads
    only the model, working directory, skills filter, and ``os_env`` sandbox
    spec; there is no gateway/provider env surface to configure.

    A hermes session with no spawn env is not inert: the wrap falls back to
    ``sandbox=none`` and the runner-wide launch directory, so the sandbox and
    workspace a session selected have to be threaded here to take effect.

    :param spec: The agent spec.
    :param cwd: Runtime working directory for the hermes subprocess — the
        session workspace, NOT the agent bundle dir. Threaded as
        ``HARNESS_HERMES_CWD``; when unset the wrap falls back to
        ``OMNIGENT_RUNNER_WORKSPACE``.
    :param workdir: The bundle's on-disk path. Accepted for signature parity
        with the sibling builders but not threaded: ``HARNESS_HERMES_BUNDLE_DIR``
        is reserved (there is no ``hermes chat`` flag for it yet), so the wrap
        would read a value it cannot pass on.
    :returns: A dict of env-var overrides.
    """
    del workdir
    env: dict[str, str] = {}
    model = _providers._resolve_spec_model(spec)
    if model is not None:
        env["HARNESS_HERMES_MODEL"] = model
    if cwd is not None:
        env["HARNESS_HERMES_CWD"] = str(cwd)
    # Always set so the wrap doesn't fall back to "all" and override an
    # explicit ``skills: none`` from the spec. Hermes turns this into its
    # ``-s`` / ``--ignore-rules`` argv (see hermes_executor._build_args).
    env["HARNESS_HERMES_SKILLS_FILTER"] = json.dumps(spec.skills_filter)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_HERMES_OS_ENV"] = os_env_payload
    _apply_harness_path_override(env, "hermes")
    return env


def _build_antigravity_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Map ``spec.executor`` fields → the ``HARNESS_ANTIGRAVITY_*`` env vars the
    antigravity harness wrap reads.

    Antigravity is Gemini-native with no OpenAI-compatible ``base_url``, so there
    is no gateway / ucode / Databricks path — only a direct API key or Vertex AI.
    API-key resolution (first wins): (1) spec ``executor.auth`` api_key; (2) the
    dedicated ``antigravity:`` config block from ``omnigent setup``; (3) an
    ambient ``GEMINI_API_KEY`` / ``ANTIGRAVITY_API_KEY``. The legacy global
    ``auth:`` block is deliberately NOT consulted: it carries the OpenAI/gateway
    key the other SDK harnesses inherit (an ``sk-…`` key), which the Gemini-native
    SDK can't use — adopting it would guarantee an auth failure / mis-billing and
    shadow the user's ambient ``GEMINI_API_KEY``. Any ``base_url`` is dropped (the
    SDK has no such field). Vertex AI is opt-in via ``executor.config``
    vertex/project/location, independent of the key path. A ``DatabricksAuth`` is
    unsupported — warned and ignored.

    Model: ``executor.model`` wins. When unset, ``antigravity.model`` from
    config is threaded so brain-picker sessions do not inherit an unintended
    SDK default.

    :param spec: The agent spec.
    :returns: Env-var overrides; may be empty (the wrap then uses the SDK's
        ambient creds and default model).
    """
    del cwd, workdir
    env: dict[str, str] = {}
    model = _providers._resolve_spec_model(spec)
    if model is None:
        # Brain-picker sessions often omit executor.model; honor antigravity.model
        # from config so the SDK does not pick an unintended provider default.
        agy_block = load_config().get("antigravity")
        if isinstance(agy_block, dict):
            agy_model = agy_block.get("model")
            if isinstance(agy_model, str) and agy_model.strip():
                model = agy_model.strip()
    if model is not None:
        env["HARNESS_ANTIGRAVITY_MODEL"] = model

    spec_auth = spec.executor.auth
    if spec_auth is not None and not isinstance(spec_auth, ApiKeyAuth):
        # Non-api_key auth implies gateway/base_url routing the SDK can't do.
        # Warn (don't drop silently — that looks like "my auth didn't take").
        _logger.warning(
            "antigravity harness: spec executor.auth is %s, but the Antigravity "
            "SDK only supports a direct API key or Vertex AI. Ignoring it and "
            "falling back to ambient Gemini credentials — configure an api_key, "
            "or executor.config vertex/project/location, instead.",
            type(spec_auth).__name__,
        )

    # Spec api-key wins; with no spec auth, fall back to the dedicated
    # ``antigravity:`` block, then an ambient Gemini key (see docstring). The
    # global ``auth:`` block is intentionally NOT consulted — it holds the
    # OpenAI/gateway key the SDK can't use. A non-api-key auth never adopts a key.
    if isinstance(spec_auth, ApiKeyAuth):
        # base_url intentionally dropped — the SDK has no such field.
        env["HARNESS_ANTIGRAVITY_API_KEY"] = spec_auth.api_key
    elif spec_auth is None:
        # Lazy import — the onboarding layer pulls in the secret store / keyring.
        from omnigent.onboarding.antigravity_auth import (
            ANTIGRAVITY_ENV_VARS,
            resolve_antigravity_api_key,
        )

        stored_key = resolve_antigravity_api_key()
        if stored_key is not None:
            env["HARNESS_ANTIGRAVITY_API_KEY"] = stored_key
        else:
            for _env_var in ANTIGRAVITY_ENV_VARS:
                if os.environ.get(_env_var):
                    env["HARNESS_ANTIGRAVITY_API_KEY"] = os.environ[_env_var]
                    break

    # Vertex AI: opt-in via executor.config (authenticated by GCP ADC).
    config = spec.executor.config
    if config.get("vertex"):
        env["HARNESS_ANTIGRAVITY_VERTEX"] = "1"
        project = config.get("project")
        if project:
            env["HARNESS_ANTIGRAVITY_PROJECT"] = str(project)
        location = config.get("location")
        if location:
            env["HARNESS_ANTIGRAVITY_LOCATION"] = str(location)

    return env


def _build_copilot_spawn_env(
    spec: AgentSpec,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
) -> dict[str, str]:
    """
    Build the ``HARNESS_COPILOT_*`` env-var dict the copilot harness wrap reads.

    Maps spec.executor fields → the ``HARNESS_COPILOT_*`` env vars defined in
    ``omnigent/inner/copilot_harness.py``. Like the cursor / antigravity
    builders there is NO gateway or Databricks-profile resolution: the GitHub
    Copilot SDK talks only to GitHub's Copilot backend (a GitHub token) and has
    no custom API base-URL override, so it never routes through the Databricks
    AI gateway. That is also why copilot is intentionally absent from
    :data:`AgentHarnessType` and the gateway/ucode dicts above.

    Auth: an explicit ``executor.auth: {type: api_key, api_key: ...}`` carries
    the GitHub token, forwarded as ``HARNESS_COPILOT_GITHUB_TOKEN`` (the copilot
    harness passes it to the SDK as its ``github_token``). When the spec declares
    no auth at all, a GitHub token registered once via ``omnigent setup`` (the
    dedicated ``copilot:`` config block — see
    :mod:`omnigent.onboarding.copilot_auth`) is used instead, so a user need not
    export it in every shell. With neither, the harness falls back to an
    inherited ``COPILOT_GITHUB_TOKEN`` / ``GH_TOKEN`` / ``GITHUB_TOKEN`` — a
    ``DatabricksAuth`` profile does not apply to copilot and is ignored.

    :param spec: The agent spec.
    :param workdir: The bundle's on-disk path, threaded as
        ``HARNESS_COPILOT_BUNDLE_DIR``.
    :returns: A dict of env-var overrides for
        :meth:`HarnessProcessManager.get_client(env=...)`.
    """
    env: dict[str, str] = {}
    model = _providers._resolve_spec_model(spec)
    if model is not None:
        env["HARNESS_COPILOT_MODEL"] = model
    # Session workspace (the selected working folder), not the bundle workdir.
    # Without this the copilot subprocess inherits the runner's launch cwd — see
    # ``HARNESS_COPILOT_CWD`` in ``omnigent/inner/copilot_harness.py``.
    if cwd is not None:
        env["HARNESS_COPILOT_CWD"] = str(cwd)
    # Auth precedence: an explicit api-key auth on the spec wins (its ``api_key``
    # is the GitHub token); with NO spec auth at all, fall back to a token
    # registered once via ``omnigent setup`` (the dedicated ``copilot:`` config
    # block), else an ambient ``COPILOT_GITHUB_TOKEN`` / ``GH_TOKEN`` /
    # ``GITHUB_TOKEN``. A Databricks / provider auth has no copilot equivalent
    # and never silently adopts a stored or ambient copilot token.
    if isinstance(spec.executor.auth, ApiKeyAuth):
        env["HARNESS_COPILOT_GITHUB_TOKEN"] = spec.executor.auth.api_key
    elif spec.executor.auth is None:
        # Imported lazily — the onboarding layer pulls in the secret store /
        # keyring, which the hot spawn-env path shouldn't import eagerly.
        from omnigent.onboarding.copilot_auth import (
            COPILOT_TOKEN_ENV_VARS,
            resolve_copilot_github_token,
        )

        stored_token = resolve_copilot_github_token()
        if stored_token is not None:
            env["HARNESS_COPILOT_GITHUB_TOKEN"] = stored_token
        else:
            for _env_var in COPILOT_TOKEN_ENV_VARS:
                if os.environ.get(_env_var):
                    env["HARNESS_COPILOT_GITHUB_TOKEN"] = os.environ[_env_var]
                    break
            # No token anywhere: leave it unset so the harness falls back to the
            # ``gh`` CLI login itself (it may run on a different host than the
            # runner, where a different ``gh`` session applies).
    # GitHub Enterprise hostname, when configured — auth and API calls must
    # target the user's own host rather than github.com.
    from omnigent.onboarding.copilot_auth import copilot_github_host

    copilot_host = copilot_github_host()
    if copilot_host is not None:
        env["HARNESS_COPILOT_GITHUB_HOST"] = copilot_host
    # Always set so the wrap doesn't fall back to ``"all"`` and override an
    # explicit ``skills: none`` from the spec (parity with the peer builders).
    env["HARNESS_COPILOT_SKILLS_FILTER"] = json.dumps(spec.skills_filter)
    if spec.name:
        env["HARNESS_COPILOT_AGENT_NAME"] = spec.name
    if workdir is not None:
        env["HARNESS_COPILOT_BUNDLE_DIR"] = str(workdir)
    os_env_payload = _serialize_os_env(spec.os_env)
    if os_env_payload is not None:
        env["HARNESS_COPILOT_OS_ENV"] = os_env_payload
    return env


def _serialize_os_env(value: OSEnvSpec | None) -> str | None:
    """
    Encode an :class:`OSEnvSpec` for the wrap's env-var input.

    JSON-encodes :func:`dataclasses.asdict` of the OSEnvSpec so
    the wrap can :func:`json.loads` it back on the harness side
    (per the per-spawn env-var pattern from §Step 5a). When
    *value* is ``None`` (no os_env declared on the spec), this
    returns ``None`` and ``_build_claude_sdk_spawn_env`` omits
    the env var entirely — the wrap then falls back to its
    enable-natives-by-default rule.

    :param value: ``spec.os_env`` — an :class:`OSEnvSpec`
        instance or ``None``.
    :returns: JSON string encoding the OSEnvSpec, or ``None``
        when *value* is ``None``.
    """
    import dataclasses

    if value is None:
        return None
    return json.dumps(dataclasses.asdict(value))


def _serialize_retry_policy(value: RetryPolicy | None) -> str | None:
    """
    Encode a :class:`RetryPolicy` for the wrap's env-var input.

    Phase 1f of ``designs/RETRY_ACROSS_HARNESSES.md``: the spec's
    ``LLMConfig.retry`` must take effect inside the CLI-harness
    subprocess. Serializing the whole policy as one JSON env var
    keeps the wire format compact (~150 bytes) and lets the
    harness round-trip via :func:`json.loads` →
    ``RetryPolicy(**dict)``. JSON over a flat fan-out of
    ``HARNESS_*_RETRY_MAX_RETRIES`` / ``..._BACKOFF_BASE_S`` /
    etc. because (a) ``RetryPolicy`` has 6 fields including a
    tuple, and a flat fan-out would multiply boilerplate at
    every wrap; (b) future field additions stay
    backwards-compatible — older wraps just ignore unknown
    keys via the ``__init__`` filter below.

    Returns ``None`` when the policy matches
    ``RetryPolicy()`` defaults so the env var is omitted —
    saves the harness wrap an unnecessary parse step on the
    common path.

    :param value: ``llm_config.retry`` — a :class:`RetryPolicy`
        or ``None``. ``None`` and ``RetryPolicy()`` (defaults)
        both produce ``None`` so the wrap falls back to the
        same baked-in default.
    :returns: JSON string encoding the policy's fields, or
        ``None`` when *value* is ``None`` or matches the
        defaults.
    """
    import dataclasses
    import json

    if value is None or value == RetryPolicy():
        return None
    payload = dataclasses.asdict(value)
    # ``retryable_status_codes`` is a tuple in the dataclass;
    # ``asdict`` converts it to a list. JSON has no tuple type —
    # the harness side reconstructs the tuple in
    # ``_deserialize_retry_policy``.
    return json.dumps(payload)


def _resolve_retry_policy(spec: AgentSpec) -> RetryPolicy | None:
    """
    Read the retry policy off a spec.

    Used by the per-harness ``_build_*_spawn_env`` builders to
    decide whether to thread a ``HARNESS_*_RETRY_POLICY`` env
    var through to the subprocess. Returns ``None`` when the
    spec has no ``llm`` block (e.g. a CLI-harness spec where
    the retry policy is implicit) so the harness wrap falls
    back to its baked-in :class:`RetryPolicy()` default.

    :param spec: The agent spec.
    :returns: ``spec.llm.retry`` if set; ``None`` otherwise.
    """
    if spec.llm is None:
        return None
    return spec.llm.retry


# ── Responses API helpers ─────────────────────────────────
