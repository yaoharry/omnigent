"""Harness launch configuration and per-spawn environment dispatch.

The runner owns the final launch envelope: harness selection, model override,
sandbox export, routing overlays, and subprocess environment cleanup. Provider
and harness-specific environment builders live in
``omnigent.harnesses.config.spawn_env`` and are selected through the harness
registry.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from omnigent.acp_cli_harnesses import ACP_CLI_HARNESSES
from omnigent.entities.session_resources import DEFAULT_ENVIRONMENT_ID
from omnigent.harness_aliases import canonicalize_harness
from omnigent.harness_availability import CODEX_CANONICAL_HARNESSES
from omnigent.harness_plugins import load_object, model_env_keys, spawn_env_builders
from omnigent.runner.subagent_routing import session_router_env, session_routing_class
from omnigent.spec import AgentSpec

if TYPE_CHECKING:
    from omnigent.runner.resource_registry import SessionResourceRegistry

_logger = logging.getLogger(__name__)

_HARNESS_MODEL_ENV_KEY = model_env_keys()


class _SpawnEnvBuilder(Protocol):
    def __call__(
        self,
        spec: object,
        *,
        cwd: Path | None,
        workdir: Path | None,
    ) -> dict[str, str]: ...


class _ModelCopyValue(Protocol):
    def model_copy(self, *, update: Mapping[str, object]) -> object: ...


def build_spawn_env_from_spec(
    spec: AgentSpec,
    harness: str,
    *,
    cwd: Path | None = None,
    workdir: Path | None = None,
    model_override: str | None = None,
    session_id: str | None = None,
    resource_registry: SessionResourceRegistry | None = None,
) -> dict[str, str] | None:
    """Build a spawn environment from a resolved agent spec.

    Native terminals build their bridge environment elsewhere. All headless
    harnesses use the registry's common builder contract; builtin ACP CLI rows
    remain explicit because their builder needs the selected row and session id.
    """
    requested_harness = harness
    harness = canonicalize_harness(harness) or harness
    if requested_harness.startswith("acp:"):
        spec = dataclasses.replace(
            spec,
            executor=dataclasses.replace(
                spec.executor,
                config={**spec.executor.config, "harness": requested_harness},
            ),
        )

    from omnigent.sandbox.copy_on_write import validate_copy_on_write_harness

    validate_copy_on_write_harness(getattr(spec, "os_env", None), harness)
    effective_spec = spec
    from omnigent.inference_config import load_runtime_inference_config, parse_inference_config

    has_inference_bindings = bool(parse_inference_config(load_runtime_inference_config()))
    if has_inference_bindings and dataclasses.is_dataclass(spec):
        declared_harness = str(spec.executor.config.get("harness") or "")
        identity = (
            requested_harness
            if requested_harness.startswith("acp:")
            else declared_harness
            if harness == "acp" and declared_harness.startswith("acp:")
            else harness
        )
        effective_spec = dataclasses.replace(
            spec,
            executor=dataclasses.replace(
                spec.executor,
                config={**spec.executor.config, "harness": identity},
                model=model_override if model_override is not None else spec.executor.model,
            ),
        )

    if model_override is not None:
        executor = getattr(spec, "executor", None)
        if (
            harness == "acp"
            and not has_inference_bindings
            and dataclasses.is_dataclass(spec)
            and dataclasses.is_dataclass(executor)
        ):
            effective_spec = dataclasses.replace(
                spec,
                executor=dataclasses.replace(spec.executor, model=model_override),
            )
        elif hasattr(spec, "model_copy") and hasattr(executor, "model_copy"):
            copied_executor = cast(_ModelCopyValue, executor).model_copy(
                update={"model": model_override}
            )
            effective_spec = cast(
                AgentSpec,
                cast(_ModelCopyValue, spec).model_copy(update={"executor": copied_executor}),
            )

    acp_default_model: str | None = None
    if harness == "acp":
        from omnigent.models.model_catalog import _acp_launch_model, validate_acp_model

        policy_spec = effective_spec if has_inference_bindings else spec
        acp_default_model = _acp_launch_model(policy_spec)
        validate_acp_model(policy_spec, acp_default_model)
        validate_acp_model(policy_spec, model_override)

    try:
        if harness in ACP_CLI_HARNESSES:
            from omnigent.harnesses.config.spawn_env import _build_acp_cli_spawn_env

            env = _build_acp_cli_spawn_env(
                effective_spec,
                harness=harness,
                cwd=cwd,
                workdir=workdir,
                session_id=session_id,
            )
        else:
            builder_path = spawn_env_builders().get(harness)
            if builder_path is None:
                # Native terminal harnesses and unknown harnesses build env elsewhere.
                return None
            builder = load_object(builder_path)
            if not callable(builder):
                raise TypeError(f"spawn environment builder {builder_path!r} is not callable")
            env = cast(_SpawnEnvBuilder, builder)(
                effective_spec,
                cwd=cwd,
                workdir=workdir,
            )
            if harness == "acp":
                # Reset uses the original default even when the process launched
                # with a session override. Empty defers to the vendor's first model.
                env["HARNESS_ACP_DEFAULT_MODEL"] = acp_default_model or ""
            if harness == "codex":
                env["HARNESS_CODEX_SKILLS_DIR"] = (
                    str(resource_registry.codex_skills_dir(session_id))
                    if resource_registry is not None and session_id is not None
                    else ""
                )
    except ImportError:
        return None

    from omnigent.inner.agent_env import desktop_session_passthrough, strip_desktop_session_env

    env = strip_desktop_session_env(env)
    env.update(desktop_session_passthrough(effective_spec.os_env))

    if (
        spec.os_env is not None
        and spec.os_env.sandbox is not None
        and any(path.copy_on_write for path in spec.os_env.sandbox.write_path_specs)
    ):
        if resource_registry is None or session_id is None:
            raise ValueError("copy_on_write harnesses require a session resource registry")
        from omnigent.sandbox.copy_on_write import (
            SHARED_ENVIRONMENT_VAR,
            export_shared_environment,
        )

        environment = resource_registry.resolve_environment(
            session_id, DEFAULT_ENVIRONMENT_ID, spec
        )
        policy = getattr(environment, "sandbox", None)
        if policy is None:
            raise ValueError("copy_on_write requires a local sandbox environment")
        environment.prepare_sandbox(policy)
        env[SHARED_ENVIRONMENT_VAR] = export_shared_environment(policy)

    if session_id:
        env.update(session_router_env(session_id, harness))
        if harness in CODEX_CANONICAL_HARNESSES:
            from omnigent.inner.codex_executor import codex_extended_catalog_env

            env.update(
                codex_extended_catalog_env(session_routing_class(session_id).routing_enabled)
            )

    if model_override:
        model_key = _HARNESS_MODEL_ENV_KEY.get(harness)
        if model_key is not None:
            env[model_key] = model_override

    prefix = f"HARNESS_{harness.upper().replace('-', '_')}"
    _logger.info(
        "%s gateway routing: gateway=%s base_url=%s profile=%s model=%s",
        harness,
        env.get(f"{prefix}_GATEWAY"),
        env.get(f"{prefix}_GATEWAY_BASE_URL") or env.get(f"{prefix}_GATEWAY_BASE_URLS"),
        env.get(f"{prefix}_DATABRICKS_PROFILE"),
        env.get(_HARNESS_MODEL_ENV_KEY.get(harness, f"{prefix}_MODEL")),
        extra={"session_id": session_id},
    )
    return env


# Compatibility name used while callers migrate to the launch module.
_build_spawn_env_from_spec = build_spawn_env_from_spec
