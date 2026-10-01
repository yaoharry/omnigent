"""Import-boundary checks for harness configuration modules."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_harness_config_import_does_not_initialize_runtime_or_server() -> None:
    repo_root = Path(__file__).parents[2]
    script = """
import sys

import omnigent.harnesses.config.providers
import omnigent.harnesses.config.spawn_env
import omnigent.runtime.services as services

for name in (
    "omnigent.server.smart_routing",
    "omnigent.server.app",
    "omnigent.runner.app",
    "omnigent.runtime.workflow",
):
    assert name not in sys.modules, name
assert services._services is None
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
