"""Observe the production monitor's actual macOS power assertions in owned processes."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import inspect
import json
import os
import platform
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

EXPECTED_HEAD = "3ea27608120cf7b80c10fb17067d9b1cc9d005c8"


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".pending")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


async def run_worker(product: Path, evidence: Path) -> None:
    from omnigent.util import keep_awake

    assert sys.platform == "darwin"
    assert keep_awake.keep_awake_enabled()
    assert Path(keep_awake.__file__).resolve() == product / "omnigent/util/keep_awake.py"
    assert shutil.which("caffeinate") == "/usr/bin/caffeinate"
    monitor = asyncio.create_task(
        keep_awake.run_keep_awake_monitor(
            (evidence / "busy").exists, runner_id=f"owned-macos-audit-{os.getpid()}"
        )
    )
    write_json(
        evidence / "ready.json",
        {
            "pid": os.getpid(),
            "platform": sys.platform,
            "monitor_source": inspect.getsourcefile(keep_awake.run_keep_awake_monitor),
            "caffeinate": shutil.which("caffeinate"),
            "poll_interval": keep_awake._POLL_INTERVAL_S,
        },
    )
    try:
        while not (evidence / "cancel_monitor").exists():
            if monitor.done():
                await monitor
                raise AssertionError("monitor unexpectedly returned")
            await asyncio.sleep(0.05)
    finally:
        monitor.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor
    write_json(evidence / "cancelled.json", {"pid": os.getpid(), "monitor_done": monitor.done()})
    while not (evidence / "shutdown").exists():
        await asyncio.sleep(0.05)


def wait_until(predicate: Callable[[], Any], description: str, timeout: float = 12) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        observed = predicate()
        if observed:
            return observed
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {description}")


def snapshot(owner_pid: int, tracked_pids: tuple[int, ...] = ()) -> dict[str, Any]:
    processes = []
    for line in subprocess.check_output(
        ["/bin/ps", "-axo", "pid=,ppid=,command="], text=True, timeout=3
    ).splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        process_pid, parent_pid = map(int, fields[:2])
        if parent_pid == owner_pid or process_pid in (owner_pid, *tracked_pids):
            processes.append({"pid": process_pid, "ppid": parent_pid, "command": fields[2]})
    return {
        "monotonic": time.monotonic(),
        "owner_pid": owner_pid,
        "owned_processes": processes,
        "pmset": subprocess.check_output(
            ["/usr/bin/pmset", "-g", "assertions"], text=True, timeout=3
        ),
    }


def has_idle_sleep_assertion(observed: dict[str, Any], process_pid: int) -> bool:
    return bool(
        re.search(
            rf"^\s*pid\s+{process_pid}\([^)]*\):[^\n]*PreventUserIdleSystemSleep",
            observed["pmset"],
            flags=re.MULTILINE,
        )
    )


def held_snapshot(owner_pid: int) -> dict[str, Any] | None:
    observed = snapshot(owner_pid)
    children = [
        process
        for process in observed["owned_processes"]
        if process["ppid"] == owner_pid
        and Path(shlex.split(process["command"])[0]).name == "caffeinate"
    ]
    if len(children) != 1:
        return None
    child = children[0]
    assert shlex.split(child["command"])[1:] == ["-i", "-w", str(owner_pid)]
    if not has_idle_sleep_assertion(observed, child["pid"]):
        return None
    observed["caffeinate_pid"] = child["pid"]
    return observed


def released_snapshot(owner_pid: int, child_pid: int) -> dict[str, Any] | None:
    observed = snapshot(owner_pid, (child_pid,))
    if any(process["pid"] == child_pid for process in observed["owned_processes"]):
        return None
    return None if has_idle_sleep_assertion(observed, child_pid) else observed


def continuously_held_snapshot(owner_pid: int, child_pid: int, duration: float) -> dict[str, Any]:
    deadline = time.monotonic() + duration
    observed = None
    while time.monotonic() < deadline:
        observed = held_snapshot(owner_pid)
        assert observed is not None, "busy monitor lost its real power assertion"
        assert observed["caffeinate_pid"] == child_pid, "busy monitor replaced its live child"
        time.sleep(0.1)
    assert observed is not None
    observed["continuously_observed_seconds"] = duration
    return observed


def run_probe(product: Path, evidence: Path) -> None:
    assert sys.platform == "darwin", "This probe requires an actual macOS host."
    assert (
        subprocess.check_output(
            ["git", "-C", str(product), "rev-parse", "HEAD"], text=True
        ).strip()
        == EXPECTED_HEAD
    )
    evidence.mkdir(parents=True, exist_ok=False)
    workers: list[subprocess.Popen[bytes]] = []
    logs = []
    results: dict[str, Any] = {
        "status": "running",
        "head": EXPECTED_HEAD,
        "platform": platform.platform(),
        "macos": subprocess.check_output(["/usr/bin/sw_vers"], text=True),
        "python": sys.version,
        "scope": (
            "real production monitor, owned worker processes, actual caffeinate/IOPM assertions"
        ),
        "limits": (
            "Controlled busy callback; no full runner/SDK/user-turn or physical-sleep journey."
        ),
        "phases": [],
        "model_or_inference_requests": 0,
    }

    def save_phase(name: str, observed: dict[str, Any]) -> None:
        write_json(evidence / f"{name}.json", observed)
        (evidence / f"{name}.pmset.txt").write_text(observed["pmset"])
        results["phases"].append({"name": name, "receipt": f"{name}.json"})
        print(f"PASS {name}", flush=True)

    def start_worker(name: str) -> tuple[subprocess.Popen[bytes], Path]:
        worker_root = evidence / name
        worker_root.mkdir()
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("OMNIGENT_", "RUNNER_"))
            and key
            not in {
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "DATABRICKS_TOKEN",
                "GH_TOKEN",
                "GITHUB_TOKEN",
            }
        }
        for key, relative in (
            ("HOME", "home"),
            ("OMNIGENT_CONFIG_HOME", "config"),
            ("OMNIGENT_DATA_DIR", "data"),
        ):
            owned_path = worker_root / relative
            owned_path.mkdir()
            environment[key] = str(owned_path)
        environment["PYTHONPATH"] = str(product)
        log = (worker_root / "worker.log").open("wb")
        logs.append(log)
        worker = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--product",
                str(product),
                "--evidence",
                str(worker_root),
                "--worker",
            ],
            cwd=product,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        workers.append(worker)

        def ready() -> bool:
            assert worker.poll() is None, f"worker exited; inspect {worker_root / 'worker.log'}"
            return (worker_root / "ready.json").is_file()

        wait_until(ready, "production monitor worker readiness")
        return worker, worker_root

    try:
        worker, worker_root = start_worker("normal")
        idle = snapshot(worker.pid)
        assert not any(process["ppid"] == worker.pid for process in idle["owned_processes"])
        save_phase("idle_without_owned_assertion", idle)
        (worker_root / "busy").touch()
        held = wait_until(lambda: held_snapshot(worker.pid), "real idle-sleep assertion")
        save_phase("busy_holds_real_assertion", held)
        first_pid = held["caffeinate_pid"]
        poll_interval = json.loads((worker_root / "ready.json").read_text())["poll_interval"]
        sustained = continuously_held_snapshot(worker.pid, first_pid, poll_interval + 1)
        save_phase("busy_retains_same_assertion_across_default_poll", sustained)
        (worker_root / "busy").unlink()
        released = wait_until(
            lambda: released_snapshot(worker.pid, first_pid), "idle releases/reaps caffeinate"
        )
        save_phase("idle_releases_assertion_and_child", released)
        (worker_root / "busy").touch()
        held_again = wait_until(lambda: held_snapshot(worker.pid), "second real assertion")
        assert held_again["caffeinate_pid"] != first_pid
        save_phase("second_turn_restarts_assertion", held_again)
        (worker_root / "cancel_monitor").touch()
        wait_until(
            lambda: (worker_root / "cancelled.json").is_file(),
            "monitor cancellation acknowledgment with owner alive",
        )
        assert worker.poll() is None
        cancelled = wait_until(
            lambda: released_snapshot(worker.pid, held_again["caffeinate_pid"]),
            "monitor cancellation releases assertion",
        )
        assert worker.poll() is None
        assert any(process["pid"] == worker.pid for process in cancelled["owned_processes"])
        save_phase("monitor_cancellation_releases_assertion_with_owner_alive", cancelled)
        (worker_root / "shutdown").touch()
        assert worker.wait(timeout=5) == 0
        crashed, crash_root = start_worker("crash")
        (crash_root / "busy").touch()
        before_crash = wait_until(lambda: held_snapshot(crashed.pid), "pre-crash assertion")
        save_phase("crash_precondition_holds_assertion", before_crash)
        crashed.kill()
        assert crashed.wait(timeout=5) == -signal.SIGKILL
        after_crash = wait_until(
            lambda: released_snapshot(crashed.pid, before_crash["caffeinate_pid"]),
            "caffeinate -w releases after owner process dies",
        )
        save_phase("owner_death_releases_assertion_and_child", after_crash)
        results["status"] = "pass"
    except BaseException as error:
        results["status"] = "fail"
        results["error"] = f"{type(error).__name__}: {error}"
        for worker in workers:
            write_json(evidence / f"failure-owner-{worker.pid}.json", snapshot(worker.pid))
        raise
    finally:
        for worker in workers:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(worker.pid, signal.SIGKILL)
            worker.wait(timeout=5)
        for log in logs:
            log.close()
        write_json(evidence / "result.json", results)
        print(json.dumps(results, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--worker", action="store_true")
    arguments = parser.parse_args()
    product = arguments.product.resolve()
    evidence = arguments.evidence.resolve()
    if arguments.worker:
        asyncio.run(run_worker(product, evidence))
    else:
        run_probe(product, evidence)


if __name__ == "__main__":
    main()
