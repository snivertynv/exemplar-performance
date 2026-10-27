# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a TaskTrace recording inside a fresh OpenSandbox sandbox (``tasktrace profile --runtime current``).

Creates one sandbox from IMAGE, uploads a ``tasktrace`` binary built for the sandbox's architecture and
the recording, runs ``tasktrace profile trace|atif --runtime current --current-container-exclusive``,
downloads the output directory into OUT_DIR and deletes the sandbox.

The replay runs from the sandbox entrypoint rather than through the OpenSandbox exec API. tasktrace
treats every process outside its own ancestry as unrelated and then marks the CPU measurement invalid;
OpenSandbox's default entrypoint (``tail -f /dev/null`` under execd, which must not be killed) and every
exec'd command are such processes. The entrypoint here waits for a ``go`` file, runs the replay, then
keeps the sandbox alive; completion is polled through the file API, which starts no processes.

Connection settings come from the OpenSandbox runtime: OPENSANDBOX_URL, OPENSANDBOX_API_KEY and the
TASKTRACE_CAPTURE_OPENSANDBOX_* request-timeout and TLS variables.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import logging
import shlex
import tarfile
import time
from datetime import timedelta
from pathlib import Path

REMOTE_DIR = "/tmp/tasktrace"


def entrypoint_script(remote_dir: str = REMOTE_DIR) -> str:
    """Sandbox entrypoint: wait for ``go``, run the replay, then keep the sandbox alive for download."""
    return (
        f"mkdir -p {remote_dir}; while [ ! -e {remote_dir}/go ]; do sleep 1; done; "
        f"/bin/sh {remote_dir}/run.sh; exec tail -f /dev/null"
    )


def replay_command(
    *,
    trajectory: str,
    trajectory_format: str,
    cpus: str,
    workdir: str,
    pacing: str,
    timeout_sec: float | None,
    continue_after_recorded_timeout: bool,
    remote_dir: str = REMOTE_DIR,
) -> str:
    """The ``tasktrace profile`` command line run inside the sandbox."""
    parts = [
        "cd",
        shlex.quote(workdir),
        "&&",
        f"{remote_dir}/tasktrace",
        "profile",
        trajectory_format,
        trajectory,
        "--runtime",
        "current",
        "--current-container-exclusive",
        "--allocated-cpus",
        shlex.quote(cpus),
        "--pacing",
        shlex.quote(pacing),
        "--workdir",
        shlex.quote(workdir),
        "--strace-mode",
        "none",
    ]
    if timeout_sec is not None:
        parts += ["--timeout-sec", f"{timeout_sec:g}"]
    if continue_after_recorded_timeout:
        if trajectory_format != "atif":
            raise ValueError("--continue-after-recorded-timeout applies to ATIF replays only")
        parts.append("--continue-after-recorded-timeout")
    parts += ["--output-dir", f"{remote_dir}/out"]
    return " ".join(parts)


def run_script(command: str, remote_dir: str = REMOTE_DIR) -> str:
    """run.sh: replay, archive the output with the log, then publish the exit code atomically."""
    return (
        f"{command} > {remote_dir}/tasktrace.log 2>&1\n"
        "rc=$?\n"
        f"mkdir -p {remote_dir}/out; cp {remote_dir}/tasktrace.log {remote_dir}/out/\n"
        f"tar czf {remote_dir}/out.tgz -C {remote_dir}/out .\n"
        f"echo $rc > {remote_dir}/done.tmp && mv {remote_dir}/done.tmp {remote_dir}/done\n"
    )


async def replay(args: argparse.Namespace) -> int:
    from opensandbox import Sandbox
    from opensandbox.models.sandboxes import PlatformSpec

    from tasktrace_capture.opensandbox_env import _connection_config

    trajectory = f"{REMOTE_DIR}/trajectory.{'jsonl' if args.format == 'trace' else 'json'}"
    command = replay_command(
        trajectory=trajectory,
        trajectory_format=args.format,
        cpus=args.cpus,
        workdir=args.workdir,
        pacing=args.pacing,
        timeout_sec=args.timeout_sec,
        continue_after_recorded_timeout=args.continue_after_recorded_timeout,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    sandbox = await Sandbox.create(
        args.image,
        platform=PlatformSpec(os="linux", arch=args.arch),
        resource={"cpu": args.cpus, "memory": args.memory},
        timeout=timedelta(minutes=args.timeout_min + 15),
        ready_timeout=timedelta(minutes=args.ready_timeout_min),
        connection_config=_connection_config(),
        entrypoint=["/bin/sh", "-c", entrypoint_script()],
        metadata={"purpose": "tasktrace-replay"},
    )
    print(f"[replay] sandbox {sandbox.id} from {args.image} ready in {time.time() - started:.0f}s", flush=True)
    try:
        await sandbox.files.write_file(f"{REMOTE_DIR}/tasktrace", Path(args.tasktrace_bin).read_bytes(), mode="755")
        await sandbox.files.write_file(trajectory, Path(args.trajectory).read_bytes(), mode="644")
        await sandbox.files.write_file(f"{REMOTE_DIR}/run.sh", run_script(command).encode(), mode="644")
        print(f"[replay] {command}", flush=True)
        replay_started = time.time()
        await sandbox.files.write_file(f"{REMOTE_DIR}/go", b"", mode="644")
        while True:
            await asyncio.sleep(10)
            try:
                rc = int((await sandbox.files.read_bytes(f"{REMOTE_DIR}/done")).decode().strip())
                break
            except Exception:
                if time.time() - replay_started > args.timeout_min * 60:
                    raise TimeoutError(f"replay did not finish in {args.timeout_min} min") from None
        print(f"[replay] finished in ~{time.time() - replay_started:.0f}s: rc={rc}", flush=True)
        data = await sandbox.files.read_bytes(f"{REMOTE_DIR}/out.tgz")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            archive.extractall(out_dir, filter="data")
        print(f"[replay] output: {out_dir}", flush=True)
        return rc
    finally:
        await sandbox.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tasktrace_bin", help="tasktrace binary for the sandbox architecture (e.g. x86_64 glibc)")
    parser.add_argument("trajectory", help="finalized capture trace (.jsonl) or ATIF trajectory.json")
    parser.add_argument("image", help="task image, as the capture used it")
    parser.add_argument("out_dir", help="local directory for the replay output")
    parser.add_argument("--format", choices=("trace", "atif"), default="trace")
    parser.add_argument("--arch", default="amd64", help="sandbox platform architecture (default amd64)")
    parser.add_argument("--cpus", default="4")
    parser.add_argument("--memory", default="4Gi")
    parser.add_argument("--workdir", default="/testbed")
    parser.add_argument("--pacing", default="unpaced")
    parser.add_argument("--timeout-sec", type=float, help="per-action limit (mini-swe-agent's live default is 30)")
    parser.add_argument(
        "--continue-after-recorded-timeout",
        action="store_true",
        help="ATIF only: keep replaying past actions that also timed out in the recording",
    )
    parser.add_argument("--timeout-min", type=int, default=90, help="replay time limit")
    parser.add_argument("--ready-timeout-min", type=int, default=10, help="sandbox start limit (image pulls)")
    args = parser.parse_args(argv)
    # Polling the done file through the file API logs every 404 until the replay finishes.
    logging.getLogger("opensandbox").setLevel(logging.CRITICAL)
    return asyncio.run(replay(args))


if __name__ == "__main__":
    raise SystemExit(main())
