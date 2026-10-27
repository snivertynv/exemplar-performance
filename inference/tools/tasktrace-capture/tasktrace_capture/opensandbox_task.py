# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Make a Harbor task runnable on OpenSandbox, which runs prebuilt images and cannot build Dockerfiles.

Copies the task and keeps only ``FROM`` and ``WORKDIR`` in ``environment/Dockerfile``. The two ``RUN``
steps of Harbor's SWE-bench template move to where they are needed: ``mkdir -p /logs`` is already done
by the OpenSandbox runtime, and the uv install used by the verifier (``uv run parser.py``) goes to the
top of ``tests/test.sh``. Any other build step is refused, since it would change the image.
``--cpus`` overrides ``[environment].cpus`` (SWE-bench Verified tasks ask for 1).
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

# RUN steps of Harbor's SWE-bench task template -> what replaces them outside the image.
KNOWN_RUN_STEPS = {
    "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh": (
        "command -v uv >/dev/null || [ -x /root/.local/bin/uv ] || "
        "curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh\n"
    ),
    "mkdir -p /logs": "",
}


def _instructions(dockerfile: str) -> list[str]:
    lines, current = [], ""
    for raw in dockerfile.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        current += line[:-1] + " " if line.endswith("\\") else line
        if not line.endswith("\\"):
            lines.append(current.strip())
            current = ""
    return lines


def convert_task(source: Path, destination: Path, cpus: int | None = None) -> tuple[list[str], list[str]]:
    """Write the OpenSandbox variant of ``source`` to ``destination``; return (kept, moved) steps."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)
    dockerfile = destination / "environment" / "Dockerfile"
    kept: list[str] = []
    moved: list[str] = []
    for instruction in _instructions(dockerfile.read_text(encoding="utf-8")):
        op, _, argument = instruction.partition(" ")
        if op.upper() in ("FROM", "WORKDIR"):
            kept.append(instruction)
        elif op.upper() == "RUN" and argument.strip() in KNOWN_RUN_STEPS:
            moved.append(KNOWN_RUN_STEPS[argument.strip()])
        else:
            shutil.rmtree(destination)
            raise ValueError(f"unsupported Dockerfile step for OpenSandbox in {source}: {instruction}")
    dockerfile.write_text("\n".join(kept) + "\n", encoding="utf-8")

    prelude = "".join(step for step in moved if step)
    if prelude:
        test_sh = destination / "tests" / "test.sh"
        body = test_sh.read_text(encoding="utf-8")
        shebang, _, rest = body.partition("\n") if body.startswith("#!") else ("", "", body)
        test_sh.write_text(
            (shebang + "\n" if shebang else "") + "# OpenSandbox: steps moved from the Dockerfile\n" + prelude + rest,
            encoding="utf-8",
        )

    if cpus:
        task_toml = destination / "task.toml"
        text, count = re.subn(r"(?m)^cpus\s*=.*$", f"cpus = {cpus}", task_toml.read_text(encoding="utf-8"))
        if not count:
            raise ValueError(f"{task_toml} has no [environment].cpus to override")
        task_toml.write_text(text, encoding="utf-8")
    return kept, [step.strip() for step in moved if step]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="Harbor task directory")
    parser.add_argument("destination", type=Path, help="output task directory (replaced)")
    parser.add_argument("--cpus", type=int, help="override [environment].cpus")
    args = parser.parse_args(argv)
    kept, moved = convert_task(args.source, args.destination, args.cpus)
    print(f"{args.destination}: Dockerfile {kept}; moved to tests/test.sh: {moved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
