# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize Harbor's mini-swe-agent ATIF trajectory for ``tasktrace profile atif``.

Both fixes keep the trajectory faithful to what mini-swe-agent executed:

1. Bind results to calls. Harbor's export writes a step's observation results in call
   order without ``source_call_id``. tasktrace accepts that only for single-call steps
   and rejects a step with parallel tool calls ("no exact observation binding").
   mini-swe-agent runs a step's calls in order, so result *i* belongs to call *i*:
   ``source_call_id`` is set where a step has as many results as calls and none is bound.
2. Drop bash arguments other than ``command``. Models sometimes add e.g.
   ``"timeout": "60"``; mini-swe-agent ignores it (its environment timeout applies),
   but tasktrace reads any ``timeout`` argument as a Claude Code millisecond timeout
   and rejects a string.

Any other inconsistency is left for tasktrace to report.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def normalize_mini_swe_atif(trajectory: dict[str, Any]) -> tuple[int, int]:
    """Normalize ``trajectory`` in place; return (steps bound, calls with arguments dropped)."""
    original_format = str(((trajectory.get("agent") or {}).get("extra") or {}).get("original_format", ""))
    mini_swe = original_format.startswith("mini-swe-agent")
    bound = dropped = 0
    for step in trajectory.get("steps", []):
        calls = step.get("tool_calls") or []
        results = (step.get("observation") or {}).get("results") or []
        if len(calls) > 1 and len(calls) == len(results) and not any("source_call_id" in r for r in results):
            for call, result in zip(calls, results):
                result["source_call_id"] = call["tool_call_id"]
            bound += 1
        if not mini_swe:
            continue
        for call in calls:
            arguments = call.get("arguments")
            if call.get("function_name") == "bash" and isinstance(arguments, dict) and set(arguments) - {"command"}:
                call["arguments"] = {"command": arguments.get("command", "")}
                dropped += 1
    return bound, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Harbor ATIF trajectory.json")
    parser.add_argument("output", help="normalized trajectory.json to write")
    args = parser.parse_args(argv)
    with open(args.input, encoding="utf-8") as handle:
        trajectory = json.load(handle)
    bound, dropped = normalize_mini_swe_atif(trajectory)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(trajectory, handle, indent=1)
    print(f"bound results by position in {bound} multi-call step(s); dropped ignored bash arguments in {dropped} call(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
