# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from tasktrace_capture.atif_normalize import normalize_mini_swe_atif
from tasktrace_capture.opensandbox_replay import entrypoint_script, replay_command, run_script
from tasktrace_capture.opensandbox_task import convert_task

SWEBENCH_DOCKERFILE = """# template comment
FROM swebench/sweb.eval.x86_64.django_1776_django-11490:latest
WORKDIR /testbed
RUN curl -LsSf https://astral.sh/uv/0.7.13/install.sh | sh
RUN mkdir -p /logs
"""


def make_task(root: Path, dockerfile: str) -> Path:
    task = root / "task"
    (task / "environment").mkdir(parents=True)
    (task / "tests").mkdir()
    (task / "environment" / "Dockerfile").write_text(dockerfile)
    (task / "tests" / "test.sh").write_text("#!/bin/bash\nuv run parser.py\n")
    (task / "task.toml").write_text("[environment]\ncpus = 1\nmemory = '4G'\n")
    return task


def test_convert_task_keeps_prebuilt_image_and_moves_template_steps(tmp_path):
    source = make_task(tmp_path, SWEBENCH_DOCKERFILE)
    kept, moved = convert_task(source, tmp_path / "out", cpus=4)

    assert kept == ["FROM swebench/sweb.eval.x86_64.django_1776_django-11490:latest", "WORKDIR /testbed"]
    assert (tmp_path / "out/environment/Dockerfile").read_text().splitlines() == kept
    test_sh = (tmp_path / "out/tests/test.sh").read_text().splitlines()
    assert test_sh[0] == "#!/bin/bash"
    assert "astral.sh/uv/0.7.13/install.sh" in test_sh[2]
    assert test_sh[-1] == "uv run parser.py"
    assert "cpus = 4" in (tmp_path / "out/task.toml").read_text()
    assert len(moved) == 1
    assert (source / "environment" / "Dockerfile").read_text() == SWEBENCH_DOCKERFILE


def test_convert_task_refuses_steps_that_change_the_image(tmp_path):
    source = make_task(tmp_path, "FROM ubuntu:24.04\nRUN apt-get update && \\\n    apt-get install -y git\n")
    with pytest.raises(ValueError, match="unsupported Dockerfile step"):
        convert_task(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def mini_swe_trajectory(calls, results):
    return {
        "agent": {"name": "mini-swe-agent", "extra": {"original_format": "mini-swe-agent-1.1"}},
        "steps": [{"step_id": 1, "tool_calls": calls, "observation": {"results": results}}],
    }


def test_normalize_binds_parallel_results_in_call_order():
    calls = [
        {"tool_call_id": "a", "function_name": "bash", "arguments": {"command": "ls"}},
        {"tool_call_id": "b", "function_name": "bash", "arguments": {"command": "pwd"}},
    ]
    trajectory = mini_swe_trajectory(calls, [{"content": "1"}, {"content": "2"}])

    assert normalize_mini_swe_atif(trajectory) == (1, 0)
    assert [r["source_call_id"] for r in trajectory["steps"][0]["observation"]["results"]] == ["a", "b"]


def test_normalize_leaves_mismatched_result_counts_for_tasktrace():
    calls = [{"tool_call_id": "a", "function_name": "bash", "arguments": {"command": "ls"}}]
    trajectory = mini_swe_trajectory(calls, [{"content": "1"}, {"content": "format error"}])

    assert normalize_mini_swe_atif(trajectory) == (0, 0)
    assert all("source_call_id" not in r for r in trajectory["steps"][0]["observation"]["results"])


def test_normalize_drops_bash_arguments_mini_swe_agent_ignores():
    calls = [{"tool_call_id": "a", "function_name": "bash", "arguments": {"command": "make", "timeout": "60"}}]
    trajectory = mini_swe_trajectory(calls, [{"content": "ok"}])

    assert normalize_mini_swe_atif(trajectory) == (0, 1)
    assert trajectory["steps"][0]["tool_calls"][0]["arguments"] == {"command": "make"}


def test_normalize_keeps_arguments_of_other_agents():
    trajectory = mini_swe_trajectory(
        [{"tool_call_id": "a", "function_name": "bash", "arguments": {"command": "make", "timeout": 60000}}],
        [{"content": "ok"}],
    )
    trajectory["agent"] = {"name": "claude-code"}

    assert normalize_mini_swe_atif(trajectory) == (0, 0)
    assert trajectory["steps"][0]["tool_calls"][0]["arguments"]["timeout"] == 60000


def test_replay_command_for_atif_with_recorded_timeouts():
    command = replay_command(
        trajectory="/tmp/tasktrace/trajectory.json",
        trajectory_format="atif",
        cpus="4",
        workdir="/testbed",
        pacing="unpaced",
        timeout_sec=30.0,
        continue_after_recorded_timeout=True,
    )
    assert command.startswith("cd /testbed && /tmp/tasktrace/tasktrace profile atif /tmp/tasktrace/trajectory.json")
    assert "--runtime current --current-container-exclusive --allocated-cpus 4" in command
    assert "--timeout-sec 30 --continue-after-recorded-timeout --output-dir /tmp/tasktrace/out" in command


def test_replay_command_rejects_continue_flag_for_capture_traces():
    with pytest.raises(ValueError, match="ATIF replays only"):
        replay_command(
            trajectory="/tmp/tasktrace/trajectory.jsonl",
            trajectory_format="trace",
            cpus="4",
            workdir="/testbed",
            pacing="unpaced",
            timeout_sec=None,
            continue_after_recorded_timeout=True,
        )


def test_entrypoint_waits_for_go_and_keeps_sandbox_alive():
    script = entrypoint_script("/r")
    assert script.index("/r/go") < script.index("/r/run.sh") < script.index("exec tail -f /dev/null")
    run = run_script("true", "/r")
    assert run.rstrip().endswith("mv /r/done.tmp /r/done")
    assert run.index("tar czf /r/out.tgz") < run.index("/r/done.tmp")
