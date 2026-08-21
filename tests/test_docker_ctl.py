"""Command assembly and state mapping. Nothing here starts a container."""

from __future__ import annotations

from app import docker_ctl
from app.docker_ctl import ContainerState, Mount


def test_run_argv_carries_this_cluster_s_conventions():
    argv = docker_ctl.build_run_argv(
        name="llmd-vllm-1",
        image="img",
        command=["vllm", "serve", "m"],
        env={"HF_HOME": "/hf"},
        mounts=[Mount("/host/cache", "/hf"), Mount("/ro", "/ro", read_only=True)],
    )
    assert argv[:4] == ["docker", "run", "--name", "llmd-vllm-1"]
    for expected in ("--runtime", "nvidia", "--gpus", "all", "--network", "host", "--ipc", "host"):
        assert expected in argv
    assert "memlock=-1" in argv and "stack=67108864" in argv
    assert "/host/cache:/hf" in argv
    assert "/ro:/ro:ro" in argv
    assert argv[argv.index("img") + 1:] == ["vllm", "serve", "m"]


def test_the_image_always_precedes_the_command():
    argv = docker_ctl.build_run_argv(
        name="n", image="img", command=["--flag", "value"], extra=["--cap-add", "SYS_PTRACE"]
    )
    assert argv.index("img") < argv.index("--flag")
    assert argv.index("--cap-add") < argv.index("img")


def test_gpu_can_be_declined_for_cpu_only_jobs():
    argv = docker_ctl.build_run_argv(name="n", image="i", command=["true"], gpu=False)
    assert "--gpus" not in argv


def test_preview_quotes_what_a_shell_would_mangle():
    argv = docker_ctl.build_run_argv(name="n", image="i", command=["sh", "-c", "echo hi there"])
    assert "'echo hi there'" in docker_ctl.preview(argv)


def test_ui_status_collapses_docker_s_state_machine():
    assert ContainerState("n", exists=False).ui_status == "absent"
    assert ContainerState("n", exists=True, running=True, status="running").ui_status == "running"
    assert ContainerState(
        "n", exists=True, running=True, status="running", health="starting"
    ).ui_status == "starting"
    assert ContainerState(
        "n", exists=True, status="exited", exit_code=0
    ).ui_status == "exited"
    assert ContainerState(
        "n", exists=True, status="exited", exit_code=1
    ).ui_status == "failed"
    assert ContainerState(
        "n", exists=True, status="exited", exit_code=137, oom_killed=True
    ).ui_status == "oom-killed"
