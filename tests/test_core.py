"""Storage, telemetry parsing and the event broker."""

from __future__ import annotations

import asyncio

import pytest

from app import db, events, jobs, telemetry


def test_settings_round_trip_json():
    db.set_setting("probe", {"a": [1, 2], "b": None})
    assert db.get_setting("probe") == {"a": [1, 2], "b": None}
    assert db.get_setting("missing", "fallback") == "fallback"


def test_loads_survives_a_corrupt_column():
    assert db.loads("{not json", {"safe": True}) == {"safe": True}
    assert db.loads(None, []) == []
    assert db.loads('{"x":1}') == {"x": 1}


def test_hydrate_decodes_only_the_named_columns():
    row = db.hydrate({"spec": '{"a":1}', "title": "x"}, ("spec",))
    assert row == {"spec": {"a": 1}, "title": "x"}


def test_meminfo_reports_a_plausible_host():
    memory = telemetry.read_meminfo()
    assert memory.total_bytes > 1 << 30
    assert 0 <= memory.available_bytes <= memory.total_bytes
    assert 0.0 <= memory.used_fraction <= 1.0


def test_na_fields_are_dropped_rather_than_shown_as_zero():
    # GB10 reports [N/A] for every memory field; a dashboard must not draw a
    # gauge from that.
    assert telemetry._coerce("temperature.gpu", "[N/A]") is None
    assert telemetry._coerce("temperature.gpu", "48") == 48.0
    assert telemetry._coerce("name", "NVIDIA GB10") == "NVIDIA GB10"
    assert telemetry._coerce("power.draw", "not-a-number") is None


def test_carriage_returns_collapse_to_the_final_state():
    assert jobs.split_carriage("10%\r50%\r100%") == "100%"
    assert jobs.split_carriage("plain") == "plain"


@pytest.mark.asyncio
async def test_broker_delivers_and_unsubscribes():
    broker = events.Broker()
    stream = broker.subscribe("t")
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.02)
    assert broker.subscriber_count("t") == 1

    await broker.publish("t", {"n": 1})
    await broker.publish("t", {"n": 2})
    assert await asyncio.wait_for(pending, timeout=2) == {"n": 1}
    assert await asyncio.wait_for(anext(stream), timeout=2) == {"n": 2}

    await stream.aclose()
    assert broker.subscriber_count("t") == 0


@pytest.mark.asyncio
async def test_a_slow_subscriber_drops_frames_instead_of_growing_forever():
    broker = events.Broker(maxsize=4)
    stream = broker.subscribe("t")
    # subscribe() only registers its queue once the generator is first driven.
    first = asyncio.create_task(anext(stream))
    await asyncio.sleep(0.02)

    for index in range(20):
        await broker.publish("t", index)

    # Twenty frames were published without the loop ever handing control back,
    # so the queue dropped from the head. What survives is the newest four —
    # a stalled browser tab loses history, never live data, and the producer's
    # memory stays bounded.
    seen = [await asyncio.wait_for(first, timeout=2)]
    seen += [await asyncio.wait_for(anext(stream), timeout=2) for _ in range(3)]
    assert seen == [16, 17, 18, 19]
    await stream.aclose()


def test_output_dir_is_sanitised():
    path = jobs.output_dir("abc123", "org/model:weird name")
    assert path.name.startswith("org-model-weird-name")
    assert path.is_dir()


def test_replayed_series_collapse_by_identity():
    # A reattached job re-reads part of its log, so accumulating series must not
    # grow a second copy of every entry.
    progress = {
        "loss_history": [[1, 2.5], [2, 2.8], [1, 2.5], [2, 2.8], [3, 1.9]],
        "trials": [{"trial": 1, "kl": 0.1}, {"trial": 1, "kl": 0.1}, {"trial": 2, "kl": 0.2}],
        "log_lines": ["a", "b", "a"],
        "step": 3,
    }
    jobs._dedupe_series(progress)
    assert progress["loss_history"] == [[1, 2.5], [2, 2.8], [3, 1.9]]
    assert progress["trials"] == [{"trial": 1, "kl": 0.1}, {"trial": 2, "kl": 0.2}]
    # A plain list that is not a series is left alone.
    assert progress["log_lines"] == ["a", "b", "a"]
    assert progress["step"] == 3


def test_a_later_entry_wins_over_an_earlier_one():
    progress = {"loss_history": [[4, 9.9], [4, 1.1]]}
    jobs._dedupe_series(progress)
    assert progress["loss_history"] == [[4, 1.1]]


def test_a_mistyped_api_route_is_a_json_404_not_the_spa_shell():
    # The catch-all that serves the shell for deep links must not swallow
    # /api/*: an HTML body there reaches the browser as a JSON parse error
    # instead of the 404 that would explain what went wrong.
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        missing = client.get("/api/definitely-not-a-route")
        assert missing.status_code == 404
        assert missing.headers["content-type"].startswith("application/json")

        deep_link = client.get("/serve")
        assert deep_link.status_code == 200
        assert deep_link.headers["content-type"].startswith("text/html")


def test_machine_lines_stay_in_the_log_file_but_out_of_the_log_pane():
    # The markers are the record of what a worker reported, so the file keeps
    # them; the pane shows the human output and the parsed progress instead.
    assert jobs.is_machine_line('@@PROGRESS@@ {"percent": 12}')
    assert jobs.is_machine_line('  @@RESULT@@ {"path": "/out"}')
    assert not jobs.is_machine_line("Loading safetensors checkpoint shards: 50%")
    assert not jobs.is_machine_line("")


def test_a_parser_can_hand_an_artefact_to_the_job_row():
    progress = {"percent": 100.0, jobs.RESULT_KEY: {"path": "/outputs/model"}}
    lifted = progress.pop(jobs.RESULT_KEY)
    assert lifted == {"path": "/outputs/model"}
    assert jobs.RESULT_KEY not in progress


def test_the_spa_catch_all_cannot_read_outside_web():
    # Joining a request path onto a directory is an arbitrary file read unless
    # the result is checked: '../../etc/passwd' resolves cleanly and is_file()
    # is perfectly happy to confirm it.
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as client:
        for escape in (
            "/../../../../etc/passwd",
            "/../../../../etc/hostname",
            "/../app/config.py",
            "/../../.ssh/authorized_keys",
        ):
            response = client.get(escape)
            assert response.status_code == 200
            assert "<title>LLM Dashboard" in response.text, f"{escape} escaped web/"

        served = client.get("/css/app.css")
        assert served.status_code == 200 and served.text.lstrip().startswith("/*")


def test_the_playground_may_reach_private_networks_and_nothing_else():
    """This guard exists because the dashboard will proxy a chat request to a
    URL somebody types. It used to list one operator's own subnets, which both
    published their network layout in a public repo and silently blocked the
    feature on every other install, whose LAN is a different /24. Private
    address space is what "engines on my own network" actually means."""
    from app.routers.chat import _url_allowed

    for allowed in ("http://127.0.0.1:8010/v1", "http://localhost:8010",
                    "http://10.0.0.2:8010/v1", "http://192.168.1.50:8000/v1",
                    "http://172.16.4.4:8000"):
        assert _url_allowed(allowed), allowed

    for blocked in ("http://8.8.8.8:80", "https://api.openai.com/v1",
                    "ftp://10.0.0.1/x", "http://[2001:4860:4860::8888]:80",
                    "not-a-url"):
        assert not _url_allowed(blocked), blocked
