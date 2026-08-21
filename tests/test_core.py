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
