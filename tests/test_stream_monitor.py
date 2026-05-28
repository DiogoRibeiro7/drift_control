import asyncio
import pandas as pd
import pytest
from drift_control.stream_monitor import StreamMonitor


def test_stream_monitor_async():
    baseline = pd.DataFrame({'x': [0, 1, 2]})
    current_batches = [pd.DataFrame({'x': [3, 4, 5]})]

    async def data_stream():
        for batch in current_batches:
            yield batch

    monitor = StreamMonitor()
    monitor.set_baseline(baseline)
    results = []

    async def run():
        async for res in monitor.monitor(data_stream()):
            results.append(res)

    asyncio.run(run())

    assert results[0]['x']['drift']


def test_stream_monitor_calls_on_drift_sync_callback():
    baseline = pd.DataFrame({'x': [0, 1, 2]})
    current_batches = [pd.DataFrame({'x': [3, 4, 5]})]
    called: list[dict] = []

    async def data_stream():
        for batch in current_batches:
            yield batch

    def on_drift(result):
        called.append(result)

    monitor = StreamMonitor(on_drift=on_drift)
    monitor.set_baseline(baseline)

    async def run():
        async for _ in monitor.monitor(data_stream()):
            pass

    asyncio.run(run())
    assert len(called) == 1
    assert called[0]['x']['drift'] is True


def test_stream_monitor_calls_on_drift_async_callback():
    baseline = pd.DataFrame({'x': [0, 1, 2]})
    current_batches = [pd.DataFrame({'x': [3, 4, 5]})]
    calls = {'n': 0}

    async def data_stream():
        for batch in current_batches:
            yield batch

    async def on_drift(_result):
        calls['n'] += 1

    monitor = StreamMonitor(on_drift=on_drift)
    monitor.set_baseline(baseline)

    async def run():
        async for _ in monitor.monitor(data_stream()):
            pass

    asyncio.run(run())
    assert calls['n'] == 1


def test_stream_monitor_schema_strict_rejects_extra_columns():
    baseline = pd.DataFrame({'x': [0, 1, 2]})
    batch = pd.DataFrame({'x': [3, 4, 5], 'y': [9, 9, 9]})
    monitor = StreamMonitor(on_schema_change="strict")
    monitor.set_baseline(baseline)
    with pytest.raises(ValueError, match="unknown columns"):
        asyncio.run(monitor.compare(batch))


def test_stream_monitor_schema_ignore_allows_extra_columns():
    baseline = pd.DataFrame({'x': [0, 1, 2]})
    batch = pd.DataFrame({'x': [3, 4, 5], 'y': [9, 9, 9]})
    monitor = StreamMonitor(on_schema_change="ignore")
    monitor.set_baseline(baseline)
    result = asyncio.run(monitor.compare(batch))
    assert 'x' in result
    assert 'y' not in result


def test_stream_monitor_schema_drop_allows_extra_columns():
    baseline = pd.DataFrame({'x': [0, 1, 2]})
    batch = pd.DataFrame({'x': [3, 4, 5], 'y': [9, 9, 9]})
    monitor = StreamMonitor(on_schema_change="drop")
    monitor.set_baseline(baseline)
    result = asyncio.run(monitor.compare(batch))
    assert 'x' in result
    assert 'y' not in result


def test_stream_monitor_schema_mode_validation():
    with pytest.raises(ValueError, match="on_schema_change"):
        StreamMonitor(on_schema_change="bad")
