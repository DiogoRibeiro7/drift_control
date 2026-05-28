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


def test_stream_monitor_fixed_baseline_does_not_change():
    baseline = pd.DataFrame({'x': [0, 1, 2]})
    batch = pd.DataFrame({'x': [10, 11, 12]})

    async def data_stream():
        yield batch

    monitor = StreamMonitor(baseline_strategy="fixed")
    monitor.set_baseline(baseline)

    async def run():
        async for _ in monitor.monitor(data_stream()):
            pass

    asyncio.run(run())
    pd.testing.assert_frame_equal(monitor.baseline.reset_index(drop=True), baseline)


def test_stream_monitor_sliding_baseline_uses_last_n_batches():
    baseline = pd.DataFrame({'x': [0, 1, 2]})
    batch1 = pd.DataFrame({'x': [10, 11]})
    batch2 = pd.DataFrame({'x': [20, 21]})
    batch3 = pd.DataFrame({'x': [30, 31]})

    async def data_stream():
        yield batch1
        yield batch2
        yield batch3

    monitor = StreamMonitor(baseline_strategy="sliding", sliding_window_batches=2)
    monitor.set_baseline(baseline)

    async def run():
        async for _ in monitor.monitor(data_stream()):
            pass

    asyncio.run(run())
    expected = pd.concat([batch2, batch3], ignore_index=True)
    pd.testing.assert_frame_equal(
        monitor.baseline.reset_index(drop=True),
        expected.reset_index(drop=True),
    )


def test_stream_monitor_ewma_baseline_blends_and_keeps_size():
    baseline = pd.DataFrame({'x': [0, 1, 2, 3, 4]})
    batch = pd.DataFrame({'x': [100, 101, 102, 103, 104]})

    async def data_stream():
        yield batch

    monitor = StreamMonitor(
        baseline_strategy="ewma",
        ewma_alpha=0.4,
        random_state=7,
    )
    monitor.set_baseline(baseline)

    async def run():
        async for _ in monitor.monitor(data_stream()):
            pass

    asyncio.run(run())
    assert monitor.baseline is not None
    assert len(monitor.baseline) == len(baseline)
    assert monitor.baseline['x'].max() >= 100
    assert monitor.baseline['x'].min() <= 4


def test_stream_monitor_baseline_strategy_validation():
    with pytest.raises(ValueError, match="baseline_strategy"):
        StreamMonitor(baseline_strategy="bad")
