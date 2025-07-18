import asyncio
import pandas as pd
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
