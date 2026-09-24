import asyncio

import pandas as pd

from drift_control import StreamMonitor

baseline = pd.DataFrame({"x": range(5)})
current_batches = [pd.DataFrame({"x": range(5, 10)})]


async def data_stream():
    for batch in current_batches:
        yield batch


async def main():
    monitor = StreamMonitor()
    monitor.set_baseline(baseline)
    async for res in monitor.monitor(data_stream()):
        print(res)


if __name__ == "__main__":
    asyncio.run(main())
