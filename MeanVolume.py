import asyncio
from tinkoff.invest import CandleInterval, AsyncClient
from tinkoff.invest.utils import now
from datetime import datetime, timedelta
from Coefficients import Static
import sys


class VolumeOperations:

    async def get_candle_data(self, client, figi):
        volume_week = 0
        candles_amount = 0
        async for candle in client.get_all_candles(
                figi=figi,
                from_=now() - timedelta(days=7),
                interval=CandleInterval.CANDLE_INTERVAL_30_MIN
        ):

            candles_amount += 1
            volume_week += candle.volume
            await asyncio.sleep(0.6)
        return volume_week / candles_amount

    async def main(self):
        async with AsyncClient(Static.token) as client:
            volume_data = {}
            for i in range(len(Static.Stock)):
                volume_data[Static.Stock[i]] = await VolumeOperations.get_candle_data(self, client, Static.Stock[i])
                sys.stdout.write(('=' * (i - 1) + ('' * (len(Static.Stock) - i + 1)) + ("\r [ %d" % (i - 1) + "% ] ")))
                sys.stdout.flush()
                await asyncio.sleep(0.1)
            return volume_data

