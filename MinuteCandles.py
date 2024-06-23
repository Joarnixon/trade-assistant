import asyncio
from LogHandler import Handling
from tinkoff.invest import AsyncClient
from tinkoff.invest import CandleInterval
from tinkoff.invest.utils import now
from tinkoff.invest.utils import quotation_to_decimal as qtd
from datetime import datetime, timedelta
from Coefficients import Static, Dynamic
TOKEN = Static.token
stocks = Static.Stock
handle_op = Handling()


async def get_candle_data(client, figi):
    async for candles in client.get_all_candles(
        figi=figi,
        from_=now() - timedelta(minutes=2),
        interval=CandleInterval.CANDLE_INTERVAL_1_MIN,
    ):
        candle = candles
        price_change = float(round(100 * (qtd(candle.close) - qtd(candle.open)) / qtd(candle.open), 3))
        buy_volume = candle.volume
        if abs(price_change) > 0.2 and buy_volume > Dynamic.MV[figi]:
            print(f'{Static.Stock_dict[figi]} : {price_change} %, volume bigger by: {buy_volume/Dynamic.MV[figi]} times!')
        return [price_change, buy_volume]

async def main():
    async with AsyncClient(TOKEN) as client:
        j = 0
        while j >= 0:
            stock_data = {}
            for i in range(len(stocks)):
                stock_data[Static.Stock[i]] = await get_candle_data(client, Static.Stock[i])
                if stock_data[Static.Stock[i]] is None:
                    stock_data[Static.Stock[i]] = [0, 0]
            filtered_dict = sorted(((k, v) for k, v in stock_data.items() if abs(v[0]) > 0.1 and v[1] > Dynamic.MV[k]),
                                   key=lambda x: abs(x[1][0]), reverse=True)
            for k, v in filtered_dict:
                print(f'Minute session{j} done')
            j += 1
            await asyncio.sleep(60)


