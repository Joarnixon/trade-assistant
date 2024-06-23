import asyncio
import time
from tinvest import AsyncClient
from Coefficients import Static


class GetChanges:

    async def get_market_orderbook(client, figi, depth):
        response = await client.get_market_orderbook(figi, depth)
        return response

    async def main(self):
        start_time = time.time()
        async with AsyncClient(Static.token) as client:
            tasks = [GetChanges.get_market_orderbook(client, figi, 1) for figi in Static.Stock]
            responses = await asyncio.gather(*tasks)
        for i, response in enumerate(responses):
            orderbook = response.payload
            Static.changes[Static.Stock_name[i]] = round(
                (orderbook.last_price - orderbook.close_price) / orderbook.close_price * 100, 2)

        sorted_changes = sorted(Static.changes.items(), key=lambda x: x[1], reverse=True)
        top_gainers = "\n".join([f"Изменение цены акции {stock} за сегодня {change} %" for stock, change in sorted_changes[:5]])
        top_losers = "\n".join([f"Изменение цены акции {stock} за сегодня {change} %" for stock, change in sorted_changes[-5:][::-1]])
        stop_time = time.time() - start_time

        return top_gainers, top_losers, stop_time


