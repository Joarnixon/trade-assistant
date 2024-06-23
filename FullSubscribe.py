import asyncio
from Coefficients import Static
from LogHandler import Handling
import time
from tinkoff.invest.utils import now, quotation_to_decimal as qtd
from tinkoff.invest import (
    AsyncClient,
    Client,
    MarketDataRequest,
    SubscribeOrderBookRequest,
    SubscribeTradesRequest,
    SubscriptionAction,
    OrderBookInstrument,
    TradeInstrument,
    TradeDirection
)

handle_op = Handling()


class TinkoffData:
    def __init__(self, figi):
        self.figi = figi
        self.bids = {}
        self.asks = {}
        self.buyers = [0, 0]
        self.weighted_bid = {}
        self.weighted_ask = {}
        self.sellers = [0, 0]
        self.timer = None
        self.price = 0
        self.last_price = None
        self.subscribe_time = None
        self.subscribed = False
        self.threshold = 300
        self.stock = Data()
        self.stop = False
        self.subscribe()

    def subscribe(self):
        if not self.subscribed:
            self.subscribed = True
            self.subscribe_time = time.time()
            self.timer = time.time()

    async def check_unsubscribe(self):
        if time.time() - self.subscribe_time > self.threshold:  # Check if 5 minutes have passed
            return False
        else:
            return True

    def log_trades_data(self, folder):
        timestamp = now().strftime("%Y-%m-%d %H:%M:%S")
        Handling.write_stock_log(handle_op, self.figi, 'BuyersLog', folder, f'{self.buyers} {timestamp}\n')
        Handling.write_stock_log(handle_op, self.figi, 'SellersLog', folder, f'{self.sellers} {timestamp}\n')
        Handling.write_stock_log(handle_op, self.figi, 'PriceLog', folder, f'{self.price} {timestamp}\n')
        Handling.write_stock_log(handle_op, self.figi, 'BidsLog', folder, f'{self.bids} ')
        Handling.write_stock_log(handle_op, self.figi, 'AsksLog', folder, f'{self.asks} ')
        Handling.write_stock_log(handle_op, self.figi, 'WeightedBidLog', folder, f'{self.weighted_bid} ')
        Handling.write_stock_log(handle_op, self.figi, 'WeightedAskLog', folder, f'{self.weighted_ask} ')
        Handling.write_stock_log(handle_op, self.figi, 'PricesLog', folder, f'{self.stock.prices} ')

    def reset_trades_data(self):
        self.stock.buyers_count += self.buyers[0]
        self.stock.sellers_count += self.sellers[0]
        self.stock.buy_count += self.buyers[1]
        self.stock.sell_count += self.sellers[1]
        self.buyers = [0, 0]
        self.sellers = [0, 0]
        self.bids = {}
        self.asks = {}
        self.weighted_ask = {}
        self.weighted_bid = {}
        self.stock.prices = {}
        self.timer = time.time()


async def trades(market_dict, folder):
    async def request_iterator():
        yield MarketDataRequest(
            subscribe_trades_request=SubscribeTradesRequest(
                subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                instruments=[TradeInstrument(figi=figi) for figi in Static.Stock])
        )
        yield MarketDataRequest(
            subscribe_order_book_request=SubscribeOrderBookRequest(
                subscription_action=SubscriptionAction.SUBSCRIPTION_ACTION_SUBSCRIBE,
                instruments=[OrderBookInstrument(figi=figi, depth=10) for figi in Static.Stock])
        )
        while True:
            await asyncio.sleep(5)

    async with AsyncClient(Static.token) as client:
        async for marketdata in client.market_data_stream.market_data_stream(
            request_iterator()
        ):
            if marketdata.trade is not None:
                task = market_dict[marketdata.trade.figi]
                task.last_price = qtd(marketdata.trade.price)
                if marketdata.trade.direction == TradeDirection.TRADE_DIRECTION_BUY:
                    task.buyers[0] += 1
                    task.buyers[1] += marketdata.trade.quantity

                if marketdata.trade.direction == TradeDirection.TRADE_DIRECTION_SELL:
                    task.sellers[0] += 1
                    task.sellers[1] += marketdata.trade.quantity

                if time.time() - task.timer > 35:
                    task.price = qtd(marketdata.trade.price)
                    task.log_trades_data(folder)
                    task.reset_trades_data()
            if marketdata.orderbook is not None:
                task = market_dict[marketdata.orderbook.figi]
                task.bids[now().strftime("%Y-%m-%d %H:%M:%S")] = sum([bid.quantity for bid in marketdata.orderbook.bids])
                task.asks[now().strftime("%Y-%m-%d %H:%M:%S")] = sum([ask.quantity for ask in marketdata.orderbook.asks])
                task.weighted_bid[now().strftime("%Y-%m-%d %H:%M:%S")] = round(sum([bid.quantity*qtd(bid.price) for bid in marketdata.orderbook.bids])/list(task.bids.values())[-1], 5)
                task.weighted_ask[now().strftime("%Y-%m-%d %H:%M:%S")] = round(sum([ask.quantity*qtd(ask.price) for ask in marketdata.orderbook.asks])/list(task.asks.values())[-1], 5)
                task.stock.prices[now().strftime("%Y-%m-%d %H:%M:%S")] = task.last_price


class Data:
    def __init__(self):
        self.buyers_count = 0
        self.sellers_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.prices = {}


async def sessions(tink_dict, folder):
    session1 = 0
    trend = {}
    while True:
        await asyncio.sleep(59)
        session1 += 1
        plus1q, minus1q, plus1, minus1 = 0, 0, 0, 0
        for i in range(len(list(tink_dict.values()))):
            plus1q += list(tink_dict.values())[i].stock.buy_count
            minus1q += list(tink_dict.values())[i].stock.sell_count
            plus1 += list(tink_dict.values())[i].stock.buyers_count
            minus1 += list(tink_dict.values())[i].stock.sellers_count
            list(tink_dict.values())[i].stock.buy_count, list(tink_dict.values())[i].stock.sell_count = 0, 0
            list(tink_dict.values())[i].stock.buyers_count, list(tink_dict.values())[i].stock.sellers_count = 0, 0

        timestamp = now().strftime("%Y-%m-%d %H:%M:%S")
        trend[timestamp] = [[plus1q, plus1], [minus1q, minus1]]
        Handling.write_trend_log(handle_op, 'TrendLog', folder, f'{trend} ')
        if plus1q > minus1q:
            print(f'Сессия {session1}, тренд РОСТ. Сила ПОКУПАТЕЛЕЙ БОЛЬШЕ в {plus1q/minus1q} раз')
            if plus1 < minus1:
                print(f'В тоже время, субьективное представление DISAGREE. Кол-во ПРОДАВЦОВ БОЛЬШЕ в {minus1/plus1} раз')
                print('===============================================================================================')
        else:
            print(f'Сессия {session1}, тренд ПАДЕНИЕ. Сила ПРОДАВЦОВ БОЛЬШЕ в {minus1q/plus1q} раз')

            if plus1 > minus1:
                print(f'В тоже время, субьективное представление DISAGREE. Кол-во ПОКУПАТЕЛЕЙ БОЛЬШЕ в {plus1/minus1} раз')
                print('===============================================================================================')
        await asyncio.sleep(1)

async def main(folder):
    tinkoff_data_dict = {figi: TinkoffData(figi) for figi in Static.Stock}
    with Client(Static.token) as client:
        for obj in list(tinkoff_data_dict.values()):
            obj.last_price = qtd(client.market_data.get_last_prices(figi=[obj.figi]).last_prices[0].price)
    sessions_task = asyncio.create_task(sessions(tinkoff_data_dict, folder=folder))
    trades_task = asyncio.create_task(trades(tinkoff_data_dict, folder=folder))
    await asyncio.gather(trades_task, sessions_task)
asyncio.run(main('StocksLog1'))



