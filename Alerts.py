import time
from tinkoff.invest import Subscribe
from typing import List, Dict, Any
from Coefficients import Static
class StockSubscriber:
    def __init__(self, token: str):
        self.token = token
        self.streaming = None
        self.subscriptions = {}  # dictionary to store the subscription data for each stock

    def __enter__(self):
        self.streaming = Streaming(token=self.token)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.streaming:
            if hasattr(self.streaming, '_client'):
                self.streaming._client.close()

    def is_trending(self, figi: str) -> bool:
        # Implement your logic to check if the stock is trending
        # For example, you can check if the stock is in a list of trending stocks
        return figi in ['BBG004S68DD6', 'BBG004S687W8', 'BBG004S687G6']

    def handle_event(self, figi: str, event: Dict[str, Any]):
        if event['event'] == 'trade':
            # handle trade event
            print(f"New trade event for {figi}: {event}")
            # add the event data to the subscription data for this stock
            self.subscriptions[figi]['events'].append(event)

            # Check for price change
            last_price = self.subscriptions[figi]['last_price']
            current_price = event['price']
            price_change = abs((current_price - last_price) / last_price)

            if price_change > 0.005:  # 0.5% price change
                self.subscriptions[figi]['last_update'] = time.time()
            self.subscriptions[figi]['last_price'] = current_price

    def subscribe(self, figi: str):
        if figi in self.subscriptions:
            print('already subscribed to this stock')

        if not self.is_trending(figi):
            print('the stock is not trending')

        self.streaming.candle.subscribe(figi, '1min', figi, lambda event: self.handle_event(figi, event))
        # add the subscription data for this stock to the dictionary
        self.subscriptions[figi] = {'events': [], 'last_price': None, 'last_update': time.time()}

    def unsubscribe(self, figi: str):
        if figi not in self.subscriptions:
            return  # not subscribed to this stock

        self.streaming.unsubscribe(figi)

        # delete the subscription data for this stock from the dictionary
        del self.subscriptions[figi]

    def monitor_subscriptions(self):
        while True:
            for figi, data in self.subscriptions.items():
                time_since_last_update = time.time() - data['last_update']
                if time_since_last_update > 5 * 60:  # 5 minutes
                    self.unsubscribe(figi)
            time.sleep(60)  # Check every minute

with StockSubscriber(token=Static.token) as subscriber:
    # Subscribe to stocks that are trending
    for figi in ['BBG004S68DD6', 'BBG004S687W8', 'BBG004S687G6']:
        subscriber.subscribe(figi)

    # Monitor subscriptions and unsubscribe if no price change >0.5% in a minute happened after 5 minutes
    subscriber.monitor_subscriptions()