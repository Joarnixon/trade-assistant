import matplotlib.dates
import os
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal
import datetime
import numpy as np
from Coefficients import Static
fig, ax = plt.subplots()


def StockControl(figi):
    bids_data = pd.read_csv(os.path.join(os.getcwd(), 'StocksLog', figi, 'BidsLog.txt'), sep=' ',
                            names=['Value', 'Date', 'Time'], parse_dates={'Timestamp': ['Date', 'Time']})
    bids_data.set_index('Timestamp', inplace=True)
    for file_name in ['PricesLog.txt', 'WeightedBidLog.txt', 'WeightedAskLog.txt']:
        # Read the file
        with open(os.path.join(os.getcwd(), 'StocksLog', figi, file_name), 'r') as f:
            # Extract the time and value information from each dictionary in the file
            data = {}
            for line in f:

                for i in range(len(line.strip('{').strip('}').split('} {'))-2):
                    for j in range(len(line.strip('{').strip('}').split('} {')[i].split(', '))):
                        time, value = line.strip('{').strip('}').split('} {')[i].split(', ')[j].split(': ')
                        data[datetime.datetime.strptime(time.strip("'"), "%Y-%m-%d %H:%M:%S")] = Decimal(eval(value))
            # Plot the data
            ax.plot(data.keys(), data.values(), label=file_name[:-4])
            if file_name == 'PricesLog.txt':
                price_start = list(data.values())[0]
                print(data)
            if file_name == 'WeightedBidLog.txt':
                buy_start = Decimal(list(data.values())[0])
                min_buy = min(list(data.values()))
            if file_name == 'WeightedAskLog.txt':
                sell_start = list(data.values())[1]

    with open(os.path.join(os.getcwd(), 'StocksLog', figi, 'PricesLog.txt'), 'r') as f:
        # Extract the time and value information from each dictionary in the file
        data2 = {}
        for line in f:
            for i in range(len(line.strip('{').strip('}').split('} {'))-2):
                for j in range(len(line.strip('{').strip('}').split('} {')[i].split(', '))):
                    time, value = line.strip('{').strip('}').split('} {')[i].split(', ')[j].split(': ')
                    data2[datetime.datetime.strptime(time.strip("'"), "%Y-%m-%d %H:%M:%S")] = Decimal(eval(value) - price_start + buy_start)
        x = matplotlib.dates.date2num(list(data2.keys()))
        y = [float(val) for val in list(data2.values())]
        print(x)
        print(y)
        y_series = pd.Series(y)
        # Calculate rolling mean with a window of 5
        rolling_mean = y_series.rolling(window=1).mean()
        ax.plot(x, rolling_mean)
    ax.plot(bids_data.index, (int(min_buy)-int(min_buy)*0.995)*np.array(bids_data['Value'])/(2*max(bids_data['Value'])) + (int(min_buy)+int(min_buy)*0.995)/2, label='Bids value')
    ax.plot(bids_data.index, [min((int(min_buy)-int(min_buy)*0.995)*np.array(bids_data['Value'])/(2*max(bids_data['Value'])) + (int(min_buy)+int(min_buy)*0.995)/2)]*len(bids_data.index))
    ax.set_title(f'{Static.Stock_dict[figi]}')

StockControl('BBG004S68CV8')
# Add legend and axis labels

# Define a function to handle mouse motion events
ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Value')
# Show the plot
plt.grid()
plt.show()