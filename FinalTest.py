import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.dates

from tinkoff.invest.utils import now

print(now() - timedelta(minutes = 1))

Data4 = np.linspace(matplotlib.dates.date2num(now() - timedelta(minutes = 1)), matplotlib.dates.date2num(now()), 60)
print(matplotlib.dates.num2date(Data4))
new_data = matplotlib.dates.num2date(Data4) + np.array(timedelta(seconds=10))
print(new_data)
print(np.size(new_data))
print(np.size(Data4))
