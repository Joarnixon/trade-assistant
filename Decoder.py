from tinvest import SyncClient
from Coefficients import Static
import time
client = SyncClient(token=Static.token)



def GetName(figi):
    start = time.time()
    instr = client.get_market_search_by_figi(figi)
    print(start - time.time())
    return instr.payload.name
GetName('BBG000SR0YS4')

