from truedata_ws.websocket.TD import TD
from copy import deepcopy
import time
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import td_username, td_password
print(td_username, td_password)
td_obj = TD(td_username, td_password)
symbols = ['NSE:TATAMOTORS25SEP670CE']
symbol_ids = td_obj.get_req_id_list(2000, len(symbols))
is_market_open = True

data = {}
live_data_objs = {}

for symbol_id, symbol in zip(symbol_ids, symbols):
    print(f"Requesting historical data for {symbol}")
    symbol_hist_data = td_obj.get_historic_data(symbol, duration='1 D', bar_size='1 min')
    data[symbol_id] = list(map(lambda x: x['c'], symbol_hist_data))[-100:]

td_obj.start_live_data(symbols, req_id=symbol_ids)

time.sleep(1)
for symbol_id in symbol_ids:
    # If you are subscribed ONLY to min data, just USE live_data NOT min_live_data
    live_data_objs[symbol_id] = deepcopy(td_obj.one_min_live_data[symbol_id])

while is_market_open:
    time.sleep(0.1)  # Adding this reduces CPU overthrottle
    for symbol_id in symbol_ids:
        # If you are subscribed ONLY to min data, just USE live_data NOT min_live_data
        if live_data_objs[symbol_id] != td_obj.one_min_live_data[symbol_id]:
            live_data_objs[symbol_id] = deepcopy(td_obj.one_min_live_data[symbol_id])
            # Here is where you can do your manipulation on the min data
            data[symbol_id] = data[symbol_id][1:] + [live_data_objs[symbol_id].close]
            if sum(data[symbol_id][-50:]) / 50 > sum(data[symbol_id]) / 100:
                print(f'{live_data_objs[symbol_id].symbol} - SEND OMS LONG SIGNAL '
                      f'{sum(data[symbol_id][-50:]) / 50:.2f} > {sum(data[symbol_id]) / 100:.2f}, '
                      f'ltp = {live_data_objs[symbol_id].close:.2f}')
            else:
                print(f'{live_data_objs[symbol_id].symbol} - SEND OMS SHORT SIGNAL '
                      f'{sum(data[symbol_id][-50:]) / 50:.2f} < {sum(data[symbol_id]) / 100:.2f}, '
                      f'ltp = {live_data_objs[symbol_id].close:.2f}')

# Graceful exit
td_obj.stop_live_data(symbols)
td_obj.disconnect()