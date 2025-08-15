def estimate_futures_costs(price, lot_size=1, is_sell=False, is_buy=False):
    """
    Estimate total trading costs for a single buy or sell of a futures contract.
    Set is_sell=True for sell leg (STT applies), is_buy=True for buy leg (stamp duty applies).
    """
    brokerage = 20
    stt = 0.000125 * price * lot_size if is_sell else 0  # Only on sell leg
    exch = 0.00019 * price * lot_size
    sebi = 0.000001 * price * lot_size
    stamp = 0.00002 * price * lot_size if is_buy else 0  # Only on buy leg
    gst = 0.18 * brokerage
    total = brokerage + stt + exch + sebi + stamp + gst
    return total

def estimate_options_costs(premium, lot_size=1, is_sell=False, is_buy=False):
    brokerage = 20
    stt = 0.000625 * premium * lot_size if is_sell else 0  # Only on sell leg, on premium
    exch = 0.0005 * premium * lot_size
    sebi = 0.000001 * premium * lot_size
    stamp = 0.00003 * premium * lot_size if is_buy else 0  # Only on buy leg
    gst = 0.18 * brokerage
    total = brokerage + stt + exch + sebi + stamp + gst
    return total

def calculate_total_trade_cost(trades):
    """
    trades: list of dicts, each with keys:
        - 'type': 'futures' or 'options'
        - 'price': price or premium
        - 'lot_size': lot size
        - 'is_buy': True/False
        - 'is_sell': True/False
    """
    total_cost = 0
    for trade in trades:
        if trade['type'] == 'futures':
            total_cost += estimate_futures_costs(
                trade['price'],
                trade['lot_size'],
                is_sell=trade.get('is_sell', False),
                is_buy=trade.get('is_buy', False)
            )
        elif trade['type'] == 'options':
            total_cost += estimate_options_costs(
                trade['price'],
                trade['lot_size'],
                is_sell=trade.get('is_sell', False),
                is_buy=trade.get('is_buy', False)
            )
    return total_cost 