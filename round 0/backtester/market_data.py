"""Load IMC Prosperity CSV logs into backtester-friendly structures.

Prices CSV columns (semicolon-delimited):
  day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;
  bid_price_3;bid_volume_3;ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;
  ask_price_3;ask_volume_3;mid_price;profit_and_loss

Trades CSV columns (semicolon-delimited):
  timestamp;buyer;seller;symbol;currency;price;quantity
"""
import csv
from collections import defaultdict

from datamodel import OrderDepth, Trade


def load_prices(path: str):
    """Return (snapshots, timestamps).

    snapshots: dict {(timestamp, symbol): OrderDepth}
    timestamps: sorted list of unique integer timestamps
    """
    snapshots = {}
    ts_set = set()

    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                ts = int(row['timestamp'])
                sym = row['product'].strip()
            except (KeyError, ValueError):
                continue

            od = OrderDepth()

            for level in (1, 2, 3):
                bp_key = f'bid_price_{level}'
                bv_key = f'bid_volume_{level}'
                ap_key = f'ask_price_{level}'
                av_key = f'ask_volume_{level}'

                bp = row.get(bp_key, '').strip()
                bv = row.get(bv_key, '').strip()
                if bp and bv:
                    try:
                        od.buy_orders[int(float(bp))] = int(float(bv))
                    except ValueError:
                        pass

                ap = row.get(ap_key, '').strip()
                av = row.get(av_key, '').strip()
                if ap and av:
                    try:
                        od.sell_orders[int(float(ap))] = -int(float(av))
                    except ValueError:
                        pass

            snapshots[(ts, sym)] = od
            ts_set.add(ts)

    timestamps = sorted(ts_set)
    return snapshots, timestamps


def load_trades(path: str):
    """Return dict {(timestamp, symbol): [Trade]}."""
    result = defaultdict(list)

    with open(path, newline='') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            try:
                ts = int(row['timestamp'])
                sym = row['symbol'].strip()
                price = float(row['price'])
                qty = int(row['quantity'])
            except (KeyError, ValueError):
                continue

            buyer = row.get('buyer', '') or ''
            seller = row.get('seller', '') or ''

            trade = Trade(
                symbol=sym,
                price=int(price),
                quantity=qty,
                buyer=buyer,
                seller=seller,
                timestamp=ts,
            )
            result[(ts, sym)].append(trade)

    return dict(result)


def get_symbols(snapshots: dict) -> list:
    """Return sorted list of unique symbols found in snapshots."""
    return sorted({sym for (_, sym) in snapshots.keys()})
