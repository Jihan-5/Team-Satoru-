"""Round 0 — three-phase trader with tuned params.

Architecture from Linear Utility (2nd place IMC Prosperity 2 round 1),
adapted for Prosperity 4 Round 0 EMERALDS/TOMATOES.

TUNED PARAMETERS (calibrated backtester, 2-day total):
  Baseline LU defaults:  3,875
  + EMERALDS clear_width=1:  +247 → 4,122
  + TOMATOES disregard_edge=2:  +71 → 4,193
  + TOMATOES reversion_beta=0:  +86 → 4,279
  Final: 4,279 backtested (vs wall_trader 3,918 = +361, +9%)

Projected live (based on wall_trader's 3,918 → 1,664 real calibration factor):
  ~1,800 on Day -1 (vs wall_trader 1,664)
  Expected improvement: ~+130 SeaShells

KEY DISCOVERY: reversion_beta=0 — TOMATOES is NOT mean-reverting on
Prosperity 4 data. Linear Utility's -0.229 was fit on Prosperity 2 STARFRUIT
and doesn't transfer. Using raw MM-mid (filtered by ≥15 volume) as fair
with zero reversion adjustment is strictly better here.

THREE PHASES:
1. TAKE — sweep orders crossed with fair ± take_width
2. CLEAR — dump position at zero-EV against resting orders (EMERALDS only,
           clear_width=1. Helps EMERALDS because FV is stable; hurts TOMATOES
           because fair drifts.)
3. MAKE — penny or join best order beyond fair ± disregard_edge, then shift
          quotes inward if position exceeds soft_position_limit
"""
from datamodel import OrderDepth, TradingState, Order
try:
    from datamodel import ProsperityEncoder
except ImportError:
    import json as _j
    class ProsperityEncoder(_j.JSONEncoder):
        def default(self, o):
            try: return o.__dict__
            except AttributeError: return str(o)

import json
from typing import Any, List


# ═════════════════════════════════════════════════════════════════════════
# Config — TUNED PARAMS
# ═════════════════════════════════════════════════════════════════════════
EMERALD = 'EMERALDS'
TOMATO = 'TOMATOES'

POS_LIMITS = {EMERALD: 80, TOMATO: 80}

PARAMS = {
    EMERALD: {
        'fair_value': 10000,
        'take_width': 1,
        'clear_width': 1,          # TUNED: +247 SeaShells
        'disregard_edge': 1,
        'join_edge': 2,
        'default_edge': 4,
        'soft_position_limit': 40,
    },
    TOMATO: {
        'take_width': 1,
        'clear_width': 0,           # Keep off — clearing hurts TOMATOES
        'prevent_adverse': True,
        'adverse_volume': 15,
        'reversion_beta': 0.0,      # TUNED: +86. No mean reversion on P4.
        'disregard_edge': 2,        # TUNED: +71. Widen ref-price filter.
        'join_edge': 0,
        'default_edge': 1,
        'soft_position_limit': 40,
    },
}

USE_MM_MID_FOR_TOMATO = True  # +1,240 TOMATOES discovery — must stay on


# ═════════════════════════════════════════════════════════════════════════
# Logger (visualizer-compliant)
# ═════════════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self) -> None:
        self.logs: str = ""
        self.max_log_length: int = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict) -> None:
        payload = {
            "state": self._compress_state(state),
            "orders": self._compress_orders(orders),
            "logs": self.logs[: self.max_log_length],
        }
        try:
            print(json.dumps(payload, cls=ProsperityEncoder,
                              separators=(",", ":"), sort_keys=True))
        except Exception:
            print(str(payload)[: self.max_log_length])
        self.logs = ""

    def _compress_state(self, state: TradingState) -> dict:
        listings = []
        for listing in getattr(state, "listings", {}).values():
            try:
                listings.append([listing.symbol, listing.product, listing.denomination])
            except AttributeError:
                pass
        order_depths = {}
        for symbol, od in state.order_depths.items():
            order_depths[symbol] = [od.buy_orders, od.sell_orders]
        return {
            "t": state.timestamp, "l": listings, "od": order_depths,
            "ot": self._compress_trades(getattr(state, "own_trades", {})),
            "mt": self._compress_trades(getattr(state, "market_trades", {})),
            "p": state.position, "o": state.observations,
        }

    def _compress_trades(self, trades: dict) -> list:
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([
                    getattr(t, "symbol", ""),
                    getattr(t, "buyer", ""),
                    getattr(t, "seller", ""),
                    getattr(t, "price", 0),
                    getattr(t, "quantity", 0),
                    getattr(t, "timestamp", 0),
                ])
        return out

    def _compress_orders(self, orders: dict) -> list:
        out = []
        for arr in orders.values():
            for o in arr:
                out.append([o.symbol, o.price, o.quantity])
        return out


logger = Logger()


# ═════════════════════════════════════════════════════════════════════════
# Trader
# ═════════════════════════════════════════════════════════════════════════
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

    def take_best_orders(
        self, product, fair_value, take_width, orders, order_depth,
        position, buy_order_volume, sell_order_volume,
        prevent_adverse=False, adverse_volume=0,
    ):
        limit = POS_LIMITS[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amt = -1 * order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(best_ask_amt) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    qty = min(best_ask_amt, limit - (position + buy_order_volume))
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        buy_order_volume += qty
                        order_depth.sell_orders[best_ask] += qty
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amt = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(best_bid_amt) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    qty = min(best_bid_amt, limit + (position - sell_order_volume))
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        sell_order_volume += qty
                        order_depth.buy_orders[best_bid] -= qty
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self, product, fair_value, width, orders, order_depth,
        position, buy_order_volume, sell_order_volume,
    ):
        limit = POS_LIMITS[product]
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_qty_cap = limit - (position + buy_order_volume)
        sell_qty_cap = limit + (position - sell_order_volume)

        if position_after_take > 0:
            clear_qty = sum(
                volume for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_qty = min(clear_qty, position_after_take)
            sent = min(sell_qty_cap, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent)))
                sell_order_volume += abs(sent)

        if position_after_take < 0:
            clear_qty = sum(
                abs(volume) for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_qty = min(clear_qty, abs(position_after_take))
            sent = min(buy_qty_cap, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_for_bid, abs(sent)))
                buy_order_volume += abs(sent)

        return buy_order_volume, sell_order_volume

    def make_orders(
        self, product, order_depth, fair_value, position,
        buy_order_volume, sell_order_volume,
        disregard_edge, join_edge, default_edge,
        manage_position=False, soft_position_limit=0,
    ):
        orders = []
        limit = POS_LIMITS[product]

        asks_above_fair = [p for p in order_depth.sell_orders.keys()
                           if p > fair_value + disregard_edge]
        bids_below_fair = [p for p in order_depth.buy_orders.keys()
                           if p < fair_value - disregard_edge]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        ask = round(fair_value + default_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - default_edge)
        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair
            else:
                bid = best_bid_below_fair + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_qty = limit - (position + buy_order_volume)
        if buy_qty > 0:
            orders.append(Order(product, round(bid), buy_qty))
        sell_qty = limit + (position - sell_order_volume)
        if sell_qty > 0:
            orders.append(Order(product, round(ask), -sell_qty))

        return orders, buy_order_volume, sell_order_volume

    def tomato_fair_value(self, order_depth, trader_obj):
        """MM-mid filtered by adverse_volume threshold."""
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None

        p = self.params[TOMATO]
        if USE_MM_MID_FOR_TOMATO:
            filtered_ask = [price for price, vol in order_depth.sell_orders.items()
                            if abs(vol) >= p['adverse_volume']]
            filtered_bid = [price for price, vol in order_depth.buy_orders.items()
                            if abs(vol) >= p['adverse_volume']]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                last = trader_obj.get('tomato_last_price')
                if last is None:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    mm_mid = (best_ask + best_bid) / 2
                else:
                    mm_mid = last
            else:
                mm_mid = (mm_ask + mm_bid) / 2
        else:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mm_mid = (best_ask + best_bid) / 2

        last = trader_obj.get('tomato_last_price')
        if last is not None and p['reversion_beta'] != 0:
            ret = (mm_mid - last) / last
            fair = mm_mid + mm_mid * (ret * p['reversion_beta'])
        else:
            fair = mm_mid
        trader_obj['tomato_last_price'] = mm_mid
        return fair

    def run(self, state: TradingState):
        trader_obj = {}
        if state.traderData:
            try:
                trader_obj = json.loads(state.traderData)
            except Exception:
                trader_obj = {}

        result = {}

        # ── EMERALDS ─────────────────────────────────────────────────────
        if EMERALD in state.order_depths:
            p = self.params[EMERALD]
            pos = state.position.get(EMERALD, 0)
            od = state.order_depths[EMERALD]
            orders: List[Order] = []
            bov = sov = 0

            bov, sov = self.take_best_orders(
                EMERALD, p['fair_value'], p['take_width'],
                orders, od, pos, bov, sov,
            )
            bov, sov = self.clear_position_order(
                EMERALD, p['fair_value'], p['clear_width'],
                orders, od, pos, bov, sov,
            )
            make_orders, _, _ = self.make_orders(
                EMERALD, od, p['fair_value'], pos, bov, sov,
                p['disregard_edge'], p['join_edge'], p['default_edge'],
                manage_position=True,
                soft_position_limit=p['soft_position_limit'],
            )
            result[EMERALD] = orders + make_orders

        # ── TOMATOES ─────────────────────────────────────────────────────
        if TOMATO in state.order_depths:
            p = self.params[TOMATO]
            pos = state.position.get(TOMATO, 0)
            od = state.order_depths[TOMATO]
            fair = self.tomato_fair_value(od, trader_obj)
            if fair is not None:
                orders = []
                bov = sov = 0

                bov, sov = self.take_best_orders(
                    TOMATO, fair, p['take_width'],
                    orders, od, pos, bov, sov,
                    prevent_adverse=p['prevent_adverse'],
                    adverse_volume=p['adverse_volume'],
                )
                bov, sov = self.clear_position_order(
                    TOMATO, fair, p['clear_width'],
                    orders, od, pos, bov, sov,
                )
                make_orders, _, _ = self.make_orders(
                    TOMATO, od, fair, pos, bov, sov,
                    p['disregard_edge'], p['join_edge'], p['default_edge'],
                    manage_position=True,
                    soft_position_limit=p['soft_position_limit'],
                )
                result[TOMATO] = orders + make_orders

        try:
            trader_data = json.dumps(trader_obj)
        except Exception:
            trader_data = ''

        try:
            logger.flush(state, result)
        except Exception:
            pass
        return result, 0, trader_data
