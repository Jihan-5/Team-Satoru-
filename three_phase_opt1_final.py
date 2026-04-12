"""Round 0 three-phase trader — FINAL, with multi-level quoting + L2-L1 signal.

Architecture from Linear Utility (2nd place IMC Prosperity 2 round 1),
adapted for Prosperity 4 Round 0 EMERALDS/TOMATOES. Extended with multi-level
passive quoting and a threshold-free L2-L1 sign signal for TOMATOES.

THREE PHASES PER PRODUCT:
1. TAKE  — sweep mispriced orders at fair ± take_width
2. CLEAR — unload position at zero-EV or better against existing orders
3. MAKE  — penny/join the best order beyond fair, adaptively

MULTI-LEVEL EXTENSION:
  EMERALDS posts at 2 levels (bid, bid−1 / ask, ask+1). Captures the
  additional fills when the book shifts through our aggressive layer.
  TOMATOES posts at 3 base levels, adjusted by L2-L1 sign signal:
    - When (L2 mid - L1 mid) > 0: 4 bid levels / 2 ask levels (catch dip)
    - When < 0: 2 bid / 4 ask (catch rise)
    - When 0: 3/3 symmetric
  The signal is threshold-free (pure sign), not tuned.

TUNING PATH (all verified in calibrated backtester, 2-day totals):
  LU defaults (single level):           3,875
  + EMERALDS clear_width=1:             4,122  (+247)
  + TOMATOES disregard_edge=1:          4,193  (+71)
  + TOMATOES reversion_beta=0:          4,279  (+86)
  + TOMATOES disregard_edge=1 kept:     4,304  (+25)
  + EMERALDS 2-level quoting:           4,868  (+564)
  + TOMATOES 2-level quoting:           5,334  (+466)
  + L2-L1 signal (TM 3-level ±1):       5,374  (+40)

Backtest total: 5,374  (vs wall_trader 3,918 = +1,456, +37%)
Per day:
  Day -1:  2,758  (EM 1,363, TM 1,395)
  Day -2:  2,616  (EM 1,234, TM 1,382)

Expected live (using wall_trader's 3,918->1,664 real calibration ratio):
  ~2,300 Day -1  (vs wall_trader live 1,664 = +640 real SeaShells)

KEY DISCOVERIES (data-driven, not hardcoded):
  1. MM-mid (L2 mid via adverse_volume filter) is the platform's internal
     fair value. Using it for TOMATOES adds ~+1,240.
  2. TOMATOES is NOT mean-reverting on Prosperity 4. reversion_beta=0.
  3. 100% of trades print at L1 exactly, 0 spillover. Penny-improve or skip.
  4. Position limit is NOT the binding constraint — we're at |pos|<10 with
     80 limit. Real constraint is queue surface area.
  5. Multi-level quoting unlocks position capacity by catching fills at
     both the aggressive and deeper price levels.
  6. L2 mid - L1 mid has 0.6+ correlation with next-tick L1 move. Used as
     sign signal to skew quote level counts on TOMATOES.
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
# Config
# ═════════════════════════════════════════════════════════════════════════
EMERALD = 'EMERALDS'
TOMATO = 'TOMATOES'

POS_LIMITS = {EMERALD: 80, TOMATO: 80}

PARAMS = {
    EMERALD: {
        'fair_value': 10000,
        'take_width': 1,
        'clear_width': 1,
        'disregard_edge': 1,
        'join_edge': 2,
        'default_edge': 4,
        'soft_position_limit': 40,
        'levels': 2,
    },
    TOMATO: {
        'take_width': 1,
        'clear_width': 0,
        'prevent_adverse': True,
        'adverse_volume': 15,
        'reversion_beta': 0.0,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': 1,
        'soft_position_limit': 10,   # tight base so skew fires
        'levels': 3,
        'trend_ema_alpha': 0.1,
        'trend_skew_gain': 20,
    },
}

USE_MM_MID_FOR_TOMATO = True


# ═════════════════════════════════════════════════════════════════════════
# Logger
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
        manage_position=False,
        soft_limit_long=0, soft_limit_short=0,
        bid_levels=1, ask_levels=1,
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
            if position > soft_limit_long:
                ask -= 1
            elif position < -soft_limit_short:
                bid += 1

        buy_qty = limit - (position + buy_order_volume)
        if buy_qty > 0 and bid_levels > 0:
            base = buy_qty // bid_levels
            for lvl in range(bid_levels):
                q = base if lvl < bid_levels - 1 else buy_qty - base * (bid_levels - 1)
                if q > 0:
                    orders.append(Order(product, round(bid) - lvl, q))

        sell_qty = limit + (position - sell_order_volume)
        if sell_qty > 0 and ask_levels > 0:
            base = sell_qty // ask_levels
            for lvl in range(ask_levels):
                q = base if lvl < ask_levels - 1 else sell_qty - base * (ask_levels - 1)
                if q > 0:
                    orders.append(Order(product, round(ask) + lvl, -q))

        return orders, buy_order_volume, sell_order_volume

    def tomato_fair_value(self, order_depth, trader_obj):
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
        if last is not None:
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
            lvl = p.get('levels', 1)
            make_orders, _, _ = self.make_orders(
                EMERALD, od, p['fair_value'], pos, bov, sov,
                p['disregard_edge'], p['join_edge'], p['default_edge'],
                manage_position=True,
                soft_limit_long=p['soft_position_limit'],
                soft_limit_short=p['soft_position_limit'],
                bid_levels=lvl, ask_levels=lvl,
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

                base_lvl = p.get('levels', 2)
                bid_lvl = ask_lvl = base_lvl
                sig = 0.0
                try:
                    sb = sorted(od.buy_orders.items(), reverse=True)
                    sa = sorted(od.sell_orders.items())
                    if len(sb) >= 2 and len(sa) >= 2:
                        l1m = (sb[0][0] + sa[0][0]) / 2
                        l2m = (sb[1][0] + sa[1][0]) / 2
                        sig = l2m - l1m
                        if sig > 0:
                            bid_lvl = base_lvl + 1
                            ask_lvl = max(1, base_lvl - 1)
                        elif sig < 0:
                            bid_lvl = max(1, base_lvl - 1)
                            ask_lvl = base_lvl + 1
                except Exception:
                    pass

                spl_base = p['soft_position_limit']
                ema_alpha = p.get('trend_ema_alpha', 0.02)
                prev_trend = trader_obj.get('tomato_trend', 0.0)
                new_trend = (1 - ema_alpha) * prev_trend + ema_alpha * sig
                trader_obj['tomato_trend'] = new_trend

                skew = int(round(new_trend * p.get('trend_skew_gain', 20)))
                skew = max(-30, min(30, skew))
                soft_long = max(5, min(70, spl_base + skew))
                soft_short = max(5, min(70, spl_base - skew))

                make_orders, _, _ = self.make_orders(
                    TOMATO, od, fair, pos, bov, sov,
                    p['disregard_edge'], p['join_edge'], p['default_edge'],
                    manage_position=True,
                    soft_limit_long=soft_long,
                    soft_limit_short=soft_short,
                    bid_levels=bid_lvl, ask_levels=ask_lvl,
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
