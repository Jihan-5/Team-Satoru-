"""Round 1 trader – ASH_COATED_OSMIUM + INTARIAN_PEPPER_ROOT.
TRIAL: minor_changes_1

Changes vs round1_trader_FINAL.py:
  ASH:   reversion_coef 0.40 → 0.60  (matches lag-1 autocorr ≈ -0.499 more exactly)
  PEPPER: fair_mode 'drift_plus_reversion' → 'drift_only'  (EMA/residual dropped)
          take_width -8 → -10  (grabs any ask, faster fill to pos=80)
          prevent_adverse True → False  (don't skip large asks)
          default_edge -6 → 3  (passive bid at fair-3)
          levels 4 → 1
          use_l2l1_signal True → False

Backtest result vs FINAL (3-day): -1,670 (worse — ASH hurt by higher reversion_coef).
Keeping for reference / further experimentation.
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


# ───────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────
ASH = 'ASH_COATED_OSMIUM'
PEPPER = 'INTARIAN_PEPPER_ROOT'

POS_LIMITS = {ASH: 80, PEPPER: 80}

# PEPPER drift: +1 price per 1000 timestamps = +0.1 per 100-ts tick
PEPPER_DRIFT_PER_TICK = 0.1

PARAMS = {
    ASH: {
        'fair_mode': 'ema_reversion',
        'ema_alpha': 0.08,
        'reversion_coef': 0.60,      # CHANGED: 0.40 → 0.60 (lag-1 autocorr ≈ -0.49)
        'imb_coef': 3.5,
        'l2l1_coef': 2.0,
        'take_width': 2,
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 15,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': 7,
        'soft_position_limit': 78,
        'levels': 2,
        'use_l2l1_signal': True,
        'buy_only': False,
    },
    PEPPER: {
        # Pure "buy to limit ASAP, never sell" — no fair-value math needed.
        # Fair = last_mid + drift. No reversion, no imbalance, no L2-L1.
        'fair_mode': 'drift_only',   # CHANGED: was 'drift_plus_reversion'
        'ema_alpha': 0.0,
        'reversion_coef': 0.0,       # CHANGED: removed (OU half-life 0.1 ticks, useless)
        'imb_coef': 0.0,
        'l2l1_coef': 0.0,
        'take_width': -10,           # CHANGED: -8 → -10 (covers all asks, faster fill)
        'clear_width': 1,
        'prevent_adverse': False,    # CHANGED: don't skip large asks
        'adverse_volume': 0,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': 3,           # CHANGED: -6 → 3 (passive bid at fair-3)
        'soft_position_limit': 80,
        'levels': 1,                 # CHANGED: 4 → 1
        'use_drift': True,
        'use_l2l1_signal': False,    # CHANGED: removed
        'buy_only': True,
    },
}


# ───────────────────────────────────────────────────────────────────────
# Logger (optional — only active when run through prosperity3bt visualizer)
# ───────────────────────────────────────────────────────────────────────
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


# ───────────────────────────────────────────────────────────────────────
# Trader
# ───────────────────────────────────────────────────────────────────────
class Trader:
    def __init__(self, params=None):
        self.params = params if params is not None else PARAMS

    def take_best_orders(
        self, product, fair_value, take_width, orders, order_depth,
        position, buy_order_volume, sell_order_volume,
        prevent_adverse=False, adverse_volume=0, buy_only=False,
    ):
        limit = POS_LIMITS[product]

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amt = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or best_ask_amt <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    qty = min(best_ask_amt, limit - (position + buy_order_volume))
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        buy_order_volume += qty
                        order_depth.sell_orders[best_ask] += qty
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if not buy_only and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amt = order_depth.buy_orders[best_bid]
            if not prevent_adverse or best_bid_amt <= adverse_volume:
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
                v for p, v in order_depth.buy_orders.items() if p >= fair_for_ask
            )
            clear_qty = min(clear_qty, position_after_take)
            sent = min(sell_qty_cap, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent)))
                sell_order_volume += abs(sent)

        if position_after_take < 0:
            clear_qty = sum(
                abs(v) for p, v in order_depth.sell_orders.items() if p <= fair_for_bid
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
        bid_levels=1, ask_levels=1,
    ):
        orders = []
        limit = POS_LIMITS[product]

        asks_above = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge]
        bids_below = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge]

        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None

        ask = round(fair_value + default_edge)
        if best_ask_above is not None:
            ask = best_ask_above if abs(best_ask_above - fair_value) <= join_edge else best_ask_above - 1

        bid = round(fair_value - default_edge)
        if best_bid_below is not None:
            bid = best_bid_below if abs(fair_value - best_bid_below) <= join_edge else best_bid_below + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
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

    def compute_fair(self, product, order_depth, trader_obj):
        p = self.params[product]
        mode = p.get('fair_mode', 'ema_reversion')
        last_fair_key = f'{product}_last_fair'

        if not order_depth.sell_orders or not order_depth.buy_orders:
            last_fair = trader_obj.get(last_fair_key)
            if last_fair is None:
                return None
            if p.get('use_drift', False):
                return last_fair + PEPPER_DRIFT_PER_TICK
            return last_fair

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        l1_mid = (best_bid + best_ask) / 2

        # PEPPER fast path: just mid + drift, no reversion math
        if mode == 'drift_only':
            fair = l1_mid + PEPPER_DRIFT_PER_TICK
            trader_obj[last_fair_key] = fair
            return fair

        # ASH path: EMA reversion + imb + L2-L1
        ema_key = f'{product}_ema'
        alpha = p.get('ema_alpha', 0.10)
        prev_ema = trader_obj.get(ema_key)
        if prev_ema is None or abs(l1_mid - prev_ema) > 50:
            ema = l1_mid
        else:
            ema = alpha * l1_mid + (1 - alpha) * prev_ema
        trader_obj[ema_key] = ema

        residual = l1_mid - ema
        bid_vol = abs(order_depth.buy_orders.get(best_bid, 0))
        ask_vol = abs(order_depth.sell_orders.get(best_ask, 0))
        total = bid_vol + ask_vol
        imb = (bid_vol - ask_vol) / total if total > 0 else 0

        fair = l1_mid
        fair -= p.get('reversion_coef', 0.50) * residual
        fair += p.get('imb_coef', 0.0) * imb

        l2l1_coef = p.get('l2l1_coef', 0.0)
        if l2l1_coef != 0.0:
            try:
                sb = sorted(order_depth.buy_orders.items(), reverse=True)
                sa = sorted(order_depth.sell_orders.items())
                if len(sb) >= 2 and len(sa) >= 2:
                    l2m = (sb[1][0] + sa[1][0]) / 2
                    fair += l2l1_coef * (l2m - l1_mid)
            except Exception:
                pass

        trader_obj[last_fair_key] = fair
        return fair

    def _l2l1_signal(self, order_depth):
        try:
            sb = sorted(order_depth.buy_orders.items(), reverse=True)
            sa = sorted(order_depth.sell_orders.items())
            if len(sb) >= 2 and len(sa) >= 2:
                return (sb[1][0] + sa[1][0]) / 2 - (sb[0][0] + sa[0][0]) / 2
        except Exception:
            pass
        return 0

    def run(self, state: TradingState):
        trader_obj = {}
        if state.traderData:
            try:
                trader_obj = json.loads(state.traderData)
            except Exception:
                pass

        result = {}

        for product in [ASH, PEPPER]:
            if product not in state.order_depths:
                continue
            p = self.params[product]
            pos = state.position.get(product, 0)
            od = state.order_depths[product]

            fair = self.compute_fair(product, od, trader_obj)
            if fair is None:
                continue

            orders: List[Order] = []
            bov = sov = 0
            buy_only = p.get('buy_only', False)

            # Phase 1: Take
            bov, sov = self.take_best_orders(
                product, fair, p['take_width'], orders, od, pos, bov, sov,
                prevent_adverse=p.get('prevent_adverse', False),
                adverse_volume=p.get('adverse_volume', 0),
                buy_only=buy_only,
            )

            # Phase 2: Clear (skip for buy_only — never unwind PEPPER longs)
            if not buy_only:
                bov, sov = self.clear_position_order(
                    product, fair, p['clear_width'], orders, od, pos, bov, sov,
                )

            # L2-L1 signal for asymmetric level selection
            base_lvl = p.get('levels', 2)
            bid_lvl = ask_lvl = base_lvl
            if p.get('use_l2l1_signal', False):
                sig = self._l2l1_signal(od)
                if sig > 0:
                    bid_lvl = base_lvl + 1
                    ask_lvl = max(1, base_lvl - 1)
                elif sig < 0:
                    bid_lvl = max(1, base_lvl - 1)
                    ask_lvl = base_lvl + 1

            if buy_only:
                ask_lvl = 0

            # ASH wide-spread regime filter
            effective_edge = p['default_edge']
            if product == ASH and od.buy_orders and od.sell_orders:
                if min(od.sell_orders) - max(od.buy_orders) <= 12:
                    effective_edge = max(effective_edge, 5)

            # Phase 3: Make
            make_orders, _, _ = self.make_orders(
                product, od, fair, pos, bov, sov,
                p['disregard_edge'], p['join_edge'], effective_edge,
                manage_position=True,
                soft_position_limit=p['soft_position_limit'],
                bid_levels=bid_lvl,
                ask_levels=ask_lvl,
            )
            result[product] = orders + make_orders

        try:
            trader_data = json.dumps(trader_obj)
        except Exception:
            trader_data = ''

        logger.flush(state, result)
        return result, 0, trader_data
