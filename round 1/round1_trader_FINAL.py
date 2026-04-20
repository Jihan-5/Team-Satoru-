"""Round 1 - AGGRESSIVE MAX-VOLUME STRATEGY

Target: Max Drawdown ~490, Recovery ~22, PnL ~11,000+

Top team analysis:
  #1: PnL 12,386, DD 423, Recovery 29, AvgFill 5.84
  #5: PnL 12,060, DD 570, Recovery 21, AvgFill 5.42
  Us: PnL  9,288, DD 344, Recovery 27, AvgFill 5.94

Problem: We have BETTER avg fill (5.94 vs 5.5) but way less volume.
We cherry-pick trades instead of running large positions.
ASH gap: ~1,746 vs top ~4,500 = we're leaving ~2,750 on the table.

Solution: Quote as tight as possible, take everything, hold max inventory.
The backtester misleads because it has no competition for fills.
On the real platform, tight quotes = more fills = more PnL.
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

ASH = 'ASH_COATED_OSMIUM'
PEPPER = 'INTARIAN_PEPPER_ROOT'
POS_LIMITS = {ASH: 80, PEPPER: 80}
PEPPER_DRIFT_PER_TICK = 0.1

PARAMS = {
    ASH: {
        'fair_mode': 'kalman_reversion',
        'ema_alpha': 0.08,
        'reversion_coef': 0.40,
        'imb_coef': 0.0,
        'l2l1_coef': 0.0,
        'use_microprice': True,
        'kf_Q': 1.0,
        'kf_R': 64.0,
        'take_width': 1,             # AGGRESSIVE: take 1-tick edge (was 5)
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 50,        # don't filter anything reasonable
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': 2,           # VERY TIGHT: penny-improve hard (was 5)
        'soft_position_limit': 80,   # NO position mgmt - hold max
        'levels': 3,
        'use_l2l1_signal': True,
        'buy_only': False,
    },
    PEPPER: {
        'fair_mode': 'drift_plus_reversion',
        'ema_alpha': 0.05,
        'reversion_coef': 0.60,
        'imb_coef': 0,
        'l2l1_coef': 0,
        'take_width': -15,
        'clear_width': 1,
        'prevent_adverse': False,
        'adverse_volume': 15,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': -10,
        'soft_position_limit': 80,
        'levels': 5,
        'use_drift': True,
        'use_l2l1_signal': True,
        'buy_only': True,
        'spike_sell_threshold': 5,
    },
}


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


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

    def bid(self):
        return 2000

    def take_best_orders(
        self, product, fair_value, take_width, orders, order_depth,
        position, buy_order_volume, sell_order_volume,
        prevent_adverse=False, adverse_volume=0, buy_only=False,
    ):
        limit = POS_LIMITS[product]

        # Take ALL sell levels that are below fair - take_width
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price > fair_value - take_width:
                    break
                ask_amt = -order_depth.sell_orders[ask_price]
                if prevent_adverse and ask_amt > adverse_volume:
                    continue
                qty = min(ask_amt, limit - (position + buy_order_volume))
                if qty > 0:
                    orders.append(Order(product, ask_price, qty))
                    buy_order_volume += qty

        # Take ALL buy levels above fair + take_width
        if not buy_only and order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price < fair_value + take_width:
                    break
                bid_amt = order_depth.buy_orders[bid_price]
                if prevent_adverse and bid_amt > adverse_volume:
                    continue
                qty = min(bid_amt, limit + (position - sell_order_volume))
                if qty > 0:
                    orders.append(Order(product, bid_price, -qty))
                    sell_order_volume += qty

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

    def compute_fair(self, product, order_depth, trader_obj, state):
        p = self.params[product]
        mode = p.get('fair_mode', 'ema_reversion')
        ema_key = f'{product}_ema'
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

        bid_vol = abs(order_depth.buy_orders.get(best_bid, 0))
        ask_vol = abs(order_depth.sell_orders.get(best_ask, 0))
        total = bid_vol + ask_vol

        # MM detection
        MM_LO, MM_HI = 8, 20
        mm_mid_key = f'{product}_mm_mid'
        mm_bid = best_bid if MM_LO <= bid_vol <= MM_HI else None
        mm_ask = best_ask if MM_LO <= ask_vol <= MM_HI else None
        if mm_bid is None:
            for px_ in sorted(order_depth.buy_orders.keys(), reverse=True):
                if MM_LO <= abs(order_depth.buy_orders[px_]) <= MM_HI:
                    mm_bid = px_
                    break
        if mm_ask is None:
            for px_ in sorted(order_depth.sell_orders.keys()):
                if MM_LO <= abs(order_depth.sell_orders[px_]) <= MM_HI:
                    mm_ask = px_
                    break
        if mm_bid is not None and mm_ask is not None:
            current_mm_mid = (mm_bid + mm_ask) / 2
            trader_obj[mm_mid_key] = current_mm_mid
        else:
            current_mm_mid = trader_obj.get(mm_mid_key)

        # Microprice
        if p.get('use_microprice', False) and total > 0:
            micro = (best_ask * bid_vol + best_bid * ask_vol) / total
        else:
            micro = l1_mid

        if mode == 'kalman_reversion' and current_mm_mid is not None:
            base = current_mm_mid
        else:
            base = micro

        # Kalman filter for ASH
        if mode == 'kalman_reversion':
            kf_x_key = f'{product}_kf_x'
            kf_P_key = f'{product}_kf_P'
            Q = p.get('kf_Q', 1.0)
            R = p.get('kf_R', 64.0)
            x_prev = trader_obj.get(kf_x_key)
            P_prev = trader_obj.get(kf_P_key)
            if x_prev is None or abs(l1_mid - x_prev) > 50:
                x_prev = l1_mid
                P_prev = R
            x_pred = x_prev
            P_pred = P_prev + Q
            K = P_pred / (P_pred + R)
            x_new = x_pred + K * (base - x_pred)
            P_new = (1.0 - K) * P_pred
            trader_obj[kf_x_key] = x_new
            trader_obj[kf_P_key] = P_new
            fair = x_new
            trader_obj[last_fair_key] = fair
            return fair

        # EMA reversion + drift for PEPPER
        alpha = p.get('ema_alpha', 0.10)
        prev_ema = trader_obj.get(ema_key)
        if prev_ema is None or abs(l1_mid - prev_ema) > 50:
            ema = l1_mid
        else:
            ema = alpha * l1_mid + (1 - alpha) * prev_ema
        trader_obj[ema_key] = ema

        residual = l1_mid - ema
        fair = base
        fair -= p.get('reversion_coef', 0.50) * residual

        if p.get('use_drift', False):
            fair += PEPPER_DRIFT_PER_TICK

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
                trader_obj = {}

        result = {}

        for product in [ASH, PEPPER]:
            if product not in state.order_depths:
                continue
            p = self.params[product]
            pos = state.position.get(product, 0)
            od = state.order_depths[product]

            fair = self.compute_fair(product, od, trader_obj, state)
            if fair is None:
                continue

            orders: List[Order] = []
            bov = sov = 0
            buy_only = p.get('buy_only', False)

            # Phase 1: Take - aggressively, ALL levels
            bov, sov = self.take_best_orders(
                product, fair, p['take_width'], orders, od, pos, bov, sov,
                prevent_adverse=p.get('prevent_adverse', False),
                adverse_volume=p.get('adverse_volume', 0),
                buy_only=buy_only,
            )

            # Phase 2: Clear
            if not buy_only:
                bov, sov = self.clear_position_order(
                    product, fair, p['clear_width'], orders, od, pos, bov, sov,
                )

            # L2-L1 asymmetric levels
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
                spike_thresh = p.get('spike_sell_threshold', 0)
                if spike_thresh > 0 and product == PEPPER:
                    if pos >= POS_LIMITS[product] and od.buy_orders and od.sell_orders:
                        l1_mid = (max(od.buy_orders) + min(od.sell_orders)) / 2
                        best_bid = max(od.buy_orders)
                        if best_bid >= l1_mid + spike_thresh:
                            sell_qty = min(10, pos + sov)
                            if sell_qty > 0:
                                orders.append(Order(product, best_bid, -sell_qty))
                                sov += sell_qty

            # Phase 3: Make - TIGHT quotes
            make_orders, _, _ = self.make_orders(
                product, od, fair, pos, bov, sov,
                p['disregard_edge'], p['join_edge'], p['default_edge'],
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
