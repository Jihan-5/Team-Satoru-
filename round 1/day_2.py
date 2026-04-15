"""Round 1 trader – Day 2: ASH_COATED_OSMIUM + INTARIAN_PEPPER_ROOT.

STRATEGY v3 — Rook-E1 feedback + inventory skewing

CHANGES FROM v2 (round1_final_changes_in_ash.py):

  Day 2 changes (Rook-E1 insights):

  1. INVENTORY SKEWING (fundamental fix):
     fair -= skew_coef * position  (skew_coef=0.3)
     At pos=+40: fair drops 12 ticks -> sell more aggressively, buy less.
     At pos=-40: fair rises 12 ticks -> buy more aggressively, sell less.
     Prevents the +46/-42 position swings seen in live logs (138570).

  2. WEIGHTED LEVEL SIZING:
     "Too little volume and nothing happens. Too much and you give away structure."
     Instead of splitting volume 50/50 across levels, front-weight at best price.
     level_weight_decay=0.4: level 0 gets ~71%, level 1 gets ~29% (for 2 levels).
     Concentrates firepower at the competitive edge while maintaining depth.

  3. VOLUME-AWARE CAPPING:
     "Add volume at the right level... just enough to shift control."
     Cap order size at any single level to max_book_ratio * opposing L1 volume.
     max_book_ratio=1.5: if opposing best has 14 units, cap at 21 per level.
     Excess capacity redistributes to deeper levels instead of being wasted.
     Prevents posting 80 against a 14-unit book (signals desperation).

  Reverted from 138570 (failed aggressive params):
  - take_width=2 (was 1; aggressive taking increased adverse fills)
  - adverse_volume=15 (was 20; larger blocks were adverse)
  - default_edge=7 (was 5; penny-improve overrides this anyway)

PEPPER: Unchanged. Pure long drift. 7,443 PnL every submission = optimal.

POSITION LIMITS: 80 per product (from Round 1 brief).
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

# Position limits — confirmed 80 per product per Round 1 brief
POS_LIMITS = {ASH: 80, PEPPER: 80}

# Drift constant for PEPPER: +100 ticks over 1000 timestamps = +0.1 per timestamp
# Timestamps in Prosperity are multiples of 100, so per-tick = 0.1
PEPPER_DRIFT_PER_TICK = 0.1

PARAMS = {
    ASH: {
        # ASH: Inventory-skewed MM with weighted sizing.
        # fair = microprice - 0.40*(l1_mid-ema) + 2.0*(l2m-l1m) - 0.3*position
        'fair_mode': 'ema_reversion',
        'ema_alpha': 0.08,
        'reversion_coef': 0.40,
        'imb_coef': 0.0,
        'l2l1_coef': 2.0,
        'use_microprice': True,
        'ofi_coef': 0.0,
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
        'skew_coef': 0.3,              # Day 2: inventory skewing
        'level_weight_decay': 0.4,     # Day 2: level 0 = 71%, level 1 = 29%
        'max_book_ratio': 1.5,         # Day 2: cap per-level size at 1.5x opposing L1
    },
    PEPPER: {
        # PEPPER: Pure long drift strategy. Never sell. UNCHANGED.
        'fair_mode': 'drift_plus_reversion',
        'ema_alpha': 0.05,
        'reversion_coef': 0.60,
        'imb_coef': 0,
        'l2l1_coef': 0,
        'take_width': -8,
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 15,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': -6,
        'soft_position_limit': 80,
        'levels': 4,
        'use_drift': True,
        'use_l2l1_signal': True,
        'buy_only': True,
    },
}


# ───────────────────────────────────────────────────────────────────────
# Logger
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
        if params is None:
            params = PARAMS
        self.params = params

    def take_best_orders(
        self, product, fair_value, take_width, orders, order_depth,
        position, buy_order_volume, sell_order_volume,
        prevent_adverse=False, adverse_volume=0, buy_only=False,
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

        if not buy_only and len(order_depth.buy_orders) != 0:
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
        soft_position_bias=0,
        bid_levels=1, ask_levels=1,
    ):
        orders = []
        limit = POS_LIMITS[product]
        p = self.params[product]

        asks_above_fair = [pr for pr in order_depth.sell_orders.keys()
                           if pr > fair_value + disregard_edge]
        bids_below_fair = [pr for pr in order_depth.buy_orders.keys()
                           if pr < fair_value - disregard_edge]

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
            effective_pos = position - soft_position_bias
            if effective_pos > soft_position_limit:
                ask -= 1
            elif effective_pos < -soft_position_limit:
                bid += 1

        # --- Day 2: Volume-aware capping ---
        # Cap per-level order size at max_book_ratio * opposing L1 volume.
        # "Too much and you give away structure" — Rook-E1
        max_book_ratio = p.get('max_book_ratio', 0)
        bid_cap = 0
        ask_cap = 0
        if max_book_ratio > 0:
            # For bids: cap based on opposing (ask) volume at L1
            if order_depth.sell_orders:
                best_ask_price = min(order_depth.sell_orders.keys())
                opposing_ask_vol = abs(order_depth.sell_orders[best_ask_price])
                bid_cap = int(max_book_ratio * opposing_ask_vol)
            # For asks: cap based on opposing (bid) volume at L1
            if order_depth.buy_orders:
                best_bid_price = max(order_depth.buy_orders.keys())
                opposing_bid_vol = abs(order_depth.buy_orders[best_bid_price])
                ask_cap = int(max_book_ratio * opposing_bid_vol)

        # --- Day 2: Weighted level sizing ---
        # Front-weight volume at best level using geometric decay.
        # decay=0.4, 2 levels: weights = [1.0, 0.4], normalized = [0.71, 0.29]
        decay = p.get('level_weight_decay', 0)

        buy_qty = limit - (position + buy_order_volume)
        if buy_qty > 0 and bid_levels > 0:
            if decay > 0 and bid_levels > 1:
                # Geometric weights: [1, decay, decay^2, ...]
                weights = [decay ** i for i in range(bid_levels)]
                w_sum = sum(weights)
                allocs = [max(1, int(buy_qty * w / w_sum)) for w in weights]
                # Fix rounding: give remainder to level 0
                allocs[0] += buy_qty - sum(allocs)
                for lvl in range(bid_levels):
                    q = allocs[lvl]
                    # Apply volume cap (redistribute excess to next level)
                    if bid_cap > 0 and q > bid_cap and lvl < bid_levels - 1:
                        allocs[lvl + 1] += q - bid_cap
                        q = bid_cap
                    if q > 0:
                        orders.append(Order(product, round(bid) - lvl, q))
            else:
                # Original even-split logic
                base = buy_qty // bid_levels
                for lvl in range(bid_levels):
                    q = base if lvl < bid_levels - 1 else buy_qty - base * (bid_levels - 1)
                    if q > 0:
                        orders.append(Order(product, round(bid) - lvl, q))

        sell_qty = limit + (position - sell_order_volume)
        if sell_qty > 0 and ask_levels > 0:
            if decay > 0 and ask_levels > 1:
                weights = [decay ** i for i in range(ask_levels)]
                w_sum = sum(weights)
                allocs = [max(1, int(sell_qty * w / w_sum)) for w in weights]
                allocs[0] += sell_qty - sum(allocs)
                for lvl in range(ask_levels):
                    q = allocs[lvl]
                    if ask_cap > 0 and q > ask_cap and lvl < ask_levels - 1:
                        allocs[lvl + 1] += q - ask_cap
                        q = ask_cap
                    if q > 0:
                        orders.append(Order(product, round(ask) + lvl, -q))
            else:
                base = sell_qty // ask_levels
                for lvl in range(ask_levels):
                    q = base if lvl < ask_levels - 1 else sell_qty - base * (ask_levels - 1)
                    if q > 0:
                        orders.append(Order(product, round(ask) + lvl, -q))

        return orders, buy_order_volume, sell_order_volume

    def compute_fair(self, product, order_depth, trader_obj, state):
        """Compute fair value with inventory skewing.

        ASH: fair = microprice - rc*(l1_mid-ema) + l2l1*(l2m-l1m) - skew*position
        PEPPER: fair = mid - rc*(mid-ema) + drift
        """
        p = self.params[product]

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

        if p.get('use_microprice', False) and total > 0:
            base = (best_ask * bid_vol + best_bid * ask_vol) / total
        else:
            base = l1_mid

        fair = base

        rev_coef = p.get('reversion_coef', 0.50)
        fair -= rev_coef * residual

        imb_coef = p.get('imb_coef', 0.0)
        fair += imb_coef * imb

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

        ofi_coef = p.get('ofi_coef', 0.0)
        if ofi_coef != 0.0:
            ofi_key = f'{product}_prev_book'
            prev_book = trader_obj.get(ofi_key)
            ofi = 0.0
            if prev_book is not None:
                pb, pbv, pa, pav = prev_book
                if abs(best_bid - pb) + abs(best_ask - pa) < 50:
                    if best_bid > pb:
                        ofi += bid_vol
                    elif best_bid == pb:
                        ofi += (bid_vol - pbv)
                    else:
                        ofi -= pbv
                    if best_ask < pa:
                        ofi -= ask_vol
                    elif best_ask == pa:
                        ofi -= (ask_vol - pav)
                    else:
                        ofi += pav
            trader_obj[f'{product}_prev_book'] = [best_bid, bid_vol, best_ask, ask_vol]
            fair += ofi_coef * ofi

        if p.get('use_drift', False):
            fair += PEPPER_DRIFT_PER_TICK

        # Inventory skewing: fair -= skew_coef * position
        # Standard MM technique to prevent inventory accumulation.
        skew_coef = p.get('skew_coef', 0.0)
        if skew_coef != 0.0:
            fair -= skew_coef * state.position.get(product, 0)

        trader_obj[last_fair_key] = fair
        return fair

    def get_l2l1_signal(self, order_depth):
        """Returns sign of (L2 mid - L1 mid) for asymmetric level selection."""
        try:
            sb = sorted(order_depth.buy_orders.items(), reverse=True)
            sa = sorted(order_depth.sell_orders.items())
            if len(sb) >= 2 and len(sa) >= 2:
                l1m = (sb[0][0] + sa[0][0]) / 2
                l2m = (sb[1][0] + sa[1][0]) / 2
                return l2m - l1m
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

            # Phase 1: Take
            bov, sov = self.take_best_orders(
                product, fair, p['take_width'], orders, od, pos, bov, sov,
                prevent_adverse=p.get('prevent_adverse', False),
                adverse_volume=p.get('adverse_volume', 0),
                buy_only=buy_only,
            )

            # Phase 2: Clear (skip for buy_only)
            if not buy_only:
                bov, sov = self.clear_position_order(
                    product, fair, p['clear_width'], orders, od, pos, bov, sov,
                )

            # L2-L1 signal for asymmetric level selection
            base_lvl = p.get('levels', 2)
            bid_lvl = ask_lvl = base_lvl
            if p.get('use_l2l1_signal', False):
                sig = self.get_l2l1_signal(od)
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
                current_spread = min(od.sell_orders) - max(od.buy_orders)
                if current_spread <= 12:
                    effective_edge = max(effective_edge, 5)

            # Phase 3: Make (with weighted sizing + volume capping)
            make_orders, _, _ = self.make_orders(
                product, od, fair, pos, bov, sov,
                p['disregard_edge'], p['join_edge'], effective_edge,
                manage_position=True,
                soft_position_limit=p['soft_position_limit'],
                soft_position_bias=0,
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
