"""Round 1 trader – ASH_COATED_OSMIUM + INTARIAN_PEPPER_ROOT.

FINAL STRATEGY (3-day backtest: 174,875 total, worst day 42,802)

ANALYSIS FINDINGS (3 days of Round 1 historical data):

  ASH_COATED_OSMIUM:
    - Pure OU process (mean-reverting to slowly-moving mean near 10,000)
    - lag-1 autocorr = -0.50, lag-2/3/5/10 all ~0 (clean AR(1))
    - Best fair value model (R²=0.45, replicates 3/3 days):
        ema[t] = 0.10 * mid + 0.90 * ema[t-1]
        fair = mid - 0.50 * (mid - ema) + 3.5 * volume_imbalance
    - L1 volume imbalance has correlation +0.64 with next-tick return
    - Realistic MM ceiling: ~3,300/day (we capture ~2,300, 70%)

  INTARIAN_PEPPER_ROOT:
    - Deterministic linear drift: +0.1 per tick (+1000 per day), replicates 3/3
    - Within the drift, lag-1 autocorr = -0.45 (OU around drift line)
    - Best fair value model (R²=0.48):
        ema[t] = 0.05 * mid + 0.95 * ema[t-1]  (slower than ASH)
        fair = mid - 0.60 * (mid - ema) + drift_per_tick
    - Realistic MM ceiling from spread alone: ~2,000/day
    - Drift accumulation captures ~45-60k/day on top of spread PnL
    - We capture the drift by quoting aggressively two-sided (edge=0, levels=4);
      as the market drifts up, our buy orders fill more than our sell orders,
      naturally accumulating a long position that marks up with the drift.

KEY DESIGN DECISIONS:

  1. Fair value is a shrunk OLS prediction, not just mid or MM-mid.
     Both products use ema_reversion / drift_plus_reversion formulas.
  2. ASH uses wider default_edge (2) because its std is larger. PEPPER
     uses edge=0 because the drift means "quoting at fair" is actually
     "quoting slightly below next-tick fair" — high fill rate.
  3. PEPPER soft_position_limit=80 (= hard limit) allows full drift
     exposure. Tightening this costs profit proportionally.
  4. PEPPER levels=4 spreads quotes across 4 price points to maximize
     capture during the drift.
  5. No directional bias flag (use_directional_bias=False). The drift
     accumulation happens naturally from aggressive two-sided quoting;
     an explicit bias would be redundant and harder to tune.
  6. L2-L1 signal flips the bid/ask level count asymmetrically when
     a wall appears on one side (Round 0 discovery, still applies).

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
    # ─────────────────────────────────────────────────────────────────────
    # FINAL TUNED PARAMS (Round 1, 3-day backtest: 174,875, worst day 42,802)
    # ─────────────────────────────────────────────────────────────────────
    ASH: {
        # ASH: OLS-derived fair value model (R²=0.45, replicates across 3 days).
        # fair = mid - 0.40*(mid - ema) + 3.5*imbalance
        # where ema = 0.10*mid + 0.90*ema[t-1]
        # Tuned: take_width=2 captures more mispriced offers, clear_width=1
        # closes positions faster, reversion_coef=0.40 is shrunk from OLS 0.50.
        'fair_mode': 'ema_reversion',
        'ema_alpha': 0.10,
        'reversion_coef': 0.40,
        'imb_coef': 3.5,
        'take_width': 2,
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 15,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': 2,
        'soft_position_limit': 40,
        'levels': 2,
        'use_l2l1_signal': True,
    },
    PEPPER: {
        # PEPPER: detrended OLS reversion + deterministic +0.1/tick drift.
        # fair = mid - 0.60*(mid - ema) + drift_per_tick
        # where ema = 0.05*mid + 0.95*ema[t-1]  (slower than ASH)
        # Tuned: default_edge=0 quotes right at fair (drift keeps us ahead),
        # levels=4 spreads quotes across many price points for max capture,
        # soft_position_limit=80 allows full exposure to drift accumulation,
        # clear_width=1 closes inventory slightly above fair.
        'fair_mode': 'drift_plus_reversion',
        'ema_alpha': 0.05,
        'reversion_coef': 0.60,
        'imb_coef': 0,
        'take_width': 1,
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 15,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': 0,
        'soft_position_limit': 80,
        'levels': 4,
        'use_drift': True,
        'use_directional_bias': False,
        'directional_bias_size': 0,
        'use_l2l1_signal': True,
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
        soft_position_bias=0,  # bias the zero point of position management
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

        # Position management with optional directional bias
        # If soft_position_bias=+10, we treat position=+10 as "neutral",
        # so we only tighten the ask when position > 10 (not when >0)
        if manage_position:
            effective_pos = position - soft_position_bias
            if effective_pos > soft_position_limit:
                ask -= 1
            elif effective_pos < -soft_position_limit:
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
        """Compute fair value using OLS-derived reversion models.

        ASH ('ema_reversion' mode):
            ema[t] = alpha*mid + (1-alpha)*ema[t-1]
            fair = mid - reversion_coef * (mid - ema) + imb_coef * volume_imbalance
                 = (1-rc)*mid + rc*ema + ic*imb
            Optimal: alpha=0.10, rc=0.50, ic=3.5. R² ~0.45.

        PEPPER ('drift_plus_reversion' mode):
            drift_per_tick = 0.1 (deterministic, confirmed 3/3 days)
            detrended_mid = mid (EMA is computed on raw mid, but reversion is
                measured in the instantaneous direction, which already accounts
                for drift naturally since drift contributes equally to mid and ema
                when alpha is in the right range)
            fair = mid - rc * (mid - ema) + ic * imb + drift_per_tick
            Optimal: alpha=0.10, rc=0.60, ic=1.5. R² ~0.48.
        """
        p = self.params[product]
        mode = p.get('fair_mode', 'ema_reversion')

        ema_key = f'{product}_ema'
        last_fair_key = f'{product}_last_fair'

        # Handle broken book — return last known fair + drift if applicable
        if not order_depth.sell_orders or not order_depth.buy_orders:
            last_fair = trader_obj.get(last_fair_key)
            if last_fair is None:
                return None
            if p.get('use_drift', False):
                return last_fair + PEPPER_DRIFT_PER_TICK
            return last_fair

        # Current best bid/ask
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        l1_mid = (best_bid + best_ask) / 2

        # Update EMA
        alpha = p.get('ema_alpha', 0.10)
        prev_ema = trader_obj.get(ema_key)
        if prev_ema is None:
            ema = l1_mid
        else:
            ema = alpha * l1_mid + (1 - alpha) * prev_ema
        trader_obj[ema_key] = ema

        # Residual from EMA
        residual = l1_mid - ema

        # Volume imbalance at L1
        bid_vol = abs(order_depth.buy_orders.get(best_bid, 0))
        ask_vol = abs(order_depth.sell_orders.get(best_ask, 0))
        total = bid_vol + ask_vol
        imb = (bid_vol - ask_vol) / total if total > 0 else 0

        # Start from the current mid
        fair = l1_mid

        # Apply reversion correction (pulls fair toward EMA when mid is far)
        rev_coef = p.get('reversion_coef', 0.50)
        fair -= rev_coef * residual

        # Apply imbalance correction
        imb_coef = p.get('imb_coef', 0.0)
        fair += imb_coef * imb

        # Apply drift (PEPPER only)
        if p.get('use_drift', False):
            fair += PEPPER_DRIFT_PER_TICK

        trader_obj[last_fair_key] = fair
        return fair

    def get_l2l1_signal(self, order_depth):
        """Returns sign of (L2 mid – L1 mid) for asymmetric level selection."""
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

            # Phase 1: Take
            bov, sov = self.take_best_orders(
                product, fair, p['take_width'], orders, od, pos, bov, sov,
                prevent_adverse=p.get('prevent_adverse', False),
                adverse_volume=p.get('adverse_volume', 0),
            )
            # Phase 2: Clear
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

            # Directional bias for PEPPER (bias long to ride +100 drift)
            soft_bias = 0
            if p.get('use_directional_bias', False):
                soft_bias = p.get('directional_bias_size', 0)

            # Phase 3: Make
            make_orders, _, _ = self.make_orders(
                product, od, fair, pos, bov, sov,
                p['disregard_edge'], p['join_edge'], p['default_edge'],
                manage_position=True,
                soft_position_limit=p['soft_position_limit'],
                soft_position_bias=soft_bias,
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
