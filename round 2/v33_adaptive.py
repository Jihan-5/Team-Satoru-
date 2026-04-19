"""Round 1 trader â ASH_COATED_OSMIUM + INTARIAN_PEPPER_ROOT.

STRATEGY v2 â 7 data-driven improvements over v1

DATA ANALYSIS (3-day deep analysis, days -2/-1/0):

  ASH_COATED_OSMIUM:
    - Pure AR(1) OU process: lag-1 autocorr = -0.499, lag-2/5 â 0 (no multi-lag signal)
    - Volume imbalance â next-tick return corr: +0.58â0.60 (all 3 days)
    - L2-L1 mid diff â next-tick return corr: +0.62â0.65 (all 3 days, independent!)
    - Spread distribution: 63% at 16 ticks, 25% at 18â19 ticks (MM posts at Â±8 from fair)
    - L1 volumes: mostly 10â15 per side (avg 14.1 bid / 14.1 ask)
    - Book moves â¥1 tick in 25% of ticks each direction
    - Fair value model: fair = mid - 0.40*(mid-ema) + 3.5*imb + 2.0*(l2m-l1m)
      OLS-derived, both signals combined push RÂ² from 0.45 toward ~0.55

  INTARIAN_PEPPER_ROOT:
    - Perfect linear drift: slope = +0.001/tick exactly (RÂ² = 1.0000), all 3 days
    - Detrended lag-1 autocorr = -0.497 (same OU as ASH)
    - OU half-life = 0.1â0.2 ticks (residuals vanish in ONE tick â no reversion profit)
    - Ask sits consistently 5.9â7.1 ticks ABOVE the trend line (98.3% of ticks)
    - Bid/ask moves in discrete Â±3 jumps (not continuous ticks)
    - CONCLUSION: Spread capture on PEPPER is near-zero; ALL profit comes from
      holding max-long and marking up with the +1000/day drift.
      Max theoretical: 80 Ã 1000 = 80,000/day.

  ASH FAIR VALUE MODEL (v2 â improved):
    - fair = microprice - 0.40*(l1_mid-ema) + 2.0*(l2m-l1m) + 0.10*ofi
    - microprice = (ask*bid_vol + bid*ask_vol)/(bid_vol+ask_vol)
               = l1_mid + (spread/2)*imb â l1_mid + 8*imb (spreadâ16)
    - OLS imb coef was â8.5 (shrunk to 3.5 previously); microprice recovers
      â8x naturally AND adapts when spread widens to 18-19 ticks.
    - ofi = delta_bid_queue - delta_ask_queue: captures order book FLOW,
      independent from static imbalance snapshot.

7 IMPROVEMENTS vs v1:

  1. EMA stale-state reset: if |mid - ema| > 50, hard-reset EMA to mid.
     Prevents blowup at day boundaries (v1 bug: day-0 EMA = 13000 carries
     into day-1 start at 14000, causing 600-tick fair-value error for ~14 ticks).

  2. ASH default_edge = 7 (was 2): MM posts at Â±8 from fair. We penny-improve
     to Â±7. Only relevant when book is empty (18% of ticks); penny-improve
     logic already handles it when book has quotes.

  3. ASH combined signal fair value: add L2-L1 coefficient (l2l1_coef=2.0).
     Slope from OLS: 0.635 Ã 3.69 / 1.03 â 2.28, shrunk to 2.0.
     Both signals are independent â combined RÂ² >> individual.

  4. ASH wide-spread regime filter: when spread â¤ 12 (unusual 9% regime),
     the MM stepped back. We stay in but raise default_edge to 5 to avoid
     quoting inside a potentially adversely-selected spread.

  5. PEPPER buy_only=True (new): never place ask orders. Every sell gives up
     drift profit (1000 ticks/day Ã 80 contracts). Two-sided MM on PEPPER
     is wrong â the OU half-life of 0.1 ticks means spread edge is near-zero.

  6. PEPPER default_edge = 3 (was 0): bid at trend â 3, aligned with the
     discrete Â±3 jump grid the MM uses. Higher fill probability vs. quoting
     at trend Â± 0.

  7. PEPPER take_width = -2: take any ask â¤ fair + 2 â trend + 2. These
     rare cheap-ask events (1.7% of ticks, ask below trend+3) are free money.

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


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Config
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
ASH = 'ASH_COATED_OSMIUM'
PEPPER = 'INTARIAN_PEPPER_ROOT'

# Position limits â confirmed 80 per product per Round 1 brief
POS_LIMITS = {ASH: 80, PEPPER: 80}

# Drift constant for PEPPER: +100 ticks over 1000 timestamps = +0.1 per timestamp
# Timestamps in Prosperity are multiples of 100, so per-tick = 0.1
PEPPER_DRIFT_PER_TICK = 0.1

PARAMS = {
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    # FINAL TUNED PARAMS (Round 1, 3-day backtest: 174,875, worst day 42,802)
    # âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
    ASH: {
        # R2-OPTIMIZED v15 params (grid-searched on R2 3-day data, +17% vs 183136 sim):
        # - l2l1_coef 2.0â0.0: BIGGEST WIN. L2-L1 signal is noise on R2's volatile data.
        #   In R1 (calm, stationary), L2-L1 captured real book skew. In R2 (volatile,
        #   changing spreads), L2 prices swing independently and add noise to fair value.
        #   Kalman microprice alone gives cleaner signal. +12% across all 3 days.
        # - soft_position_limit 78â50: R2 is more volatile (|dev|>=15 happens ~1%/tick).
        #   Tighter position management prevents accumulating huge losing positions during drift.
        # - default_edge 7â6: tighter quotes capture more fills.
        # - kf_Q 1.0â3.0, kf_R 64â32: faster Kalman adaptation for volatile R2 data.
        # - take_width 1â2: slightly more selective takes (avoid marginal-edge trades).
        'fair_mode': 'kalman_reversion',
        'ema_alpha': 0.08,
        'reversion_coef': 0.40,
        'imb_coef': 0.0,
        'l2l1_coef': 0.0,                 # R2-TUNED (was 2.0): L2-L1 is noise in volatile R2 data
        'use_microprice': True,
        'ofi_coef': 0.0,
        'kf_Q': 3.0,                # R2 TUNED (was 1.0): faster adaptation
        'kf_R': 32.0,               # R2 TUNED (was 64): trust observations more
        'take_width': 2,            # R2 TUNED (was 1): more selective takes
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 20,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': 5,          # v31: tighter passive quotes for more fills (user's hypothesis)
        'soft_position_limit': 50,  # R2 TUNED (was 78): CRITICAL â prevents runaway positions
        'levels': 2,
        'use_l2l1_signal': True,
    },
    PEPPER: {
        # PEPPER: Pure long drift strategy. Never sell.
        # OU half-life = 0.1 ticks â spread edge is near-zero.
        # All profit from holding max-long into +1000/day drift.
        # TUNED: take_width=-8 captures asks at trend+7-8 (was missing at -7);
        #        default_edge=-6 tighter passive bid for earlier position build.
        # buy_only=True: no ask orders ever.
        'fair_mode': 'drift_plus_reversion',
        'ema_alpha': 0.05,
        'reversion_coef': 0.60,
        'imb_coef': 0,
        'l2l1_coef': 0,
        'take_width': -8,           # TUNED: take asks â¤ fair+8 (was -7; catches trend+7-8 asks)
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 15,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': -6,         # TUNED: bid at fair+6 â trend+6 (was -5)
        'soft_position_limit': 80,
        'levels': 4,
        'use_drift': True,
        'use_l2l1_signal': True,
        'buy_only': True,           # never place ask orders
        'spike_sell_threshold': 5,  # sell into bids >= l1_mid+5 when at max pos
    },
}


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Logger
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
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


# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
# Trader
# âââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââââ
class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

    def bid(self):
        """
        Round 2 Market Access Fee bid â first-price sealed-bid auction.
        Top 50% win; winners pay their own bid.

        GAME THEORY APPROACH:
        We target the top 10% of bids (not just top 50%) to maximize win probability
        while keeping the bid below our true valuation of the extra market access.

        VALUATION MODEL:
          Extra 25% market flow â +25% on ASH only (PEPPER position-capped at 80).
          Our v15 baseline ASH PnL â 1,700 (sandbox) Ã 9.23 (final scale) = 15,700.
          Tighter quotes (edge=5) convert extra flow to fills more efficiently.
          â MAF value estimate: 5,000â10,000 XIRECs (conservative midpoint = 7,500).

        BID DISTRIBUTION MODEL (640 participants, our prior):
          40% of teams bid 0       (didn't optimize MAF)
          30% bid 100â2,000        (low-effort optimizers)
          20% bid 2,000â7,000      (competitive optimizers)
          10% bid 7,000+           (aggressive overbidders)
          â estimated 90th percentile bid: ~6,000
          â estimated median: ~1,500

        DECISION:
          To target top 10%: bid â 90th-percentile threshold = 6,000.
          To stay profitable: bid â¤ 0.9 Ã valuation = 6,750.
          Optimal bid = min(top_10pct_threshold, max_profitable_bid) = 6,000.

        RISK-AWARE ADJUSTMENT:
          Underbid risk: lose 5,000â10,000 opportunity (asymmetric loss)
          Overbid risk:  pay extra 1,000â2,000 if win
          â lean toward slightly higher than pure EV-optimal.
        """
        # --- Value estimation (in XIRECs) ---
        ASH_SANDBOX_PNL = 1700            # our proven ASH sandbox baseline
        FINAL_SCALE = 9.23                # sandbox â final simulation multiplier
        MAF_FLOW_BOOST = 0.25             # +25% extra market flow
        PEPPER_CAPPED = True              # PEPPER at position limit â no boost

        ash_final = ASH_SANDBOX_PNL * FINAL_SCALE
        ash_boost = ash_final * MAF_FLOW_BOOST
        pep_boost = 0 if PEPPER_CAPPED else 500  # PEPPER barely benefits
        maf_value = ash_boost + pep_boost          # â 3,925 pure ASH

        # With tighter-margin strategy (edge=5), flow conversion is more efficient.
        # Multiply value by strategy efficiency factor (empirical).
        STRATEGY_MULTIPLIER = 1.8          # tighter quotes capture more flow
        effective_value = maf_value * STRATEGY_MULTIPLIER   # â 7,065

        # --- Bid distribution prior (90th percentile estimation) ---
        # Pessimistic about others being lazy: most bid low.
        TOP_10_PCT_BID = 6000              # our belief about 90th-percentile bid
        MAX_BID_FRACTION = 0.90            # never bid more than 90% of our value

        max_profitable = int(effective_value * MAX_BID_FRACTION)

        # --- Final bid: min of aggressive-win target and max-profitable ---
        final_bid = min(TOP_10_PCT_BID, max_profitable)

        # Clamp to non-negative integer
        return max(0, int(final_bid))

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

        # FIX Bug 3: skip sell side entirely for buy_only products
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
        soft_position_bias=0,  # bias the zero point of position management
        bid_levels=1, ask_levels=1,
        bid_edge=None, ask_edge=None,  # v33: asymmetric edges
    ):
        orders = []
        limit = POS_LIMITS[product]

        asks_above_fair = [p for p in order_depth.sell_orders.keys()
                           if p > fair_value + disregard_edge]
        bids_below_fair = [p for p in order_depth.buy_orders.keys()
                           if p < fair_value - disregard_edge]

        best_ask_above_fair = min(asks_above_fair) if asks_above_fair else None
        best_bid_below_fair = max(bids_below_fair) if bids_below_fair else None

        # v33: asymmetric edges from signal
        _bid_edge = bid_edge if bid_edge is not None else default_edge
        _ask_edge = ask_edge if ask_edge is not None else default_edge

        ask = round(fair_value + _ask_edge)
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair
            else:
                ask = best_ask_above_fair - 1

        bid = round(fair_value - _bid_edge)
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
        """Compute fair value.

        ASH ('kalman_reversion' mode):
            Kalman filter estimates true fair value from noisy microprice observations.
            Two-step predict/update cycle with adaptive gain K = P/(P+R):
              - High noise tick  â K small â filter stays stable, ignores spike
              - Trending tick    â K rises â filter tracks price faster
              - Settled tick     â K â Q/(Q+R) â 0.015 â smooth like EMA Î±=0.015
            fair = kalman_x + l2l1_coef*(l2m-l1m)
            Replaces fixed EMA + reversion_coef with adaptive equivalent.

        PEPPER ('drift_plus_reversion' mode):
            Unchanged â EMA reversion + deterministic drift.
            drift_per_tick = 0.1 (RÂ²=1.0000, confirmed 3/3 days).
            fair = mid - rc*(mid-ema) + drift_per_tick
        """
        p = self.params[product]
        mode = p.get('fair_mode', 'ema_reversion')

        ema_key = f'{product}_ema'
        last_fair_key = f'{product}_last_fair'

        # Handle broken book â return last known fair + drift if applicable
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

        # Volume at L1
        bid_vol = abs(order_depth.buy_orders.get(best_bid, 0))
        ask_vol = abs(order_depth.sell_orders.get(best_ask, 0))
        total = bid_vol + ask_vol
        imb = (bid_vol - ask_vol) / total if total > 0 else 0

        # v31: MM DETECTION â find MM bot's mid (volume 8-20 signature for ASH)
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

        # Microprice (fallback)
        if p.get('use_microprice', False) and total > 0:
            micro = (best_ask * bid_vol + best_bid * ask_vol) / total
        else:
            micro = l1_mid

        # v31: use MM-mid when available, fall back to microprice
        if mode == 'kalman_reversion' and current_mm_mid is not None:
            base = current_mm_mid
        else:
            base = micro

        # ââ ASH: Kalman filter replaces EMA reversion ââââââââââââââââââââââ
        if mode == 'kalman_reversion':
            kf_x_key = f'{product}_kf_x'
            kf_P_key = f'{product}_kf_P'

            Q = p.get('kf_Q', 1.0)   # process noise: how much true price moves per tick
            R = p.get('kf_R', 64.0)  # measurement noise: microprice uncertainty â (spread/2)Â²

            x_prev = trader_obj.get(kf_x_key)
            P_prev = trader_obj.get(kf_P_key)

            # Day-boundary / first-tick reset: if filter is far from current price, reinitialise
            if x_prev is None or abs(l1_mid - x_prev) > 50:
                x_prev = l1_mid
                P_prev = R  # start with full measurement uncertainty

            # Step 1 â Predict (no drift assumed for ASH; OU mean reverts)
            x_pred = x_prev
            P_pred = P_prev + Q

            # Step 2 â Update with microprice observation
            K = P_pred / (P_pred + R)          # Kalman gain: adapts automatically
            x_new = x_pred + K * (base - x_pred)
            P_new = (1.0 - K) * P_pred

            trader_obj[kf_x_key] = x_new
            trader_obj[kf_P_key] = P_new

            fair = x_new

            # L2-L1 mid signal (unchanged â independent predictor, corr 0.63-0.65)
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

        # ââ PEPPER and fallback: original EMA reversion (unchanged) ââââââââ
        alpha = p.get('ema_alpha', 0.10)
        prev_ema = trader_obj.get(ema_key)
        if prev_ema is None or abs(l1_mid - prev_ema) > 50:
            ema = l1_mid
        else:
            ema = alpha * l1_mid + (1 - alpha) * prev_ema
        trader_obj[ema_key] = ema

        residual = l1_mid - ema
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

        trader_obj[last_fair_key] = fair
        return fair

    def get_l2l1_signal(self, order_depth):
        """Returns sign of (L2 mid â L1 mid) for asymmetric level selection."""
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
            # For buy_only products, sell side is skipped entirely via buy_only flag.
            # take_width=-7: buys any ask â¤ fair+7 â trend+7, capturing the typical ask.
            bov, sov = self.take_best_orders(
                product, fair, p['take_width'], orders, od, pos, bov, sov,
                prevent_adverse=p.get('prevent_adverse', False),
                adverse_volume=p.get('adverse_volume', 0),
                buy_only=buy_only,
            )

            # Phase 2: Clear â skip for buy_only products (improvement #5)
            # Never unwind a long PEPPER position; every unit held = drift profit.
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

            # buy_only: never quote asks (improvement #5)
            # EXCEPTION: Pepper spike-sell when at max position.
            # Use l1_mid (actual market price) not fair (which includes drift inflation).
            # Sell into bids that are >= l1_mid + spike_threshold â genuine price spikes.
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

            # ASH wide-spread regime filter (improvement #4):
            # When spread â¤ 12 (unusual â 9% of ticks), the normal MM has
            # stepped back. Widen our default_edge to 5 to avoid quoting
            # inside a potentially adversely-selected spread.
            effective_edge = p['default_edge']
            if product == ASH and od.buy_orders and od.sell_orders:
                current_spread = min(od.sell_orders) - max(od.buy_orders)
                if current_spread <= 12:
                    effective_edge = max(effective_edge, 5)

            # v33: signal-based asymmetric edges for ASH
            # When imb > 0.5: price likely UP (98% accurate) â widen ask, tighten bid
            # When imb < -0.5: price likely DOWN â widen bid, tighten ask
            bid_edge = effective_edge
            ask_edge = effective_edge
            if product == ASH and od.buy_orders and od.sell_orders:
                best_bid_ = max(od.buy_orders)
                best_ask_ = min(od.sell_orders)
                bv_ = abs(od.buy_orders[best_bid_])
                av_ = abs(od.sell_orders[best_ask_])
                tot_ = bv_ + av_
                if tot_ > 0:
                    imb_ = (bv_ - av_) / tot_
                    if imb_ > 0.5:
                        # Price going UP: don't sell cheap, buy eagerly
                        ask_edge = effective_edge + 2   # widen ask (avoid adverse)
                        bid_edge = effective_edge - 1   # tighten bid slightly
                    elif imb_ < -0.5:
                        # Price going DOWN: don't buy high, sell eagerly
                        bid_edge = effective_edge + 2   # widen bid
                        ask_edge = effective_edge - 1   # tighten ask

            # Phase 3: Make
            make_orders, _, _ = self.make_orders(
                product, od, fair, pos, bov, sov,
                p['disregard_edge'], p['join_edge'], effective_edge,
                manage_position=True,
                soft_position_limit=p['soft_position_limit'],
                soft_position_bias=0,
                bid_levels=bid_lvl,
                ask_levels=ask_lvl,
                bid_edge=bid_edge, ask_edge=ask_edge,
            )
            result[product] = orders + make_orders

        # ââ ASH DIAGNOSTICS â logs to help identify the PnL gap ââââââââââ
        ash_pos = state.position.get(ASH, 0)
        diag = trader_obj.setdefault('_ash_diag', {
            'pos_time': {'0-20': 0, '20-40': 0, '40-60': 0, '60-80': 0, 'neg': 0},
        })

        abs_pos = abs(ash_pos)
        if ash_pos < 0:
            diag['pos_time']['neg'] += 1
        elif abs_pos <= 20:
            diag['pos_time']['0-20'] += 1
        elif abs_pos <= 40:
            diag['pos_time']['20-40'] += 1
        elif abs_pos <= 60:
            diag['pos_time']['40-60'] += 1
        else:
            diag['pos_time']['60-80'] += 1

        # Log summary near end of each day
        if state.timestamp % 100000 >= 99800:
            logger.print(f"ASH_DIAG pos_time={diag['pos_time']}")

        try:
            trader_data = json.dumps(trader_obj)
        except Exception:
            trader_data = ''

        logger.flush(state, result)
        return result, 0, trader_data
