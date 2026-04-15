"""Round 1 trader – ASH_COATED_OSMIUM + INTARIAN_PEPPER_ROOT.

STRATEGY v2 — 7 data-driven improvements over v1

DATA ANALYSIS (3-day deep analysis, days -2/-1/0):

  ASH_COATED_OSMIUM:
    - Pure AR(1) OU process: lag-1 autocorr = -0.499, lag-2/5 ≈ 0 (no multi-lag signal)
    - Volume imbalance → next-tick return corr: +0.58–0.60 (all 3 days)
    - L2-L1 mid diff → next-tick return corr: +0.62–0.65 (all 3 days, independent!)
    - Spread distribution: 63% at 16 ticks, 25% at 18–19 ticks (MM posts at ±8 from fair)
    - L1 volumes: mostly 10–15 per side (avg 14.1 bid / 14.1 ask)
    - Book moves ≥1 tick in 25% of ticks each direction
    - Fair value model: fair = mid - 0.40*(mid-ema) + 3.5*imb + 2.0*(l2m-l1m)
      OLS-derived, both signals combined push R² from 0.45 toward ~0.55

  INTARIAN_PEPPER_ROOT:
    - Perfect linear drift: slope = +0.001/tick exactly (R² = 1.0000), all 3 days
    - Detrended lag-1 autocorr = -0.497 (same OU as ASH)
    - OU half-life = 0.1–0.2 ticks (residuals vanish in ONE tick — no reversion profit)
    - Ask sits consistently 5.9–7.1 ticks ABOVE the trend line (98.3% of ticks)
    - Bid/ask moves in discrete ±3 jumps (not continuous ticks)
    - CONCLUSION: Spread capture on PEPPER is near-zero; ALL profit comes from
      holding max-long and marking up with the +1000/day drift.
      Max theoretical: 80 × 1000 = 80,000/day.

  ASH FAIR VALUE MODEL (v2 — improved):
    - fair = microprice - 0.40*(l1_mid-ema) + 2.0*(l2m-l1m) + 0.10*ofi
    - microprice = (ask*bid_vol + bid*ask_vol)/(bid_vol+ask_vol)
               = l1_mid + (spread/2)*imb ≈ l1_mid + 8*imb (spread≈16)
    - OLS imb coef was ≈8.5 (shrunk to 3.5 previously); microprice recovers
      ≈8x naturally AND adapts when spread widens to 18-19 ticks.
    - ofi = delta_bid_queue - delta_ask_queue: captures order book FLOW,
      independent from static imbalance snapshot.

7 IMPROVEMENTS vs v1:

  1. EMA stale-state reset: if |mid - ema| > 50, hard-reset EMA to mid.
     Prevents blowup at day boundaries (v1 bug: day-0 EMA = 13000 carries
     into day-1 start at 14000, causing 600-tick fair-value error for ~14 ticks).

  2. ASH default_edge = 7 (was 2): MM posts at ±8 from fair. We penny-improve
     to ±7. Only relevant when book is empty (18% of ticks); penny-improve
     logic already handles it when book has quotes.

  3. ASH combined signal fair value: add L2-L1 coefficient (l2l1_coef=2.0).
     Slope from OLS: 0.635 × 3.69 / 1.03 ≈ 2.28, shrunk to 2.0.
     Both signals are independent — combined R² >> individual.

  4. ASH wide-spread regime filter: when spread ≤ 12 (unusual 9% regime),
     the MM stepped back. We stay in but raise default_edge to 5 to avoid
     quoting inside a potentially adversely-selected spread.

  5. PEPPER buy_only=True (new): never place ask orders. Every sell gives up
     drift profit (1000 ticks/day × 80 contracts). Two-sided MM on PEPPER
     is wrong — the OU half-life of 0.1 ticks means spread edge is near-zero.

  6. PEPPER default_edge = 3 (was 0): bid at trend − 3, aligned with the
     discrete ±3 jump grid the MM uses. Higher fill probability vs. quoting
     at trend ± 0.

  7. PEPPER take_width = -2: take any ask ≤ fair + 2 ≈ trend + 2. These
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
        # ASH: Max-aggression fair value — microprice base + EMA reversion + L2-L1.
        # fair = microprice - 0.40*(l1_mid-ema) + 2.0*(l2m-l1m)
        # microprice = (ask*bid_vol + bid*ask_vol)/(bid_vol+ask_vol)
        #            = l1_mid + (spread/2)*imb ≈ l1_mid + 8*imb (spread≈16)
        # OFI removed: live test showed it hurt performance (−123 PnL, 136945 vs 136480).
        # default_edge=5: quote at ±5 vs MM at ±8, deeper inside spread → more fills.
        #   At 5-tick edge, each round-trip earns 5; at spread=16 that's 62.5% of half-spread.
        # take_width=1: take any ask ≤ fair-1 (slightly more aggressive capture).
        # adverse_volume=20: ASH avg L1 volume ≈14; was skipping 15-20 blocks unnecessarily.
        'fair_mode': 'ema_reversion',
        'ema_alpha': 0.08,
        'reversion_coef': 0.40,
        'imb_coef': 0.0,            # zeroed: microprice captures L1 imbalance
        'l2l1_coef': 2.0,           # independent signal (corr 0.63-0.65)
        'use_microprice': True,     # vol-weighted mid: mid + (spread/2)*imb
        'ofi_coef': 0.0,            # OFI removed: calibration-free, hurt in live test
        'take_width': 1,            # AGGRESSIVE: was 2; take asks ≤ fair-1
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 20,       # AGGRESSIVE: was 15; trade bigger blocks
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': 5,          # AGGRESSIVE: was 7; quote at ±5 vs MM ±8
        'soft_position_limit': 78,
        'levels': 2,
        'use_l2l1_signal': True,
    },
    PEPPER: {
        # PEPPER: Pure long drift strategy. Never sell.
        # OU half-life = 0.1 ticks → spread edge is near-zero.
        # All profit from holding max-long into +1000/day drift.
        # TUNED: take_width=-8 captures asks at trend+7-8 (was missing at -7);
        #        default_edge=-6 tighter passive bid for earlier position build.
        # buy_only=True: no ask orders ever.
        'fair_mode': 'drift_plus_reversion',
        'ema_alpha': 0.05,
        'reversion_coef': 0.60,
        'imb_coef': 0,
        'l2l1_coef': 0,
        'take_width': -8,           # TUNED: take asks ≤ fair+8 (was -7; catches trend+7-8 asks)
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': 15,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': -6,         # TUNED: bid at fair+6 ≈ trend+6 (was -5)
        'soft_position_limit': 80,
        'levels': 4,
        'use_drift': True,
        'use_l2l1_signal': True,
        'buy_only': True,           # never place ask orders
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
            ema[t] = alpha*l1_mid + (1-alpha)*ema[t-1]  (EMA on simple mid)
            microprice = (ask*bid_vol + bid*ask_vol) / (bid_vol + ask_vol)
                       = l1_mid + (spread/2)*imb  [vol-weighted mid]
            fair = microprice - rc*(l1_mid-ema) + l2l1_coef*(l2m-l1m) + ofi_coef*ofi
            OFI = delta_bid_queue - delta_ask_queue (price-adjusted, stored in traderData).

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

        # Update EMA — with stale-state reset guard (improvement #1)
        # If EMA is >50 from mid (e.g. day boundary jump), reset it immediately
        # to avoid a multi-tick fair-value error at the start of each new day.
        alpha = p.get('ema_alpha', 0.10)
        prev_ema = trader_obj.get(ema_key)
        if prev_ema is None or abs(l1_mid - prev_ema) > 50:
            ema = l1_mid
        else:
            ema = alpha * l1_mid + (1 - alpha) * prev_ema
        trader_obj[ema_key] = ema

        # Residual from EMA
        residual = l1_mid - ema

        # Volume at L1
        bid_vol = abs(order_depth.buy_orders.get(best_bid, 0))
        ask_vol = abs(order_depth.sell_orders.get(best_ask, 0))
        total = bid_vol + ask_vol
        imb = (bid_vol - ask_vol) / total if total > 0 else 0

        # Base: microprice (vol-weighted mid) if enabled, else simple mid.
        # microprice = l1_mid + (spread/2)*imb ≈ l1_mid + 8*imb for ASH (spread≈16).
        # OLS estimated imb_coef = 8.5; microprice recovers ≈8x and adapts to spread.
        if p.get('use_microprice', False) and total > 0:
            base = (best_ask * bid_vol + best_bid * ask_vol) / total
        else:
            base = l1_mid

        fair = base

        # Apply reversion correction (pulls fair toward EMA when mid is far)
        rev_coef = p.get('reversion_coef', 0.50)
        fair -= rev_coef * residual

        # Apply explicit imbalance correction (0 when using microprice, which already captures it)
        imb_coef = p.get('imb_coef', 0.0)
        fair += imb_coef * imb

        # Apply L2-L1 mid signal (improvement #3 — independent predictor for ASH)
        # L2-L1 diff has corr 0.63-0.65 with next-tick return (all 3 days).
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

        # OFI (Order Flow Imbalance): delta bid queue − delta ask queue tick-to-tick.
        # Captures order book FLOW — independent from static imbalance snapshot.
        # Standard formula: price-adjusted queue changes at the best level.
        ofi_coef = p.get('ofi_coef', 0.0)
        if ofi_coef != 0.0:
            ofi_key = f'{product}_prev_book'
            prev_book = trader_obj.get(ofi_key)
            ofi = 0.0
            if prev_book is not None:
                pb, pbv, pa, pav = prev_book
                # Guard against day-boundary price jumps contaminating OFI
                if abs(best_bid - pb) + abs(best_ask - pa) < 50:
                    if best_bid > pb:
                        ofi += bid_vol          # bid price rose → all new queue
                    elif best_bid == pb:
                        ofi += (bid_vol - pbv)  # same level → delta volume
                    else:
                        ofi -= pbv              # bid price fell → lost old queue
                    if best_ask < pa:
                        ofi -= ask_vol          # ask price fell → all new queue
                    elif best_ask == pa:
                        ofi -= (ask_vol - pav)  # same level → delta volume
                    else:
                        ofi += pav              # ask price rose → lost old queue
            trader_obj[f'{product}_prev_book'] = [best_bid, bid_vol, best_ask, ask_vol]
            fair += ofi_coef * ofi

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
            buy_only = p.get('buy_only', False)

            # Phase 1: Take
            # For buy_only products, sell side is skipped entirely via buy_only flag.
            # take_width=-7: buys any ask ≤ fair+7 ≈ trend+7, capturing the typical ask.
            bov, sov = self.take_best_orders(
                product, fair, p['take_width'], orders, od, pos, bov, sov,
                prevent_adverse=p.get('prevent_adverse', False),
                adverse_volume=p.get('adverse_volume', 0),
                buy_only=buy_only,
            )

            # Phase 2: Clear — skip for buy_only products (improvement #5)
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
            if buy_only:
                ask_lvl = 0

            # ASH wide-spread regime filter (improvement #4):
            # When spread ≤ 12 (unusual — 9% of ticks), the normal MM has
            # stepped back. Widen our default_edge to 5 to avoid quoting
            # inside a potentially adversely-selected spread.
            effective_edge = p['default_edge']
            if product == ASH and od.buy_orders and od.sell_orders:
                current_spread = min(od.sell_orders) - max(od.buy_orders)
                if current_spread <= 12:
                    effective_edge = max(effective_edge, 5)

            # Phase 3: Make
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
