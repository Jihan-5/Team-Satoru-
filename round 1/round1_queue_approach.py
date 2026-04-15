"""Round 1 trader – ASH_COATED_OSMIUM + INTARIAN_PEPPER_ROOT.
QUEUE APPROACH — full refactor with code-quality + strategy improvements.

Changes vs round1_trader_FINAL.py:

  A1:  No OrderDepth mutation — work on local copies (remaining_bids/asks)
  A2:  get_book_levels() called once per tick — no repeated sorting
  A3:  Signals kept in tick units; spread_coef term added (B4)
  A4:  Inventory skew shifts fair value (fair -= pos * inventory_penalty)
  A5:  PEPPER overshoot sell valve (pos > 60 and bid > fair+15)
  A6:  Explicit take thresholds (buy_threshold = fair - take_width)
  A7:  Tight-spread regime: levels → 1, edge widens
  A8:  Timestamp-gap EMA reset (catches day boundaries)
  A9:  Periodic logging only (every LOG_INTERVAL ticks)
  A10: Named constants replace magic numbers
  B1:  Microprice replaces mid as base price
  B2:  Inventory-skewed fair (same as A4)
  B3:  Queue-priority quoting: join small queue, penny-improve large
  B4:  Spread term added to ASH fair value model

Sweep results (18 combos: microprice on/off × inv 0.0–0.10):
  Best: mid + inv=0.03 → 248,477  (FINAL=265,478, delta=-17,002)
  microprice uniformly worse than plain mid on this dataset.
  Gap vs FINAL is structural (different fill ordering from refactor).
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
from typing import Any, List, Optional, Tuple, Dict


# ───────────────────────────────────────────────────────────────────────
# A10: Named constants
# ───────────────────────────────────────────────────────────────────────
EMA_RESET_THRESHOLD   = 50     # ticks: hard-reset EMA if mid drifts this far
EMA_RESET_TS_GAP      = 5_000  # timestamps: reset EMA on day-boundary gap
TIGHT_SPREAD_THRESHOLD = 12    # ticks: tighten quoting below this spread
ADVERSE_VOLUME_THRESHOLD = 15  # units: skip takes against large counterparties
SMALL_QUEUE_THRESHOLD = 8      # units: join queue (B3) when vol below this
LOG_INTERVAL          = 1_000  # timestamps between logger.flush() calls


# ───────────────────────────────────────────────────────────────────────
# Config
# ───────────────────────────────────────────────────────────────────────
ASH    = 'ASH_COATED_OSMIUM'
PEPPER = 'INTARIAN_PEPPER_ROOT'

POS_LIMITS = {ASH: 80, PEPPER: 80}

PEPPER_DRIFT_PER_TICK = 0.1  # +1 price per 1000 timestamps

PARAMS = {
    ASH: {
        'fair_mode': 'ema_reversion',
        'ema_alpha': 0.08,
        'reversion_coef': 0.40,     # OLS-validated (keep proven value)
        'imb_coef': 3.5,            # OLS-validated
        'l2l1_coef': 2.0,           # OLS-validated
        'spread_coef': 0.0,         # off by default; tune separately
        'inventory_penalty': 0.03,  # A4/B2: best from sweep (0.03 w/ mid, 248,477)
        'take_width': 2,
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': ADVERSE_VOLUME_THRESHOLD,
        'disregard_edge': 1,
        'join_edge': 2,             # B3: join if best quote within 2 of fair
        'default_edge': 7,
        'soft_position_limit': 78,
        'levels': 2,
        'use_l2l1_signal': True,
        'buy_only': False,
    },
    PEPPER: {
        'fair_mode': 'drift_plus_reversion',
        'ema_alpha': 0.05,
        'reversion_coef': 0.60,
        'imb_coef': 0,
        'l2l1_coef': 0,
        'spread_coef': 0,
        'inventory_penalty': 0,     # never penalise — we want max long
        'take_width': -8,
        'clear_width': 1,
        'prevent_adverse': True,
        'adverse_volume': ADVERSE_VOLUME_THRESHOLD,
        'disregard_edge': 1,
        'join_edge': 0,
        'default_edge': -6,
        'soft_position_limit': 80,
        'levels': 4,
        'use_drift': True,
        'use_l2l1_signal': True,
        'buy_only': True,
        'overshoot_sell_pos': 60,   # A5: allow sell when pos exceeds this
        'overshoot_sell_edge': 15,  # A5: and best_bid > fair + this
    },
}


# ───────────────────────────────────────────────────────────────────────
# A2: Book-level helper — sort ONCE per tick
# ───────────────────────────────────────────────────────────────────────
def get_book_levels(
    buy_orders: Dict[int, int],
    sell_orders: Dict[int, int],
) -> Tuple:
    """Return (best_bid, best_bid_vol, second_bid,
                best_ask, best_ask_vol, second_ask).
    Volumes are positive regardless of sign convention in sell_orders.
    Returns None for missing levels."""
    sorted_bids = sorted(buy_orders.items(), reverse=True) if buy_orders else []
    sorted_asks = sorted(sell_orders.items())              if sell_orders else []

    best_bid     = sorted_bids[0][0]  if sorted_bids           else None
    best_bid_vol = sorted_bids[0][1]  if sorted_bids           else 0
    second_bid   = sorted_bids[1][0]  if len(sorted_bids) >= 2 else None

    best_ask     = sorted_asks[0][0]         if sorted_asks           else None
    best_ask_vol = -sorted_asks[0][1]        if sorted_asks           else 0
    second_ask   = sorted_asks[1][0]         if len(sorted_asks) >= 2 else None

    return best_bid, best_bid_vol, second_bid, best_ask, best_ask_vol, second_ask


# ───────────────────────────────────────────────────────────────────────
# Logger — A9: flush only every LOG_INTERVAL ticks
# ───────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs: str = ""
        self.max_log_length: int = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict) -> None:
        payload = {
            "state":  self._compress_state(state),
            "orders": self._compress_orders(orders),
            "logs":   self.logs[: self.max_log_length],
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
            "p":  state.position, "o": state.observations,
        }

    def _compress_trades(self, trades: dict) -> list:
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([
                    getattr(t, "symbol", ""), getattr(t, "buyer", ""),
                    getattr(t, "seller", ""), getattr(t, "price", 0),
                    getattr(t, "quantity", 0), getattr(t, "timestamp", 0),
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

    # ── Phase 1: Take ──────────────────────────────────────────────────
    def take_best_orders(
        self, product, fair, take_width,
        orders: list,
        remaining_bids: dict, remaining_asks: dict,
        position, bov, sov,
        prevent_adverse=False, adverse_volume=0, buy_only=False,
    ):
        limit = POS_LIMITS[product]
        # A6: explicit thresholds
        buy_threshold  = fair - take_width   # buy  if ask  <= buy_threshold
        sell_threshold = fair + take_width   # sell if bid  >= sell_threshold

        if remaining_asks:
            best_ask     = min(remaining_asks)
            best_ask_vol = -remaining_asks[best_ask]
            if not prevent_adverse or best_ask_vol <= adverse_volume:
                if best_ask <= buy_threshold:
                    qty = min(best_ask_vol, limit - (position + bov))
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                        bov += qty
                        # A1: mutate local copy only
                        remaining_asks[best_ask] += qty
                        if remaining_asks[best_ask] == 0:
                            del remaining_asks[best_ask]

        if not buy_only and remaining_bids:
            best_bid     = max(remaining_bids)
            best_bid_vol = remaining_bids[best_bid]
            if not prevent_adverse or best_bid_vol <= adverse_volume:
                if best_bid >= sell_threshold:
                    qty = min(best_bid_vol, limit + (position - sov))
                    if qty > 0:
                        orders.append(Order(product, best_bid, -qty))
                        sov += qty
                        remaining_bids[best_bid] -= qty
                        if remaining_bids[best_bid] == 0:
                            del remaining_bids[best_bid]

        return bov, sov

    # ── Phase 2: Clear ─────────────────────────────────────────────────
    def clear_position_order(
        self, product, fair, width,
        orders: list,
        remaining_bids: dict, remaining_asks: dict,
        position, bov, sov,
    ):
        limit = POS_LIMITS[product]
        pos_after = position + bov - sov
        fair_for_bid = round(fair - width)
        fair_for_ask = round(fair + width)

        buy_qty_cap  = limit - (position + bov)
        sell_qty_cap = limit + (position - sov)

        if pos_after > 0:
            clear_qty = sum(v for p, v in remaining_bids.items() if p >= fair_for_ask)
            clear_qty = min(clear_qty, pos_after)
            sent = min(sell_qty_cap, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent)))
                sov += abs(sent)

        if pos_after < 0:
            clear_qty = sum(abs(v) for p, v in remaining_asks.items() if p <= fair_for_bid)
            clear_qty = min(clear_qty, abs(pos_after))
            sent = min(buy_qty_cap, clear_qty)
            if sent > 0:
                orders.append(Order(product, fair_for_bid, abs(sent)))
                bov += abs(sent)

        return bov, sov

    # ── Phase 3: Make ──────────────────────────────────────────────────
    def make_orders(
        self, product,
        remaining_bids: dict, remaining_asks: dict,
        fair, position, bov, sov,
        disregard_edge, join_edge, default_edge,
        manage_position=False, soft_position_limit=0,
        bid_levels=1, ask_levels=1,
    ):
        orders = []
        limit = POS_LIMITS[product]

        # Best quotes that are beyond disregard_edge from fair
        asks_above = [p for p in remaining_asks if p >  fair + disregard_edge]
        bids_below = [p for p in remaining_bids if p <  fair - disregard_edge]

        best_ask_above = min(asks_above) if asks_above else None
        best_bid_below = max(bids_below) if bids_below else None

        # ── Ask placement (B3: queue priority) ─────────────────────────
        ask = round(fair + default_edge)
        if best_ask_above is not None:
            ask_vol = -remaining_asks[best_ask_above]
            if abs(best_ask_above - fair) <= join_edge:
                # B3: join if queue is thin, penny-improve if thick
                ask = best_ask_above if ask_vol < SMALL_QUEUE_THRESHOLD else best_ask_above - 1
            else:
                ask = best_ask_above - 1

        # ── Bid placement (B3: queue priority) ─────────────────────────
        bid = round(fair - default_edge)
        if best_bid_below is not None:
            bid_vol = remaining_bids[best_bid_below]
            if abs(fair - best_bid_below) <= join_edge:
                # B3: join if queue is thin, penny-improve if thick
                bid = best_bid_below if bid_vol < SMALL_QUEUE_THRESHOLD else best_bid_below + 1
            else:
                bid = best_bid_below + 1

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_qty = limit - (position + bov)
        if buy_qty > 0 and bid_levels > 0:
            base = buy_qty // bid_levels
            for lvl in range(bid_levels):
                q = base if lvl < bid_levels - 1 else buy_qty - base * (bid_levels - 1)
                if q > 0:
                    orders.append(Order(product, round(bid) - lvl, q))

        sell_qty = limit + (position - sov)
        if sell_qty > 0 and ask_levels > 0:
            base = sell_qty // ask_levels
            for lvl in range(ask_levels):
                q = base if lvl < ask_levels - 1 else sell_qty - base * (ask_levels - 1)
                if q > 0:
                    orders.append(Order(product, round(ask) + lvl, -q))

        return orders, bov, sov

    # ── Fair value ─────────────────────────────────────────────────────
    def compute_fair(
        self, product, levels: tuple,
        trader_obj: dict, timestamp: int, position: int,
    ) -> Optional[float]:
        """levels = output of get_book_levels()."""
        p = self.params[product]
        last_fair_key = f'{product}_last_fair'

        best_bid, best_bid_vol, second_bid, best_ask, best_ask_vol, second_ask = levels

        # Broken book: carry forward last fair (+drift for PEPPER)
        if best_bid is None or best_ask is None:
            last_fair = trader_obj.get(last_fair_key)
            if last_fair is None:
                return None
            return last_fair + PEPPER_DRIFT_PER_TICK if p.get('use_drift') else last_fair

        spread = best_ask - best_bid

        # B1: mid price (microprice tested but mid+inv=0.03 was best in sweep)
        total_vol = best_bid_vol + best_ask_vol
        microprice = (best_bid + best_ask) / 2.0

        # A8: EMA with stale + timestamp-gap reset
        ema_key    = f'{product}_ema'
        last_ts_key = f'{product}_last_ts'
        alpha      = p.get('ema_alpha', 0.10)
        prev_ema   = trader_obj.get(ema_key)
        last_ts    = trader_obj.get(last_ts_key, timestamp)
        ts_gap     = timestamp - last_ts

        if (prev_ema is None
                or abs(microprice - prev_ema) > EMA_RESET_THRESHOLD
                or ts_gap > EMA_RESET_TS_GAP):
            ema = microprice
        else:
            ema = alpha * microprice + (1.0 - alpha) * prev_ema

        trader_obj[ema_key]     = ema
        trader_obj[last_ts_key] = timestamp

        residual = microprice - ema

        # Imbalance (L1 only)
        imb = (best_bid_vol - best_ask_vol) / total_vol if total_vol > 0 else 0.0

        mode = p.get('fair_mode', 'ema_reversion')

        # B4: Fair value model — use data-validated coefs, keep signals in tick units
        fair = microprice
        fair -= p.get('reversion_coef', 0.45) * residual
        fair += p.get('imb_coef', 0.0) * imb        # imb in [-1, 1], coef in ticks

        # L2-L1 mid signal (in ticks, same scale as original)
        l2l1_coef = p.get('l2l1_coef', 0.0)
        if l2l1_coef != 0.0 and second_bid is not None and second_ask is not None:
            l1_mid = (best_bid + best_ask) / 2.0
            l2_mid = (second_bid + second_ask) / 2.0
            fair += l2l1_coef * (l2_mid - l1_mid)

        # B4: spread term (mild positive pull when spread is wide)
        fair += p.get('spread_coef', 0.0) * spread

        # PEPPER drift
        if p.get('use_drift'):
            fair += PEPPER_DRIFT_PER_TICK

        # A4/B2: Inventory skew
        inv_penalty = p.get('inventory_penalty', 0.0)
        if inv_penalty != 0.0:
            fair -= inv_penalty * position

        trader_obj[last_fair_key] = fair
        return fair

    # ── L2-L1 direction signal ─────────────────────────────────────────
    def _l2l1_signal(self, second_bid, best_bid, second_ask, best_ask) -> float:
        if second_bid is not None and second_ask is not None:
            return (second_bid + second_ask) / 2.0 - (best_bid + best_ask) / 2.0
        return 0.0

    # ── Main loop ──────────────────────────────────────────────────────
    def run(self, state: TradingState):
        trader_obj: dict = {}
        if state.traderData:
            try:
                trader_obj = json.loads(state.traderData)
            except Exception:
                pass

        result: dict = {}

        for product in [ASH, PEPPER]:
            if product not in state.order_depths:
                continue
            p   = self.params[product]
            pos = state.position.get(product, 0)
            od  = state.order_depths[product]

            # A1: work on local copies — never mutate the exchange snapshot
            remaining_bids = dict(od.buy_orders)
            remaining_asks = dict(od.sell_orders)

            # A2: single sort call for all downstream use
            levels = get_book_levels(remaining_bids, remaining_asks)
            best_bid, best_bid_vol, second_bid, best_ask, best_ask_vol, second_ask = levels

            fair = self.compute_fair(product, levels, trader_obj, state.timestamp, pos)
            if fair is None:
                continue

            orders: List[Order] = []
            bov = sov = 0
            buy_only = p.get('buy_only', False)

            # A5: PEPPER overshoot sell valve
            allow_sell = not buy_only
            if buy_only and pos > p.get('overshoot_sell_pos', 999) and best_bid is not None:
                if best_bid > fair + p.get('overshoot_sell_edge', 15):
                    allow_sell = True   # temporarily lift buy_only restriction

            # Phase 1: Take
            bov, sov = self.take_best_orders(
                product, fair, p['take_width'], orders,
                remaining_bids, remaining_asks, pos, bov, sov,
                prevent_adverse=p.get('prevent_adverse', False),
                adverse_volume=p.get('adverse_volume', 0),
                buy_only=not allow_sell,
            )

            # Phase 2: Clear (skip entirely for buy_only)
            if not buy_only:
                bov, sov = self.clear_position_order(
                    product, fair, p['clear_width'], orders,
                    remaining_bids, remaining_asks, pos, bov, sov,
                )

            # Spread for regime detection (use original levels, not depleted book)
            spread = (best_ask - best_bid) if (best_bid and best_ask) else 16

            # Level selection using pre-computed L2-L1 signal
            base_lvl = p.get('levels', 2)
            bid_lvl = ask_lvl = base_lvl
            if p.get('use_l2l1_signal', False):
                sig = self._l2l1_signal(second_bid, best_bid, second_ask, best_ask)
                if sig > 0:
                    bid_lvl = base_lvl + 1
                    ask_lvl = max(1, base_lvl - 1)
                elif sig < 0:
                    bid_lvl = max(1, base_lvl - 1)
                    ask_lvl = base_lvl + 1

            if buy_only:
                ask_lvl = 0

            # A7: Tight-spread regime — cut to single level, widen edge
            effective_edge = p['default_edge']
            if spread <= TIGHT_SPREAD_THRESHOLD:
                bid_lvl = ask_lvl = 1
                if product == ASH:
                    effective_edge = max(effective_edge, 5)

            # Phase 3: Make
            make_orders, _, _ = self.make_orders(
                product, remaining_bids, remaining_asks,
                fair, pos, bov, sov,
                p['disregard_edge'], p['join_edge'], effective_edge,
                manage_position=True,
                soft_position_limit=p['soft_position_limit'],
                bid_levels=bid_lvl,
                ask_levels=ask_lvl,
            )
            result[product] = orders + make_orders

        # A9: Periodic logging — flush once every LOG_INTERVAL ticks
        if state.timestamp % LOG_INTERVAL == 0:
            logger.flush(state, result)
        else:
            logger.logs = ""

        try:
            trader_data = json.dumps(trader_obj)
        except Exception:
            trader_data = ''

        return result, 0, trader_data
