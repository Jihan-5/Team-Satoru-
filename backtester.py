from datamodel import OrderDepth, TradingState, Order
import json

# ── Product symbols & position limits ────────────────────────────────────────
EMERALD = 'EMERALDS'
TOMATO  = 'TOMATOES'
POS_LIMITS = {EMERALD: 80, TOMATO: 80}

# ── Emerald parameters (Stoikov market maker, σ ≈ 0) ─────────────────────────
EMERALD_FV     = 10000   # known pegged fair value – never moves
EMERALD_OFFSET = 8
EMERALD_SKEW   = 0       # σ≈0 → no reservation shift needed; capacity caps handle inventory

# ── Tomato parameters (regime-adaptive) ──────────────────────────────────────
# Regime classification using fast/slow EMA of mid-price:
#   STRONG TREND   → one-sided market making (only quote in trend direction)
#                    + build directional position toward ±DIR_SZ
#   MILD TREND     → one-sided passive quoting (no aggression)
#   NEUTRAL / FLAT → symmetric market making at best bid/ask (captures micro-reversals)
#   VOLUME IMBALANCE → small scalp bet even in neutral regime
#
# CRITICAL: when already AT target directional position → HOLD (zero new orders).
# Churning (posting both sides when at-target) creates massive slippage losses.
TOMATO_EMA_FAST    = 9
TOMATO_EMA_SLOW    = 16
TOMATO_STRONG_THR  = 3.5  # |fast − slow| to enter strong-trend mode (Day -1 style)
TOMATO_MILD_THR    = 0.05
TOMATO_VOL_THR     = 3.0  # bid_L1_vol / ask_L1_vol for micro imbalance signal
TOMATO_WARMUP      = 20
TOMATO_DIR_SZ      = 65   # target position in strong trend
TOMATO_SCALP_SZ    = 12   # target position on volume-imbalance only
TOMATO_CHUNK       = 25   # max contracts to sweep per timestamp (limits slippage)

LONG, SHORT, NEUTRAL = 1, -1, 0


def ema_step(prev: float, val: float, n: int) -> float:
    """Standard exponential moving average update."""
    a = 2.0 / (n + 1)
    return a * val + (1.0 - a) * prev


# ─────────────────────────────────────────────────────────────────────────────
# EmeraldTrader
# Stoikov-Avellaneda market maker for the perfectly pegged EMERALDS.
#
# σ = 0  →  reservation price r = FV = 10000 always.
# We still apply an inventory skew so we rebalance faster and never pin
# ourselves against the position limit for long stretches.
#
# Architecture:
#   1. TAKE  – capture any rogue orders crossing FV (free edge);
#              also close excess inventory at FV when at limit.
#   2. MAKE  – post bid at (FV − offset − skew) and ask at (FV + offset − skew)
#              with all remaining capacity.
# ─────────────────────────────────────────────────────────────────────────────
class EmeraldTrader:

    def __init__(self, state: TradingState, prints: dict, new_td: dict):
        self.state   = state
        self.prints  = prints
        self.new_td  = new_td
        self.last_td = self._load_td()

        self.lim  = POS_LIMITS[EMERALD]
        self.pos  = state.position.get(EMERALD, 0)
        self.orders: list[Order] = []

        od = state.order_depths.get(EMERALD, OrderDepth())
        # normalise to positive volumes
        self.bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  reverse=True)}
        self.asks = {p: abs(v) for p, v in sorted(od.sell_orders.items())}

        self.cap_buy  = self.lim - self.pos   # remaining buy capacity
        self.cap_sell = self.lim + self.pos   # remaining sell capacity

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_td(self) -> dict:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except:
            pass
        return {}

    def _buy(self, price: int, vol: int):
        vol = min(int(abs(vol)), self.cap_buy)
        if vol > 0:
            self.orders.append(Order(EMERALD, int(price), vol))
            self.cap_buy -= vol

    def _sell(self, price: int, vol: int):
        vol = min(int(abs(vol)), self.cap_sell)
        if vol > 0:
            self.orders.append(Order(EMERALD, int(price), -vol))
            self.cap_sell -= vol

    # ── main logic ────────────────────────────────────────────────────────────

    def get_orders(self) -> dict:

        # Stoikov reservation price skew (σ≈0, so shift is purely inventory-driven)
        skew = round(self.pos * EMERALD_SKEW)
        bid_q = EMERALD_FV - EMERALD_OFFSET - skew   # e.g. 9993 when flat
        ask_q = EMERALD_FV + EMERALD_OFFSET - skew   # e.g. 10007 when flat

        # ── 1. TAKING ─────────────────────────────────────────────────────────
        # Buy any ask that crosses below FV (never observed in normal conditions,
        # but fires on the rare rogue ask sitting at 10000 when we are short).
        for ap, av in self.asks.items():
            if ap > EMERALD_FV:
                break  # nothing cheap left
            if ap < EMERALD_FV:
                # Strictly below FV: free money – take everything
                self._buy(ap, av)
            elif ap == EMERALD_FV:
                # At FV: only take if short (faster path to flat than waiting for 9993)
                if self.pos < 0:
                    self._buy(ap, min(av, abs(self.pos)))

        # Sell into any bid that crosses above FV.
        for bp, bv in self.bids.items():
            if bp < EMERALD_FV:
                break  # nothing expensive left
            if bp > EMERALD_FV:
                self._sell(bp, bv)
            elif bp == EMERALD_FV:
                # At FV: only sell if long (faster rebalancing)
                if self.pos > 0:
                    self._sell(bp, min(bv, self.pos))

        # ── 2. MAKING ─────────────────────────────────────────────────────────
        self._buy (bid_q, self.cap_buy)
        self._sell(ask_q, self.cap_sell)

        self.prints[EMERALD] = {
            'POS': self.pos, 'SKEW': skew,
            'BID_Q': bid_q,  'ASK_Q': ask_q,
        }
        return {EMERALD: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
# TomatoTrader
# Hybrid trend-follower + market maker for volatile TOMATOES.
#
# Key insight (Stoikov extended with drift):
#   In a trending market, adverse selection destroys market-maker P&L.
#   The optimal response is ONE-SIDED quoting:
#     • Downtrend → only offer (ask), accumulate short
#     • Uptrend   → only bid,        accumulate long
#     • Flat      → symmetric quotes at best bid/ask
#
# Trend is identified via fast/slow EMA crossover of mid-price.
# L1 volume imbalance provides a faster micro-signal between EMA updates.
#
# Execution limits order size per timestamp (TOMATO_CHUNK) to avoid
# sweeping multiple order-book levels and suffering excess slippage.
# ─────────────────────────────────────────────────────────────────────────────
class TomatoTrader:

    def __init__(self, state: TradingState, prints: dict, new_td: dict):
        self.state   = state
        self.prints  = prints
        self.new_td  = new_td
        self.last_td = self._load_td()

        self.lim  = POS_LIMITS[TOMATO]
        self.pos  = state.position.get(TOMATO, 0)
        self.orders: list[Order] = []

        od = state.order_depths.get(TOMATO, OrderDepth())
        self.bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  reverse=True)}
        self.asks = {p: abs(v) for p, v in sorted(od.sell_orders.items())}

        self.cap_buy  = self.lim - self.pos
        self.cap_sell = self.lim + self.pos

        self.best_bid = max(self.bids.keys()) if self.bids else None
        self.best_ask = min(self.asks.keys()) if self.asks else None
        self.bid_wall = min(self.bids.keys()) if self.bids else None  # L3 bid (deepest)
        self.ask_wall = max(self.asks.keys()) if self.asks else None  # L3 ask (deepest)
        self.best_bid_vol = self.bids.get(self.best_bid, 0) if self.best_bid else 0
        self.best_ask_vol = self.asks.get(self.best_ask, 0) if self.best_ask else 0
        self.mid = (self.best_bid + self.best_ask) / 2 if (self.best_bid and self.best_ask) else None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_td(self) -> dict:
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except:
            pass
        return {}

    def _post_bid(self, price: int, vol: int):
        """Passive limit order – no CHUNK cap (we want full queue depth)."""
        vol = min(int(abs(vol)), self.cap_buy)
        if vol > 0:
            self.orders.append(Order(TOMATO, int(price), vol))
            self.cap_buy -= vol

    def _post_ask(self, price: int, vol: int):
        """Passive limit order – no CHUNK cap."""
        vol = min(int(abs(vol)), self.cap_sell)
        if vol > 0:
            self.orders.append(Order(TOMATO, int(price), -vol))
            self.cap_sell -= vol

    def _buy(self, price: int, vol: int):
        """Aggressive sweep – capped at TOMATO_CHUNK to limit slippage."""
        vol = min(int(abs(vol)), self.cap_buy, TOMATO_CHUNK)
        if vol > 0:
            self.orders.append(Order(TOMATO, int(price), vol))
            self.cap_buy -= vol

    def _sell(self, price: int, vol: int):
        """Aggressive sweep – capped at TOMATO_CHUNK to limit slippage."""
        vol = min(int(abs(vol)), self.cap_sell, TOMATO_CHUNK)
        if vol > 0:
            self.orders.append(Order(TOMATO, int(price), -vol))
            self.cap_sell -= vol

    # ── EMA update ────────────────────────────────────────────────────────────

    def _update_emas(self):
        n    = self.last_td.get('t_n', 0)
        fast = self.last_td.get('t_fast', self.mid)
        slow = self.last_td.get('t_slow', self.mid)

        if self.mid is not None and fast is not None and slow is not None:
            fast = ema_step(fast, self.mid, TOMATO_EMA_FAST)
            slow = ema_step(slow, self.mid, TOMATO_EMA_SLOW)
            n   += 1
        elif self.mid is not None:
            fast = self.mid
            slow = self.mid
            n   += 1

        self.new_td['t_fast'] = fast
        self.new_td['t_slow'] = slow
        self.new_td['t_n']    = n
        self.new_td['t_mid']  = self.mid   # persist for 1-tick momentum signal
        return fast, slow, n

    # ── main logic ────────────────────────────────────────────────────────────

    def get_orders(self) -> dict:

        if self.best_bid is None or self.best_ask is None:
            return {TOMATO: self.orders}

        fast_ema, slow_ema, n_ticks = self._update_emas()

        # ── EMA regime classification ─────────────────────────────────────────
        # Test results (2-day combined Tomatoes P&L):
        #   L1 bid / L1 ask (symmetric):    17,836  ← best for neutral / uptrend
        #   L2 bid / L1 ask (asymmetric):   18,623  ← best for downtrend (day -1 10,855)
        #   L1 bid / L2 ask (asymmetric):   12,035  ← worse overall
        #
        # EMA-adaptive strategy:
        #   Bearish (fast < slow − MILD_THR) → L2 bid / L1 ask
        #     Rationale: reduces long accumulation in downtrend (fewer fills at L2),
        #     while L1 ask captures bounces instantly. Day-1 result: ~10,855.
        #   Bullish or neutral → L1 bid / L1 ask (pure symmetric)
        #     Rationale: higher fill rate on both sides in uptrend / quiet market.
        #     Day-2 result: 9,654.  Combined expected: >20,000 for Tomatoes.
        last_mid = self.last_td.get('t_mid', self.mid)
        momentum_bearish = (
            last_mid is not None and self.mid is not None
            and self.mid < last_mid
        )
        bearish = (
            momentum_bearish
            or (fast_ema is not None and slow_ema is not None
                and n_ticks >= TOMATO_WARMUP
                and (fast_ema - slow_ema) < -TOMATO_MILD_THR)
        )

        if self.bid_wall is not None and self.ask_wall is not None:
            if bearish:
                # Downtrend: L2 bid (avoid over-buying dips) + L1 ask (sell bounces fast)
                self._post_bid(self.bid_wall + 1, self.cap_buy)
                self._post_ask(self.best_ask,     self.cap_sell)
            else:
                # Neutral / bullish: symmetric L1 market making — maximum fill rate
                self._post_bid(self.best_bid, self.cap_buy)
                self._post_ask(self.best_ask, self.cap_sell)

        self.prints[TOMATO] = {
            'POS': self.pos, 'BEAR': bearish,
            'FAST': round(fast_ema, 2) if fast_ema else None,
            'SLOW': round(slow_ema, 2) if slow_ema else None,
            'N': n_ticks,
        }
        return {TOMATO: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
# Main Trader
# ─────────────────────────────────────────────────────────────────────────────
class Trader:

    def run(self, state: TradingState):

        new_td  = {}
        prints  = {'GENERAL': {'TS': state.timestamp, 'POS': state.position}}
        result  = {}

        for cls in (EmeraldTrader, TomatoTrader):
            try:
                t = cls(state, prints, new_td)
                result.update(t.get_orders())
            except:
                pass

        try:
            trader_data = json.dumps(new_td)
        except:
            trader_data = ''

        try:
            print(json.dumps(prints))
        except:
            pass

        return result, 0, trader_data
