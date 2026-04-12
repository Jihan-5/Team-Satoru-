"""Round 0 GLFT strategy -- EMERALDS (pegged) + TOMATOES (drifting).

Strategy that scored 1,569 live (submission 82831).
GLFT = Gueant-Lehalle-Fernandez-Tapia (2012) closed-form market-making quotes.
"""
from datamodel import OrderDepth, TradingState, Order
import json
import math
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# IMC Prosperity Visualizer-compatible logger
# ─────────────────────────────────────────────────────────────────────────────
class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict) -> None:
        if self.logs:
            print(self.logs, end="")
        self.logs = ""



logger = Logger()

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

EMERALD = 'EMERALDS'
TOMATO = 'TOMATOES'
POS_LIMITS = {EMERALD: 80, TOMATO: 80}

# -- EMERALDS (pegged-FV GLFT) -----------------------------------------------
EMERALD_FV = 10000
EMERALD_GAMMA = 0.30
EMERALD_SIGMA = 1.2
EMERALD_A = 0.9
EMERALD_K = 0.08
EMERALD_MIN_EDGE = 4
EMERALD_MAX_EDGE = 8
EMERALD_QUEUE_IMPROVE = True

# -- TOMATOES (drift-aware GLFT) ---------------------------------------------
TOMATO_GAMMA = 0.08
TOMATO_SIGMA_WINDOW = 30
TOMATO_SIGMA_MIN = 1.5
TOMATO_A = 0.5
TOMATO_K = 0.20
TOMATO_DRIFT_WINDOW = 16
TOMATO_DRIFT_SCALE = 1.5
TOMATO_MIN_EDGE = 2
TOMATO_MAX_EDGE = 20
TOMATO_WARMUP = 25


# ─────────────────────────────────────────────────────────────────────────────
# GLFT math
# ─────────────────────────────────────────────────────────────────────────────
def glft_quotes(gamma: float, sigma: float, A: float, k: float,
                inventory: int, inventory_limit: int):
    base = (1.0 / gamma) * math.log(1.0 + gamma / k)

    q_norm = inventory / max(inventory_limit, 1) * 10

    inv_scale = math.sqrt((sigma ** 2 * gamma) / (2.0 * k * A))
    power_term = (1.0 + gamma / k) ** (1.0 + k / gamma)
    skew_magnitude = inv_scale * power_term

    delta_bid = base + ((2 * q_norm + 1) / 2.0) * skew_magnitude
    delta_ask = base - ((2 * q_norm - 1) / 2.0) * skew_magnitude

    delta_bid = max(0.5, delta_bid)
    delta_ask = max(0.5, delta_ask)
    return delta_bid, delta_ask


def ema_step(prev: float, val: float, n: int) -> float:
    a = 2.0 / (n + 1)
    return a * val + (1.0 - a) * prev


# ─────────────────────────────────────────────────────────────────────────────
# EMERALDS
# ─────────────────────────────────────────────────────────────────────────────
class EmeraldTrader:
    def __init__(self, state: TradingState, prints: dict, new_td: dict):
        self.state = state
        self.prints = prints
        self.new_td = new_td
        self.last_td = self._load_td()

        self.lim = POS_LIMITS[EMERALD]
        self.pos = state.position.get(EMERALD, 0)
        self.orders: list = []

        od = state.order_depths.get(EMERALD, OrderDepth())
        self.bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(), reverse=True)}
        self.asks = {p: abs(v) for p, v in sorted(od.sell_orders.items())}

        self.cap_buy = self.lim - self.pos
        self.cap_sell = self.lim + self.pos

        self.best_bid = max(self.bids.keys()) if self.bids else None
        self.best_ask = min(self.asks.keys()) if self.asks else None

    def _load_td(self):
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception:
            pass
        return {}

    def _buy(self, price, vol):
        vol = min(int(abs(vol)), self.cap_buy)
        if vol > 0:
            self.orders.append(Order(EMERALD, int(price), vol))
            self.cap_buy -= vol

    def _sell(self, price, vol):
        vol = min(int(abs(vol)), self.cap_sell)
        if vol > 0:
            self.orders.append(Order(EMERALD, int(price), -vol))
            self.cap_sell -= vol

    def get_orders(self):
        # PHASE 1: TAKE free edge (rogue orders crossing FV)
        for ap, av in self.asks.items():
            if ap > EMERALD_FV:
                break
            if ap < EMERALD_FV:
                self._buy(ap, av)
            elif ap == EMERALD_FV and self.pos < 0:
                self._buy(ap, min(av, abs(self.pos)))

        for bp, bv in self.bids.items():
            if bp < EMERALD_FV:
                break
            if bp > EMERALD_FV:
                self._sell(bp, bv)
            elif bp == EMERALD_FV and self.pos > 0:
                self._sell(bp, min(bv, self.pos))

        # PHASE 2: POST GLFT quotes around FV
        delta_bid, delta_ask = glft_quotes(
            gamma=EMERALD_GAMMA,
            sigma=EMERALD_SIGMA,
            A=EMERALD_A,
            k=EMERALD_K,
            inventory=self.pos,
            inventory_limit=self.lim,
        )

        bid_offset = max(EMERALD_MIN_EDGE, min(EMERALD_MAX_EDGE, round(delta_bid)))
        ask_offset = max(EMERALD_MIN_EDGE, min(EMERALD_MAX_EDGE, round(delta_ask)))

        bid_price = EMERALD_FV - bid_offset
        ask_price = EMERALD_FV + ask_offset

        # PHASE 3: Queue-improvement front-running
        if EMERALD_QUEUE_IMPROVE:
            if self.best_bid is not None and self.best_bid >= bid_price:
                improved = self.best_bid + 1
                if improved <= EMERALD_FV - EMERALD_MIN_EDGE:
                    bid_price = improved
            if self.best_ask is not None and self.best_ask <= ask_price:
                improved = self.best_ask - 1
                if improved >= EMERALD_FV + EMERALD_MIN_EDGE:
                    ask_price = improved

        self._buy(bid_price, self.cap_buy)
        self._sell(ask_price, self.cap_sell)

        mid = ((self.best_bid + self.best_ask) / 2
               if (self.best_bid and self.best_ask) else EMERALD_FV)
        logger.print(
            f"[EM] ts={self.state.timestamp}"
            f" pos={self.pos}"
            f" mid={mid:.1f}"
            f" bid={bid_price}(d={round(mid - bid_price, 1)})"
            f" ask={ask_price}(d={round(ask_price - mid, 1)})"
            f" glft_db={round(delta_bid, 2)}"
            f" glft_da={round(delta_ask, 2)}"
        )

        return {EMERALD: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
# TOMATOES
# ─────────────────────────────────────────────────────────────────────────────
class TomatoTrader:
    def __init__(self, state: TradingState, prints: dict, new_td: dict):
        self.state = state
        self.prints = prints
        self.new_td = new_td
        self.last_td = self._load_td()

        self.lim = POS_LIMITS[TOMATO]
        self.pos = state.position.get(TOMATO, 0)
        self.orders: list = []

        od = state.order_depths.get(TOMATO, OrderDepth())
        self.bids = {p: abs(v) for p, v in sorted(od.buy_orders.items(), reverse=True)}
        self.asks = {p: abs(v) for p, v in sorted(od.sell_orders.items())}

        self.cap_buy = self.lim - self.pos
        self.cap_sell = self.lim + self.pos

        self.best_bid = max(self.bids.keys()) if self.bids else None
        self.best_ask = min(self.asks.keys()) if self.asks else None
        self.mid = ((self.best_bid + self.best_ask) / 2
                    if (self.best_bid and self.best_ask) else None)

    def _load_td(self):
        try:
            if self.state.traderData:
                return json.loads(self.state.traderData)
        except Exception:
            pass
        return {}

    def _buy(self, price, vol):
        vol = min(int(abs(vol)), self.cap_buy)
        if vol > 0:
            self.orders.append(Order(TOMATO, int(price), vol))
            self.cap_buy -= vol

    def _sell(self, price, vol):
        vol = min(int(abs(vol)), self.cap_sell)
        if vol > 0:
            self.orders.append(Order(TOMATO, int(price), -vol))
            self.cap_sell -= vol

    def _update_state(self):
        n = self.last_td.get('t_n', 0)
        mid_ema_fast = self.last_td.get('t_mid_fast', self.mid)
        mid_ema_slow = self.last_td.get('t_mid_slow', self.mid)
        sigma_sq = self.last_td.get('t_sigma_sq', TOMATO_SIGMA_MIN ** 2)
        last_mid = self.last_td.get('t_last_mid', self.mid)

        if self.mid is None:
            return mid_ema_fast, mid_ema_slow, sigma_sq, n

        if mid_ema_fast is None:
            mid_ema_fast = self.mid
            mid_ema_slow = self.mid
        else:
            mid_ema_fast = ema_step(mid_ema_fast, self.mid, TOMATO_DRIFT_WINDOW)
            mid_ema_slow = ema_step(mid_ema_slow, self.mid, TOMATO_DRIFT_WINDOW * 3)

        if last_mid is not None:
            ret = self.mid - last_mid
            sigma_sq = ema_step(sigma_sq, ret * ret, TOMATO_SIGMA_WINDOW)

        n += 1

        self.new_td['t_n'] = n
        self.new_td['t_mid_fast'] = mid_ema_fast
        self.new_td['t_mid_slow'] = mid_ema_slow
        self.new_td['t_sigma_sq'] = sigma_sq
        self.new_td['t_last_mid'] = self.mid

        return mid_ema_fast, mid_ema_slow, sigma_sq, n

    def get_orders(self):
        if self.best_bid is None or self.best_ask is None or self.mid is None:
            return {TOMATO: self.orders}

        mid_fast, mid_slow, sigma_sq, n_ticks = self._update_state()

        if n_ticks < TOMATO_WARMUP:
            logger.print(f"[TM] ts={self.state.timestamp} WARMUP {n_ticks}/{TOMATO_WARMUP}")
            return {TOMATO: self.orders}

        drift = (mid_fast - mid_slow) if (mid_fast and mid_slow) else 0.0
        drift *= TOMATO_DRIFT_SCALE

        sigma = max(math.sqrt(max(sigma_sq, TOMATO_SIGMA_MIN ** 2)), TOMATO_SIGMA_MIN)

        delta_bid, delta_ask = glft_quotes(
            gamma=TOMATO_GAMMA,
            sigma=sigma,
            A=TOMATO_A,
            k=TOMATO_K,
            inventory=self.pos,
            inventory_limit=self.lim,
        )

        ref_price = self.mid + drift

        bid_offset = max(TOMATO_MIN_EDGE, min(TOMATO_MAX_EDGE, round(delta_bid)))
        ask_offset = max(TOMATO_MIN_EDGE, min(TOMATO_MAX_EDGE, round(delta_ask)))

        bid_price = int(round(ref_price - bid_offset))
        ask_price = int(round(ref_price + ask_offset))

        if bid_price >= ask_price:
            bid_price = int(math.floor(ref_price)) - 1
            ask_price = int(math.ceil(ref_price)) + 1

        bid_price = min(bid_price, self.best_ask - 1)
        ask_price = max(ask_price, self.best_bid + 1)

        self._buy(bid_price, self.cap_buy)
        self._sell(ask_price, self.cap_sell)

        ema_diff = round(mid_fast - mid_slow, 3) if (mid_fast and mid_slow) else 0.0
        trend = ("UP" if ema_diff > 0.05 else "DOWN" if ema_diff < -0.05 else "FLAT")
        logger.print(
            f"[TM] ts={self.state.timestamp}"
            f" pos={self.pos}"
            f" mid={round(self.mid, 1)}"
            f" ref={round(ref_price, 1)}"
            f" drift={round(drift, 3)}"
            f" sigma={round(sigma, 2)}"
            f" ema_diff={ema_diff}"
            f" trend={trend}"
            f" bid={bid_price}(d={round(self.mid - bid_price, 1)})"
            f" ask={ask_price}(d={round(ask_price - self.mid, 1)})"
            f" glft_db={round(delta_bid, 2)}"
            f" glft_da={round(delta_ask, 2)}"
        )

        return {TOMATO: self.orders}


# ─────────────────────────────────────────────────────────────────────────────
# Main Trader
# ─────────────────────────────────────────────────────────────────────────────
class Trader:
    def run(self, state: TradingState):
        new_td = {}
        prints = {}
        result = {}

        for cls in (EmeraldTrader, TomatoTrader):
            try:
                t = cls(state, prints, new_td)
                result.update(t.get_orders())
            except Exception:
                import traceback
                logger.print(f"[ERROR] {traceback.format_exc()}")

        try:
            trader_data = json.dumps(new_td)
        except Exception:
            trader_data = ''

        try:
            logger.flush(state, result)
        except Exception:
            pass
        return result, 0, trader_data
