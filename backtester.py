from datamodel import OrderDepth, TradingState, Order
import json

####### GENERAL ####### GENERAL ####### GENERAL ####### GENERAL ####### GENERAL

STATIC_SYMBOL  = 'EMERALDS'   # stable fair-value product  (~9992)
DYNAMIC_SYMBOL = 'TOMATOES'   # volatile product            (~5000)

POS_LIMITS = {
    STATIC_SYMBOL:  80,
    DYNAMIC_SYMBOL: 80,
}

LONG, NEUTRAL, SHORT = 1, 0, -1

INFORMED_TRADER_ID = 'Olivia'


# ──────────────────────────────────────────────────────────────────────────────
# Base ProductTrader
# ──────────────────────────────────────────────────────────────────────────────

class ProductTrader:

    def __init__(self, name, state, prints, new_trader_data, product_group=None):

        self.orders = []

        self.name = name
        self.state = state
        self.prints = prints
        self.new_trader_data = new_trader_data
        self.product_group = name if product_group is None else product_group

        self.last_traderData = self._load_last_trader_data()

        self.position_limit    = POS_LIMITS.get(self.name, 0)
        self.initial_position  = self.state.position.get(self.name, 0)
        self.expected_position = self.initial_position

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self._get_walls()
        self.best_bid, self.best_ask = self._get_best_bid_ask()

        self.max_allowed_buy_volume, self.max_allowed_sell_volume = self._get_max_allowed_volume()
        self.total_mkt_buy_volume, self.total_mkt_sell_volume = self._get_total_market_volume()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _load_last_trader_data(self):
        try:
            if self.state.traderData != '':
                return json.loads(self.state.traderData)
        except:
            self.log("ERROR", 'td')
        return {}

    def _get_best_bid_ask(self):
        best_bid = best_ask = None
        try:
            if self.mkt_buy_orders:
                best_bid = max(self.mkt_buy_orders.keys())
            if self.mkt_sell_orders:
                best_ask = min(self.mkt_sell_orders.keys())
        except:
            pass
        return best_bid, best_ask

    def _get_walls(self):
        bid_wall = wall_mid = ask_wall = None
        try: bid_wall = min(self.mkt_buy_orders.keys())
        except: pass
        try: ask_wall = max(self.mkt_sell_orders.keys())
        except: pass
        try: wall_mid = (bid_wall + ask_wall) / 2
        except: pass
        return bid_wall, wall_mid, ask_wall

    def _get_total_market_volume(self):
        try:
            buy_vol  = sum(self.mkt_buy_orders.values())
            sell_vol = sum(self.mkt_sell_orders.values())
            return buy_vol, sell_vol
        except:
            return 0, 0

    def _get_max_allowed_volume(self):
        return (
            self.position_limit - self.initial_position,
            self.position_limit + self.initial_position,
        )

    def _get_order_depth(self):
        buy_orders = sell_orders = {}
        try:
            od: OrderDepth = self.state.order_depths[self.name]
            buy_orders  = {p: abs(v) for p, v in sorted(od.buy_orders.items(),  key=lambda x: x[0], reverse=True)}
            sell_orders = {p: abs(v) for p, v in sorted(od.sell_orders.items(), key=lambda x: x[0])}
        except:
            pass
        return buy_orders, sell_orders

    # ── order helpers ─────────────────────────────────────────────────────────

    def bid(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        order = Order(self.name, int(price), abs_volume)
        if logging:
            self.log("BUYO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_buy_volume -= abs_volume
        self.orders.append(order)

    def ask(self, price, volume, logging=True):
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        order = Order(self.name, int(price), -abs_volume)
        if logging:
            self.log("SELLO", {"p": price, "s": self.name, "v": int(volume)}, product_group='ORDERS')
        self.max_allowed_sell_volume -= abs_volume
        self.orders.append(order)

    def log(self, kind, message, product_group=None):
        if product_group is None:
            product_group = self.product_group
        if product_group == 'ORDERS':
            group = self.prints.get(product_group, [])
            group.append({kind: message})
        else:
            group = self.prints.get(product_group, {})
            group[kind] = message
        self.prints[product_group] = group

    # ── informed trader detection ─────────────────────────────────────────────

    def check_for_informed(self):
        informed_bought_ts, informed_sold_ts = self.last_traderData.get(self.name, [None, None])

        trades = (
            self.state.market_trades.get(self.name, []) +
            self.state.own_trades.get(self.name, [])
        )
        for trade in trades:
            if trade.buyer  == INFORMED_TRADER_ID:
                informed_bought_ts = trade.timestamp
            if trade.seller == INFORMED_TRADER_ID:
                informed_sold_ts   = trade.timestamp

        self.new_trader_data[self.name] = [informed_bought_ts, informed_sold_ts]

        if not informed_bought_ts and not informed_sold_ts:
            direction = NEUTRAL
        elif not informed_bought_ts:
            direction = SHORT
        elif not informed_sold_ts:
            direction = LONG
        elif informed_sold_ts > informed_bought_ts:
            direction = SHORT
        elif informed_sold_ts < informed_bought_ts:
            direction = LONG
        else:
            direction = NEUTRAL

        self.log('TD', self.new_trader_data[self.name])
        self.log('ID', direction)
        return direction, informed_bought_ts, informed_sold_ts

    def get_orders(self):
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# StaticTrader  —  EMERALDS
# Wall-based market making: take anything mispriced vs wall_mid, then post
# tight quotes just inside the walls.
# ──────────────────────────────────────────────────────────────────────────────

class StaticTrader(ProductTrader):

    def __init__(self, state, prints, new_trader_data):
        super().__init__(STATIC_SYMBOL, state, prints, new_trader_data)

    def get_orders(self):

        if self.wall_mid is None:
            return {self.name: self.orders}

        # ── 1. TAKING ────────────────────────────────────────────────────────
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= self.wall_mid - 1:
                self.bid(sp, sv, logging=False)
            elif sp <= self.wall_mid and self.initial_position < 0:
                self.bid(sp, min(sv, abs(self.initial_position)), logging=False)

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= self.wall_mid + 1:
                self.ask(bp, bv, logging=False)
            elif bp >= self.wall_mid and self.initial_position > 0:
                self.ask(bp, min(bv, self.initial_position), logging=False)

        # ── 2. MAKING ────────────────────────────────────────────────────────
        bid_price = int(self.bid_wall + 1)
        ask_price = int(self.ask_wall - 1)

        for bp, bv in self.mkt_buy_orders.items():
            overbid = bp + 1
            if bv > 1 and overbid < self.wall_mid:
                bid_price = max(bid_price, overbid)
                break
            elif bp < self.wall_mid:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in self.mkt_sell_orders.items():
            underask = sp - 1
            if sv > 1 and underask > self.wall_mid:
                ask_price = min(ask_price, underask)
                break
            elif sp > self.wall_mid:
                ask_price = min(ask_price, sp)
                break

        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)

        return {self.name: self.orders}


# ──────────────────────────────────────────────────────────────────────────────
# DynamicTrader  —  TOMATOES
# Informed-trader-aware market making.  Without buyer/seller data in Round 0
# the informed detection stays NEUTRAL, giving plain tight-spread market making.
# ──────────────────────────────────────────────────────────────────────────────

class DynamicTrader(ProductTrader):

    def __init__(self, state, prints, new_trader_data):
        super().__init__(DYNAMIC_SYMBOL, state, prints, new_trader_data)
        self.informed_direction, self.informed_bought_ts, self.informed_sold_ts = self.check_for_informed()

    def get_orders(self):

        if self.wall_mid is None:
            return {self.name: self.orders}

        # ── BID ───────────────────────────────────────────────────────────────
        bid_price  = self.bid_wall + 1
        bid_volume = self.max_allowed_buy_volume

        if (self.informed_bought_ts is not None and
                self.informed_bought_ts + 500 >= self.state.timestamp):
            # Informed buyer recently active — chase ask wall to build long
            if self.initial_position < self.position_limit * 0.8:
                bid_price  = self.ask_wall
                bid_volume = int(self.position_limit * 0.8) - self.initial_position
        else:
            if (self.wall_mid - bid_price < 1 and
                    self.informed_direction == SHORT and
                    self.initial_position > -self.position_limit * 0.8):
                bid_price = self.bid_wall

        self.bid(bid_price, bid_volume)

        # ── ASK ───────────────────────────────────────────────────────────────
        ask_price  = self.ask_wall - 1
        ask_volume = self.max_allowed_sell_volume

        if (self.informed_sold_ts is not None and
                self.informed_sold_ts + 500 >= self.state.timestamp):
            # Informed seller recently active — chase bid wall to build short
            if self.initial_position > -self.position_limit * 0.8:
                ask_price  = self.bid_wall
                ask_volume = self.initial_position + int(self.position_limit * 0.8)

        if (ask_price - self.wall_mid < 1 and
                self.informed_direction == LONG and
                self.initial_position < self.position_limit * 0.8):
            ask_price = self.ask_wall

        self.ask(ask_price, ask_volume)

        return {self.name: self.orders}


# ──────────────────────────────────────────────────────────────────────────────
# Main Trader
# ──────────────────────────────────────────────────────────────────────────────

class Trader:

    def run(self, state: TradingState):

        new_trader_data = {}
        prints = {
            "GENERAL": {
                "TIMESTAMP": state.timestamp,
                "POSITIONS": state.position,
            },
        }

        def export(data):
            try:
                print(json.dumps(data))
            except:
                pass

        product_traders = {
            STATIC_SYMBOL:  StaticTrader,
            DYNAMIC_SYMBOL: DynamicTrader,
        }

        result = {}
        for symbol, trader_cls in product_traders.items():
            if symbol in state.order_depths:
                try:
                    trader = trader_cls(state, prints, new_trader_data)
                    result.update(trader.get_orders())
                except:
                    pass

        try:
            final_trader_data = json.dumps(new_trader_data)
        except:
            final_trader_data = ''

        export(prints)
        return result, 0, final_trader_data
