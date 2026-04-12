"""Realistic backtester runner using FillEngine v3."""
import importlib.util
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional

from datamodel import TradingState, Observation
from fill_engine import FillEngine, FillConfig, Fill, liquidation_value
from market_data import load_prices, load_trades, get_symbols


POS_LIMITS_DEFAULT = {
    'EMERALDS': 80,
    'TOMATOES': 80,
    'RAINFOREST_RESIN': 50,
    'KELP': 50,
    'SQUID_INK': 50,
}


@dataclass
class BacktestResult:
    fills: list = field(default_factory=list)
    pnl_realized: dict = field(default_factory=dict)
    pnl_liquidation: dict = field(default_factory=dict)
    pnl_total: float = 0.0
    positions_over_time: list = field(default_factory=list)
    pnl_over_time: list = field(default_factory=list)
    drawdown: float = 0.0
    num_fills: int = 0
    adverse_fills: int = 0
    take_fills: int = 0
    passive_fills: int = 0
    errors: list = field(default_factory=list)
    config: Optional[FillConfig] = None
    final_positions: dict = field(default_factory=dict)


def load_trader(path: str, param_overrides: Optional[dict] = None):
    spec = importlib.util.spec_from_file_location("user_trader", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["user_trader"] = mod
    spec.loader.exec_module(mod)
    if param_overrides:
        for k, v in param_overrides.items():
            if hasattr(mod, k):
                setattr(mod, k, v)
    return mod.Trader()


def run_backtest(trader_path, prices_csv, trades_csv,
                 config=None, position_limits=None, seed=42, verbose=False,
                 param_overrides=None):
    if config is None:
        config = FillConfig()
    if position_limits is None:
        position_limits = POS_LIMITS_DEFAULT

    import random
    rng = random.Random(seed)

    snapshots, timestamps = load_prices(prices_csv)
    market_trades_map = load_trades(trades_csv)
    symbols = get_symbols(snapshots)

    trader = load_trader(trader_path, param_overrides=param_overrides)
    engine = FillEngine(config, position_limits, rng=rng)

    positions = {s: 0 for s in symbols}
    cash = {s: 0.0 for s in symbols}
    trader_data = ""
    own_trades_by_sym = {s: [] for s in symbols}

    result = BacktestResult(config=config)

    mid_series = {s: {} for s in symbols}
    for (ts, sym), od in snapshots.items():
        if od.buy_orders and od.sell_orders:
            bb = max(od.buy_orders.keys())
            ba = min(od.sell_orders.keys())
            mid_series[sym][ts] = (bb + ba) / 2

    for i, ts in enumerate(timestamps):
        order_depths = {}
        mkt_trades = {}
        for sym in symbols:
            if (ts, sym) in snapshots:
                order_depths[sym] = snapshots[(ts, sym)]
            mkt_trades[sym] = market_trades_map.get((ts, sym), [])

        state = TradingState(
            timestamp=ts,
            traderData=trader_data,
            listings={},
            order_depths=order_depths,
            own_trades={s: own_trades_by_sym[s][-5:] for s in symbols},
            market_trades=mkt_trades,
            position=dict(positions),
            observations=Observation({}, {}),
        )

        try:
            orders_dict, conversions, trader_data = trader.run(state)
            if not isinstance(trader_data, str):
                trader_data = ""
        except Exception:
            err = traceback.format_exc()
            result.errors.append((ts, err))
            if verbose:
                print(f"[ERROR @ ts={ts}] {err}")
            orders_dict = {}

        for sym, orders in orders_dict.items():
            for o in orders:
                if o.quantity != 0:
                    engine.submit(ts, sym, o.price, o.quantity)

        next_ts = timestamps[i + 1] if i + 1 < len(timestamps) else None
        next_mid_by_sym = {
            s: (mid_series[s].get(next_ts) if next_ts else None) for s in symbols
        }
        next_order_depth_by_sym = {}
        if next_ts is not None:
            for sym in symbols:
                if (next_ts, sym) in snapshots:
                    next_order_depth_by_sym[sym] = snapshots[(next_ts, sym)]
        fills = engine.resolve_tick(
            ts, order_depths, mkt_trades, next_mid_by_sym, positions,
            next_order_depth_by_sym=next_order_depth_by_sym,
        )
        for f in fills:
            cash[f.symbol] -= f.side * f.price * f.quantity
            own_trades_by_sym[f.symbol].append(f)
            result.fills.append(f)
            result.num_fills += 1
            if f.adverse:
                result.adverse_fills += 1
            if f.reason == "take":
                result.take_fills += 1
            elif f.reason == "passive":
                result.passive_fills += 1

        result.positions_over_time.append((ts, dict(positions)))
        running_pnl = 0.0
        for sym in symbols:
            if sym in order_depths:
                running_pnl += cash[sym] + liquidation_value(
                    positions[sym], order_depths[sym], config.liquidation_penalty_ticks
                )
        result.pnl_over_time.append((ts, running_pnl))

    final_ts = timestamps[-1]
    for sym in symbols:
        result.pnl_realized[sym] = cash[sym]
        if (final_ts, sym) in snapshots:
            result.pnl_liquidation[sym] = liquidation_value(
                positions[sym], snapshots[(final_ts, sym)], config.liquidation_penalty_ticks
            )
        else:
            result.pnl_liquidation[sym] = 0.0
    result.pnl_total = sum(result.pnl_realized.values()) + sum(result.pnl_liquidation.values())
    result.final_positions = dict(positions)

    peak = float('-inf')
    for _, p in result.pnl_over_time:
        if p > peak:
            peak = p
        result.drawdown = max(result.drawdown, peak - p)

    return result
