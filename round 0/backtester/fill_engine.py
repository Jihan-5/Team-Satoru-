"""Realistic fill engine – v3, calibrated against real submission logs.

Changes from v2:
  - Passive fills now match against the NEXT TICK's order book. If I post a
    bid at P, and next-tick's best_ask <= P, I get filled at P for up to my
    full quoted volume (capped by the L1 volume that crosses me). This matches
    how the real platform fills orders (book-crossing), not trade-flow sharing.
  - Trade-flow matching is kept as a secondary path when the book doesn't
    move through my price but prints happen at my level.
  - Stochastic rounding on queue shares so small-volume x small-capture
    fraction doesn't always truncate to zero.
  - Better same-tick fill detection: I can also be filled by trades this tick
    printing at my price level.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Fill:
    timestamp: int
    symbol: str
    side: int
    price: int
    quantity: int
    reason: str
    adverse: bool = False


@dataclass
class FillConfig:
    latency_ticks: int = 1
    queue_capture_frac: float = 0.20       # base share at same level as best
    price_improvement_capture: float = 1.0 # share when I improved the book
    adverse_skip_prob: float = 0.50
    liquidation_penalty_ticks: float = 0.5
    max_trade_follow: int = 3


class FillEngine:
    def __init__(self, config: FillConfig, position_limits: dict, rng=None):
        self.config = config
        self.position_limits = position_limits
        import random
        self.rng = rng if rng is not None else random.Random(42)
        self.pending = {}

    def submit(self, submit_ts: int, symbol: str, price: int, quantity: int):
        arrive_ts = submit_ts + self.config.latency_ticks * 100
        self.pending.setdefault(arrive_ts, []).append((symbol, price, quantity))

    def resolve_tick(self, ts, order_depth_by_sym, market_trades_by_sym,
                     next_mid_by_sym, positions,
                     next_order_depth_by_sym=None):
        fills_this_tick = []
        arrived = self.pending.pop(ts, [])
        if not arrived:
            return fills_this_tick

        by_sym_side = {}
        for sym, price, qty in arrived:
            side = 1 if qty > 0 else -1
            by_sym_side.setdefault((sym, side), []).append([price, abs(qty)])

        for (sym, side), orders in by_sym_side.items():
            if sym not in order_depth_by_sym:
                continue
            od = order_depth_by_sym[sym]
            lim = self.position_limits.get(sym, 80)
            cur_pos = positions.get(sym, 0)
            if side > 0:
                remaining_capacity = lim - cur_pos
            else:
                remaining_capacity = lim + cur_pos
            if remaining_capacity <= 0:
                continue

            # Best-priced orders first
            orders.sort(key=lambda x: -x[0] if side > 0 else x[0])

            sorted_bids = sorted(od.buy_orders.items(), key=lambda x: -x[0])
            sorted_asks = sorted(od.sell_orders.items(), key=lambda x: x[0])
            best_bid = sorted_bids[0][0] if sorted_bids else None
            best_ask = sorted_asks[0][0] if sorted_asks else None
            mid = (best_bid + best_ask) / 2 if (best_bid and best_ask) else None
            next_mid = next_mid_by_sym.get(sym)
            mkt_trades = market_trades_by_sym.get(sym, [])

            # Track how much of each market trade has been consumed by my orders
            trade_consumed = [0] * len(mkt_trades)

            for order in orders:
                if remaining_capacity <= 0:
                    break
                price, qty = order
                qty = min(qty, remaining_capacity)
                if qty <= 0:
                    continue

                # ── TAKING ──────────────────────────────────────────────────
                if side > 0:
                    for ap, av in sorted_asks:
                        if qty <= 0 or price < ap:
                            break
                        take = min(qty, abs(av))
                        if take > 0:
                            fills_this_tick.append(Fill(ts, sym, +1, ap, take, "take"))
                            qty -= take
                            remaining_capacity -= take
                            cur_pos += take
                else:
                    for bp, bv in sorted_bids:
                        if qty <= 0 or price > bp:
                            break
                        take = min(qty, abs(bv))
                        if take > 0:
                            fills_this_tick.append(Fill(ts, sym, -1, bp, take, "take"))
                            qty -= take
                            remaining_capacity -= take
                            cur_pos -= take

                if qty <= 0 or remaining_capacity <= 0:
                    continue

                # Queue regime: "improved" = my price beats current best
                if side > 0:
                    improved = (best_bid is None) or (price > best_bid)
                else:
                    improved = (best_ask is None) or (price < best_ask)
                capture_frac = (self.config.price_improvement_capture if improved
                                else self.config.queue_capture_frac)

                # ── PASSIVE: match against NEXT-TICK order depth ─────────────
                passive_filled = 0
                if next_order_depth_by_sym:
                    next_od = next_order_depth_by_sym.get(sym)
                    if next_od:
                        if side > 0:
                            next_asks = sorted(next_od.sell_orders.items())
                            for ap, av in next_asks:
                                if ap > price:
                                    break
                                crossing_vol = abs(av)
                                my_share_exact = crossing_vol * capture_frac
                                my_share = int(my_share_exact)
                                if self.rng.random() < (my_share_exact - my_share):
                                    my_share += 1
                                my_share = min(my_share, qty - passive_filled,
                                               remaining_capacity - passive_filled)
                                if my_share > 0:
                                    passive_filled += my_share
                                if passive_filled >= qty or passive_filled >= remaining_capacity:
                                    break
                        else:
                            next_bids = sorted(next_od.buy_orders.items(),
                                               key=lambda x: -x[0])
                            for bp, bv in next_bids:
                                if bp < price:
                                    break
                                crossing_vol = abs(bv)
                                my_share_exact = crossing_vol * capture_frac
                                my_share = int(my_share_exact)
                                if self.rng.random() < (my_share_exact - my_share):
                                    my_share += 1
                                my_share = min(my_share, qty - passive_filled,
                                               remaining_capacity - passive_filled)
                                if my_share > 0:
                                    passive_filled += my_share
                                if passive_filled >= qty or passive_filled >= remaining_capacity:
                                    break

                # ── PASSIVE: also match against same-tick trades ─────────────
                for i, tr in enumerate(mkt_trades):
                    if passive_filled >= qty or passive_filled >= remaining_capacity:
                        break
                    avail_trade = tr.quantity - trade_consumed[i]
                    if avail_trade <= 0:
                        continue
                    if side > 0 and tr.price <= price:
                        share_exact = avail_trade * capture_frac
                        share = int(share_exact)
                        if self.rng.random() < (share_exact - share):
                            share += 1
                        share = min(share, qty - passive_filled,
                                    remaining_capacity - passive_filled)
                        if share > 0:
                            passive_filled += share
                            trade_consumed[i] += share
                    elif side < 0 and tr.price >= price:
                        share_exact = avail_trade * capture_frac
                        share = int(share_exact)
                        if self.rng.random() < (share_exact - share):
                            share += 1
                        share = min(share, qty - passive_filled,
                                    remaining_capacity - passive_filled)
                        if share > 0:
                            passive_filled += share
                            trade_consumed[i] += share

                if passive_filled > 0:
                    adverse = False
                    if next_mid is not None and mid is not None:
                        if side > 0 and next_mid < mid:
                            adverse = True
                        elif side < 0 and next_mid > mid:
                            adverse = True
                    if adverse and self.rng.random() < self.config.adverse_skip_prob:
                        pass
                    else:
                        fills_this_tick.append(Fill(
                            ts, sym, side, price, passive_filled, "passive", adverse
                        ))
                        qty -= passive_filled
                        remaining_capacity -= passive_filled
                        cur_pos += side * passive_filled

            positions[sym] = cur_pos

        return fills_this_tick


def liquidation_value(position: int, order_depth, penalty_ticks: float) -> float:
    if position == 0:
        return 0.0
    if position > 0:
        bids = sorted(order_depth.buy_orders.items(), key=lambda x: -x[0])
        if not bids:
            return 0.0
        remaining = position
        proceeds = 0.0
        for bp, bv in bids:
            take = min(remaining, abs(bv))
            proceeds += (bp - penalty_ticks) * take
            remaining -= take
            if remaining == 0:
                break
        if remaining > 0:
            worst = bids[-1][0]
            proceeds += (worst - penalty_ticks * 4) * remaining
        return proceeds
    else:
        asks = sorted(order_depth.sell_orders.items(), key=lambda x: x[0])
        if not asks:
            return 0.0
        remaining = -position
        cost = 0.0
        for ap, av in asks:
            take = min(remaining, abs(av))
            cost += (ap + penalty_ticks) * take
            remaining -= take
            if remaining == 0:
                break
        if remaining > 0:
            worst = asks[-1][0]
            cost += (worst + penalty_ticks * 4) * remaining
        return -cost
