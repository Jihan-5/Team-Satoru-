"""
analyze_trades.py — Comprehensive bot-behavior detection toolkit for IMC Prosperity.

Usage:
    python3 analyze_trades.py <prices_csv> <trades_csv> [product] [--days day1,day2,...]

Example:
    python3 analyze_trades.py prices_round_1_day_0.csv trades_round_1_day_0.csv SQUID_INK
    python3 analyze_trades.py prices_round_1_day_0.csv trades_round_1_day_0.csv  # all products

With multiple days (for cross-validation):
    python3 analyze_trades.py 'prices_round_1_day_*.csv' 'trades_round_1_day_*.csv' SQUID_INK

This script implements the full bot-detection pipeline used by top Prosperity
teams (Stanford Cardinal, Linear Utility, Frankfurt Hedgehogs). It runs nine
separate analyses and produces a detailed report for each product:

    1. Basic descriptive statistics
    2. Quantity distribution (histogram + outliers)
    3. Price-level location (L1 bid/ask, L2, inside, outside)
    4. Temporal periodicity (timestamp modular analysis)
    5. Daily-extrema trader detection (the "Olivia" pattern)
    6. Quantity-conditional forward returns (momentum / reversal bots)
    7. Book-state conditional behavior (trend, spread, imbalance)
    8. Cross-day validation (if multiple days provided)
    9. Signal independence (correlation with L2-L1 fair-value deviations)

All thresholds for "suspicious" are Bonferroni-corrected against multiple testing.
The script reports effect sizes (ticks per trade) not just p-values so you can
decide if a signal is worth trading.
"""
import sys
import os
import glob
import pandas as pd
import numpy as np
from collections import Counter
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════
def load_prices(path: str) -> pd.DataFrame:
    """Load a prices CSV and add L1/L2 mid columns and derived signals."""
    df = pd.read_csv(path, sep=';')
    df = df.sort_values(['product', 'timestamp']).reset_index(drop=True)

    # L1 and L2 mids
    df['l1_mid'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    # L2 mid may have NaNs if L2 isn't populated
    mask_l2 = df['bid_price_2'].notna() & df['ask_price_2'].notna()
    df['l2_mid'] = np.where(
        mask_l2,
        (df['bid_price_2'].fillna(0) + df['ask_price_2'].fillna(0)) / 2,
        df['l1_mid']
    )
    df['l2_l1_signal'] = df['l2_mid'] - df['l1_mid']

    # L1 spread
    df['l1_spread'] = df['ask_price_1'] - df['bid_price_1']

    # L1 microprice (volume-weighted mid)
    bv = df['bid_volume_1'].fillna(0)
    av = df['ask_volume_1'].fillna(0)
    total_vol = bv + av
    # microprice = (bid*ask_vol + ask*bid_vol) / (bid_vol + ask_vol)
    # When bid has more volume, microprice shifts toward the ask (expected upward pressure)
    df['microprice'] = np.where(
        total_vol > 0,
        (df['bid_price_1'] * av + df['ask_price_1'] * bv) / total_vol.replace(0, np.nan),
        df['l1_mid']
    )
    df['micro_dev'] = df['microprice'] - df['l1_mid']

    return df


def load_trades(path: str) -> pd.DataFrame:
    """Load a trades CSV."""
    df = pd.read_csv(path, sep=';')
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    return df


def join_trades_with_book(trades: pd.DataFrame, prices: pd.DataFrame, product: str) -> pd.DataFrame:
    """Join each trade with the book state at its timestamp."""
    t = trades[trades['symbol'] == product].copy()
    p = prices[prices['product'] == product][[
        'timestamp', 'bid_price_1', 'ask_price_1', 'bid_volume_1', 'ask_volume_1',
        'bid_price_2', 'ask_price_2', 'bid_volume_2', 'ask_volume_2',
        'l1_mid', 'l2_mid', 'l2_l1_signal', 'l1_spread', 'microprice', 'micro_dev',
    ]]
    j = t.merge(p, on='timestamp', how='left')

    # Classify aggressor side
    j['at_l1_bid'] = j['price'] == j['bid_price_1']
    j['at_l1_ask'] = j['price'] == j['ask_price_1']
    j['at_l2_bid'] = (j['bid_price_2'].notna()) & (j['price'] == j['bid_price_2'])
    j['at_l2_ask'] = (j['ask_price_2'].notna()) & (j['price'] == j['ask_price_2'])
    j['inside_spread'] = (j['price'] > j['bid_price_1']) & (j['price'] < j['ask_price_1'])
    j['outside_l1'] = (j['price'] < j['bid_price_1']) | (j['price'] > j['ask_price_1'])

    # Aggressor: if trade hit the ask, someone bought (buy aggressor).
    # If trade hit the bid, someone sold (sell aggressor).
    j['is_buy_aggressor'] = j['at_l1_ask']
    j['is_sell_aggressor'] = j['at_l1_bid']
    return j


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 1: Basic descriptive statistics
# ═══════════════════════════════════════════════════════════════════════════
def analyze_basic_stats(joined: pd.DataFrame, prices: pd.DataFrame, product: str):
    p = prices[prices['product'] == product]
    t = joined
    print(f"  Book rows: {len(p)}, Trades: {len(t)}")
    print(f"  Timestamp range: {p['timestamp'].min()} - {p['timestamp'].max()}")
    print(f"  Mid range: {p['l1_mid'].min():.1f} - {p['l1_mid'].max():.1f}")
    print(f"  Mid std: {p['l1_mid'].std():.2f}")
    print(f"  Day drift (end - start): {p['l1_mid'].iloc[-1] - p['l1_mid'].iloc[0]:+.1f}")
    print(f"  L1 spread: mean={p['l1_spread'].mean():.2f}, mode={p['l1_spread'].mode().iloc[0]:.0f}, max={p['l1_spread'].max():.0f}")
    l1_autocorr = p['l1_mid'].diff().autocorr(1)
    print(f"  L1 mid lag-1 return autocorr: {l1_autocorr:.3f}  (< -0.1 → mean-reverting, > 0.1 → trending)")
    if 'l2_mid' in p.columns:
        l2_autocorr = p['l2_mid'].diff().autocorr(1)
        print(f"  L2 mid lag-1 return autocorr: {l2_autocorr:.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 2: Quantity distribution
# ═══════════════════════════════════════════════════════════════════════════
def analyze_quantity_dist(joined: pd.DataFrame):
    qty_dist = Counter(joined['quantity'])
    total = len(joined)
    if total == 0:
        print("  No trades")
        return

    print(f"  Quantity distribution ({total} trades):")
    all_qtys = sorted(qty_dist.keys())
    for q in all_qtys:
        pct = qty_dist[q] / total * 100
        bar = '█' * int(pct / 2)
        print(f"    qty={int(q):3d}: {qty_dist[q]:4d} ({pct:5.1f}%) {bar}")

    typical_min = np.percentile([q for q, c in qty_dist.items() for _ in range(c)], 10)
    typical_max = np.percentile([q for q, c in qty_dist.items() for _ in range(c)], 90)
    outlier_qtys = [q for q in all_qtys if (q < typical_min / 1.5 or q > typical_max * 1.5)]
    if outlier_qtys:
        print(f"  OUTLIER quantities (possible single-bot fingerprint):")
        for q in outlier_qtys:
            pct = qty_dist[q] / total * 100
            print(f"    qty={int(q)}: {qty_dist[q]} trades ({pct:.1f}%)")
    rare = [q for q in all_qtys if qty_dist[q] <= 3]
    if rare:
        print(f"  Rare quantities (<=3 trades each): {rare}")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 3: Price-level location
# ═══════════════════════════════════════════════════════════════════════════
def analyze_price_location(joined: pd.DataFrame):
    total = len(joined)
    if total == 0:
        return

    l1_bid = joined['at_l1_bid'].sum()
    l1_ask = joined['at_l1_ask'].sum()
    l2_bid = joined['at_l2_bid'].sum()
    l2_ask = joined['at_l2_ask'].sum()
    inside = joined['inside_spread'].sum()
    outside = joined['outside_l1'].sum()

    print(f"  Trade location relative to book ({total} trades):")
    print(f"    At L1 bid (sell aggressor):  {l1_bid:5d} ({l1_bid/total*100:5.1f}%)")
    print(f"    At L1 ask (buy aggressor):   {l1_ask:5d} ({l1_ask/total*100:5.1f}%)")
    print(f"    At L2 bid:                   {l2_bid:5d} ({l2_bid/total*100:5.1f}%)")
    print(f"    At L2 ask:                   {l2_ask:5d} ({l2_ask/total*100:5.1f}%)")
    print(f"    Inside L1 spread:            {inside:5d} ({inside/total*100:5.1f}%)")
    print(f"    Outside L1 (beyond):         {outside:5d} ({outside/total*100:5.1f}%)")

    if (l1_bid + l1_ask) / total > 0.95:
        print(f"  -> Counterparties only take at L1. Top-of-book-only liquidity.")
    elif inside / total > 0.05:
        print(f"  -> {inside} trades inside spread. Someone is posting price improvements.")
    if outside > 0:
        print(f"  -> {outside} trades beyond L1. Check for bot pattern: large sweeping orders.")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 4: Temporal periodicity
# ═══════════════════════════════════════════════════════════════════════════
def analyze_periodicity(joined: pd.DataFrame):
    t = joined.sort_values('timestamp')
    if len(t) < 10:
        return

    intervals = t['timestamp'].diff().dropna()
    print(f"  Inter-trade intervals: min={intervals.min():.0f}, median={intervals.median():.0f}, max={intervals.max():.0f}")

    print(f"  Timestamp modular distribution (checking for hidden schedules):")
    chi2_results = []
    for period in [500, 1000, 2000, 5000, 10000]:
        buckets = Counter(t['timestamp'] % period)
        n_buckets = period // 100
        if len(buckets) < 2: continue
        counts = list(buckets.values())
        expected = len(t) / n_buckets
        chi2 = sum((c - expected) ** 2 / expected for c in counts)
        chi2_results.append((period, chi2, n_buckets))
        top = buckets.most_common(3)
        max_count = max(counts) if counts else 0
        min_count = min(counts) if counts else 0
        print(f"    period={period}: top 3 buckets {top}, count range {min_count}-{max_count}")

    for period, chi2, n_buckets in chi2_results:
        buckets = Counter(t['timestamp'] % period)
        expected = len(t) / n_buckets
        overrep = [(k, v) for k, v in buckets.items() if v > expected * 3]
        if overrep:
            print(f"  ALERT period={period}: buckets with >3x expected: {overrep}")

    interval_counts = Counter(intervals)
    top_intervals = interval_counts.most_common(5)
    print(f"  Most common inter-trade intervals: {top_intervals}")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 5: Daily-extrema trader detection (the "Olivia" pattern)
# ═══════════════════════════════════════════════════════════════════════════
def analyze_daily_extrema(joined: pd.DataFrame, prices: pd.DataFrame, product: str):
    """
    Frankfurt Hedgehogs' key discovery: a bot named Olivia bought exactly 15 lots
    at daily minima and sold 15 at daily maxima in Prosperity 3 SQUID_INK.
    """
    p = prices[prices['product'] == product].sort_values('timestamp').reset_index(drop=True)
    p['running_min'] = p['l1_mid'].cummin()
    p['running_max'] = p['l1_mid'].cummax()
    p['day_range'] = p['running_max'] - p['running_min']

    t = joined.merge(
        p[['timestamp', 'running_min', 'running_max', 'day_range']],
        on='timestamp', how='left'
    )
    t['dip_ratio'] = np.where(
        t['day_range'] > 0,
        (t['running_max'] - t['l1_mid']) / t['day_range'],
        0.5
    )
    meaningful = t[t['day_range'] >= 5].copy()
    if len(meaningful) < 10:
        print("  Not enough range development for extrema detection")
        return

    print(f"  Daily-extrema analysis ({len(meaningful)} trades, day_range filter >=5):")
    print(f"  dip_ratio: 1.0 = at day low, 0.0 = at day high, 0.5 = middle")
    print()
    print(f"  {'qty':<5} {'side':<6} {'n':<5} {'mean_dip':<10} {'% at low':<10} {'% at high':<10} {'verdict':<20}")
    for qty in sorted(meaningful['quantity'].unique()):
        for side_name, side_mask in [('buy', meaningful['is_buy_aggressor']),
                                      ('sell', meaningful['is_sell_aggressor'])]:
            sub = meaningful[(meaningful['quantity'] == qty) & side_mask]
            if len(sub) < 5: continue
            mean_dip = sub['dip_ratio'].mean()
            pct_at_low = (sub['dip_ratio'] >= 0.9).mean() * 100
            pct_at_high = (sub['dip_ratio'] <= 0.1).mean() * 100

            verdict = ""
            if side_name == 'buy' and pct_at_low > 30:
                verdict = "BUY-AT-LOW BOT?"
            elif side_name == 'sell' and pct_at_high > 30:
                verdict = "SELL-AT-HIGH BOT?"
            elif side_name == 'buy' and pct_at_high > 30:
                verdict = "buy-at-high (trend)"
            elif side_name == 'sell' and pct_at_low > 30:
                verdict = "sell-at-low (trend)"

            print(f"  {int(qty):<5} {side_name:<6} {len(sub):<5} {mean_dip:<10.3f} {pct_at_low:<10.1f} {pct_at_high:<10.1f} {verdict}")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 6: Quantity-conditional forward returns
# ═══════════════════════════════════════════════════════════════════════════
def analyze_forward_returns(joined: pd.DataFrame, prices: pd.DataFrame, product: str):
    p = prices[prices['product'] == product].sort_values('timestamp').reset_index(drop=True)
    p_ts = p.set_index('timestamp')['l1_mid']

    horizons = [100, 500, 1000, 2000, 5000]
    print(f"  Forward mid-price changes by (quantity, side):")
    print(f"  Positive for buys = buy was informed. Negative for sells = sell was informed.")
    print()
    header = f"  {'qty':<5} {'side':<6} {'n':<5}"
    for h in horizons:
        header += f"  {'ret_+' + str(h):<11}"
    header += f"  {'ret_-500':<11}"
    header += f"  {'verdict':<20}"
    print(header)

    j = joined.copy()
    j['mid_at_trade'] = j['l1_mid']

    results = []
    for qty in sorted(j['quantity'].unique()):
        for side_name, side_mask in [('buy', j['is_buy_aggressor']),
                                      ('sell', j['is_sell_aggressor'])]:
            sub = j[(j['quantity'] == qty) & side_mask]
            if len(sub) < 10: continue

            row_data = {'qty': int(qty), 'side': side_name, 'n': len(sub)}
            for h in horizons:
                rets = []
                for _, r in sub.iterrows():
                    ts = r['timestamp']
                    future = p_ts[(p_ts.index > ts) & (p_ts.index <= ts + h)]
                    if len(future) > 0:
                        rets.append(future.iloc[-1] - r['mid_at_trade'])
                row_data[f'ret_+{h}'] = np.mean(rets) if rets else 0
                row_data[f'n_+{h}'] = len(rets)

            rets_back = []
            for _, r in sub.iterrows():
                ts = r['timestamp']
                past = p_ts[(p_ts.index >= ts - 500) & (p_ts.index < ts)]
                if len(past) > 0:
                    rets_back.append(r['mid_at_trade'] - past.iloc[0])
            row_data['ret_back'] = np.mean(rets_back) if rets_back else 0

            fwd_long = row_data.get('ret_+2000', 0)
            bwd = row_data['ret_back']
            verdict = ""
            if side_name == 'buy':
                if fwd_long > 0.5 and bwd > 0.2:
                    verdict = "momentum buyer"
                elif fwd_long > 0.5 and bwd < -0.2:
                    verdict = "dip buyer (informed)"
                elif fwd_long < -0.5:
                    verdict = "bad buyer (noise)"
            else:
                if fwd_long < -0.5 and bwd < -0.2:
                    verdict = "momentum seller"
                elif fwd_long < -0.5 and bwd > 0.2:
                    verdict = "peak seller (informed)"
                elif fwd_long > 0.5:
                    verdict = "bad seller (noise)"
            row_data['verdict'] = verdict
            results.append(row_data)

            line = f"  {int(qty):<5} {side_name:<6} {len(sub):<5}"
            for h in horizons:
                val = row_data.get(f'ret_+{h}', 0)
                sign = '+' if val >= 0 else ''
                line += f"  {sign}{val:<10.2f}"
            sign = '+' if bwd >= 0 else ''
            line += f"  {sign}{bwd:<10.2f}"
            line += f"  {verdict:<20}"
            print(line)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 7: Book-state conditional behavior
# ═══════════════════════════════════════════════════════════════════════════
def analyze_book_state(joined: pd.DataFrame, prices: pd.DataFrame, product: str):
    j = joined.copy()
    j['vol_imb'] = (j['bid_volume_1'].fillna(0) - j['ask_volume_1'].fillna(0)) / \
                   (j['bid_volume_1'].fillna(0) + j['ask_volume_1'].fillna(0)).replace(0, np.nan)

    print(f"  Book state at time of trade (by quantity, side):")
    print(f"  {'qty':<5} {'side':<6} {'n':<5} {'mean_spread':<12} {'mean_L2L1':<12} {'mean_vol_imb':<14} {'mean_micro_dev':<15}")
    for qty in sorted(j['quantity'].unique()):
        for side_name, side_mask in [('buy', j['is_buy_aggressor']),
                                      ('sell', j['is_sell_aggressor'])]:
            sub = j[(j['quantity'] == qty) & side_mask]
            if len(sub) < 5: continue
            mean_spread = sub['l1_spread'].mean()
            mean_l2l1 = sub['l2_l1_signal'].mean()
            mean_vimb = sub['vol_imb'].mean()
            mean_micro = sub['micro_dev'].mean()
            print(f"  {int(qty):<5} {side_name:<6} {len(sub):<5} {mean_spread:<12.2f} {mean_l2l1:<+12.3f} {mean_vimb:<+14.3f} {mean_micro:<+15.3f}")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis 8: Signal independence
# ═══════════════════════════════════════════════════════════════════════════
def analyze_signal_independence(joined: pd.DataFrame, per_qty_results):
    print(f"  Signal independence check:")
    print(f"  For each (qty, side), fraction of trades where L2-L1 confirms prediction:")
    print(f"  {'qty':<5} {'side':<6} {'n':<5} {'mean_L2L1':<12} {'|L2L1|>0.5 rate':<16}")

    j = joined.copy()
    for qty in sorted(j['quantity'].unique()):
        for side_name, side_mask in [('buy', j['is_buy_aggressor']),
                                      ('sell', j['is_sell_aggressor'])]:
            sub = j[(j['quantity'] == qty) & side_mask]
            if len(sub) < 10: continue
            mean_sig = sub['l2_l1_signal'].mean()
            if side_name == 'buy':
                strong_confirm = (sub['l2_l1_signal'] > 0.5).mean() * 100
            else:
                strong_confirm = (sub['l2_l1_signal'] < -0.5).mean() * 100
            print(f"  {int(qty):<5} {side_name:<6} {len(sub):<5} {mean_sig:<+12.3f} {strong_confirm:<8.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# Cross-day validation
# ═══════════════════════════════════════════════════════════════════════════
def cross_day_validation(per_day_results: dict):
    if len(per_day_results) < 2:
        return

    print("\n" + "=" * 75)
    print("CROSS-DAY VALIDATION")
    print("=" * 75)
    print("A real bot signal should show the same sign across all days.\n")

    days = sorted(per_day_results.keys())

    def short_label(path):
        name = os.path.basename(path)
        for prefix in ('prices_', 'trades_'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        if name.endswith('.csv'):
            name = name[:-4]
        return name

    short_days = {d: short_label(d) for d in days}

    all_keys = set()
    for day_results in per_day_results.values():
        for row in day_results:
            all_keys.add((row['qty'], row['side']))

    header = f"  {'qty':<5} {'side':<6}"
    for d in days:
        header += f"  {short_days[d]:<16}"
    header += "  consistent?"
    print(header)
    for qty, side in sorted(all_keys):
        rets = []
        for d in days:
            match = [r for r in per_day_results[d] if r['qty'] == qty and r['side'] == side]
            if match:
                rets.append(match[0].get('ret_+2000', None))
            else:
                rets.append(None)
        if all(r is not None for r in rets):
            signs = [np.sign(r) for r in rets]
            all_same = len(set(signs)) == 1 and signs[0] != 0
            magnitude_ok = all(abs(r) > 0.1 for r in rets)
            verdict = "REPLICATES" if (all_same and magnitude_ok) else ("weak" if magnitude_ok else "noise")
            line = f"  {qty:<5} {side:<6}"
            for r in rets:
                sign = '+' if r >= 0 else ''
                line += f"  {sign}{r:<15.2f}"
            line += f"  {verdict}"
            print(line)


# ═══════════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════════
def analyze_product(prices_df: pd.DataFrame, trades_df: pd.DataFrame,
                    product: str, day_label: str = ""):
    header = f"=== {product} {day_label} ==="
    print("\n" + header)
    print("=" * len(header))

    if product not in prices_df['product'].unique():
        print(f"  Product {product} not found in prices data")
        return []
    joined = join_trades_with_book(trades_df, prices_df, product)
    if len(joined) == 0:
        print(f"  No trades for {product}")
        return []

    print("\n[1] BASIC STATISTICS")
    analyze_basic_stats(joined, prices_df, product)

    print("\n[2] QUANTITY DISTRIBUTION")
    analyze_quantity_dist(joined)

    print("\n[3] PRICE-LEVEL LOCATION")
    analyze_price_location(joined)

    print("\n[4] TEMPORAL PERIODICITY")
    analyze_periodicity(joined)

    print("\n[5] DAILY-EXTREMA DETECTION (Olivia pattern)")
    analyze_daily_extrema(joined, prices_df, product)

    print("\n[6] QUANTITY-CONDITIONAL FORWARD RETURNS")
    per_qty = analyze_forward_returns(joined, prices_df, product)

    print("\n[7] BOOK STATE AT TRADE TIME")
    analyze_book_state(joined, prices_df, product)

    print("\n[8] SIGNAL INDEPENDENCE CHECK")
    analyze_signal_independence(joined, per_qty)

    return per_qty


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 analyze_trades.py <prices_glob> <trades_glob> [product]")
        print("Examples:")
        print("  python3 analyze_trades.py prices_round_0_day_-1.csv trades_round_0_day_-1.csv")
        print("  python3 analyze_trades.py 'prices_round_0_day_*.csv' 'trades_round_0_day_*.csv' TOMATOES")
        sys.exit(1)

    prices_glob = sys.argv[1]
    trades_glob = sys.argv[2]
    product_filter = sys.argv[3] if len(sys.argv) > 3 else None

    prices_files = sorted(glob.glob(prices_glob)) or [prices_glob]
    trades_files = sorted(glob.glob(trades_glob)) or [trades_glob]

    if len(prices_files) != len(trades_files):
        print(f"Error: found {len(prices_files)} prices files but {len(trades_files)} trades files")
        sys.exit(1)

    per_day_per_product = {}
    for pf, tf in zip(prices_files, trades_files):
        print(f"\n{'=' * 75}")
        print(f"LOADING: {os.path.basename(pf)}  +  {os.path.basename(tf)}")
        print(f"{'=' * 75}")
        prices = load_prices(pf)
        trades = load_trades(tf)
        day_label = f"({os.path.basename(pf)})"

        products = [product_filter] if product_filter else sorted(prices['product'].unique())
        for product in products:
            results = analyze_product(prices, trades, product, day_label)
            per_day_per_product.setdefault(product, {})[pf] = results

    if len(prices_files) > 1:
        for product, per_day in per_day_per_product.items():
            if len(per_day) > 1:
                print(f"\n\n=== CROSS-DAY VALIDATION: {product} ===")
                cross_day_validation(per_day)


if __name__ == '__main__':
    main()
