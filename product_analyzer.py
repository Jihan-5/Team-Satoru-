"""
product_analyzer.py — Full product diagnostic and strategy recommender.

Goes beyond bot detection. For any Prosperity product, answers:

    1. What KIND of product is it? (pegged / drifting / mean-reverting / trending)
    2. What's the fair value process? (L1 mid / L2 mid / microprice / constant)
    3. What's the short-term predictability? (autocorrelation, momentum, imbalance)
    4. What's the volatility structure? (constant / clustered / regime-shifting)
    5. What's the counterparty flow profile? (taker rate, size distribution)
    6. What's the theoretical PnL ceiling for market making?
    7. What STRATEGY ARCHETYPE fits? (pure MM / directional / mean-reversion / arb)
    8. What STARTING PARAMETERS should I use? (take_width, edge, levels, position limit)
    9. What forward-return predictors exist? (returns ~ imbalance, L2L1, trade flow)
   10. A predicted next-500-tick price path using a linear model

Usage:
    python3 product_analyzer.py <prices_csv> <trades_csv> <product>
    python3 product_analyzer.py 'prices_round_1_day_*.csv' 'trades_round_1_day_*.csv' SQUID_INK

Output: detailed report + "strategy card" with starting params you can paste
into a three-phase trader file.
"""
import sys
import os
import glob
import pandas as pd
import numpy as np
from collections import Counter


# ═════════════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════════════
def load_prices(path):
    df = pd.read_csv(path, sep=';')
    df = df.sort_values(['product', 'timestamp']).reset_index(drop=True)
    df['l1_mid'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    mask_l2 = df['bid_price_2'].notna() & df['ask_price_2'].notna()
    df['l2_mid'] = np.where(
        mask_l2,
        (df['bid_price_2'].fillna(0) + df['ask_price_2'].fillna(0)) / 2,
        df['l1_mid']
    )
    df['l2_l1'] = df['l2_mid'] - df['l1_mid']
    df['l1_spread'] = df['ask_price_1'] - df['bid_price_1']
    bv = df['bid_volume_1'].fillna(0)
    av = df['ask_volume_1'].fillna(0)
    tot = bv + av
    df['microprice'] = np.where(
        tot > 0,
        (df['bid_price_1'] * av + df['ask_price_1'] * bv) / tot.replace(0, np.nan),
        df['l1_mid']
    )
    df['micro_dev'] = df['microprice'] - df['l1_mid']
    df['vol_imb'] = np.where(tot > 0, (bv - av) / tot.replace(0, np.nan), 0)
    return df


def load_trades(path):
    df = pd.read_csv(path, sep=';')
    df = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    return df


# ═════════════════════════════════════════════════════════════════════════
# Section 1: Classify product archetype
# ═════════════════════════════════════════════════════════════════════════
def classify_archetype(p):
    """
    Returns one of:
        PEGGED         — std < 2, constant mean, trivial to MM
        DRIFTING       — std 2-30, random walk, no autocorrelation
        MEAN_REVERTING — lag-1 autocorr < -0.2, oscillates around mean
        TRENDING       — lag-1 autocorr > +0.15, persistent moves
        VOLATILE       — std > 30, hard to classify without more data
    """
    std = p['l1_mid'].std()
    mid_range = p['l1_mid'].max() - p['l1_mid'].min()
    returns = p['l1_mid'].diff().dropna()
    ac1 = returns.autocorr(1) if len(returns) > 10 else 0
    drift = p['l1_mid'].iloc[-1] - p['l1_mid'].iloc[0]

    # Variance ratio test: σ²(k-step returns) / k / σ²(1-step returns)
    # VR = 1 → random walk
    # VR < 1 → mean reverting
    # VR > 1 → trending
    ret_5 = p['l1_mid'].diff(5).dropna()
    ret_1 = p['l1_mid'].diff(1).dropna()
    vr_5 = ret_5.var() / (5 * ret_1.var()) if ret_1.var() > 0 else 1.0

    ret_20 = p['l1_mid'].diff(20).dropna()
    vr_20 = ret_20.var() / (20 * ret_1.var()) if ret_1.var() > 0 else 1.0

    # Classify
    if std < 2:
        archetype = 'PEGGED'
        reason = f"std={std:.2f} < 2, mid range only {mid_range:.0f} ticks"
    elif ac1 < -0.25:
        archetype = 'MEAN_REVERTING'
        reason = f"lag-1 autocorr={ac1:.3f} strongly negative"
    elif ac1 > 0.15:
        archetype = 'TRENDING'
        reason = f"lag-1 autocorr={ac1:.3f} positive"
    elif std > 30:
        archetype = 'VOLATILE'
        reason = f"std={std:.2f} very high, uncertain archetype"
    else:
        archetype = 'DRIFTING'
        reason = f"std={std:.2f}, autocorr={ac1:.3f} near zero (random walk)"

    return {
        'archetype': archetype,
        'reason': reason,
        'std': std,
        'autocorr_1': ac1,
        'drift': drift,
        'variance_ratio_5': vr_5,
        'variance_ratio_20': vr_20,
        'mid_range': mid_range,
    }


# ═════════════════════════════════════════════════════════════════════════
# Section 2: Fair value process
# ═════════════════════════════════════════════════════════════════════════
def analyze_fair_value(p):
    """
    Which estimator of 'fair' best predicts next-tick L1 mid?
        - L1 mid (naive)
        - L2 mid (MM-mid filter, used by LU)
        - Microprice (volume-weighted)
        - Volume-weighted average of L1 and L2

    Metric: MSE of (next_l1_mid - estimate). Lower = better.
    """
    next_l1 = p['l1_mid'].shift(-1)
    mask = next_l1.notna()

    estimators = {
        'L1 mid': p['l1_mid'],
        'L2 mid (wall)': p['l2_mid'],
        'Microprice': p['microprice'],
        'L1+L2 avg': (p['l1_mid'] + p['l2_mid']) / 2,
    }

    results = []
    for name, est in estimators.items():
        m = mask & est.notna()
        if m.sum() < 10:
            continue
        mse = ((next_l1[m] - est[m]) ** 2).mean()
        # bias: are we systematically above or below?
        bias = (next_l1[m] - est[m]).mean()
        results.append((name, mse, bias))

    results.sort(key=lambda x: x[1])
    return results


# ═════════════════════════════════════════════════════════════════════════
# Section 3: Volatility structure
# ═════════════════════════════════════════════════════════════════════════
def analyze_volatility(p):
    """
    Is volatility constant across the day, or does it cluster?
    Constant vol → simple MM. Clustered vol → widen quotes in high-vol regimes.
    """
    returns = p['l1_mid'].diff().dropna()
    abs_ret = returns.abs()

    # Split day into quarters, compute std in each
    n = len(returns)
    quarters = []
    for q in range(4):
        chunk = returns.iloc[q * n // 4:(q + 1) * n // 4]
        quarters.append(chunk.std())

    # GARCH-like check: correlation of |ret| with |prev ret|
    abs_ac = abs_ret.autocorr(1) if len(abs_ret) > 10 else 0

    return {
        'overall_std': returns.std(),
        'q1_std': quarters[0],
        'q2_std': quarters[1],
        'q3_std': quarters[2],
        'q4_std': quarters[3],
        'abs_return_autocorr': abs_ac,
        'max_abs_move': abs_ret.max(),
        'pct_zero_moves': (returns == 0).mean() * 100,
    }


# ═════════════════════════════════════════════════════════════════════════
# Section 4: Forward return predictors (what signals predict the future?)
# ═════════════════════════════════════════════════════════════════════════
def analyze_predictors(p, trades, product):
    """
    Run linear regression of next-K-tick returns against candidate signals.
    Returns correlation, R², and regression coefficient for each.

    Predictors tested:
        - Lag-1 return (for momentum/reversal)
        - L2-L1 signal
        - Microprice deviation
        - Volume imbalance
        - L1 spread width
        - Rolling 20-tick return
        - Trade flow (signed volume in last 10 ticks)
    """
    df = p.copy()
    df['ret_1'] = df['l1_mid'].diff()
    df['ret_20'] = df['l1_mid'].diff(20)

    # Compute signed trade flow in rolling windows
    t = trades[trades['symbol'] == product].copy()
    if len(t) > 0:
        t_by_ts = t.set_index('timestamp')
        # For each book tick, sum signed volume of recent trades
        # We need to know the aggressor side: if trade price == ask, buy aggressor (+qty)
        # We need book context to determine this
        t_joined = t.merge(df[['timestamp', 'bid_price_1', 'ask_price_1']], on='timestamp', how='left')
        t_joined['signed_qty'] = np.where(
            t_joined['price'] == t_joined['ask_price_1'],  t_joined['quantity'],
            np.where(t_joined['price'] == t_joined['bid_price_1'], -t_joined['quantity'], 0)
        )
        flow_by_ts = t_joined.groupby('timestamp')['signed_qty'].sum()
        df = df.merge(flow_by_ts.rename('trade_flow_1').reset_index(), on='timestamp', how='left')
        df['trade_flow_1'] = df['trade_flow_1'].fillna(0)
        df['trade_flow_10'] = df['trade_flow_1'].rolling(10, min_periods=1).sum()
    else:
        df['trade_flow_1'] = 0
        df['trade_flow_10'] = 0

    # Targets: forward returns at horizons
    horizons = [1, 5, 20, 100]
    for h in horizons:
        df[f'fwd_ret_{h}'] = df['l1_mid'].shift(-h) - df['l1_mid']

    predictors = [
        ('lag_1_return', df['ret_1']),
        ('lag_20_return', df['ret_20']),
        ('l2_l1_signal', df['l2_l1']),
        ('microprice_dev', df['micro_dev']),
        ('volume_imbalance', df['vol_imb']),
        ('l1_spread', df['l1_spread']),
        ('trade_flow_1tick', df['trade_flow_1']),
        ('trade_flow_10tick', df['trade_flow_10']),
    ]

    results = {}
    for h in horizons:
        target = df[f'fwd_ret_{h}']
        rows = []
        for name, pred in predictors:
            mask = target.notna() & pred.notna() & np.isfinite(pred)
            if mask.sum() < 100:
                continue
            x = pred[mask].values
            y = target[mask].values
            if x.std() == 0:
                continue
            corr = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0
            # OLS coefficient
            beta = (np.sum((x - x.mean()) * (y - y.mean())) /
                    np.sum((x - x.mean()) ** 2))
            r_squared = corr ** 2
            rows.append((name, corr, r_squared, beta))
        rows.sort(key=lambda r: -abs(r[1]))
        results[h] = rows

    return results


# ═════════════════════════════════════════════════════════════════════════
# Section 5: Counterparty flow profile
# ═════════════════════════════════════════════════════════════════════════
def analyze_flow(trades, product, prices):
    t = trades[trades['symbol'] == product]
    p = prices[prices['product'] == product]
    if len(t) == 0:
        return None

    n_ticks = len(p)
    n_trades = len(t)

    # Join to classify aggressor
    j = t.merge(
        p[['timestamp', 'bid_price_1', 'ask_price_1', 'l1_mid']],
        on='timestamp', how='left'
    )
    is_buy = j['price'] == j['ask_price_1']
    is_sell = j['price'] == j['bid_price_1']

    total_vol = j['quantity'].sum()
    buy_vol = j[is_buy]['quantity'].sum()
    sell_vol = j[is_sell]['quantity'].sum()

    return {
        'n_trades': n_trades,
        'n_ticks': n_ticks,
        'trade_rate': n_trades / n_ticks,
        'total_volume': total_vol,
        'buy_volume': buy_vol,
        'sell_volume': sell_vol,
        'volume_per_tick': total_vol / n_ticks,
        'avg_trade_size': j['quantity'].mean(),
        'median_trade_size': j['quantity'].median(),
        'buy_ratio': buy_vol / total_vol if total_vol > 0 else 0,
    }


# ═════════════════════════════════════════════════════════════════════════
# Section 6: Market making PnL ceiling
# ═════════════════════════════════════════════════════════════════════════
def estimate_mm_ceiling(flow, p, product):
    """
    Upper bound estimate for a pure market maker:
        Half the spread × (expected fraction of volume you capture) × (total volume)

    Assumptions:
        - You capture ~40-50% of total volume (typical for a tight MM with multi-level)
        - You earn half the L1 spread per round-trip on average
        - Adverse selection eats ~1-2 ticks per round-trip
    """
    if flow is None or flow['total_volume'] == 0:
        return None

    prices = p[p['product'] == product]
    mean_spread = prices['l1_spread'].mean()
    edge_per_trade = (mean_spread / 2) - 1.5  # subtract adverse selection
    edge_per_trade = max(edge_per_trade, 1.0)  # floor at 1 tick

    # Upper bound: capture 100% of volume, earn full half-spread
    upper = (flow['total_volume'] / 2) * (mean_spread / 2)
    # Realistic: capture 45%, earn half_spread minus adverse
    realistic = (flow['total_volume'] / 2) * edge_per_trade * 0.45

    return {
        'mean_spread': mean_spread,
        'edge_per_trade_est': edge_per_trade,
        'upper_bound': upper,
        'realistic_estimate': realistic,
    }


# ═════════════════════════════════════════════════════════════════════════
# Section 7: Strategy recommender
# ═════════════════════════════════════════════════════════════════════════
def recommend_strategy(classify, fair_val, vol, predictors, flow, ceiling, product):
    """
    Based on all the analyses, recommend a strategy archetype and starting params.
    """
    arch = classify['archetype']
    std = classify['std']
    ac1 = classify['autocorr_1']
    mean_spread = ceiling['mean_spread'] if ceiling else 5

    rec = {}

    # STRATEGY ARCHETYPE
    if arch == 'PEGGED':
        rec['strategy'] = 'Pure market making around constant fair value'
        rec['fair_value_method'] = f'Constant at {classify.get("median", "determine from data")}'
        rec['why'] = 'Low volatility, no drift. Maximize edge per fill.'
    elif arch == 'MEAN_REVERTING':
        rec['strategy'] = 'Market making + light reversion bias'
        rec['fair_value_method'] = 'L2 mid (wall) with optional EMA smoothing'
        rec['why'] = f'Autocorr {ac1:.2f} strongly negative. Post both sides; let inventory flip naturally.'
    elif arch == 'DRIFTING':
        rec['strategy'] = 'Market making with MM-mid fair value'
        rec['fair_value_method'] = 'L2 mid (filter by high adverse_volume)'
        rec['why'] = 'Random walk — best estimate of next price is current L2 mid. Linear Utility recipe.'
    elif arch == 'TRENDING':
        rec['strategy'] = 'Market making + trend bias (asymmetric levels)'
        rec['fair_value_method'] = 'L2 mid + drift term'
        rec['why'] = f'Autocorr {ac1:.2f} positive. Follow-through probable — bias position pro-trend.'
    elif arch == 'VOLATILE':
        rec['strategy'] = 'Wide-spread MM with regime detection'
        rec['fair_value_method'] = 'L2 mid, reset on large jumps'
        rec['why'] = f'std={std:.1f} very high. Must widen quotes to avoid adverse selection.'

    # STARTING PARAMS (suitable for a three-phase trader)
    # take_width: how aggressively to take mispriced orders
    # default_edge: how far from fair to quote when book is empty
    # levels: how many price levels deep to post
    if arch == 'PEGGED':
        rec['params'] = {
            'take_width': 1,
            'clear_width': 1,
            'default_edge': max(2, int(mean_spread / 3)),
            'disregard_edge': 1,
            'join_edge': 2,
            'levels': 2,
            'soft_position_limit': 40,
            'prevent_adverse': False,
        }
    elif arch == 'MEAN_REVERTING':
        rec['params'] = {
            'take_width': 1,
            'clear_width': 0,
            'default_edge': max(1, int(mean_spread / 4)),
            'disregard_edge': 1,
            'join_edge': 0,
            'levels': 2,
            'soft_position_limit': 30,
            'prevent_adverse': True,
            'adverse_volume': 15,
            'reversion_beta': max(-0.5, min(0.0, ac1 * 0.6)),
        }
    elif arch == 'DRIFTING':
        rec['params'] = {
            'take_width': 1,
            'clear_width': 0,
            'default_edge': 1,
            'disregard_edge': 1,
            'join_edge': 0,
            'levels': 3,
            'soft_position_limit': 40,
            'prevent_adverse': True,
            'adverse_volume': 15,
            'reversion_beta': 0.0,
        }
    elif arch == 'TRENDING':
        rec['params'] = {
            'take_width': 1,
            'clear_width': 0,
            'default_edge': 1,
            'disregard_edge': 1,
            'join_edge': 0,
            'levels': 3,
            'soft_position_limit': 50,
            'prevent_adverse': True,
            'adverse_volume': 15,
            'reversion_beta': 0.0,
            'bias_mode': 'trend_following',
        }
    elif arch == 'VOLATILE':
        rec['params'] = {
            'take_width': 2,
            'clear_width': 1,
            'default_edge': max(3, int(mean_spread / 2)),
            'disregard_edge': 2,
            'join_edge': 1,
            'levels': 2,
            'soft_position_limit': 20,
            'prevent_adverse': True,
            'adverse_volume': 15,
        }

    # SIGNALS TO ADD
    rec['signals_to_add'] = []
    if predictors and 1 in predictors:
        strong = [row for row in predictors[1] if abs(row[1]) > 0.3]
        medium = [row for row in predictors[1] if 0.15 < abs(row[1]) <= 0.3]
        for row in strong[:2]:
            rec['signals_to_add'].append(
                f"STRONG: {row[0]} (corr={row[1]:.3f}, R²={row[2]:.3f}) — use as fair value shift"
            )
        for row in medium[:2]:
            rec['signals_to_add'].append(
                f"MEDIUM: {row[0]} (corr={row[1]:.3f}) — use as quote-skew signal"
            )

    return rec


# ═════════════════════════════════════════════════════════════════════════
# Section 8: Forward price path prediction
# ═════════════════════════════════════════════════════════════════════════
def predict_next_500_ticks(p, predictors_result):
    """
    Simple linear combination of top predictors → predicted mid change over 500 ticks.
    This is NOT a tradeable forecast, it's a sanity check on the predictors.
    """
    if 100 not in predictors_result:
        return None
    best = predictors_result[100][:3]
    if not best:
        return None

    last_row = p.iloc[-1]
    # Compute each predictor's last value
    explanation = []
    total = 0
    for name, corr, r2, beta in best:
        if name == 'lag_1_return':
            val = p['l1_mid'].diff().iloc[-1]
        elif name == 'lag_20_return':
            val = p['l1_mid'].diff(20).iloc[-1]
        elif name == 'l2_l1_signal':
            val = last_row['l2_l1']
        elif name == 'microprice_dev':
            val = last_row['micro_dev']
        elif name == 'volume_imbalance':
            val = last_row['vol_imb']
        elif name == 'l1_spread':
            val = last_row['l1_spread']
        else:
            continue
        if pd.isna(val):
            continue
        contribution = beta * val
        total += contribution
        explanation.append(f"  {name}={val:+.3f} × β={beta:+.3f} → {contribution:+.3f}")

    return {
        'predicted_move_500': total,
        'current_mid': last_row['l1_mid'],
        'predicted_mid_500': last_row['l1_mid'] + total,
        'explanation': explanation,
    }


# ═════════════════════════════════════════════════════════════════════════
# Report writer
# ═════════════════════════════════════════════════════════════════════════
def print_report(product, prices, trades, day_label=""):
    p = prices[prices['product'] == product]
    if len(p) == 0:
        print(f"{product} not in data")
        return

    print()
    print("═" * 78)
    print(f" PRODUCT: {product}   {day_label}")
    print("═" * 78)

    # 1. Classify
    print("\n[1] PRODUCT ARCHETYPE CLASSIFICATION")
    c = classify_archetype(p)
    print(f"  Archetype:       {c['archetype']}")
    print(f"  Why:             {c['reason']}")
    print(f"  Mid std:         {c['std']:.2f}")
    print(f"  Mid range:       {c['mid_range']:.0f} ticks")
    print(f"  Day drift:       {c['drift']:+.1f}")
    print(f"  Lag-1 autocorr:  {c['autocorr_1']:.3f}")
    print(f"  Variance ratio (5):  {c['variance_ratio_5']:.3f}  (1=random walk, <1=reverting, >1=trending)")
    print(f"  Variance ratio (20): {c['variance_ratio_20']:.3f}")

    # 2. Fair value
    print("\n[2] FAIR VALUE ESTIMATORS (lower MSE = better predictor of next-tick L1 mid)")
    fv = analyze_fair_value(p)
    print(f"  {'Estimator':<18} {'MSE':<10} {'Bias':<10}")
    for name, mse, bias in fv:
        marker = "  ← BEST" if name == fv[0][0] else ""
        sign = '+' if bias >= 0 else ''
        print(f"  {name:<18} {mse:<10.4f} {sign}{bias:<9.4f}{marker}")

    # 3. Volatility
    print("\n[3] VOLATILITY STRUCTURE")
    v = analyze_volatility(p)
    print(f"  Overall return std:      {v['overall_std']:.3f}")
    print(f"  Quarter stds:            Q1={v['q1_std']:.2f}  Q2={v['q2_std']:.2f}  Q3={v['q3_std']:.2f}  Q4={v['q4_std']:.2f}")
    print(f"  |return| autocorr:       {v['abs_return_autocorr']:.3f}  (>0.1 = volatility clustering)")
    print(f"  Max abs move (1-tick):   {v['max_abs_move']:.1f}")
    print(f"  % zero-moves:            {v['pct_zero_moves']:.1f}%")

    if v['abs_return_autocorr'] > 0.1:
        print(f"  → Volatility clusters. Consider regime-based quote width.")
    q_max = max(v['q1_std'], v['q2_std'], v['q3_std'], v['q4_std'])
    q_min = min(v['q1_std'], v['q2_std'], v['q3_std'], v['q4_std'])
    if q_max > q_min * 2:
        print(f"  → Volatility varies by quarter (max/min = {q_max/q_min:.1f}x). Time-of-day effect.")

    # 4. Predictors
    print("\n[4] FORWARD-RETURN PREDICTORS (linear regression)")
    print("    Tests how well each signal predicts N-tick-ahead returns")
    preds = analyze_predictors(p, trades, product)
    for h in [1, 5, 20, 100]:
        if h not in preds or not preds[h]:
            continue
        print(f"\n  Horizon: {h}-tick forward return")
        print(f"  {'Predictor':<22} {'Corr':<10} {'R²':<10} {'Beta':<12}")
        for name, corr, r2, beta in preds[h][:5]:
            marker = ""
            if abs(corr) > 0.3:
                marker = "  ← STRONG"
            elif abs(corr) > 0.15:
                marker = "  ← medium"
            sign = '+' if corr >= 0 else ''
            sign_b = '+' if beta >= 0 else ''
            print(f"  {name:<22} {sign}{corr:<9.4f} {r2:<10.4f} {sign_b}{beta:<11.4f}{marker}")

    # 5. Flow
    print("\n[5] COUNTERPARTY FLOW PROFILE")
    f = analyze_flow(trades, product, prices)
    if f:
        print(f"  Total trades:        {f['n_trades']}")
        print(f"  Total volume:        {f['total_volume']}")
        print(f"  Volume per tick:     {f['volume_per_tick']:.3f}")
        print(f"  Trade rate (per tick): {f['trade_rate']:.4f}")
        print(f"  Avg trade size:      {f['avg_trade_size']:.1f}")
        print(f"  Median trade size:   {f['median_trade_size']:.0f}")
        print(f"  Buy volume %:        {f['buy_ratio']*100:.1f}%   (50% = balanced)")
        if abs(f['buy_ratio'] - 0.5) > 0.05:
            direction = "BUY" if f['buy_ratio'] > 0.5 else "SELL"
            print(f"  → Imbalanced flow: more {direction} volume. Possible directional bias this day.")

    # 6. Ceiling
    print("\n[6] MARKET MAKING PNL CEILING")
    ceil = estimate_mm_ceiling(f, prices, product)
    if ceil:
        print(f"  Mean L1 spread:      {ceil['mean_spread']:.2f}")
        print(f"  Est. edge per RT:    {ceil['edge_per_trade_est']:.2f} ticks (half_spread - adverse)")
        print(f"  Theoretical upper:   {ceil['upper_bound']:,.0f} SeaShells (capture 100% volume)")
        print(f"  Realistic estimate:  {ceil['realistic_estimate']:,.0f} SeaShells (capture 45%)")
        print(f"  → If your live PnL approaches this, you're near the MM ceiling.")

    # 7. Strategy recommendation
    print("\n[7] RECOMMENDED STRATEGY")
    rec = recommend_strategy(c, fv, v, preds, f, ceil, product)
    print(f"  Strategy:            {rec['strategy']}")
    print(f"  Fair value method:   {rec['fair_value_method']}")
    print(f"  Reasoning:           {rec['why']}")
    print()
    print(f"  Starting parameters (paste into three-phase trader PARAMS dict):")
    print(f"  {product}: {{")
    for k, val in rec['params'].items():
        if isinstance(val, str):
            print(f"      '{k}': '{val}',")
        elif isinstance(val, float):
            print(f"      '{k}': {val:.3f},")
        else:
            print(f"      '{k}': {val},")
    print(f"      'levels': {rec['params']['levels']},")
    print(f"  }},")

    if rec.get('signals_to_add'):
        print()
        print(f"  Signals to incorporate:")
        for s in rec['signals_to_add']:
            print(f"    • {s}")

    # 8. Forward prediction
    print("\n[8] NEXT-500-TICK PRICE PREDICTION (linear model, for sanity check only)")
    pred = predict_next_500_ticks(p, preds)
    if pred:
        sign = '+' if pred['predicted_move_500'] >= 0 else ''
        print(f"  Current mid:         {pred['current_mid']:.2f}")
        print(f"  Predicted move:      {sign}{pred['predicted_move_500']:.2f} ticks")
        print(f"  Predicted mid:       {pred['predicted_mid_500']:.2f}")
        print(f"  Contributors:")
        for line in pred['explanation']:
            print(line)
        print(f"  → NOT a trading signal. Shows which current book features point where.")


def main():
    if len(sys.argv) < 4:
        print("Usage: python3 product_analyzer.py <prices_glob> <trades_glob> <product>")
        print("Example: python3 product_analyzer.py prices_round_0_day_-1.csv trades_round_0_day_-1.csv TOMATOES")
        sys.exit(1)

    prices_glob = sys.argv[1]
    trades_glob = sys.argv[2]
    product = sys.argv[3]

    prices_files = sorted(glob.glob(prices_glob)) or [prices_glob]
    trades_files = sorted(glob.glob(trades_glob)) or [trades_glob]

    for pf, tf in zip(prices_files, trades_files):
        if not os.path.exists(pf) or not os.path.exists(tf):
            print(f"File not found: {pf} or {tf}")
            continue
        prices = load_prices(pf)
        trades = load_trades(tf)
        label = f"[{os.path.basename(pf)}]"
        print_report(product, prices, trades, day_label=label)


if __name__ == '__main__':
    main()
