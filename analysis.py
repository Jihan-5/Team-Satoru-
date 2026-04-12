"""
IMC Prosperity 4 — Multi-panel analysis dashboard (v2)
Outputs 4 self-contained HTML files:
  analysis_EMERALDS_day-2.html   analysis_EMERALDS_day-1.html
  analysis_TOMATOES_day-2.html   analysis_TOMATOES_day-1.html

5 vertically-stacked panels, shared x-axis, spike-crosshair on hover:
  1. Price      — mid + best bid/ask + L2/L3 band + EMA fast/slow +
                  regime shading + bucketed fill triangles + rug plot
  2. Algo action — thin strip: gray=warmup, blue=neutral MM, red=bearish posting
  3. Position   — filled area, auto-zoom y, ±limit annotations, target step line
  4. P&L        — total (green) + realized (dotted) + drawdown shading + fill ticks
  5. Signal     — EMA diff + ±STRONG/MILD threshold lines + L1 imbalance (2nd y)
"""

import json, os, re, subprocess, sys, tempfile
from collections import deque
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Mirror backtester constants ───────────────────────────────────────────────
EMERALD_FV     = 10_000
EMERALD_OFFSET = 8
POS_LIMIT      = 80
EMA_FAST_N     = 9
EMA_SLOW_N     = 16
WARMUP_TICKS   = 20
MILD_THR       = 0.05   # actual value in backtester.py — NOT 1.5
STRONG_THR     = 3.5

ALGO  = "backtester.py"
ROUND = "0"


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA ACQUISITION
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest() -> str:
    tmp = tempfile.mktemp(suffix=".log")
    r   = subprocess.run(
        [sys.executable, "-m", "prosperity4bt", ALGO, ROUND, "--out", tmp],
        capture_output=True, text=True,
    )
    print(r.stdout.strip())
    if not os.path.exists(tmp):
        raise RuntimeError(r.stderr)
    return tmp


def parse_log(path: str):
    with open(path) as f:
        raw = f.read()

    act_start = raw.index("Activities log:") + len("Activities log:")
    act_end   = raw.rfind("\n[")
    rows = []
    for line in raw[act_start:act_end].strip().splitlines():
        if line.startswith("day"):
            continue
        p = line.split(";")
        if len(p) < 16:
            continue
        fv = lambda x: float(x) if x else None
        iv = lambda x: int(x)   if x else None
        try:
            rows.append({
                "day": int(p[0]),  "ts":  int(p[1]),   "product": p[2],
                "bp1": fv(p[3]),   "bv1": iv(p[4]),
                "bp2": fv(p[5]),   "bv2": iv(p[6]),
                "bp3": fv(p[7]),   "bv3": iv(p[8]),
                "ap1": fv(p[9]),   "av1": iv(p[10]),
                "ap2": fv(p[11]),  "av2": iv(p[12]),
                "ap3": fv(p[13]),  "av3": iv(p[14]),
                "mid": fv(p[15]),
                "pnl": float(p[16]) if len(p) > 16 and p[16] else 0.0,
            })
        except (ValueError, IndexError):
            continue

    trade_raw = raw[act_end:].strip()
    trade_raw = re.sub(r",(\s*[}\]])", r"\1", trade_raw)
    trades    = json.loads(trade_raw)
    return rows, trades


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SIGNAL COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def ema(arr: np.ndarray, n: int) -> np.ndarray:
    a   = 2.0 / (n + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1.0 - a) * out[i - 1]
    return out


def regime_array(mid: np.ndarray, fast: np.ndarray, slow: np.ndarray) -> np.ndarray:
    """
    0 = neutral      |diff| < MILD_THR and no momentum
    1 = mild bearish  momentum-only or -STRONG ≤ diff < -MILD after warmup
    2 = strong bearish diff < -STRONG after warmup
    3 = mild bullish  diff > MILD after warmup
    4 = strong bullish diff > STRONG after warmup
    """
    n    = len(mid)
    diff = fast - slow
    mom  = np.zeros(n, bool)
    mom[1:] = mid[1:] < mid[:-1]
    warm = np.arange(n) >= WARMUP_TICKS

    r = np.zeros(n, int)
    r[warm & (diff >  MILD_THR)]   = 3
    r[warm & (diff >  STRONG_THR)] = 4
    r[mom  & ~(warm & (diff > MILD_THR))] = 1
    r[warm & (diff < -MILD_THR)]   = 1
    r[warm & (diff < -STRONG_THR)] = 2
    return r


def fifo_realized(fills: list) -> list:
    longs, shorts = deque(), deque()
    realized, cum = 0.0, []
    for f in fills:
        qty, price = f["qty"], f["price"]
        if qty > 0:
            rem = qty
            while rem > 0 and shorts:
                sq, sp = shorts[0]
                m = min(sq, rem);  realized += m * (sp - price)
                rem -= m;  sq -= m
                if sq == 0: shorts.popleft()
                else:       shorts[0] = (sq, sp)
            if rem > 0: longs.append((rem, price))
        else:
            rem = -qty
            while rem > 0 and longs:
                lq, lp = longs[0]
                m = min(lq, rem);  realized += m * (price - lp)
                rem -= m;  lq -= m
                if lq == 0: longs.popleft()
                else:       longs[0] = (lq, lp)
            if rem > 0: shorts.append((rem, price))
        cum.append(realized)
    return cum


def pos_from_fills(fills: list, tick_ts: np.ndarray) -> np.ndarray:
    pos, arr, fi = 0, np.zeros(len(tick_ts), int), 0
    for i, t in enumerate(tick_ts):
        while fi < len(fills) and fills[fi]["ts"] <= t:
            pos += fills[fi]["qty"];  fi += 1
        arr[i] = pos
    return arr


def bucket_fills(fills: list, window: float = 5.0) -> list:
    """Group fills into window-second buckets."""
    if not fills:
        return []
    buckets: dict = {}
    for f in fills:
        b = int(f["ts"] / window)
        if b not in buckets:
            buckets[b] = {"ts": (b + 0.5) * window, "net": 0,
                          "buys": 0, "sells": 0, "prices": []}
        buckets[b]["net"]    += f["qty"]
        buckets[b]["prices"].append(f["price"])
        if f["qty"] > 0: buckets[b]["buys"]  += f["qty"]
        else:            buckets[b]["sells"] += abs(f["qty"])
    result = []
    for b in sorted(buckets):
        bk = buckets[b]
        bk["avg_price"] = float(np.mean(bk["prices"]))
        result.append(bk)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 3.  PER-DAY DATA EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract(rows, trades, product, day, day_idx):
    book = sorted(
        [r for r in rows if r["product"] == product and r["day"] == day],
        key=lambda r: r["ts"],
    )
    if not book:
        return None

    t0  = book[0]["ts"]
    ts  = np.array([(r["ts"] - t0) / 1000.0 for r in book])

    def col(k): return np.array(
        [r[k] if r[k] is not None else np.nan for r in book])

    mid  = col("mid")
    bp1, bv1 = col("bp1"), col("bv1")
    bp2, bv2 = col("bp2"), col("bv2")
    bp3, bv3 = col("bp3"), col("bv3")
    ap1, av1 = col("ap1"), col("av1")
    ap2, av2 = col("ap2"), col("av2")
    ap3, av3 = col("ap3"), col("av3")

    nans = np.isnan(mid)
    if nans.any() and (~nans).any():
        mid[nans] = np.interp(np.flatnonzero(nans),
                              np.flatnonzero(~nans), mid[~nans])

    pnl_raw = np.array([r["pnl"] for r in book])
    pnl     = pnl_raw - pnl_raw[0]

    fast_arr = ema(mid, EMA_FAST_N)
    slow_arr = ema(mid, EMA_SLOW_N)
    diff_arr = fast_arr - slow_arr
    reg      = regime_array(mid, fast_arr, slow_arr)

    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = np.where((bv1 > 0) & (av1 > 0),
                             (bv1 - av1) / (bv1 + av1), 0.0)

    # Fills
    day_offset_ms = day_idx * 1_000_000
    fills_raw = []
    for t in trades:
        if t["symbol"] != product:
            continue
        lms = t["timestamp"] - day_offset_ms
        if not (0 <= lms < 1_000_000):
            continue
        is_buy  = t.get("buyer")  == "SUBMISSION"
        is_sell = t.get("seller") == "SUBMISSION"
        if not (is_buy or is_sell):
            continue
        fills_raw.append({
            "ts":    lms / 1000.0,
            "price": float(t["price"]),
            "qty":   int(t["quantity"]) * (1 if is_buy else -1),
        })
    fills_raw.sort(key=lambda f: f["ts"])

    rpnl_vals = fifo_realized(fills_raw)
    rpnl_ts   = [f["ts"] for f in fills_raw]
    rpnl_tick = (np.interp(ts, rpnl_ts, rpnl_vals)
                 if rpnl_ts else np.zeros(len(ts)))

    pos_tick  = pos_from_fills(fills_raw, ts)

    # ── EMERALDS-specific signals ─────────────────────────────────────────────
    # bid_dist / ask_dist: how far best bid/ask is from FV (9992→ -8, 10001→ +1)
    # Positive ask_dist_inv means ask crossed below FV → free take-buy opportunity
    em_bid_dist     = np.where(np.isnan(bp1), np.nan, bp1 - EMERALD_FV)
    em_ask_dist     = np.where(np.isnan(ap1), np.nan, ap1 - EMERALD_FV)
    em_take_buy     = (~np.isnan(ap1)) & (ap1 < EMERALD_FV)   # ask below FV
    em_take_sell    = (~np.isnan(bp1)) & (bp1 > EMERALD_FV)   # bid above FV
    em_spread       = np.where(~(np.isnan(ap1) | np.isnan(bp1)), ap1 - bp1, np.nan)

    # Algo action: 0=warmup/idle, 1=neutral posting, 2=bearish posting
    if product == "TOMATOES":
        action = np.ones(len(ts), int)
        action[:WARMUP_TICKS] = 0
        action[(np.arange(len(ts)) >= WARMUP_TICKS) & (reg >= 1)] = 2
    else:  # EMERALDS: 0=take opportunity, 1=making passive, 2=near limit
        action = np.ones(len(ts), int)
        action[em_take_buy | em_take_sell] = 0
        action[np.abs(pos_tick) > int(POS_LIMIT * 0.75)] = 2

    # Target position
    if product == "TOMATOES":
        target = np.where(reg >= 1, float(-POS_LIMIT * 0.5), 0.0)
    else:
        target = np.zeros(len(ts))

    # Summary stats
    dd = float(np.max(np.maximum.accumulate(pnl) - pnl)) if len(pnl) else 0.0
    tick_ms = 100
    rt = {
        "neutral":        int(np.sum(reg == 0)) * tick_ms / 1000,
        "mild_bearish":   int(np.sum(reg == 1)) * tick_ms / 1000,
        "strong_bearish": int(np.sum(reg == 2)) * tick_ms / 1000,
        "mild_bullish":   int(np.sum(reg == 3)) * tick_ms / 1000,
        "strong_bullish": int(np.sum(reg == 4)) * tick_ms / 1000,
    }

    return dict(
        ts=ts, mid=mid, pnl=pnl, rpnl=rpnl_tick,
        bp1=bp1, bv1=bv1, bp2=bp2, bv2=bv2, bp3=bp3, bv3=bv3,
        ap1=ap1, av1=av1, ap2=ap2, av2=av2, ap3=ap3, av3=av3,
        fast=fast_arr, slow=slow_arr, diff=diff_arr,
        reg=reg, imbalance=imbalance,
        pos=pos_tick, target=target, action=action,
        fills=fills_raw,
        # EMERALDS-specific
        em_bid_dist=em_bid_dist, em_ask_dist=em_ask_dist,
        em_take_buy=em_take_buy, em_take_sell=em_take_sell,
        em_spread=em_spread,
        total_fills=len(fills_raw),
        total_volume=sum(abs(f["qty"]) for f in fills_raw),
        final_pnl=float(pnl[-1]) if len(pnl) else 0.0,
        drawdown=dd,
        regime_time=rt,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CHART HELPERS
# ══════════════════════════════════════════════════════════════════════════════

REGIME_BG = {
    1: "rgba(239,68,68,0.07)",
    2: "rgba(185,28,28,0.12)",
    3: "rgba(34,197,94,0.07)",
    4: "rgba(21,128,62,0.12)",
}
ACTION_COLORS = {
    0: "rgba(100,116,139,0.55)",   # gray — warmup/idle  OR  emerald take opportunity
    1: "rgba(56,189,248,0.45)",    # sky blue — neutral/making posting
    2: "rgba(249,115,22,0.55)",    # orange — bearish posting  OR  near limit
}

ACTION_LABELS_TOMATOES = {
    0: "Warmup / idle",
    1: "Passive MM — neutral (L1/L1)",
    2: "Passive MM — bearish (deep bid + L1 ask)",
}
ACTION_LABELS_EMERALDS = {
    0: "Take opportunity (crossing FV=10000)",
    1: "Making — posting 9992 / 10008",
    2: "Near position limit (|pos| > 60)",
}


def _build_spans(arr, ts):
    """Return list of (x0, x1, value) contiguous spans, merging spans < 1s."""
    if len(arr) == 0:
        return []
    spans, start, cur = [], float(ts[0]), arr[0]
    for t, v in zip(ts[1:], arr[1:]):
        if v != cur:
            spans.append([float(start), float(t), int(cur)])
            start, cur = float(t), v
    spans.append([float(start), float(ts[-1]), int(cur)])
    # Merge tiny adjacent same-value spans
    merged = []
    for s in spans:
        if merged and s[2] == merged[-1][2]:
            merged[-1][1] = s[1]
        elif s[1] - s[0] < 1.0 and merged:
            merged[-1][1] = s[1]   # absorb into previous
        else:
            merged.append(s)
    return merged


def add_vrects_from_spans(fig, spans, color_map, row, opacity_override=None):
    for x0, x1, val in spans:
        color = color_map.get(val)
        if not color:
            continue
        if opacity_override is not None:
            # Replace rgba alpha
            import re as _re
            color = _re.sub(r"(rgba\(\d+,\d+,\d+,)([\d.]+)(\))",
                            lambda m: f"{m.group(1)}{opacity_override}{m.group(3)}",
                            color)
        fig.add_vrect(x0=x0, x1=x1, fillcolor=color,
                      layer="below", line_width=0, row=row, col=1)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURE CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_figure(d, product, day):
    ts  = d["ts"]
    pnl = d["pnl"]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = make_subplots(
        rows=5, cols=1,
        shared_xaxes=True,
        row_heights=[0.42, 0.04, 0.17, 0.22, 0.15],
        vertical_spacing=0.012,
        specs=[
            [{"secondary_y": False}],  # 1. price
            [{"secondary_y": False}],  # 2. algo action strip
            [{"secondary_y": False}],  # 3. position
            [{"secondary_y": False}],  # 4. pnl
            [{"secondary_y": True}],   # 5. signal + imbalance
        ],
    )

    # ── Master hover aggregator (invisible, on price panel) ───────────────────
    is_em = (product == "EMERALDS")
    cdata = np.column_stack([
        d["mid"],
        np.where(np.isnan(d["bp1"]), 0, d["bp1"]),
        np.where(np.isnan(d["ap1"]), 0, d["ap1"]),
        d["fast"],
        d["slow"],
        d["diff"],
        d["pos"].astype(float),
        pnl,
        d["rpnl"],
        np.where(np.isnan(d["em_bid_dist"]), 0, d["em_bid_dist"]),
        np.where(np.isnan(d["em_ask_dist"]), 0, d["em_ask_dist"]),
    ])

    if is_em:
        MASTER_HOVER = (
            "<b>t = %{x:.1f}s</b><br>"
            "Mid: %{customdata[0]:.2f}  FV=10000<br>"
            "Bid: %{customdata[1]:.2f}  ·  Ask: %{customdata[2]:.2f}<br>"
            "Bid dist from FV: %{customdata[9]:+.2f}<br>"
            "Ask dist from FV: %{customdata[10]:+.2f}<br>"
            "Position: %{customdata[6]:.0f}<br>"
            "Total PnL: %{customdata[7]:.0f}<br>"
            "Realized PnL: %{customdata[8]:.0f}"
            "<extra></extra>"
        )
    else:
        MASTER_HOVER = (
            "<b>t = %{x:.1f}s</b><br>"
            "Mid: %{customdata[0]:.2f}<br>"
            "Bid: %{customdata[1]:.2f}  ·  Ask: %{customdata[2]:.2f}<br>"
            "EMA fast: %{customdata[3]:.4f}<br>"
            "EMA slow: %{customdata[4]:.4f}<br>"
            "EMA diff: %{customdata[5]:.4f}<br>"
            "Position: %{customdata[6]:.0f}<br>"
            "Total PnL: %{customdata[7]:.0f}<br>"
            "Realized PnL: %{customdata[8]:.0f}"
            "<extra></extra>"
        )
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["mid"].tolist(),
        mode="markers",
        marker=dict(opacity=0, size=1),
        showlegend=False,
        name="__hover__",
        customdata=cdata,
        hovertemplate=MASTER_HOVER,
    ), row=1, col=1)

    # ══════════════════════════════════
    #  ROW 1: PRICE PANEL
    # ══════════════════════════════════

    # Regime background shading
    reg_spans = _build_spans(d["reg"], ts)
    add_vrects_from_spans(fig, reg_spans, REGIME_BG, row=1)

    # L2/L3 depth band (very faint fill)
    valid3 = ~(np.isnan(d["bp3"]) | np.isnan(d["ap3"]))
    if valid3.any():
        tv = ts[valid3].tolist()
        b3v = d["bp3"][valid3].tolist()
        a3v = d["ap3"][valid3].tolist()
        fig.add_trace(go.Scatter(
            x=tv + tv[::-1], y=b3v + a3v[::-1],
            fill="toself", fillcolor="rgba(148,163,184,0.06)",
            line=dict(width=0),
            name="L2–L3 depth", legendgroup="depth",
            hoverinfo="skip",
        ), row=1, col=1)

    # EMERALDS: FV reference line + take-opportunity highlights
    if is_em:
        fig.add_shape(type="line",
                      x0=float(ts[0]), x1=float(ts[-1]),
                      y0=EMERALD_FV, y1=EMERALD_FV,
                      line=dict(color="#fbbf24", width=1.2, dash="dot"),
                      row=1, col=1)
        fig.add_annotation(x=float(ts[-1]), y=EMERALD_FV,
                           text=" FV=10000", font=dict(size=8, color="#fbbf24"),
                           showarrow=False, xanchor="left", row=1, col=1)
        # Shade take-buy windows (ask < FV) — bright green flash
        buy_spans = _build_spans(d["em_take_buy"].astype(int), ts)
        for x0, x1, v in buy_spans:
            if v == 1:
                fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(74,222,128,0.25)",
                              layer="above", line_width=0, row=1, col=1)
        # Shade take-sell windows (bid > FV) — bright red flash
        sell_spans = _build_spans(d["em_take_sell"].astype(int), ts)
        for x0, x1, v in sell_spans:
            if v == 1:
                fig.add_vrect(x0=x0, x1=x1, fillcolor="rgba(248,113,113,0.25)",
                              layer="above", line_width=0, row=1, col=1)

    # Best bid / ask — thin, translucent
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["bp1"].tolist(), mode="lines",
        line=dict(color="rgba(56,189,248,0.30)", width=1),
        name="Best bid", hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["ap1"].tolist(), mode="lines",
        line=dict(color="rgba(248,113,113,0.30)", width=1),
        name="Best ask", hoverinfo="skip",
    ), row=1, col=1)

    if is_em:
        # EMERALDS: show our passive quote levels instead of EMAs
        for lvl, clr, nm in [
            (EMERALD_FV - EMERALD_OFFSET, "#4ade80", f"Our bid ({EMERALD_FV - EMERALD_OFFSET})"),
            (EMERALD_FV + EMERALD_OFFSET, "#f87171", f"Our ask ({EMERALD_FV + EMERALD_OFFSET})"),
        ]:
            fig.add_shape(type="line",
                          x0=float(ts[0]), x1=float(ts[-1]),
                          y0=lvl, y1=lvl,
                          line=dict(color=clr, width=1.5, dash="dash"),
                          row=1, col=1)
            fig.add_annotation(x=float(ts[-1]), y=lvl,
                               text=f" {nm}", font=dict(size=8, color=clr),
                               showarrow=False, xanchor="left", row=1, col=1)
    else:
        # EMA fast — bright yellow, prominent
        fig.add_trace(go.Scatter(
            x=ts.tolist(), y=d["fast"].tolist(), mode="lines",
            line=dict(color="#facc15", width=2.5),
            name=f"EMA {EMA_FAST_N} (fast)",
            hovertemplate=f"EMA{EMA_FAST_N}: %{{y:.3f}}<extra></extra>",
        ), row=1, col=1)
        # EMA slow — bright purple, prominent
        fig.add_trace(go.Scatter(
            x=ts.tolist(), y=d["slow"].tolist(), mode="lines",
            line=dict(color="#c084fc", width=2.5),
            name=f"EMA {EMA_SLOW_N} (slow)",
            hovertemplate=f"EMA{EMA_SLOW_N}: %{{y:.3f}}<extra></extra>",
        ), row=1, col=1)

    # Mid price — main focus, thick dark line
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["mid"].tolist(), mode="lines",
        line=dict(color="#f8fafc", width=2.2),
        name="Mid price",
        hoverinfo="skip",
    ), row=1, col=1)

    # Bucketed fill triangles (5-second windows), placed at actual fill price
    bucketed = bucket_fills(d["fills"], window=5.0)
    buy_bkts  = [b for b in bucketed if b["net"] > 0]
    sell_bkts = [b for b in bucketed if b["net"] < 0]
    mix_bkts  = [b for b in bucketed if b["net"] == 0 and b["buys"] > 0]

    for bkts, sym, clr, nm in [
        (buy_bkts,  "triangle-up",   "#22c55e", "Buy (5s bucket)"),
        (sell_bkts, "triangle-down", "#ef4444", "Sell (5s bucket)"),
        (mix_bkts,  "circle",        "#94a3b8", "Mixed (5s bucket)"),
    ]:
        if not bkts:
            continue
        fig.add_trace(go.Scatter(
            x=[b["ts"] for b in bkts],
            y=[b["avg_price"] for b in bkts],
            mode="markers",
            marker=dict(
                symbol=sym, color=clr,
                size=[max(abs(b["net"]) * 1.8, 5) for b in bkts],
                opacity=0.70,
                line=dict(width=0.5, color="rgba(255,255,255,0.5)"),
            ),
            name=nm,
            hovertemplate=(
                f"<b>{nm}</b><br>"
                "t=%{x:.1f}s  avg_px=%{y:.2f}<br>"
                "net=%{customdata[0]}  buys=%{customdata[1]}  sells=%{customdata[2]}"
                "<extra></extra>"
            ),
            customdata=[[b["net"], b["buys"], b["sells"]] for b in bkts],
        ), row=1, col=1)

    # Rug plot — individual fill marks along the bottom of the price panel
    if d["fills"]:
        p_min = float(np.nanmin(d["bp3"][~np.isnan(d["bp3"])])) if (~np.isnan(d["bp3"])).any() else float(np.nanmin(d["mid"]))
        p_max = float(np.nanmax(d["ap3"][~np.isnan(d["ap3"])])) if (~np.isnan(d["ap3"])).any() else float(np.nanmax(d["mid"]))
        rug_y = p_min - (p_max - p_min) * 0.012
        rug_cols = ["#22c55e" if f["qty"] > 0 else "#ef4444" for f in d["fills"]]
        fig.add_trace(go.Scatter(
            x=[f["ts"] for f in d["fills"]],
            y=[rug_y] * len(d["fills"]),
            mode="markers",
            marker=dict(symbol="line-ns", size=8,
                        color=rug_cols,
                        line=dict(width=1.2, color=rug_cols)),
            name="Fill rug",
            showlegend=False,
            hovertemplate="Fill: qty=%{customdata[0]} @ %{customdata[1]}<extra></extra>",
            customdata=[[abs(f["qty"]), f["price"]] for f in d["fills"]],
        ), row=1, col=1)

    # ══════════════════════════════════
    #  ROW 2: ALGO ACTION STRIP
    # ══════════════════════════════════

    # Invisible anchor trace so the subplot axis exists
    fig.add_trace(go.Scatter(
        x=[ts[0], ts[-1]], y=[0.5, 0.5],
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=2, col=1)

    action_spans = _build_spans(d["action"], ts)
    add_vrects_from_spans(fig, action_spans, ACTION_COLORS, row=2)

    # Legend entries for algo action (invisible dots in legend)
    action_labels = ACTION_LABELS_EMERALDS if is_em else ACTION_LABELS_TOMATOES
    for val, label in action_labels.items():
        color = ACTION_COLORS[val].replace("0.55", "0.8").replace("0.45", "0.8")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=color, symbol="square"),
            name=label, legendgroup=f"action_{val}",
        ), row=2, col=1)

    # ══════════════════════════════════
    #  ROW 3: POSITION PANEL
    # ══════════════════════════════════

    pos  = d["pos"].astype(float)
    pmin = float(pos.min())
    pmax = float(pos.max())
    pad  = max((pmax - pmin) * 0.25, 5.0)
    y_lo = max(pmin - pad, -POS_LIMIT - 10)
    y_hi = min(pmax + pad,  POS_LIMIT + 10)

    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=pos.tolist(),
        mode="lines",
        line=dict(color="#38bdf8", width=1.5),
        fill="tozeroy", fillcolor="rgba(56,189,248,0.10)",
        name="Position",
        hovertemplate="Position: %{y:.0f}<extra></extra>",
    ), row=3, col=1)

    # Target step line
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["target"].tolist(),
        mode="lines",
        line=dict(color="#94a3b8", width=1.2, dash="dash"),
        name="Target position",
        hovertemplate="Target: %{y:.0f}<extra></extra>",
        line_shape="hv",
    ), row=3, col=1)

    # ±Limit as shapes + annotations (not gridlines)
    for lim, clr in [(POS_LIMIT, "#f87171"), (-POS_LIMIT, "#f87171")]:
        fig.add_shape(type="line",
                      x0=float(ts[0]), x1=float(ts[-1]), y0=lim, y1=lim,
                      line=dict(color=clr, width=0.9, dash="dash"),
                      row=3, col=1)
        fig.add_annotation(
            x=float(ts[-1]), y=float(lim),
            text=f" {lim:+d} limit",
            font=dict(size=8, color=clr),
            showarrow=False, xanchor="left", yanchor="middle",
            row=3, col=1,
        )
    fig.add_shape(type="line",
                  x0=float(ts[0]), x1=float(ts[-1]), y0=0, y1=0,
                  line=dict(color="#475569", width=0.5, dash="dot"),
                  row=3, col=1)

    # ══════════════════════════════════
    #  ROW 4: P&L PANEL
    # ══════════════════════════════════

    running_max = np.maximum.accumulate(pnl)
    pnl_lo = float(np.min(pnl))

    # Drawdown shading: fill between running_max and pnl
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=running_max.tolist(),
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
        name="__dd_top__",
    ), row=4, col=1)
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=pnl.tolist(),
        fill="tonexty", fillcolor="rgba(239,68,68,0.13)",
        mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
        name="__dd_bot__",
    ), row=4, col=1)

    # Realized PnL
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["rpnl"].tolist(),
        mode="lines", line=dict(color="#64748b", width=1.1, dash="dot"),
        name="Realized PnL",
        hovertemplate="Realized: %{y:.0f}<extra></extra>",
    ), row=4, col=1)

    # Total PnL
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=pnl.tolist(),
        mode="lines", line=dict(color="#4ade80", width=2.2),
        name="Total PnL",
        hovertemplate="Total PnL: %{y:.0f}<extra></extra>",
    ), row=4, col=1)

    # Final value annotation
    fig.add_annotation(
        x=float(ts[-1]), y=float(pnl[-1]),
        text=f"  {d['final_pnl']:+,.0f}",
        font=dict(size=11, color="#4ade80", family="monospace"),
        showarrow=False, xanchor="left", row=4, col=1,
    )

    # Fill tick marks along bottom of PnL panel
    if d["fills"]:
        rug_pnl = pnl_lo - abs(pnl_lo) * 0.04 - 5
        fig.add_trace(go.Scatter(
            x=[f["ts"] for f in d["fills"]],
            y=[rug_pnl] * len(d["fills"]),
            mode="markers",
            marker=dict(symbol="line-ns-open", size=7,
                        line=dict(width=1, color="rgba(148,163,184,0.45)")),
            showlegend=False, hoverinfo="skip", name="__fill_ticks__",
        ), row=4, col=1)

    # ══════════════════════════════════
    #  ROW 5: SIGNAL PANEL  (product-specific)
    # ══════════════════════════════════

    if is_em:
        # ── EMERALDS signal: bid/ask distance from FV ─────────────────────────
        # bid_dist = bp1 − 10000  (normally ~ −8; positive = bid crossed FV → free edge)
        # ask_dist = ap1 − 10000  (normally ~ +8; negative = ask crossed FV → free edge)
        fig.add_shape(type="line",
                      x0=float(ts[0]), x1=float(ts[-1]), y0=0, y1=0,
                      line=dict(color="#fbbf24", width=1.0, dash="dash"),
                      row=5, col=1)
        fig.add_annotation(x=float(ts[-1]), y=0, text=" FV crossing",
                           font=dict(size=8, color="#fbbf24"),
                           showarrow=False, xanchor="left", row=5, col=1)
        for lvl, clr, lbl in [
            (-EMERALD_OFFSET, "#4ade80", f" Our bid dist (−{EMERALD_OFFSET})"),
            ( EMERALD_OFFSET, "#f87171", f" Our ask dist (+{EMERALD_OFFSET})"),
        ]:
            fig.add_shape(type="line",
                          x0=float(ts[0]), x1=float(ts[-1]), y0=lvl, y1=lvl,
                          line=dict(color=clr, width=0.8, dash="dot"),
                          row=5, col=1)
            fig.add_annotation(x=float(ts[-1]), y=lvl, text=lbl,
                               font=dict(size=8, color=clr),
                               showarrow=False, xanchor="left", row=5, col=1)

        fig.add_trace(go.Scatter(
            x=ts.tolist(), y=d["em_bid_dist"].tolist(),
            mode="lines", line=dict(color="#38bdf8", width=1.8),
            name="Best bid − FV",
            hovertemplate="Bid dist: %{y:+.2f}<extra></extra>",
        ), row=5, col=1)
        fig.add_trace(go.Scatter(
            x=ts.tolist(), y=d["em_ask_dist"].tolist(),
            mode="lines", line=dict(color="#f87171", width=1.8),
            name="Best ask − FV",
            hovertemplate="Ask dist: %{y:+.2f}<extra></extra>",
        ), row=5, col=1)

        # Spread (ap1 − bp1) on secondary y
        fig.add_trace(go.Scatter(
            x=ts.tolist(), y=d["em_spread"].tolist(),
            mode="lines", line=dict(color="#a78bfa", width=0.9, dash="dot"),
            name="Market spread (ask−bid)",
            opacity=0.65,
            hovertemplate="Spread: %{y:.2f}<extra></extra>",
        ), row=5, col=1, secondary_y=True)

        fig.update_yaxes(title_text="Dist from FV", row=5, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Spread",        row=5, col=1, secondary_y=True,
                         range=[0, 30],
                         tickfont=dict(color="#a78bfa", size=9),
                         gridcolor="#1e3a5f")

    else:
        # ── TOMATOES signal: EMA diff + imbalance ─────────────────────────────
        for thr, clr, lbl in [
            ( STRONG_THR, "#22c55e",  f"+STRONG = {STRONG_THR}"),
            ( MILD_THR,   "#86efac",  f"+MILD = {MILD_THR}"),
            ( 0,          "#475569",  "0"),
            (-MILD_THR,   "#fca5a5",  f"−MILD = {MILD_THR}"),
            (-STRONG_THR, "#ef4444",  f"−STRONG = {STRONG_THR}"),
        ]:
            fig.add_shape(type="line",
                          x0=float(ts[0]), x1=float(ts[-1]),
                          y0=float(thr), y1=float(thr),
                          line=dict(color=clr, width=0.8, dash="dash"),
                          row=5, col=1)
            if thr != 0:
                fig.add_annotation(
                    x=float(ts[-1]), y=float(thr),
                    text=f" {lbl}", font=dict(size=8, color=clr),
                    showarrow=False, xanchor="left", yanchor="middle",
                    row=5, col=1,
                )

        fig.add_trace(go.Scatter(
            x=ts.tolist(), y=d["diff"].tolist(),
            mode="lines", line=dict(color="#facc15", width=1.8),
            name="EMA diff (fast−slow)",
            hovertemplate="Diff: %{y:.4f}<extra></extra>",
        ), row=5, col=1)

        fig.add_trace(go.Scatter(
            x=ts.tolist(), y=d["imbalance"].tolist(),
            mode="lines", line=dict(color="#38bdf8", width=0.8, dash="dot"),
            name="L1 vol imbalance", opacity=0.55,
            hovertemplate="Imbalance: %{y:.2f}<extra></extra>",
        ), row=5, col=1, secondary_y=True)

        fig.update_yaxes(title_text="EMA diff",  row=5, col=1, secondary_y=False)
        fig.update_yaxes(title_text="Imbalance", row=5, col=1, secondary_y=True,
                         range=[-1.5, 1.5],
                         tickfont=dict(color="#38bdf8", size=9),
                         gridcolor="#1e3a5f")

    # ══════════════════════════════════
    #  GLOBAL LAYOUT
    # ══════════════════════════════════

    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#cbd5e1", size=11, family="monospace"),
        hovermode="x",
        height=980,
        title=dict(
            text=f"<b>{product}</b>  ·  Day {day:+d}  ·  Round 0",
            font=dict(size=15, color="#f1f5f9"),
            x=0.5,
        ),
        legend=dict(
            orientation="v", x=1.01, y=0.98,
            bgcolor="rgba(15,23,42,0.9)",
            bordercolor="#334155", borderwidth=1,
            font=dict(size=9.5),
        ),
        margin=dict(l=70, r=210, t=55, b=50),
    )

    # Spike crosshair on all x-axes (draws vertical line across all panels)
    spike_kw = dict(showspikes=True, spikemode="across",
                    spikethickness=1, spikecolor="#64748b",
                    spikedash="dot", spikesnap="cursor")
    grid_kw  = dict(gridcolor="#1e3a5f", zerolinecolor="#334155")

    for r in range(1, 6):
        fig.update_xaxes(row=r, col=1, **spike_kw, **grid_kw)
        fig.update_yaxes(row=r, col=1, **grid_kw)

    # Per-axis customisation
    fig.update_yaxes(title_text="Price",    row=1, col=1)
    fig.update_yaxes(row=2, col=1,
                     showticklabels=False, showgrid=False,
                     zeroline=False, showline=False, range=[0, 1])
    fig.update_yaxes(title_text="Position", row=3, col=1, range=[y_lo, y_hi])
    fig.update_yaxes(title_text="P&L",      row=4, col=1)
    # Row 5 y-axis titles set inside the product-specific branches above
    fig.update_xaxes(title_text="Time (seconds into day)", row=5, col=1)

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6.  HTML OUTPUT
# ══════════════════════════════════════════════════════════════════════════════

def _summary_html(d, product, day) -> str:
    """Build the per-day stats block (returned as an HTML string)."""
    rt   = d["regime_time"]
    ttot = max(sum(rt.values()), 1)

    em_note = ""
    if product == "EMERALDS":
        take_buys  = int(np.sum(d["em_take_buy"]))
        take_sells = int(np.sum(d["em_take_sell"]))
        em_note = (
            f'<div style="background:#1c2f4a;border:1px solid #334155;border-radius:6px;'
            f'padding:10px 16px;margin-top:10px;font-size:12px;color:#94a3b8;">'
            f'⚡ <b style="color:#f1f5f9">EMERALDS — Stoikov MM on a pegged price</b><br>'
            f'FV=10,000. Passive quotes: bid 9992 / ask 10008 (±8 ticks).<br>'
            f'<b style="color:#4ade80">Green flashes</b>: ask&lt;FV (take-buy) — {take_buys} ticks · '
            f'<b style="color:#f87171">Red flashes</b>: bid&gt;FV (take-sell) — {take_sells} ticks<br>'
            f'Signal panel: bid/ask dist from FV + market spread. '
            f'Algo strip: gray=take opp, blue=making, orange=near limit.</div>'
        )

    return (
        f'<div style="font-family:monospace;background:#1e293b;color:#cbd5e1;'
        f'padding:16px 28px;border-radius:8px;line-height:1.85;font-size:13px;">'
        f'<b style="color:#f1f5f9;font-size:14px;">{product} · Day {day:+d} · Summary</b>'
        f'{em_note}<br>'
        f'<table style="border-collapse:collapse;width:100%;margin-top:4px">'
        f'<tr><td style="padding:2px 18px 2px 0;color:#94a3b8">Total fills</td>'
        f'<td><b style="color:#f1f5f9">{d["total_fills"]}</b></td>'
        f'<td style="padding:2px 18px 2px 24px;color:#94a3b8">Total volume</td>'
        f'<td><b style="color:#f1f5f9">{d["total_volume"]}</b></td></tr>'
        f'<tr><td style="color:#94a3b8">Final PnL</td>'
        f'<td><b style="color:#4ade80">{d["final_pnl"]:+,.0f}</b></td>'
        f'<td style="padding:2px 18px 2px 24px;color:#94a3b8">Max drawdown</td>'
        f'<td><b style="color:#f87171">{d["drawdown"]:,.0f}</b></td></tr>'
        f'</table><br>'
        f'<b style="color:#94a3b8">Regime (MILD={MILD_THR}, STRONG={STRONG_THR}):</b><br>'
        f'<span style="color:#64748b"> Neutral </span><b>{rt["neutral"]:.0f}s ({rt["neutral"]/ttot*100:.0f}%)</b> · '
        f'<span style="color:#fca5a5">Mild bear </span><b>{rt["mild_bearish"]:.0f}s ({rt["mild_bearish"]/ttot*100:.0f}%)</b> · '
        f'<span style="color:#ef4444">Strong bear </span><b>{rt["strong_bearish"]:.0f}s ({rt["strong_bearish"]/ttot*100:.0f}%)</b><br>'
        f'<span style="color:#86efac">Mild bull </span><b>{rt["mild_bullish"]:.0f}s ({rt["mild_bullish"]/ttot*100:.0f}%)</b> · '
        f'<span style="color:#4ade80">Strong bull </span><b>{rt["strong_bullish"]:.0f}s ({rt["strong_bullish"]/ttot*100:.0f}%)</b>'
        f'</div>'
    )


def make_combined_html(days_data: list, product: str) -> str:
    """
    days_data: list of (day, d, fig) tuples — one per day.
    Returns a single HTML with a dropdown to switch between days.
    Uses Plotly.react() to swap figures without reloading.
    """
    import json as _json

    options_html = "\n".join(
        f'    <option value="{i}">Day {day:+d}</option>'
        for i, (day, _, _) in enumerate(days_data)
    )

    summaries_js = _json.dumps({
        str(i): _summary_html(d, product, day)
        for i, (day, d, _) in enumerate(days_data)
    })

    # Serialize each figure to JSON for Plotly.react()
    figs_js = "[\n" + ",\n".join(
        fig.to_json() for _, _, fig in days_data
    ) + "\n]"

    hint = "↓ Price · Algo action · Position · P&amp;L · Signal"
    if product == "TOMATOES":
        hint += " (EMA diff + imbalance)"
    else:
        hint += " (bid/ask dist from FV + spread)"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>IMC P4 · {product}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *{{box-sizing:border-box;margin:0;padding:0;}}
    body{{background:#0f172a;font-family:monospace;}}
    .hdr{{background:#1e293b;border-bottom:1px solid #334155;
          padding:14px 28px;display:flex;align-items:center;
          justify-content:space-between;gap:20px;}}
    .hdr h1{{font-size:1.05rem;color:#f1f5f9;}}
    .hdr-right{{display:flex;align-items:center;gap:14px;}}
    .hdr span{{font-size:0.8rem;color:#64748b;}}
    select#day-select{{
      background:#0f172a;color:#f1f5f9;border:1px solid #475569;
      border-radius:6px;padding:6px 28px 6px 12px;font-size:0.95rem;
      font-family:monospace;cursor:pointer;outline:none;
      appearance:none;-webkit-appearance:none;
      background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath fill='%2394a3b8' d='M6 8L0 0h12z'/%3E%3C/svg%3E");
      background-repeat:no-repeat;background-position:right 10px center;
    }}
    select#day-select:hover{{border-color:#7c3aed;}}
    #summary-box{{padding:12px 28px;}}
    .hint{{text-align:center;font-size:0.75rem;color:#475569;padding:3px;}}
    #chart-div{{width:100%;}}
  </style>
</head>
<body>
  <div class="hdr">
    <h1>IMC Prosperity 4 · Round 0 · {product}</h1>
    <div class="hdr-right">
      <select id="day-select">{options_html}
      </select>
      <span>Hover → crosshair · Scroll = zoom · Drag = pan</span>
    </div>
  </div>

  <div id="summary-box"></div>
  <div class="hint">{hint}</div>
  <div id="chart-div"></div>

  <script>
    const FIGS     = {figs_js};
    const SUMMARIES = {summaries_js};
    const CONFIG   = {{scrollZoom:true,displaylogo:false,
                       modeBarButtonsToRemove:["lasso2d","select2d"]}};

    function switchDay(idx) {{
      const f = FIGS[idx];
      Plotly.react("chart-div", f.data, f.layout, CONFIG);
      document.getElementById("summary-box").innerHTML = SUMMARIES[String(idx)];
    }}

    document.getElementById("day-select").addEventListener("change", function() {{
      switchDay(parseInt(this.value));
    }});

    // Initial render
    switchDay(0);
  </script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Running backtest…")
    log_path = run_backtest()

    print("Parsing log…")
    rows, trades = parse_log(log_path)
    try:
        os.remove(log_path)
    except OSError:
        pass

    days = sorted({r["day"] for r in rows})
    print(f"Days found: {days}\n")

    for product in ["EMERALDS", "TOMATOES"]:
        days_data = []   # list of (day, d, fig)
        for day_idx, day in enumerate(days):
            label = f"{product} Day {day:+d}"
            print(f"  [{label}]  extracting…", end=" ", flush=True)
            d = extract(rows, trades, product, day, day_idx)
            if d is None:
                print("no data — skipped.")
                continue
            print(f"fills={d['total_fills']}  PnL={d['final_pnl']:+,.0f}",
                  end="  building…", flush=True)
            fig = build_figure(d, product, day)
            days_data.append((day, d, fig))
            print(" ✓")

        if not days_data:
            continue

        fname = f"analysis_{product}.html"
        html  = make_combined_html(days_data, product)
        with open(fname, "w") as f:
            f.write(html)
        print(f"  → {fname}\n")

    print("Done.")
    print("  analysis_EMERALDS.html")
    print("  analysis_TOMATOES.html")


if __name__ == "__main__":
    main()
