"""
IMC Prosperity 4 — Multi-panel analysis dashboard
Generates 4 self-contained HTML files:
  analysis_EMERALDS_day-2.html
  analysis_EMERALDS_day-1.html
  analysis_TOMATOES_day-2.html
  analysis_TOMATOES_day-1.html

Each file has 4 vertically-stacked panels with shared x-axis and
synchronized crosshair hover:
  1. Price  — mid + best bid/ask + L2/L3 bands + EMA fast/slow +
               regime shading + fill triangles sized by volume
  2. Position — filled area + ±limit lines + target position
  3. PnL    — total + realized-only
  4. Signal  — EMA diff (fast−slow) + regime thresholds +
               L1 volume imbalance (secondary axis)
"""

import json, os, re, subprocess, sys, tempfile
from collections import deque
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Constants (mirror backtester.py) ─────────────────────────────────────────
EMERALD_FV      = 10_000
EMERALD_OFFSET  = 8
POS_LIMIT       = 80
EMA_FAST        = 9
EMA_SLOW        = 16
WARMUP          = 20
MILD_THR        = 0.05
STRONG_THR      = 3.5

ALGO  = "backtester.py"
ROUND = "0"


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════════════════

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

    # ── Activities CSV ────────────────────────────────────────────────────────
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

    # ── Trade history ─────────────────────────────────────────────────────────
    trade_raw = raw[act_end:].strip()
    trade_raw = re.sub(r",(\s*[}\]])", r"\1", trade_raw)
    trades    = json.loads(trade_raw)

    return rows, trades


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  SIGNAL COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ema(arr: np.ndarray, n: int) -> np.ndarray:
    a   = 2.0 / (n + 1)
    out = np.empty_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = a * arr[i] + (1.0 - a) * out[i - 1]
    return out


def regime_array(mid: np.ndarray, fast: np.ndarray, slow: np.ndarray
                 ) -> np.ndarray:
    """
    Returns int array:
      0 = neutral (|diff| < MILD_THR and no momentum)
      1 = mild bearish  (momentum only, OR -STRONG_THR ≤ diff < -MILD_THR after warmup)
      2 = strong bearish (diff < -STRONG_THR after warmup)
      3 = mild bullish  (diff > MILD_THR after warmup)
      4 = strong bullish (diff > STRONG_THR after warmup)
    """
    n    = len(mid)
    diff = fast - slow
    mom  = np.zeros(n, bool)
    mom[1:] = mid[1:] < mid[:-1]
    warm = np.arange(n) >= WARMUP

    regime = np.zeros(n, int)
    # Bullish
    regime[warm & (diff > MILD_THR)]   = 3
    regime[warm & (diff > STRONG_THR)] = 4
    # Bearish (overrides bullish at same tick — shouldn't conflict)
    regime[mom & ~(warm & (diff > MILD_THR))] = 1          # momentum only
    regime[warm & (diff < -MILD_THR)]  = 1                  # mild bearish
    regime[warm & (diff < -STRONG_THR)] = 2                 # strong bearish
    return regime


def fifo_realized(fills_sorted: list) -> list:
    """
    FIFO realized P&L.  fills_sorted: list of dicts with 'qty' (signed) and 'price'.
    Returns list of cumulative realized P&L values (same length as input).
    """
    longs  = deque()   # (qty, price)
    shorts = deque()
    realized = 0.0
    cum = []
    for f in fills_sorted:
        qty, price = f["qty"], f["price"]
        if qty > 0:                     # buy: close shorts first
            rem = qty
            while rem > 0 and shorts:
                sq, sp = shorts[0]
                m = min(sq, rem)
                realized += m * (sp - price)
                rem -= m; sq -= m
                if sq == 0: shorts.popleft()
                else:       shorts[0] = (sq, sp)
            if rem > 0:
                longs.append((rem, price))
        else:                           # sell: close longs first
            rem = -qty
            while rem > 0 and longs:
                lq, lp = longs[0]
                m = min(lq, rem)
                realized += m * (price - lp)
                rem -= m; lq -= m
                if lq == 0: longs.popleft()
                else:       longs[0] = (lq, lp)
            if rem > 0:
                shorts.append((rem, price))
        cum.append(realized)
    return cum


def position_from_fills(fills_sorted: list, tick_ts: np.ndarray) -> np.ndarray:
    """
    Forward-fill position from fills into a tick-resolution array.
    """
    pos = 0
    pos_arr = np.zeros(len(tick_ts), int)
    fi = 0
    for i, t in enumerate(tick_ts):
        while fi < len(fills_sorted) and fills_sorted[fi]["ts"] <= t:
            pos += fills_sorted[fi]["qty"]
            fi  += 1
        pos_arr[i] = pos
    return pos_arr


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  PER-PRODUCT / PER-DAY DATA EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract(rows, trades, product, day, day_idx):
    book = [r for r in rows if r["product"] == product and r["day"] == day]
    book.sort(key=lambda r: r["ts"])
    if not book:
        return None

    # x-axis: seconds from start of day
    t0  = book[0]["ts"]
    ts  = np.array([(r["ts"] - t0) / 1000.0 for r in book])

    # Price levels
    def col(key): return np.array(
        [r[key] if r[key] is not None else np.nan for r in book])

    mid  = col("mid")
    bp1, bv1 = col("bp1"), col("bv1")
    bp2, bv2 = col("bp2"), col("bv2")
    bp3, bv3 = col("bp3"), col("bv3")
    ap1, av1 = col("ap1"), col("av1")
    ap2, av2 = col("ap2"), col("av2")
    ap3, av3 = col("ap3"), col("av3")

    # Fill NaN mid with linear interp before EMA
    nans = np.isnan(mid)
    if nans.any() and (~nans).any():
        mid[nans] = np.interp(np.flatnonzero(nans),
                              np.flatnonzero(~nans), mid[~nans])

    # PnL: normalize to start at 0 each day
    pnl_raw = np.array([r["pnl"] for r in book])
    pnl     = pnl_raw - pnl_raw[0]

    # EMAs and regime (computed locally — no lambda-log dependency)
    fast = compute_ema(mid, EMA_FAST)
    slow = compute_ema(mid, EMA_SLOW)
    diff = fast - slow
    reg  = regime_array(mid, fast, slow)

    # L1 volume imbalance
    with np.errstate(divide="ignore", invalid="ignore"):
        imbalance = np.where(
            (bv1 > 0) & (av1 > 0),
            (bv1 - av1) / (bv1 + av1),
            0.0,
        )

    # Fills — use global timestamps, subtract day_offset for local seconds
    day_offset_ms = day_idx * 1_000_000
    fills_raw = []
    for t in trades:
        if t["symbol"] != product:
            continue
        local_ms = t["timestamp"] - day_offset_ms
        if not (0 <= local_ms < 1_000_000):
            continue
        is_buy  = t.get("buyer")  == "SUBMISSION"
        is_sell = t.get("seller") == "SUBMISSION"
        if not (is_buy or is_sell):
            continue
        fills_raw.append({
            "ts":    local_ms / 1000.0,
            "price": float(t["price"]),
            "qty":   int(t["quantity"]) * (1 if is_buy else -1),
        })
    fills_raw.sort(key=lambda f: f["ts"])

    # Realized PnL (FIFO)
    rpnl_vals = fifo_realized(fills_raw)
    rpnl_ts   = [f["ts"] for f in fills_raw]
    # Forward-fill to tick resolution
    rpnl_tick = np.interp(ts, rpnl_ts if rpnl_ts else [0], rpnl_vals if rpnl_vals else [0])

    # Position at tick resolution
    pos_tick  = position_from_fills(fills_raw, ts)

    # Target position (TOMATOES regime-based, EMERALDS = 0)
    if product == "TOMATOES":
        # Bearish → short bias (strategy posts deep bid, rarely fills → net short)
        # Not a hard target but indicative: neutral → 0, bearish → negative
        target = np.where(reg >= 1, -POS_LIMIT * 0.5, 0).astype(float)
    else:
        target = np.zeros(len(ts))

    # Summary stats
    total_fills  = sum(1 for f in fills_raw)
    total_volume = sum(abs(f["qty"]) for f in fills_raw)
    final_pnl    = pnl[-1] if len(pnl) else 0
    drawdown     = np.max(np.maximum.accumulate(pnl) - pnl) if len(pnl) else 0
    tick_count   = len(ts)
    regime_time  = {
        "neutral":        int(np.sum(reg == 0)) * 100 / 1000,
        "mild_bearish":   int(np.sum(reg == 1)) * 100 / 1000,
        "strong_bearish": int(np.sum(reg == 2)) * 100 / 1000,
        "mild_bullish":   int(np.sum(reg == 3)) * 100 / 1000,
        "strong_bullish": int(np.sum(reg == 4)) * 100 / 1000,
    }

    return dict(
        ts=ts, mid=mid, pnl=pnl, rpnl=rpnl_tick,
        bp1=bp1, bv1=bv1, bp2=bp2, bv2=bv2, bp3=bp3, bv3=bv3,
        ap1=ap1, av1=av1, ap2=ap2, av2=av2, ap3=ap3, av3=av3,
        fast=fast, slow=slow, diff=diff, reg=reg, imbalance=imbalance,
        pos=pos_tick, target=target,
        fills=fills_raw,
        # summary
        total_fills=total_fills, total_volume=total_volume,
        final_pnl=final_pnl, drawdown=drawdown, regime_time=regime_time,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  CHART CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

REGIME_COLORS = {
    0: None,
    1: "rgba(239,68,68,0.06)",    # mild bearish — faint red
    2: "rgba(220,38,38,0.13)",    # strong bearish — red
    3: "rgba(34,197,94,0.06)",    # mild bullish — faint green
    4: "rgba(21,128,62,0.11)",    # strong bullish — green
}
REGIME_LABELS = {
    0: "Neutral",
    1: "Mild bearish",
    2: "Strong bearish",
    3: "Mild bullish",
    4: "Strong bullish",
}


def add_regime_shading(fig, ts, reg, row, col=1):
    """Add background shading spans for each regime change."""
    if len(reg) == 0:
        return
    spans = []
    start, cur = ts[0], reg[0]
    for t, r in zip(ts[1:], reg[1:]):
        if r != cur:
            spans.append((start, t, cur))
            start, cur = t, r
    spans.append((start, ts[-1], cur))
    # Merge tiny spans (< 1s) into neighbours to avoid thousands of rects
    merged = []
    for span in spans:
        if merged and span[2] == merged[-1][2]:
            merged[-1] = (merged[-1][0], span[1], span[2])
        elif span[1] - span[0] < 1.0 and merged:
            # absorb into previous
            merged[-1] = (merged[-1][0], span[1], merged[-1][2])
        else:
            merged.append(list(span))
    for x0, x1, r in merged:
        color = REGIME_COLORS.get(r)
        if color:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=color,
                          layer="below", line_width=0,
                          row=row, col=col)


def build_figure(d, product, day):
    """
    d: dict from extract()
    Returns a Plotly figure with 4 stacked panels.
    """
    ts   = d["ts"]
    ts_s = [f"{t:.1f}s" for t in ts]   # string labels for hover

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        row_heights=[0.45, 0.18, 0.22, 0.15],
        vertical_spacing=0.025,
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": False}],
            [{"secondary_y": True}],
        ],
    )

    # ── Build a flat customdata array for the unified hover tooltip ───────────
    # Fields: [mid, bp1, ap1, fast, slow, diff, regime_label, pos, pnl, rpnl]
    def safe(arr, i):
        v = arr[i] if i < len(arr) else np.nan
        return float(v) if v is not None and not np.isnan(float(v)) else 0.0

    cdata = []
    for i in range(len(ts)):
        cdata.append([
            safe(d["mid"],  i),
            safe(d["bp1"],  i),
            safe(d["ap1"],  i),
            safe(d["fast"], i),
            safe(d["slow"], i),
            safe(d["diff"], i),
            REGIME_LABELS.get(int(d["reg"][i]), "?"),
            int(d["pos"][i]),
            safe(d["pnl"],  i),
            safe(d["rpnl"], i),
        ])

    HOVER = (
        "<b>t=%{x:.1f}s</b><br>"
        "Mid: %{customdata[0]:.2f}<br>"
        "Bid: %{customdata[1]:.2f}  Ask: %{customdata[2]:.2f}<br>"
        "EMA fast: %{customdata[3]:.3f}<br>"
        "EMA slow: %{customdata[4]:.3f}<br>"
        "EMA diff: %{customdata[5]:.3f}<br>"
        "Regime: %{customdata[6]}<br>"
        "Position: %{customdata[7]}<br>"
        "Total PnL: %{customdata[8]:.0f}<br>"
        "Realized PnL: %{customdata[9]:.0f}"
        "<extra></extra>"
    )

    # ── Panel 1: PRICE ────────────────────────────────────────────────────────
    add_regime_shading(fig, ts, d["reg"], row=1)

    # L2/L3 depth band (very faint)
    valid = ~(np.isnan(d["bp3"]) | np.isnan(d["ap3"]))
    if valid.any():
        ts_v   = ts[valid].tolist()
        bid3_v = d["bp3"][valid].tolist()
        ask3_v = d["ap3"][valid].tolist()
        fig.add_trace(go.Scatter(
            x=ts_v + ts_v[::-1],
            y=bid3_v + ask3_v[::-1],
            fill="toself",
            fillcolor="rgba(148,163,184,0.07)",
            line=dict(width=0),
            name="L2-L3 depth band",
            legendgroup="depth",
            showlegend=True,
            hoverinfo="skip",
        ), row=1, col=1)

    # Best bid / ask thin lines
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["bp1"].tolist(),
        mode="lines",
        line=dict(color="rgba(37,99,235,0.35)", width=1),
        name="Best bid",
        legendgroup="bestbid",
        hovertemplate="Bid: %{y:.2f}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["ap1"].tolist(),
        mode="lines",
        line=dict(color="rgba(220,38,38,0.35)", width=1),
        name="Best ask",
        legendgroup="bestask",
        hovertemplate="Ask: %{y:.2f}<extra></extra>",
    ), row=1, col=1)

    # EMA lines
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["fast"].tolist(),
        mode="lines",
        line=dict(color="#f59e0b", width=1.5, dash="solid"),
        name=f"EMA fast ({EMA_FAST})",
        hovertemplate=f"EMA{EMA_FAST}: %{{y:.3f}}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["slow"].tolist(),
        mode="lines",
        line=dict(color="#8b5cf6", width=1.5, dash="solid"),
        name=f"EMA slow ({EMA_SLOW})",
        hovertemplate=f"EMA{EMA_SLOW}: %{{y:.3f}}<extra></extra>",
    ), row=1, col=1)

    # Mid price — main focus
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["mid"].tolist(),
        mode="lines",
        line=dict(color="#1e293b", width=2),
        name="Mid price",
        customdata=cdata,
        hovertemplate=HOVER,
    ), row=1, col=1)

    # Fill triangles (sized by |qty|, min size 6)
    buys  = [f for f in d["fills"] if f["qty"] > 0]
    sells = [f for f in d["fills"] if f["qty"] < 0]

    if buys:
        # Place triangles at mid price interpolated to fill timestamp
        buy_mid = np.interp([f["ts"] for f in buys], ts, d["mid"])
        fig.add_trace(go.Scatter(
            x=[f["ts"] for f in buys],
            y=buy_mid.tolist(),
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                color="#16a34a",
                size=[max(abs(f["qty"]) * 2.5, 6) for f in buys],
                line=dict(width=0.5, color="white"),
            ),
            name="Buy fills",
            hovertemplate=(
                "<b>BUY</b><br>"
                "t=%{x:.1f}s  price=%{customdata[0]}<br>"
                "qty=%{customdata[1]}<extra></extra>"
            ),
            customdata=[[f["price"], abs(f["qty"])] for f in buys],
        ), row=1, col=1)

    if sells:
        sell_mid = np.interp([f["ts"] for f in sells], ts, d["mid"])
        fig.add_trace(go.Scatter(
            x=[f["ts"] for f in sells],
            y=sell_mid.tolist(),
            mode="markers",
            marker=dict(
                symbol="triangle-down",
                color="#dc2626",
                size=[max(abs(f["qty"]) * 2.5, 6) for f in sells],
                line=dict(width=0.5, color="white"),
            ),
            name="Sell fills",
            hovertemplate=(
                "<b>SELL</b><br>"
                "t=%{x:.1f}s  price=%{customdata[0]}<br>"
                "qty=%{customdata[1]}<extra></extra>"
            ),
            customdata=[[f["price"], abs(f["qty"])] for f in sells],
        ), row=1, col=1)

    # ── Panel 2: POSITION ─────────────────────────────────────────────────────
    pos = d["pos"].astype(float).tolist()
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=pos,
        mode="lines",
        line=dict(color="#0ea5e9", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(14,165,233,0.12)",
        name="Position",
        customdata=cdata,
        hovertemplate="Position: %{y}<br>PnL: %{customdata[8]:.0f}<extra></extra>",
    ), row=2, col=1)

    # Target position
    if product == "TOMATOES":
        fig.add_trace(go.Scatter(
            x=ts.tolist(), y=d["target"].tolist(),
            mode="lines",
            line=dict(color="#94a3b8", width=1, dash="dot"),
            name="Target position",
            hoverinfo="skip",
        ), row=2, col=1)

    # Limit lines
    for lim, color in [(POS_LIMIT, "#ef4444"), (-POS_LIMIT, "#ef4444")]:
        fig.add_hline(y=lim, line_dash="dash", line_color=color,
                      line_width=0.8, row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8",
                  line_width=0.6, row=2, col=1)

    # ── Panel 3: PnL ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["rpnl"].tolist(),
        mode="lines",
        line=dict(color="#94a3b8", width=1, dash="dot"),
        name="Realized PnL",
        hovertemplate="Realized: %{y:.0f}<extra></extra>",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["pnl"].tolist(),
        mode="lines",
        line=dict(color="#22c55e", width=2),
        name="Total PnL",
        customdata=cdata,
        hovertemplate="Total PnL: %{y:.0f}<br>Realized: %{customdata[9]:.0f}<extra></extra>",
    ), row=3, col=1)
    # Annotate final value
    fig.add_annotation(
        x=ts[-1], y=float(d["pnl"][-1]),
        text=f"  {d['final_pnl']:+,.0f}",
        font=dict(size=11, color="#22c55e", family="monospace"),
        showarrow=False,
        xanchor="left",
        row=3, col=1,
    )

    # ── Panel 4: SIGNAL (EMA diff + thresholds + imbalance) ──────────────────
    fig.add_hline(y=0, line_color="#94a3b8", line_width=0.6,
                  line_dash="solid", row=4, col=1)
    for thr, color, label in [
        ( STRONG_THR, "#16a34a", f"+STRONG ({STRONG_THR})"),
        ( MILD_THR,   "#86efac", f"+MILD ({MILD_THR})"),
        (-MILD_THR,   "#fca5a5", f"−MILD ({MILD_THR})"),
        (-STRONG_THR, "#dc2626", f"−STRONG ({STRONG_THR})"),
    ]:
        fig.add_hline(y=thr, line_dash="dash", line_color=color,
                      line_width=0.9, row=4, col=1,
                      annotation_text=label,
                      annotation_position="right",
                      annotation_font_size=8,
                      annotation_font_color=color)

    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["diff"].tolist(),
        mode="lines",
        line=dict(color="#f59e0b", width=1.5),
        name="EMA diff (fast−slow)",
        hovertemplate="EMA diff: %{y:.4f}<extra></extra>",
    ), row=4, col=1)

    # L1 volume imbalance on secondary y-axis (row 4)
    fig.add_trace(go.Scatter(
        x=ts.tolist(), y=d["imbalance"].tolist(),
        mode="lines",
        line=dict(color="#38bdf8", width=0.8, dash="dot"),
        name="L1 vol imbalance",
        opacity=0.55,
        hovertemplate="Imbalance: %{y:.2f}<extra></extra>",
        yaxis="y8",
    ), row=4, col=1, secondary_y=True)

    # ── Global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#cbd5e1", size=11, family="monospace"),
        hovermode="x unified",
        height=900,
        title=dict(
            text=f"<b>{product}</b>  —  Day {day:+d}  |  Round 0",
            font=dict(size=15, color="#f1f5f9"),
            x=0.5,
        ),
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            bgcolor="rgba(30,41,59,0.9)",
            bordercolor="#334155",
            borderwidth=1,
            font=dict(size=10),
        ),
        margin=dict(l=70, r=190, t=60, b=50),
    )

    # Axis styling
    ax_style = dict(
        gridcolor="#334155", zerolinecolor="#475569",
        showspikes=True, spikemode="across", spikethickness=1,
        spikecolor="#64748b", spikedash="dot",
    )
    for row_i in range(1, 5):
        fig.update_xaxes(row=row_i, col=1, **ax_style)
        fig.update_yaxes(row=row_i, col=1,
                         gridcolor="#334155", zerolinecolor="#475569")

    fig.update_xaxes(title_text="Time (seconds into day)", row=4, col=1)
    fig.update_yaxes(title_text="Price",     row=1, col=1)
    fig.update_yaxes(title_text="Position",  row=2, col=1, range=[-95, 95])
    fig.update_yaxes(title_text="P&L",       row=3, col=1)
    fig.update_yaxes(title_text="EMA diff",  row=4, col=1,
                     secondary_y=False)
    fig.update_yaxes(title_text="Imbalance", row=4, col=1,
                     secondary_y=True,
                     range=[-1.5, 1.5],
                     tickfont=dict(color="#38bdf8", size=9))

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  HTML OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def make_html(fig, d, product, day) -> str:
    chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn",
                              config={"scrollZoom": True, "displaylogo": False})

    rt = d["regime_time"]
    total_t = sum(rt.values()) or 1

    summary = f"""
<div style="font-family:monospace;background:#1e293b;color:#cbd5e1;
            padding:18px 24px;border-radius:8px;margin:20px auto;
            max-width:820px;line-height:1.8;font-size:13px;">
  <b style="color:#f1f5f9;font-size:15px;">
    {product} · Day {day:+d} · Summary
  </b><br><br>
  <table style="border-collapse:collapse;width:100%">
    <tr>
      <td style="padding:3px 16px 3px 0"><span style="color:#94a3b8">Total fills</span></td>
      <td><b style="color:#f1f5f9">{d['total_fills']}</b></td>
      <td style="padding:3px 16px 3px 32px"><span style="color:#94a3b8">Total volume</span></td>
      <td><b style="color:#f1f5f9">{d['total_volume']}</b></td>
    </tr>
    <tr>
      <td style="padding:3px 16px 3px 0"><span style="color:#94a3b8">Final PnL</span></td>
      <td><b style="color:#22c55e">{d['final_pnl']:+,.0f}</b></td>
      <td style="padding:3px 16px 3px 32px"><span style="color:#94a3b8">Max drawdown</span></td>
      <td><b style="color:#ef4444">{d['drawdown']:,.0f}</b></td>
    </tr>
  </table>
  <br>
  <b style="color:#94a3b8">Time in regime:</b><br>
  <span style="color:#64748b">  Neutral:        </span>
  <b>{rt['neutral']:.0f}s ({rt['neutral']/total_t*100:.0f}%)</b><br>
  <span style="color:#fca5a5">  Mild bearish:   </span>
  <b>{rt['mild_bearish']:.0f}s ({rt['mild_bearish']/total_t*100:.0f}%)</b><br>
  <span style="color:#ef4444">  Strong bearish: </span>
  <b>{rt['strong_bearish']:.0f}s ({rt['strong_bearish']/total_t*100:.0f}%)</b><br>
  <span style="color:#86efac">  Mild bullish:   </span>
  <b>{rt['mild_bullish']:.0f}s ({rt['mild_bullish']/total_t*100:.0f}%)</b><br>
  <span style="color:#16a34a">  Strong bullish: </span>
  <b>{rt['strong_bullish']:.0f}s ({rt['strong_bullish']/total_t*100:.0f}%)</b>
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>IMC P4 — {product} Day {day:+d}</title>
  <style>
    body {{background:#0f172a;margin:0;padding:0;}}
    .header {{background:#1e293b;border-bottom:1px solid #334155;
              padding:16px 32px;color:#f1f5f9;font-family:monospace;
              display:flex;align-items:center;gap:20px;}}
    .header h1 {{font-size:1.1rem;margin:0;}}
    .header span {{color:#94a3b8;font-size:0.85rem;}}
    .hint {{font-size:0.78rem;color:#64748b;text-align:center;
            padding:6px;font-family:monospace;}}
  </style>
</head>
<body>
  <div class="header">
    <h1>IMC Prosperity 4 · Round 0 · {product} · Day {day:+d}</h1>
    <span>Scroll to zoom · Drag to pan · Hover for unified tooltip</span>
  </div>
  {summary}
  <div class="hint">
    Panel order ↓: Price (mid + bid/ask + EMAs + fills) · Position · P&L · Signal (EMA diff + imbalance)
  </div>
  {chart_html}
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

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
    print(f"Days: {days}")

    for product in ["EMERALDS", "TOMATOES"]:
        for day_idx, day in enumerate(days):
            print(f"  Building {product} Day {day:+d}…", end=" ", flush=True)
            d = extract(rows, trades, product, day, day_idx)
            if d is None:
                print("no data, skipping.")
                continue

            fig  = build_figure(d, product, day)
            html = make_html(fig, d, product, day)

            fname = f"analysis_{product}_day{day}.html"
            with open(fname, "w") as f:
                f.write(html)
            print(f"→ {fname}  (fills: {d['total_fills']}, PnL: {d['final_pnl']:+,.0f})")

    print("\nDone. Open the 4 HTML files in your browser.")


if __name__ == "__main__":
    main()
