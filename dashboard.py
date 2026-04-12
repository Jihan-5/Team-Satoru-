"""
IMC Prosperity 4 — Interactive Dashboard
Generates dashboard.html with two interactive charts (EMERALDS + TOMATOES).
Hover any point to see price, volume, timestamp, and regime.

Usage:
    python3 dashboard.py
"""

import json, os, re, subprocess, sys, tempfile
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ALGO  = "backtester.py"
ROUND = "0"
OUT   = "dashboard.html"

# ── Run backtest ──────────────────────────────────────────────────────────────

def run_backtest() -> str:
    tmp = tempfile.mktemp(suffix=".log")
    cmd = [sys.executable, "-m", "prosperity4bt", ALGO, ROUND, "--out", tmp]
    print("Running backtest…")
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout.strip())
    if not os.path.exists(tmp):
        raise RuntimeError(r.stderr)
    return tmp

# ── Parse log ────────────────────────────────────────────────────────────────

def parse_log(path: str):
    with open(path) as f:
        raw = f.read()

    # Activities CSV
    act_start = raw.index("Activities log:") + len("Activities log:")
    act_end   = raw.rfind("\n[")
    act_csv   = raw[act_start:act_end].strip()

    rows = []
    for line in act_csv.splitlines():
        if line.startswith("day"):
            continue
        p = line.split(";")
        if len(p) < 16:
            continue
        fv = lambda x: float(x) if x else None
        iv = lambda x: int(x)   if x else None
        rows.append({
            "day": int(p[0]), "ts": int(p[1]), "product": p[2],
            "bp1": fv(p[3]),  "bv1": iv(p[4]),
            "bp2": fv(p[5]),  "bv2": iv(p[6]),
            "bp3": fv(p[7]),  "bv3": iv(p[8]),
            "ap1": fv(p[9]),  "av1": iv(p[10]),
            "ap2": fv(p[11]), "av2": iv(p[12]),
            "ap3": fv(p[13]), "av3": iv(p[14]),
            "mid": fv(p[15]),
            "pnl": float(p[16]) if len(p) > 16 and p[16] else 0.0,
        })

    # Trade history
    trade_raw = raw[act_end:].strip()
    trade_raw = re.sub(r",(\s*[}\]])", r"\1", trade_raw)
    trades = json.loads(trade_raw)

    # Lambda logs
    sandbox_raw = raw[:raw.index("Activities log:")]
    lambda_rows = []
    depth, buf = 0, []
    for line in sandbox_raw.splitlines():
        s = line.strip()
        if s == "{":
            depth, buf = 1, [line]
        elif depth > 0:
            buf.append(line)
            depth += s.count("{") - s.count("}")
            if depth == 0:
                try:
                    obj = json.loads("\n".join(buf))
                    ll  = obj.get("lambdaLog", "")
                    if ll:
                        lambda_rows.append((obj["timestamp"], json.loads(ll)))
                except Exception:
                    pass
                buf = []

    lambda_by_day, cur, prev = [], {}, -1
    for ts, pl in lambda_rows:
        if ts < prev:
            lambda_by_day.append(cur)
            cur = {}
        cur[ts] = pl
        prev = ts
    lambda_by_day.append(cur)

    return rows, trades, lambda_by_day


# ── Extract per-day data ──────────────────────────────────────────────────────

def extract(rows, trades, lambda_by_day, product, day, day_idx, day_offset):
    book = [r for r in rows if r["product"] == product and r["day"] == day]
    book.sort(key=lambda r: r["ts"])
    if not book:
        return None
    t0 = book[0]["ts"]
    book = [{**r, "ts": r["ts"] - t0} for r in book]

    lmap = lambda_by_day[day_idx] if day_idx < len(lambda_by_day) else {}

    # Build arrays
    ts  = [r["ts"] / 1000 for r in book]   # seconds within day
    mid = [r["mid"] for r in book]
    pnl = [r["pnl"] - book[0]["pnl"] for r in book]

    # 3 bid levels
    bids = [
        {"price": [r["bp1"] for r in book], "vol": [r["bv1"] or 0 for r in book], "label": "Bid L1"},
        {"price": [r["bp2"] for r in book], "vol": [r["bv2"] or 0 for r in book], "label": "Bid L2"},
        {"price": [r["bp3"] for r in book], "vol": [r["bv3"] or 0 for r in book], "label": "Bid L3"},
    ]
    asks = [
        {"price": [r["ap1"] for r in book], "vol": [r["av1"] or 0 for r in book], "label": "Ask L1"},
        {"price": [r["ap2"] for r in book], "vol": [r["av2"] or 0 for r in book], "label": "Ask L2"},
        {"price": [r["ap3"] for r in book], "vol": [r["av3"] or 0 for r in book], "label": "Ask L3"},
    ]

    # Our quotes from lambda (EMERALDS only)
    our_bid_ts, our_bid_px = [], []
    our_ask_ts, our_ask_px = [], []
    ema_fast_ts, ema_fast_px = [], []
    ema_slow_ts, ema_slow_px = [], []
    regime_ts, regime_bear = [], []
    pos_ts, pos_val = [], []

    for r in book:
        ll = lmap.get(r["ts"] + t0 - (day_idx * 1_000_000), lmap.get(r["ts"], {}))
        # Try both local and offset-adjusted ts
        em = ll.get("EMERALDS", {}) if product == "EMERALDS" else {}
        to = ll.get("TOMATOES", {}) if product == "TOMATOES" else {}
        gen = ll.get("GENERAL", {})

        if product == "EMERALDS":
            bq = em.get("BID_Q")
            aq = em.get("ASK_Q")
            if bq: our_bid_ts.append(r["ts"] / 1000); our_bid_px.append(bq)
            if aq: our_ask_ts.append(r["ts"] / 1000); our_ask_px.append(aq)
            p = gen.get("POS", {}).get("EMERALDS", 0) if gen else 0
        else:
            f = to.get("FAST")
            s = to.get("SLOW")
            b = to.get("BEAR")
            if f is not None: ema_fast_ts.append(r["ts"] / 1000); ema_fast_px.append(f)
            if s is not None: ema_slow_ts.append(r["ts"] / 1000); ema_slow_px.append(s)
            if b is not None: regime_ts.append(r["ts"] / 1000); regime_bear.append(b)
            p = gen.get("POS", {}).get("TOMATOES", 0) if gen else 0

        pos_ts.append(r["ts"] / 1000)
        pos_val.append(p)

    # Fills
    fills_buy, fills_sell = [], []
    for t in trades:
        if t["symbol"] != product:
            continue
        local = t["timestamp"] - day_offset
        if not (0 <= local < 1_000_000):
            continue
        entry = {"ts": local / 1000, "price": t["price"], "qty": t["quantity"]}
        if t["buyer"] == "SUBMISSION":
            fills_buy.append(entry)
        elif t["seller"] == "SUBMISSION":
            fills_sell.append(entry)

    return {
        "ts": ts, "mid": mid, "pnl": pnl,
        "bids": bids, "asks": asks,
        "our_bid_ts": our_bid_ts, "our_bid_px": our_bid_px,
        "our_ask_ts": our_ask_ts, "our_ask_px": our_ask_px,
        "ema_fast_ts": ema_fast_ts, "ema_fast_px": ema_fast_px,
        "ema_slow_ts": ema_slow_ts, "ema_slow_px": ema_slow_px,
        "regime_ts": regime_ts, "regime_bear": regime_bear,
        "pos_ts": pos_ts, "pos_val": pos_val,
        "fills_buy": fills_buy, "fills_sell": fills_sell,
    }


# ── Build chart for one product (both days stacked) ──────────────────────────

def build_chart(product, days_data):
    """
    days_data: list of dicts (one per day) from extract()
    Returns a plotly Figure.
    """
    day_labels  = [f"Day {d['day']:+d}" for d in days_data]
    n           = len(days_data)

    # Row layout: price panel (large) + position panel (small), per day side by side
    col_widths = [1 / n] * n
    fig = make_subplots(
        rows=2, cols=n,
        shared_xaxes=False,
        row_heights=[0.72, 0.28],
        column_widths=col_widths,
        subplot_titles=[f"{product}  ·  {lbl}" for lbl in day_labels] + [""] * n,
        vertical_spacing=0.07,
        horizontal_spacing=0.06,
    )

    BID_COLORS = ["#1d4ed8", "#3b82f6", "#93c5fd"]   # dark → light blue
    ASK_COLORS = ["#b91c1c", "#ef4444", "#fca5a5"]   # dark → light red
    FILL_BUY   = "#f97316"
    FILL_SELL  = "#f97316"
    EMA_FAST   = "#facc15"
    EMA_SLOW   = "#a78bfa"
    OUR_QUOTE  = "#000000"
    MID_COLOR  = "#6b7280"

    for ci, d in enumerate(days_data, start=1):
        ts = d["ts"]
        shown = set()   # track legend entries shown

        def leg(name):
            if name in shown:
                return False
            shown.add(name)
            return True

        # ── Bid book levels ──────────────────────────────────────────────────
        for i, bd in enumerate(d["bids"]):
            valid = [(t, p, v) for t, p, v in zip(ts, bd["price"], bd["vol"])
                     if p is not None]
            if not valid:
                continue
            xt, yp, yv = zip(*valid)
            fig.add_trace(go.Scatter(
                x=list(xt), y=list(yp),
                mode="markers",
                marker=dict(
                    color=BID_COLORS[i],
                    size=[max(v * 0.6, 4) for v in yv],
                    opacity=0.6 - i * 0.12,
                    line=dict(width=0),
                ),
                name=bd["label"],
                legendgroup=bd["label"],
                showlegend=leg(bd["label"]),
                hovertemplate=(
                    f"<b>{bd['label']}</b><br>"
                    "Time: %{x:.1f}s<br>Price: %{y}<br>"
                    "Volume: %{customdata}<extra></extra>"
                ),
                customdata=list(yv),
            ), row=1, col=ci)

        # ── Ask book levels ──────────────────────────────────────────────────
        for i, ak in enumerate(d["asks"]):
            valid = [(t, p, v) for t, p, v in zip(ts, ak["price"], ak["vol"])
                     if p is not None]
            if not valid:
                continue
            xt, yp, yv = zip(*valid)
            fig.add_trace(go.Scatter(
                x=list(xt), y=list(yp),
                mode="markers",
                marker=dict(
                    color=ASK_COLORS[i],
                    size=[max(v * 0.6, 4) for v in yv],
                    opacity=0.6 - i * 0.12,
                    line=dict(width=0),
                ),
                name=ak["label"],
                legendgroup=ak["label"],
                showlegend=leg(ak["label"]),
                hovertemplate=(
                    f"<b>{ak['label']}</b><br>"
                    "Time: %{x:.1f}s<br>Price: %{y}<br>"
                    "Volume: %{customdata}<extra></extra>"
                ),
                customdata=list(yv),
            ), row=1, col=ci)

        # ── Mid-price line ───────────────────────────────────────────────────
        valid_mid = [(t, m) for t, m in zip(ts, d["mid"]) if m is not None]
        if valid_mid:
            xt, ym = zip(*valid_mid)
            fig.add_trace(go.Scatter(
                x=list(xt), y=list(ym),
                mode="lines",
                line=dict(color=MID_COLOR, width=1, dash="dot"),
                name="Mid price",
                legendgroup="Mid price",
                showlegend=leg("Mid price"),
                hovertemplate="<b>Mid</b><br>Time: %{x:.1f}s<br>Price: %{y:.1f}<extra></extra>",
            ), row=1, col=ci)

        # ── EMERALDS: our passive quotes ─────────────────────────────────────
        if d["our_bid_ts"]:
            fig.add_trace(go.Scatter(
                x=d["our_bid_ts"][::5], y=d["our_bid_px"][::5],
                mode="markers",
                marker=dict(color=OUR_QUOTE, symbol="star", size=7, opacity=0.8),
                name="Our bid quote",
                legendgroup="Our bid quote",
                showlegend=leg("Our bid quote"),
                hovertemplate="<b>Our bid</b><br>Time: %{x:.1f}s<br>@%{y}<extra></extra>",
            ), row=1, col=ci)
        if d["our_ask_ts"]:
            fig.add_trace(go.Scatter(
                x=d["our_ask_ts"][::5], y=d["our_ask_px"][::5],
                mode="markers",
                marker=dict(color=OUR_QUOTE, symbol="star", size=7, opacity=0.8),
                name="Our ask quote",
                legendgroup="Our ask quote",
                showlegend=leg("Our ask quote"),
                hovertemplate="<b>Our ask</b><br>Time: %{x:.1f}s<br>@%{y}<extra></extra>",
            ), row=1, col=ci)

        # ── TOMATOES: EMA lines ──────────────────────────────────────────────
        if d["ema_fast_ts"]:
            fig.add_trace(go.Scatter(
                x=d["ema_fast_ts"], y=d["ema_fast_px"],
                mode="lines",
                line=dict(color=EMA_FAST, width=1.5),
                name="EMA fast (9)",
                legendgroup="EMA fast",
                showlegend=leg("EMA fast (9)"),
                hovertemplate="<b>EMA fast</b><br>Time: %{x:.1f}s<br>%{y:.2f}<extra></extra>",
            ), row=1, col=ci)
        if d["ema_slow_ts"]:
            fig.add_trace(go.Scatter(
                x=d["ema_slow_ts"], y=d["ema_slow_px"],
                mode="lines",
                line=dict(color=EMA_SLOW, width=1.5),
                name="EMA slow (16)",
                legendgroup="EMA slow",
                showlegend=leg("EMA slow (16)"),
                hovertemplate="<b>EMA slow</b><br>Time: %{x:.1f}s<br>%{y:.2f}<extra></extra>",
            ), row=1, col=ci)

        # ── Regime shading for TOMATOES ──────────────────────────────────────
        # Use a regime scatter trace (much faster than hundreds of vrects).
        # Add a filled area trace at the bottom of the price panel for regime.
        if d["regime_ts"] and product == "TOMATOES":
            bear_arr = list(d["regime_bear"])
            ts_arr   = list(d["regime_ts"])
            # Build contiguous spans, merge spans < 2s wide to reduce noise
            spans, start, cur_bear = [], ts_arr[0], bear_arr[0]
            for t, b in zip(ts_arr[1:], bear_arr[1:]):
                if b != cur_bear:
                    spans.append((start, t, cur_bear))
                    start, cur_bear = t, b
            spans.append((start, ts_arr[-1], cur_bear))
            # Only keep spans >= 2 seconds to avoid thousands of tiny rects
            spans = [(x0, x1, b) for x0, x1, b in spans if (x1 - x0) >= 2.0]

            for x0, x1, bear in spans:
                fig.add_vrect(
                    x0=x0, x1=x1,
                    fillcolor="#ef4444" if bear else "#22c55e",
                    opacity=0.06,
                    layer="below",
                    line_width=0,
                    row=1, col=ci,
                )

        # ── Our fills ────────────────────────────────────────────────────────
        for flist, sym, color, name, grp in [
            (d["fills_buy"],  "triangle-up",   FILL_BUY,  "Our buy fills",  "fill_buy"),
            (d["fills_sell"], "triangle-down",  FILL_SELL, "Our sell fills", "fill_sell"),
        ]:
            if flist:
                fig.add_trace(go.Scatter(
                    x=[f["ts"]    for f in flist],
                    y=[f["price"] for f in flist],
                    mode="markers",
                    marker=dict(color=color, symbol=sym, size=9,
                                line=dict(width=1, color="white")),
                    name=name,
                    legendgroup=grp,
                    showlegend=leg(name),
                    hovertemplate=(
                        f"<b>{name}</b><br>"
                        "Time: %{x:.1f}s<br>Price: %{y}<br>"
                        "Qty: %{customdata}<extra></extra>"
                    ),
                    customdata=[f["qty"] for f in flist],
                ), row=1, col=ci)

        # ── Position panel ───────────────────────────────────────────────────
        fig.add_trace(go.Scatter(
            x=d["pos_ts"], y=d["pos_val"],
            mode="lines",
            line=dict(color="#38bdf8", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(56,189,248,0.15)",
            name="Position",
            legendgroup="Position",
            showlegend=leg("Position"),
            hovertemplate="<b>Position</b><br>Time: %{x:.1f}s<br>Contracts: %{y}<extra></extra>",
        ), row=2, col=ci)

        # ── Position ±80 limit lines ─────────────────────────────────────────
        for limit in [80, -80]:
            fig.add_hline(y=limit, line_dash="dash", line_color="#64748b",
                          line_width=0.8, row=2, col=ci)

    # ── Layout ───────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>{product}</b>  —  Orderbook & Strategy  |  Round 0",
            font=dict(size=16, color="#1e293b"),
            x=0.5,
        ),
        paper_bgcolor="#f8fafc",
        plot_bgcolor="#f1f5f9",
        hovermode="x unified",
        height=700,
        legend=dict(
            orientation="v",
            x=1.01, y=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=160, t=80, b=50),
    )

    # Y-axis labels
    for ci in range(1, n + 1):
        suffix = f"{ci}" if ci > 1 else ""
        fig.update_yaxes(title_text="Price", row=1, col=ci,
                         gridcolor="#e2e8f0", zerolinecolor="#cbd5e1")
        fig.update_yaxes(title_text="Position", row=2, col=ci,
                         range=[-90, 90], gridcolor="#e2e8f0")
        fig.update_xaxes(title_text="Time (s)", row=2, col=ci,
                         gridcolor="#e2e8f0")
        fig.update_xaxes(showticklabels=False, row=1, col=ci,
                         gridcolor="#e2e8f0")

    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = run_backtest()
    print("Parsing log…")
    rows, trades, lambda_by_day = parse_log(log_path)

    days = sorted({r["day"] for r in rows})
    print(f"Days: {days}")

    figs = {}
    for product in ["EMERALDS", "TOMATOES"]:
        days_data = []
        for di, day in enumerate(days):
            day_offset = di * 1_000_000
            d = extract(rows, trades, lambda_by_day, product, day, di, day_offset)
            if d:
                d["day"] = day
                days_data.append(d)
        figs[product] = build_chart(product, days_data)

    # ── Write single HTML with two tabs ──────────────────────────────────────
    em_html = figs["EMERALDS"].to_html(full_html=False, include_plotlyjs="cdn",
                                        div_id="em_chart")
    to_html = figs["TOMATOES"].to_html(full_html=False, include_plotlyjs=False,
                                        div_id="to_chart")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>IMC Prosperity 4 — Strategy Dashboard</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f172a; color: #e2e8f0; }}
    header {{ padding: 18px 32px; background: #1e293b;
              border-bottom: 1px solid #334155;
              display: flex; align-items: center; gap: 16px; }}
    header h1 {{ font-size: 1.25rem; font-weight: 700; color: #f1f5f9; }}
    header span {{ font-size: 0.85rem; color: #94a3b8; }}
    .tabs {{ display: flex; gap: 4px; padding: 16px 32px 0; }}
    .tab {{ padding: 10px 28px; border-radius: 8px 8px 0 0; cursor: pointer;
            font-size: 0.95rem; font-weight: 600; border: 1px solid #334155;
            border-bottom: none; background: #1e293b; color: #94a3b8;
            transition: all 0.15s; }}
    .tab.active {{ background: #f8fafc; color: #0f172a; border-color: #cbd5e1; }}
    .tab:hover:not(.active) {{ background: #273549; color: #e2e8f0; }}
    .panel {{ display: none; background: #f8fafc; border: 1px solid #cbd5e1;
              border-radius: 0 8px 8px 8px; margin: 0 32px 32px;
              padding: 8px; }}
    .panel.active {{ display: block; }}
    .hint {{ font-size: 0.78rem; color: #64748b; padding: 6px 12px;
             background: #f1f5f9; border-radius: 4px; margin-bottom: 4px;
             display: inline-block; }}
  </style>
</head>
<body>
  <header>
    <h1>IMC Prosperity 4 · Round 0 · Strategy Dashboard</h1>
    <span>Total P&amp;L: 36,454 · EMERALDS: 17,080 · TOMATOES: 19,374</span>
  </header>

  <div class="tabs">
    <div class="tab active" onclick="show('em')">💎 EMERALDS</div>
    <div class="tab"        onclick="show('to')">🍅 TOMATOES</div>
  </div>

  <div id="em" class="panel active">
    <span class="hint">💡 Hover to see price level, volume, and timestamp ·
      Scroll to zoom · Drag to pan · Click legend to toggle layers</span>
    {em_html}
  </div>
  <div id="to" class="panel">
    <span class="hint">💡 Green shading = bullish regime (L1/L1 quotes) ·
      Red shading = bearish regime (deep bid + L1 ask) ·
      Yellow = fast EMA · Purple = slow EMA</span>
    {to_html}
  </div>

  <script>
    function show(id) {{
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
      document.getElementById(id).classList.add('active');
      event.target.classList.add('active');
    }}
  </script>
</body>
</html>"""

    with open(OUT, "w") as f:
        f.write(html)
    print(f"Saved → {OUT}")

    try:
        os.remove(log_path)
    except OSError:
        pass

    # Auto-open in browser
    import webbrowser
    webbrowser.open(f"file://{os.path.abspath(OUT)}")


if __name__ == "__main__":
    main()
