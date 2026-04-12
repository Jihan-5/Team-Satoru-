"""
IMC Prosperity 4 — Orderbook Visualizer (TimoDiehm style)
Generates orderbook_emeralds.png and orderbook_tomatoes.png

Usage:
    python3 visual.py
"""

import json, os, re, subprocess, sys, tempfile
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ALGO  = "backtester.py"
ROUND = "0"

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "axes.labelcolor":   "#333333",
    "xtick.color":       "#666666",
    "ytick.color":       "#666666",
    "text.color":        "#222222",
    "grid.color":        "#e8e8e8",
    "grid.linewidth":    0.7,
    "legend.facecolor":  "white",
    "legend.edgecolor":  "#cccccc",
    "font.family":       "sans-serif",
    "font.size":         9,
})

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

    # ── Activities (CSV) ──────────────────────────────────────────────────────
    act_start = raw.index("Activities log:") + len("Activities log:")
    act_end   = raw.rfind("\n[")          # trade history starts with \n[
    act_csv   = raw[act_start:act_end].strip()

    rows = []
    for line in act_csv.splitlines():
        if line.startswith("day"):
            continue
        p = line.split(";")
        if len(p) < 16:
            continue
        def fv(x): return float(x) if x else None
        def iv(x): return int(x)   if x else None
        rows.append({
            "day": int(p[0]), "ts": int(p[1]), "product": p[2],
            "bp1": fv(p[3]),  "bv1": iv(p[4]),
            "bp2": fv(p[5]),  "bv2": iv(p[6]),
            "bp3": fv(p[7]),  "bv3": iv(p[8]),
            "ap1": fv(p[9]),  "av1": iv(p[10]),
            "ap2": fv(p[11]), "av2": iv(p[12]),
            "ap3": fv(p[13]), "av3": iv(p[14]),
            "mid": fv(p[15]), "pnl": float(p[16]) if len(p) > 16 and p[16] else 0.0,
        })

    # ── Trade history (trailing-comma JSON) ───────────────────────────────────
    trade_raw = raw[act_end:].strip()
    trade_raw = re.sub(r",(\s*[}\]])", r"\1", trade_raw)   # fix trailing commas
    trades = json.loads(trade_raw)

    # ── Sandbox lambda logs ───────────────────────────────────────────────────
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

    # Split lambda rows by day (ts restarts at 0 each day)
    lambda_by_day, cur, prev = [], {}, -1
    for ts, pl in lambda_rows:
        if ts < prev:
            lambda_by_day.append(cur)
            cur = {}
        cur[ts] = pl
        prev = ts
    lambda_by_day.append(cur)

    return rows, trades, lambda_by_day


# ── Build book data per product per day ──────────────────────────────────────

def book_for(rows, product, day):
    subset = [r for r in rows if r["product"] == product and r["day"] == day]
    subset.sort(key=lambda r: r["ts"])
    # Normalise to 0-relative timestamps so all days share the same x-axis
    if subset:
        t0 = subset[0]["ts"]
        subset = [{**r, "ts": r["ts"] - t0} for r in subset]
    return subset


def subsample(lst, every=10):
    return lst[::every]


# ── Orderbook scatter plot ────────────────────────────────────────────────────

def plot_orderbook(ax, book, fills_buy, fills_sell, our_bids, our_asks,
                   title, normalize=False):
    """
    book       : list of row dicts
    fills_buy  : list of (ts, price) — our buys
    fills_sell : list of (ts, price) — our sells
    our_bids   : list of (ts, price) — our passive bid quotes
    our_asks   : list of (ts, price) — our passive ask quotes
    """

    sub = subsample(book, every=5)  # plot every 5th tick for clarity

    def scatter_level(price_key, vol_key, color, alpha=0.55):
        xs, ys, ss = [], [], []
        for r in sub:
            p = r[price_key]
            v = r[vol_key]
            ref = (r["mid"] or 0) if normalize else 0
            if p is not None and v is not None:
                xs.append(r["ts"] / 1000)
                ys.append(p - ref)
                ss.append(max(v * 3.5, 8))
        if xs:
            ax.scatter(xs, ys, s=ss, color=color, alpha=alpha,
                       linewidths=0, zorder=2)

    # Bid levels (blue shades deepening with depth)
    scatter_level("bp1", "bv1", "#2563eb", 0.65)
    scatter_level("bp2", "bv2", "#3b82f6", 0.45)
    scatter_level("bp3", "bv3", "#93c5fd", 0.30)

    # Ask levels (red shades deepening with depth)
    scatter_level("ap1", "av1", "#dc2626", 0.65)
    scatter_level("ap2", "av2", "#ef4444", 0.45)
    scatter_level("ap3", "av3", "#fca5a5", 0.30)

    # Mid-price line (thin grey)
    if not normalize:
        mids = [(r["ts"] / 1000, r["mid"]) for r in sub if r["mid"]]
        if mids:
            xs, ys = zip(*mids)
            ax.plot(xs, ys, color="#999999", lw=0.6, zorder=1, alpha=0.7)

    # Our quotes (★ black stars)
    ref_map = {r["ts"]: r["mid"] for r in book if r["mid"]}
    def plot_quotes(qlist, marker, color, zorder=5):
        xs, ys = [], []
        for ts, price in qlist:
            ref = ref_map.get(ts, 0) if normalize else 0
            xs.append(ts / 1000)
            ys.append(price - ref)
        if xs:
            ax.scatter(xs, ys, s=28, marker=marker, color=color,
                       zorder=zorder, linewidths=0.4, edgecolors="black")

    plot_quotes(our_bids, "*", "black")
    plot_quotes(our_asks, "*", "black")

    # Our fills (orange crosses = profitable trades / fills we received)
    def plot_fills(flist, marker, color, label):
        xs, ys = [], []
        for ts, price in flist:
            ref = ref_map.get(ts, 0) if normalize else 0
            xs.append(ts / 1000)
            ys.append(price - ref)
        if xs:
            ax.scatter(xs, ys, s=60, marker=marker, color=color,
                       zorder=6, linewidths=1.2, edgecolors=color, label=label)

    plot_fills(fills_buy,  "P", "#f97316", "Our buys")   # orange plus
    plot_fills(fills_sell, "X", "#f97316", "Our sells")  # orange X

    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)
    ax.set_xlabel("Timestamp (s)", fontsize=8)
    ax.set_ylabel("Price (relative to mid)" if normalize else "Price", fontsize=8)
    ax.grid(True, alpha=0.5)

    # Legend
    legend_elems = [
        mpatches.Patch(color="#2563eb", alpha=0.7, label="Bid L1"),
        mpatches.Patch(color="#93c5fd", alpha=0.7, label="Bid L2/L3"),
        mpatches.Patch(color="#dc2626", alpha=0.7, label="Ask L1"),
        mpatches.Patch(color="#fca5a5", alpha=0.7, label="Ask L2/L3"),
        mlines.Line2D([0],[0], marker="*", color="w", markerfacecolor="black",
               markersize=8, label="Our quotes"),
        mlines.Line2D([0],[0], marker="P", color="w", markerfacecolor="#f97316",
               markersize=8, label="Our fills (buy)"),
        mlines.Line2D([0],[0], marker="X", color="w", markerfacecolor="#f97316",
               markersize=8, label="Our fills (sell)"),
    ]
    ax.legend(handles=legend_elems, fontsize=7, loc="upper right",
              ncol=2, framealpha=0.9)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = run_backtest()
    print("Parsing log…")
    rows, trades, lambda_by_day = parse_log(log_path)

    days = sorted({r["day"] for r in rows})

    for di, day in enumerate(days):
        lmap   = lambda_by_day[di] if di < len(lambda_by_day) else {}
        em_bk  = book_for(rows, "EMERALDS", day)
        to_bk  = book_for(rows, "TOMATOES", day)

        # ── Collect our quotes from lambda log ───────────────────────────────
        em_bids, em_asks = [], []
        for r in em_bk:
            ll = lmap.get(r["ts"], {})
            bq = ll.get("EMERALDS", {}).get("BID_Q")
            aq = ll.get("EMERALDS", {}).get("ASK_Q")
            if bq: em_bids.append((r["ts"], bq))
            if aq: em_asks.append((r["ts"], aq))

        # ── Collect our fills from trade history ─────────────────────────────
        # Timestamp in trade history is global (day -2: 0-999900, day -1: 1000000-1999900)
        day_offset = (di) * 1_000_000   # day -2 → 0, day -1 → 1_000_000

        def get_fills(symbol, is_buy):
            result = []
            for t in trades:
                if t["symbol"] != symbol:
                    continue
                is_ours = (t["buyer"] == "SUBMISSION") if is_buy else (t["seller"] == "SUBMISSION")
                if not is_ours:
                    continue
                local_ts = t["timestamp"] - day_offset
                if 0 <= local_ts < 1_000_000:
                    result.append((local_ts, t["price"]))
            return result

        em_fb  = get_fills("EMERALDS", True)
        em_fs  = get_fills("EMERALDS", False)
        to_fb  = get_fills("TOMATOES", True)
        to_fs  = get_fills("TOMATOES", False)

        # ── Figure: EMERALDS orderbook ───────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(14, 5))
        fig1.suptitle(f"EMERALDS Orderbook — Day {day:+d}  |  "
                      f"Fills: {len(em_fb)+len(em_fs)}  "
                      f"(buy {len(em_fb)} / sell {len(em_fs)})",
                      fontsize=11, fontweight="bold")
        plot_orderbook(ax1, em_bk, em_fb, em_fs, em_bids, em_asks,
                       "Price levels over time (circle size = order volume)")
        fig1.tight_layout()
        fn1 = f"orderbook_emeralds_day{day}.png"
        fig1.savefig(fn1, dpi=150, bbox_inches="tight")
        print(f"Saved → {fn1}")
        plt.close(fig1)

        # ── Figure: TOMATOES — raw + normalized ──────────────────────────────
        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(14, 10))
        fig2.suptitle(f"TOMATOES Orderbook — Day {day:+d}  |  "
                      f"Fills: {len(to_fb)+len(to_fs)}  "
                      f"(buy {len(to_fb)} / sell {len(to_fs)})",
                      fontsize=11, fontweight="bold")

        plot_orderbook(ax2a, to_bk, to_fb, to_fs, [], [],
                       "Raw price — shows trend direction",
                       normalize=False)

        plot_orderbook(ax2b, to_bk, to_fb, to_fs, [], [],
                       "Normalized (price − mid) — shows spread structure",
                       normalize=True)

        fig2.tight_layout()
        fn2 = f"orderbook_tomatoes_day{day}.png"
        fig2.savefig(fn2, dpi=150, bbox_inches="tight")
        print(f"Saved → {fn2}")
        plt.close(fig2)

    try:
        os.remove(log_path)
    except OSError:
        pass
    print("Done.")


if __name__ == "__main__":
    main()
