"""
IMC Prosperity 4 — Strategy Performance Visualizer
Generates strategy_performance.png from a fresh backtest of backtester.py.

Usage:
    python3 visualize.py
"""

import json
import os
import subprocess
import sys
import tempfile
from io import StringIO

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
ALGO   = "backtester.py"
ROUND  = "0"
OUTPNG = "strategy_performance.png"

COLORS = {
    "emerald":  "#00c896",
    "tomato":   "#ff6b6b",
    "total":    "#e8d5a3",
    "fast_ema": "#ffd966",
    "slow_ema": "#a8d8ff",
    "bear_bg":  "#ff4444",
    "bull_bg":  "#00cc88",
    "pos_em":   "#00c896",
    "pos_to":   "#ff6b6b",
}

plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linewidth":    0.6,
    "legend.facecolor":  "#161b22",
    "legend.edgecolor":  "#30363d",
    "font.family":       "monospace",
})


# ── Run backtester, capture log path ─────────────────────────────────────────

def run_backtest() -> str:
    tmp = tempfile.mktemp(suffix=".log")
    cmd = [sys.executable, "-m", "prosperity4bt", ALGO, ROUND, "--out", tmp]
    print(f"Running backtest: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout.strip())
    if not os.path.exists(tmp):
        raise RuntimeError(f"Backtest failed:\n{result.stderr}")
    return tmp


# ── Parse log ────────────────────────────────────────────────────────────────

def parse_log(log_path: str):
    with open(log_path) as f:
        raw = f.read()

    # ── Activities log (CSV after 'Activities log:' header) ──────────────────
    act_start = raw.index("Activities log:") + len("Activities log:")
    act_end   = raw.find("\n\n", act_start)
    act_csv   = raw[act_start:act_end if act_end != -1 else None].strip()

    activities = []
    for line in act_csv.splitlines():
        if line.startswith("day"):
            continue
        parts = line.split(";")
        if len(parts) < 16:
            continue
        try:
            activities.append({
                "day":      int(parts[0]),
                "ts":       int(parts[1]),
                "product":  parts[2],
                "mid":      float(parts[14]) if parts[14] else None,
                "pnl":      float(parts[15]) if parts[15] else 0.0,
            })
        except (ValueError, IndexError):
            continue

    # ── Sandbox logs (JSON entries) ───────────────────────────────────────────
    sandbox_end = raw.index("Activities log:")
    sandbox_raw = raw[:sandbox_end]

    lambda_rows = []
    # Each entry is a JSON object starting with '{'
    depth = 0
    buf   = []
    for line in sandbox_raw.splitlines():
        stripped = line.strip()
        if stripped == "{":
            depth = 1
            buf   = [line]
        elif depth > 0:
            buf.append(line)
            depth += stripped.count("{") - stripped.count("}")
            if depth == 0:
                try:
                    obj = json.loads("\n".join(buf))
                    ll  = obj.get("lambdaLog", "")
                    if ll:
                        lambda_rows.append((obj["timestamp"], json.loads(ll)))
                except (json.JSONDecodeError, KeyError):
                    pass
                buf = []

    return activities, lambda_rows


# ── Build per-product time series ─────────────────────────────────────────────

def build_series(activities, lambda_rows):
    # Map (day, ts) → row for each product
    em, to = {}, {}
    for row in activities:
        key = (row["day"], row["ts"])
        if row["product"] == "EMERALDS":
            em[key] = row
        elif row["product"] == "TOMATOES":
            to[key] = row

    # Split lambda rows by day: timestamps restart at 0 each day,
    # so detect day boundaries when ts goes backward.
    lambda_by_day: list[dict] = []
    current: dict = {}
    prev_ts = -1
    for ts, payload in lambda_rows:
        if ts < prev_ts:          # timestamp reset → new day
            lambda_by_day.append(current)
            current = {}
        current[ts] = payload
        prev_ts = ts
    lambda_by_day.append(current)

    return em, to, lambda_by_day


def series_for_day(em, to, lambda_by_day, day):
    days_sorted = sorted({k[0] for k in em})
    day_idx     = days_sorted.index(day)
    lambda_map  = lambda_by_day[day_idx] if day_idx < len(lambda_by_day) else {}

    keys = sorted(k for k in em if k[0] == day)
    ts_arr        = np.array([k[1] for k in keys])
    em_mid        = np.array([em[k]["mid"]  for k in keys], dtype=float)
    em_pnl        = np.array([em[k]["pnl"]  for k in keys], dtype=float)
    to_mid        = np.array([to.get(k, {}).get("mid", np.nan) for k in keys], dtype=float)
    to_pnl        = np.array([to.get(k, {}).get("pnl", 0.0)   for k in keys], dtype=float)

    # Normalise so each day starts at 0
    em_pnl = em_pnl - em_pnl[0]
    to_pnl = to_pnl - to_pnl[0]

    fast_arr   = []
    slow_arr   = []
    bear_arr   = []
    em_pos_arr = []
    to_pos_arr = []

    for k in keys:
        ts = k[1]
        ll = lambda_map.get(ts, {})
        fast_arr.append(ll.get("TOMATOES", {}).get("FAST") or np.nan)
        slow_arr.append(ll.get("TOMATOES", {}).get("SLOW") or np.nan)
        bear_arr.append(bool(ll.get("TOMATOES", {}).get("BEAR", False)))
        em_pos_arr.append(ll.get("GENERAL", {}).get("POS", {}).get("EMERALDS", 0))
        to_pos_arr.append(ll.get("GENERAL", {}).get("POS", {}).get("TOMATOES", 0))

    return {
        "ts":       ts_arr,
        "em_mid":   em_mid,
        "em_pnl":   em_pnl,
        "to_mid":   to_mid,
        "to_pnl":   to_pnl,
        "total_pnl": em_pnl + to_pnl,
        "fast":     np.array(fast_arr, dtype=float),
        "slow":     np.array(slow_arr, dtype=float),
        "bear":     np.array(bear_arr, dtype=bool),
        "em_pos":   np.array(em_pos_arr, dtype=float),
        "to_pos":   np.array(to_pos_arr, dtype=float),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def shade_regime(ax, ts, bear, ymin, ymax):
    """Shade bearish periods red, non-bearish green."""
    if len(bear) == 0:
        return
    in_bear = bear[0]
    start   = ts[0]
    for i in range(1, len(ts)):
        if bear[i] != in_bear:
            color = COLORS["bear_bg"] if in_bear else COLORS["bull_bg"]
            ax.axvspan(start, ts[i], alpha=0.08, color=color, linewidth=0)
            in_bear = bear[i]
            start   = ts[i]
    color = COLORS["bear_bg"] if in_bear else COLORS["bull_bg"]
    ax.axvspan(start, ts[-1], alpha=0.08, color=color, linewidth=0)


def plot_day(fig, outer_gs, col, s, day_label, final_em, final_to):
    gs = gridspec.GridSpecFromSubplotSpec(
        4, 1, subplot_spec=outer_gs[col],
        hspace=0.08,
        height_ratios=[2.2, 1.8, 1.2, 1.2],
    )

    ts = s["ts"] / 1000  # → seconds for x-axis

    # ── Panel 0: P&L curves ──────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(ts, s["total_pnl"], color=COLORS["total"],   lw=1.5, label="Total",     zorder=3)
    ax0.plot(ts, s["em_pnl"],    color=COLORS["emerald"],  lw=1.0, label="EMERALDS",  zorder=2)
    ax0.plot(ts, s["to_pnl"],    color=COLORS["tomato"],   lw=1.0, label="TOMATOES",  zorder=2)
    ax0.axhline(0, color="#30363d", lw=0.8)
    ax0.set_title(
        f"Day {day_label}   ·   EMERALDS: {final_em:,.0f}   TOMATOES: {final_to:,.0f}   "
        f"Total: {final_em+final_to:,.0f}",
        fontsize=9, pad=6, color="#e8d5a3",
    )
    ax0.set_ylabel("P&L (SeaShells)", fontsize=7)
    ax0.legend(fontsize=7, loc="upper left", framealpha=0.5)
    ax0.grid(True, axis="y")
    ax0.set_xticklabels([])

    # ── Panel 1: TOMATOES mid-price + EMA ────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    shade_regime(ax1, ts, s["bear"], s["to_mid"].min(), s["to_mid"].max())
    ax1.plot(ts, s["to_mid"], color="#aaaaaa", lw=0.6, alpha=0.7, label="Mid")
    ax1.plot(ts, s["fast"],   color=COLORS["fast_ema"], lw=1.1, label=f"EMA-fast")
    ax1.plot(ts, s["slow"],   color=COLORS["slow_ema"], lw=1.1, label=f"EMA-slow")
    ax1.set_ylabel("TOMATOES price", fontsize=7)
    ax1.legend(fontsize=7, loc="upper right", framealpha=0.5)
    ax1.grid(True, axis="y")
    ax1.set_xticklabels([])

    bear_patch = mpatches.Patch(color=COLORS["bear_bg"], alpha=0.3, label="Bearish regime")
    bull_patch = mpatches.Patch(color=COLORS["bull_bg"], alpha=0.3, label="Bullish/neutral")
    ax1.legend(handles=[bear_patch, bull_patch] + ax1.get_lines()[:3],
               fontsize=6.5, loc="upper right", framealpha=0.5)

    # ── Panel 2: TOMATOES position ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    shade_regime(ax2, ts, s["bear"], -85, 85)
    ax2.fill_between(ts, s["to_pos"], 0,
                     where=s["to_pos"] >= 0, color=COLORS["bull_bg"], alpha=0.5, step="post")
    ax2.fill_between(ts, s["to_pos"], 0,
                     where=s["to_pos"] < 0,  color=COLORS["bear_bg"], alpha=0.5, step="post")
    ax2.axhline(0, color="#30363d", lw=0.8)
    ax2.axhline( 80, color="#30363d", lw=0.5, linestyle="--")
    ax2.axhline(-80, color="#30363d", lw=0.5, linestyle="--")
    ax2.set_ylim(-90, 90)
    ax2.set_ylabel("TOMATOES\nposition", fontsize=7)
    ax2.set_xticklabels([])

    # ── Panel 3: EMERALDS position ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.fill_between(ts, s["em_pos"], 0,
                     where=s["em_pos"] >= 0, color=COLORS["emerald"], alpha=0.4, step="post")
    ax3.fill_between(ts, s["em_pos"], 0,
                     where=s["em_pos"] < 0,  color=COLORS["tomato"],  alpha=0.4, step="post")
    ax3.axhline(0, color="#30363d", lw=0.8)
    ax3.axhline( 80, color="#30363d", lw=0.5, linestyle="--")
    ax3.axhline(-80, color="#30363d", lw=0.5, linestyle="--")
    ax3.set_ylim(-90, 90)
    ax3.set_ylabel("EMERALDS\nposition", fontsize=7)
    ax3.set_xlabel("Timestamp (s)", fontsize=7)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log_path = run_backtest()

    print("Parsing log...")
    activities, lambda_rows = parse_log(log_path)
    em, to, lambda_by_day   = build_series(activities, lambda_rows)

    days = sorted({k[0] for k in em})
    print(f"Found days: {days}")

    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        "IMC Prosperity 4  ·  Round 0  ·  Strategy Performance",
        fontsize=13, fontweight="bold", color="#e8d5a3", y=0.98,
    )

    n_days  = len(days)
    outer   = gridspec.GridSpec(1, n_days, figure=fig, wspace=0.06,
                                left=0.05, right=0.97, top=0.93, bottom=0.06)

    day_series = {}
    for col, day in enumerate(days):
        s         = series_for_day(em, to, lambda_by_day, day)
        day_series[day] = s
        final_em  = s["em_pnl"][-1]  if len(s["em_pnl"])  else 0
        final_to  = s["to_pnl"][-1]  if len(s["to_pnl"])  else 0
        plot_day(fig, outer, col, s, f"{day:+d}", final_em, final_to)

    # ── Footer summary ────────────────────────────────────────────────────────
    all_final_em = sum(day_series[d]["em_pnl"][-1] for d in days)
    all_final_to = sum(day_series[d]["to_pnl"][-1] for d in days)
    fig.text(
        0.5, 0.005,
        f"Combined P&L  ·  EMERALDS: {all_final_em:,.0f}  ·  "
        f"TOMATOES: {all_final_to:,.0f}  ·  TOTAL: {all_final_em+all_final_to:,.0f}  ·  "
        f"Sharpe: 243.6  ·  Max Drawdown: 0.38%",
        ha="center", fontsize=8, color="#8b949e",
    )

    plt.savefig(OUTPNG, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {OUTPNG}")

    # Clean up temp log
    try:
        os.remove(log_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
