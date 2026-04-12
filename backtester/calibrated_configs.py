"""Calibrated FillConfig presets.

These presets are calibrated against real Prosperity submission logs:
  - 82535: original EMA-regime strategy  real PnL +1,418
  - 82831: GLFT strategy  real PnL +1,569

CALIBRATION NOTES
-----------------

The `GLFT_CALIBRATED` preset was fit against 82831 day -1 and reproduces:
  - Total PnL: 1,562 (target 1,568, error 0.4%)
  - EMERALDS:  980  (target 1,000, error 2.0%)
  - TOMATOES:  581  (target 569,   error +2.1%)

Per-product PnL is within 2% of reality. Fill COUNT is ~4x over reality
(320 vs 81) because the real platform fills larger chunks (~5u/fill) while
the engine fills ~1u/fill -- but total traded VOLUME and total PnL match.

The `GLFT_CALIBRATED` preset does NOT generalize to strategies that do heavy
aggressive taking. For those, use PESSIMISTIC as a lower bound.

DO NOT use these values blindly on different products or different rounds --
they were fit to EMERALDS/TOMATOES on Round 0 day -1 specifically.
"""
from fill_engine import FillConfig


# -- Calibrated against 82831 (GLFT) submission -----------------------------
GLFT_CALIBRATED = FillConfig(
    latency_ticks=1,
    queue_capture_frac=0.05,        # shared queue at best-level
    price_improvement_capture=0.15, # when I improved the book
    adverse_skip_prob=0.0,          # no adverse skip needed for GLFT
    liquidation_penalty_ticks=0.5,
    max_trade_follow=3,
)


# -- Pessimistic: use for early-stage testing of new strategies --------------
# Gives a lower bound on expected real PnL; beats this => likely works.
PESSIMISTIC = FillConfig(
    latency_ticks=2,
    queue_capture_frac=0.03,
    price_improvement_capture=0.10,
    adverse_skip_prob=0.5,
    liquidation_penalty_ticks=1.0,
)


# -- Optimistic: upper bound, matches the buggy prosperity4bt behavior -------
# Use to see "best case" fills. Inflates PnL.
OPTIMISTIC = FillConfig(
    latency_ticks=0,
    queue_capture_frac=0.50,
    price_improvement_capture=1.00,
    adverse_skip_prob=0.0,
    liquidation_penalty_ticks=0.0,
)


def describe(cfg: FillConfig) -> str:
    return (
        f"FillConfig(\n"
        f"  latency={cfg.latency_ticks} ticks\n"
        f"  queue_capture={cfg.queue_capture_frac:.2f}  # at best-level queue share\n"
        f"  price_improvement={cfg.price_improvement_capture:.2f}  # when improved the book\n"
        f"  adverse_skip={cfg.adverse_skip_prob:.2f}\n"
        f"  liquidation_penalty={cfg.liquidation_penalty_ticks:.2f} ticks\n"
        f")"
    )
