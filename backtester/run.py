from realistic_bt import run_backtest
from calibrated_configs import GLFT_CALIBRATED

r = run_backtest(
    trader_path='glft_trader_round0.py',
    prices_csv='prices_round_0_day_-1.csv',
    trades_csv='trades_round_0_day_-1.csv',
    config=GLFT_CALIBRATED,
)

print(f"Total PnL:  {r.pnl_total:.0f}")
print(f"EMERALDS:   {r.pnl_realized['EMERALDS'] + r.pnl_liquidation['EMERALDS']:.0f}")
print(f"TOMATOES:   {r.pnl_realized['TOMATOES'] + r.pnl_liquidation['TOMATOES']:.0f}")
print(f"Fills:      {r.num_fills}")
print(f"Drawdown:   {r.drawdown:.0f}")
print(f"Errors:     {len(r.errors)}")
