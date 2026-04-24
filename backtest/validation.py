import pandas as pd


def simple_objective(summary: pd.Series) -> float:
    sharpe = float(summary.get("Sharpe Ratio", 0.0) or 0.0)
    sortino = float(summary.get("Sortino Ratio", 0.0) or 0.0)
    pf = float(summary.get("Profit Factor", 0.0) or 0.0)
    mdd = abs(float(summary.get("Max Drawdown [%]", 0.0) or 0.0))
    trades = float(summary.get("Total Trades", 0.0) or 0.0)
    trade_penalty = 0.0 if trades >= 30 else 2.0
    return 0.4 * sharpe + 0.3 * sortino + 0.3 * pf - 0.05 * mdd - trade_penalty
