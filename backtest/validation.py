import pandas as pd


def simple_objective(summary) -> float:
    if not isinstance(summary, pd.Series):
        summary = pd.Series(summary)

    pf = float(summary.get("profit_factor", summary.get("Profit Factor", 0.0)) or 0.0)
    win_rate = float(summary.get("win_rate", summary.get("Win Rate [%]", 0.0)) or 0.0)
    total_return = float(summary.get("total_return", summary.get("Total Return [%]", 0.0)) or 0.0)
    mdd = abs(float(summary.get("max_drawdown", summary.get("Max Drawdown [%]", 0.0)) or 0.0))
    trades = float(summary.get("total_trades", summary.get("Total Trades", 0.0)) or 0.0)
    trade_penalty = 0.0 if trades >= 10 else 1.0

    return 0.4 * win_rate + 0.3 * pf + 0.2 * (total_return / 100) - 0.05 * mdd - trade_penalty
