def risk_adjusted_reward(pnl: float, drawdown: float, turnover: float) -> float:
    return pnl - 0.5 * abs(drawdown) - 0.1 * turnover
