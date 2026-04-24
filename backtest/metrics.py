import pandas as pd


def summarize_portfolio(portfolio) -> pd.Series:
    stats = portfolio.stats()
    keep = [
        c for c in [
            "Total Return [%]",
            "Win Rate [%]",
            "Sharpe Ratio",
            "Calmar Ratio",
            "Sortino Ratio",
            "Max Drawdown [%]",
            "Total Trades",
            "Profit Factor",
        ] if c in stats.index
    ]
    return stats.loc[keep]
