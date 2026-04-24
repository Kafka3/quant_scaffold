import pandas as pd


def summarize_portfolio(portfolio) -> pd.Series:
    if hasattr(portfolio, "summary") and isinstance(portfolio.summary, dict):
        return pd.Series(portfolio.summary)

    if hasattr(portfolio, "stats"):
        stats = portfolio.stats()
        keep = [
            c
            for c in [
                "Total Return [%]",
                "Win Rate [%]",
                "Sharpe Ratio",
                "Calmar Ratio",
                "Sortino Ratio",
                "Max Drawdown [%]",
                "Total Trades",
                "Profit Factor",
            ]
            if c in stats.index
        ]
        return stats.loc[keep]

    raise ValueError("Unsupported portfolio type for summarize_portfolio")
