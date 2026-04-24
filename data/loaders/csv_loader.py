from pathlib import Path
import pandas as pd
from typing import Union

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_ohlcv_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.sort_index()
