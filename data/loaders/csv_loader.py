from pathlib import Path
from typing import Union

import pandas as pd

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
TIME_COLUMNS = ["datetime", "timestamp", "date", "time", "open_time", "close_time"]
COLUMN_ALIASES = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "vol": "Volume",
}


def load_ohlcv_csv(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    # Detect time column and convert to DatetimeIndex.
    time_col = None
    lower_cols = {col.lower(): col for col in df.columns}
    for candidate in TIME_COLUMNS:
        if candidate in lower_cols:
            time_col = lower_cols[candidate]
            break

    if time_col is not None:
        col = df[time_col]
        if pd.api.types.is_numeric_dtype(col):
            max_val = col.max()
            if max_val > 1e12:
                df[time_col] = pd.to_datetime(col, unit="ms", errors="raise")
            elif max_val > 1e9:
                df[time_col] = pd.to_datetime(col, unit="s", errors="raise")
            else:
                df[time_col] = pd.to_datetime(col, errors="raise")
        else:
            df[time_col] = pd.to_datetime(col, errors="raise")
        df = df.set_index(time_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except (ValueError, TypeError):
            raise ValueError("Could not parse a DatetimeIndex from the CSV file.")

    # Normalize common OHLCV column names.
    rename_map = {}
    for raw_name, standard_name in COLUMN_ALIASES.items():
        if raw_name in lower_cols:
            rename_map[lower_cols[raw_name]] = standard_name
    df = df.rename(columns=rename_map)

    if "Volume" not in df.columns:
        df["Volume"] = 0

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    return df[REQUIRED_COLUMNS].copy()
