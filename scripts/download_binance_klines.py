#!/usr/bin/env python3
"""
Download historical klines from Binance Spot REST API.

Usage:
    python scripts/download_binance_klines.py
    python scripts/download_binance_klines.py --start 2024-06-01 --end 2024-09-01
    python scripts/download_binance_klines.py --symbol ETHUSDT --interval 1h
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

API_URL = "https://api.binance.com/api/v3/klines"
LIMIT = 1000


def parse_date_to_ms(date_str: str) -> int:
    """Parse 'YYYY-MM-DD' to UTC timestamp in milliseconds."""
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def ms_to_iso(ts_ms: int) -> str:
    """Convert millisecond timestamp to UTC ISO string."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> List[List]:
    """Fetch one page of klines from Binance API."""
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": LIMIT,
    }
    resp = requests.get(API_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_all_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Paginate through Binance klines API and return a DataFrame."""
    all_rows: List[List] = []
    current_start = start_ms
    page = 0

    while current_start < end_ms:
        page += 1
        batch = fetch_klines(symbol, interval, current_start, end_ms)
        if not batch:
            break

        all_rows.extend(batch)

        # Next start = last open_time + 1 ms
        last_open_time = batch[-1][0]
        current_start = last_open_time + 1

        if len(batch) < LIMIT:
            break

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])

    # Build DataFrame
    df = pd.DataFrame(all_rows)
    df = df.iloc[:, :6]  # Keep only open_time, open, high, low, close, volume
    df.columns = ["open_time", "Open", "High", "Low", "Close", "Volume"]

    # Convert types
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Binance historical klines")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--interval", default="5m", help="Kline interval (default: 5m)")
    parser.add_argument("--start", default="2024-01-01", help="Start date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: now)")
    parser.add_argument("--out", default="data/raw/BTCUSDT_5m.csv", help="Output CSV path")
    args = parser.parse_args()

    # Resolve output path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Time range
    start_ms = parse_date_to_ms(args.start)
    if args.end:
        end_ms = parse_date_to_ms(args.end)
    else:
        end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print("=" * 50)
    print("Binance Klines Downloader")
    print("=" * 50)
    print(f"Symbol:   {args.symbol}")
    print(f"Interval: {args.interval}")
    print(f"Start:    {ms_to_iso(start_ms)} ({args.start})")
    print(f"End:      {ms_to_iso(end_ms)} ({args.end or 'now'})")
    print(f"Output:   {out_path.resolve()}")
    print("=" * 50)

    # Download
    df = download_all_klines(args.symbol, args.interval, start_ms, end_ms)

    if df.empty:
        print("No data returned. Check date range or symbol.")
        sys.exit(1)

    # Deduplicate & sort
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Save
    df.to_csv(out_path, index=False)

    print(f"Total bars downloaded: {len(df)}")
    print(f"First bar: {df['timestamp'].iloc[0]}")
    print(f"Last bar:  {df['timestamp'].iloc[-1]}")
    print(f"Saved to:  {out_path.resolve()}")
    print("=" * 50)
    print("Next steps:")
    print(f'  python main.py --data {out_path}')
    print(f'  python optimize/grid_search.py --data {out_path}')


if __name__ == "__main__":
    main()
