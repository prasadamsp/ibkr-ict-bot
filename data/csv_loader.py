"""
CSV Data Loader — load OHLCV bar data from local CSV files.

Enables offline backtesting without an IBKR connection.

Supported CSV formats:
  1. Standard (datetime as first column or index):
         datetime,open,high,low,close,volume
         2024-01-01 00:00:00,25.0,25.1,24.9,25.05,1000

  2. Unix timestamps (column named 'timestamp' or 'time'):
         timestamp,open,high,low,close,volume
         1704067200,25.0,25.1,24.9,25.05,1000

All outputs are timezone-aware (UTC) DataFrames with DatetimeIndex
and columns: open, high, low, close, volume.
"""

from pathlib import Path
from typing import Optional
import pandas as pd


REQUIRED_COLUMNS = {"open", "high", "low", "close"}
OPTIONAL_COLUMNS = {"volume"}


def load_csv(path: str, timeframe: str = "unknown") -> pd.DataFrame:
    """
    Load an OHLCV CSV file and return a clean, UTC-indexed DataFrame.

    Args:
        path:       Path to the CSV file.
        timeframe:  Label for logging purposes (e.g. "M15", "H1").

    Returns:
        pd.DataFrame with DatetimeIndex (UTC) and OHLCV columns.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing or data is empty.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)

    # ── Identify and set the datetime index ──────────────────────────────────
    df = _set_datetime_index(df)

    # ── Validate required columns ─────────────────────────────────────────────
    missing = REQUIRED_COLUMNS - set(df.columns.str.lower())
    if missing:
        raise ValueError(
            f"CSV '{path}' missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    # Normalise column names to lowercase
    df.columns = [c.lower() for c in df.columns]

    # Ensure volume column exists (default 0 if absent)
    if "volume" not in df.columns:
        df["volume"] = 0.0

    # Keep only OHLCV
    df = df[["open", "high", "low", "close", "volume"]].copy()

    # ── Type coercion ─────────────────────────────────────────────────────────
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with NaN in price columns
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    # Remove duplicates, sort ascending
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    if df.empty:
        raise ValueError(f"CSV '{path}' produced an empty DataFrame after cleaning.")

    print(
        f"[csv_loader] Loaded {len(df)} {timeframe} bars from '{p.name}' "
        f"[{df.index[0]} → {df.index[-1]}]"
    )
    return df


def _set_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and parse the datetime column, returning df with DatetimeIndex (UTC).

    Detection priority:
        1. Column named 'datetime'
        2. Column named 'date'
        3. Column named 'timestamp' or 'time' (treated as Unix seconds)
        4. Existing index if it looks like dates
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # 1. Named datetime/date column
    for candidate in ("datetime", "date"):
        if candidate in cols_lower:
            col = cols_lower[candidate]
            df = df.copy()
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            df.set_index(col, inplace=True)
            df.index.name = "datetime"
            return df

    # 2. Unix timestamp column
    for candidate in ("timestamp", "time"):
        if candidate in cols_lower:
            col = cols_lower[candidate]
            df = df.copy()
            ts_series = pd.to_numeric(df[col], errors="coerce")
            df["datetime"] = pd.to_datetime(ts_series, unit="s", utc=True)
            df.drop(columns=[col], inplace=True)
            df.set_index("datetime", inplace=True)
            return df

    # 3. Try the existing index
    try:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        if idx.notna().all():
            df = df.copy()
            df.index = idx
            df.index.name = "datetime"
            return df
    except Exception:
        pass

    # 4. Assume first column is datetime
    first_col = df.columns[0]
    df = df.copy()
    df[first_col] = pd.to_datetime(df[first_col], utc=True, errors="coerce")
    df.set_index(first_col, inplace=True)
    df.index.name = "datetime"
    return df


def load_pair(
    m15_path: str,
    h1_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both M15 and H1 CSV files and return (m15_df, h1_df).

    Validates that the two datasets cover an overlapping time range.
    """
    m15 = load_csv(m15_path, timeframe="M15")
    h1 = load_csv(h1_path, timeframe="H1")

    # Warn if no overlap
    m15_start, m15_end = m15.index[0], m15.index[-1]
    h1_start, h1_end = h1.index[0], h1.index[-1]

    overlap_start = max(m15_start, h1_start)
    overlap_end = min(m15_end, h1_end)

    if overlap_start >= overlap_end:
        raise ValueError(
            f"M15 and H1 CSVs have no overlapping time range.\n"
            f"  M15: {m15_start} → {m15_end}\n"
            f"  H1:  {h1_start} → {h1_end}"
        )

    print(
        f"[csv_loader] Overlap: {overlap_start} → {overlap_end} "
        f"({(overlap_end - overlap_start).days} days)"
    )
    return m15, h1
