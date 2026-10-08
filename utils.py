from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import pandas as pd

DATE_COLUMN = "period_end"
CREATIONS_COLUMN = "TikTok Creations"
MERGE_KEYS = ["Artist ID", "Track ID", "Region", DATE_COLUMN]
IDENTITY_COLUMNS = ["Artist", "Artist ID", "Project", "Project ID", "Track", "Track ID", "Region"]

DEFAULT_STREAM_METRICS = [
    "All Partner Streams (Non-UGC)",
    "Spotify Streams",
    "Apple Music Streams",
    "Amazon Streams",
    "YouTube Streams",
    "Pandora Streams",
]


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    raw = uploaded_file.read()
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=encoding, skipinitialspace=True)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise ValueError("Không thể đọc CSV bằng các encoding phổ biến.") from last_error


def clean_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def validate_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} thiếu cột: {', '.join(missing)}")


def available_stream_metrics(df: pd.DataFrame) -> list[str]:
    preferred = [c for c in DEFAULT_STREAM_METRICS if c in df.columns]
    extra = [
        c for c in df.columns
        if c.endswith(" Streams") and "growth" not in c.lower() and "RTD" not in c and c not in preferred
    ]
    return preferred + sorted(extra)


def prepare_data(creations_raw: pd.DataFrame, streams_raw: pd.DataFrame, stream_metric: str) -> pd.DataFrame:
    required_base = ["Artist", "Artist ID", "Track", "Track ID", "Region", DATE_COLUMN]
    validate_columns(creations_raw, required_base + [CREATIONS_COLUMN], "Creations file")
    validate_columns(streams_raw, required_base + [stream_metric], "Streams file")

    creation_cols = [c for c in IDENTITY_COLUMNS + [DATE_COLUMN, CREATIONS_COLUMN] if c in creations_raw.columns]
    stream_cols = [c for c in IDENTITY_COLUMNS + [DATE_COLUMN, stream_metric] if c in streams_raw.columns]

    creations = creations_raw[creation_cols].copy()
    streams = streams_raw[stream_cols].copy()

    creations[DATE_COLUMN] = pd.to_datetime(creations[DATE_COLUMN], errors="coerce")
    streams[DATE_COLUMN] = pd.to_datetime(streams[DATE_COLUMN], errors="coerce")
    creations[CREATIONS_COLUMN] = clean_numeric(creations[CREATIONS_COLUMN])
    streams[stream_metric] = clean_numeric(streams[stream_metric])

    creations = creations.dropna(subset=[DATE_COLUMN]).drop_duplicates(MERGE_KEYS, keep="last")
    streams = streams.dropna(subset=[DATE_COLUMN]).drop_duplicates(MERGE_KEYS, keep="last")

    merged = pd.merge(
        creations,
        streams[MERGE_KEYS + [stream_metric]],
        on=MERGE_KEYS,
        how="outer",
        suffixes=("", "_streams"),
    )
    return merged.sort_values(DATE_COLUMN).reset_index(drop=True)


def filter_entity(df: pd.DataFrame, artist: str, track: str, region: str) -> pd.DataFrame:
    out = df[
        df["Artist"].astype(str).eq(artist)
        & df["Track"].astype(str).eq(track)
        & df["Region"].astype(str).eq(region)
    ].copy()
    return out.sort_values(DATE_COLUMN).reset_index(drop=True)


def trim_leading_empty(df: pd.DataFrame, stream_metric: str) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df[CREATIONS_COLUMN].fillna(0).ne(0) | df[stream_metric].fillna(0).ne(0)
    if mask.any():
        first = int(np.flatnonzero(mask.to_numpy())[0])
        df = df.iloc[first:]
    return df.reset_index(drop=True)


def calculate_lag_correlations(df: pd.DataFrame, stream_metric: str, max_lag: int) -> pd.DataFrame:
    work = df[[DATE_COLUMN, CREATIONS_COLUMN, stream_metric]].sort_values(DATE_COLUMN).reset_index(drop=True)
    rows = []
    for lag in range(max_lag + 1):
        pair = pd.DataFrame({
            "creations": work[CREATIONS_COLUMN],
            "future_streams": work[stream_metric].shift(-lag),
        }).dropna()
        pair = pair[~(pair["creations"].eq(0) & pair["future_streams"].eq(0))]
        valid = len(pair) >= 4 and pair["creations"].nunique() > 1 and pair["future_streams"].nunique() > 1
        corr = pair["creations"].corr(pair["future_streams"]) if valid else np.nan
        rows.append({"lag_weeks": lag, "correlation": corr, "observations": len(pair)})
    return pd.DataFrame(rows)


def best_lag(lag_df: pd.DataFrame) -> tuple[int | None, float | None, int]:
    valid = lag_df.dropna(subset=["correlation"])
    if valid.empty:
        return None, None, 0
    row = valid.loc[valid["correlation"].idxmax()]
    return int(row["lag_weeks"]), float(row["correlation"]), int(row["observations"])


def compact_number(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    v = float(value)
    av = abs(v)
    if av >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if av >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{v:,.0f}"


def pct_change(series: pd.Series) -> float | None:
    valid = series.dropna()
    if len(valid) < 2 or valid.iloc[-2] == 0:
        return None
    return float((valid.iloc[-1] - valid.iloc[-2]) / abs(valid.iloc[-2]) * 100)


def min_max_index(series: pd.Series) -> pd.Series:
    values = series.astype(float)
    valid = values.dropna()
    if valid.empty:
        return pd.Series(np.nan, index=series.index)
    lo, hi = valid.min(), valid.max()
    if lo == hi:
        return pd.Series(0.0, index=series.index)
    return (values - lo) / (hi - lo) * 100
