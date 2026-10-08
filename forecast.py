from __future__ import annotations

import numpy as np
import pandas as pd

from utils import CREATIONS_COLUMN, DATE_COLUMN


def _baseline_forecast(series: pd.Series, recent_weeks: int = 6) -> dict:
    y = pd.to_numeric(series, errors="coerce").dropna().clip(lower=0).to_numpy(float)
    if len(y) == 0:
        return {"latest": np.nan, "forecast": np.nan, "low": np.nan, "high": np.nan, "method": "Unavailable"}
    if len(y) == 1:
        return {"latest": y[-1], "forecast": y[-1], "low": max(0, y[-1] * 0.75), "high": y[-1] * 1.25, "method": "Naive level"}

    window = y[-min(recent_weeks, len(y)):]
    log_y = np.log1p(window)
    x = np.arange(len(window), dtype=float)
    weights = np.linspace(0.65, 1.0, len(window))
    slope, intercept = np.polyfit(x, log_y, 1, w=weights)

    # Damping avoids extrapolating a viral spike indefinitely.
    recent_diffs = np.diff(log_y[-min(4, len(log_y)):])
    local_slope = float(np.median(recent_diffs)) if len(recent_diffs) else slope
    blended_slope = 0.7 * local_slope + 0.3 * slope
    damped_slope = float(np.clip(blended_slope * 0.45, -0.40, 0.34))
    trend_pred = float(window[-1] * np.exp(damped_slope))
    ewma_level = float(pd.Series(window).ewm(alpha=0.62, adjust=False).mean().iloc[-1])
    forecast = max(0.0, 0.72 * trend_pred + 0.28 * ewma_level)
    forecast = float(np.clip(forecast, window[-1] * 0.60, window[-1] * 1.40))

    fitted = np.expm1(intercept + slope * x)
    residual_log = np.log1p(window) - np.log1p(np.clip(fitted, 0, None))
    sigma = float(np.std(residual_log, ddof=1)) if len(residual_log) >= 3 else 0.22
    sigma = max(0.12, min(sigma, 0.65))
    low = max(0.0, float(np.expm1(np.log1p(forecast) - 1.28 * sigma)))
    high = max(low, float(np.expm1(np.log1p(forecast) + 1.28 * sigma)))
    return {"latest": y[-1], "forecast": forecast, "low": low, "high": high, "method": "Damped log-trend + EWMA"}


def next_week_prediction(df: pd.DataFrame, stream_metric: str, current_phase: str | None = None) -> pd.DataFrame:
    next_date = pd.Timestamp(df[DATE_COLUMN].max()) + pd.Timedelta(days=7)
    metrics = [("TikTok Creations", CREATIONS_COLUMN), (stream_metric, stream_metric)]
    rows = []
    for label, column in metrics:
        result = _baseline_forecast(df[column])
        latest = result["latest"]
        forecast = result["forecast"]
        change = ((forecast - latest) / latest * 100) if np.isfinite(latest) and latest > 0 else np.nan
        rows.append({
            "Week": next_date.date(),
            "Metric": label,
            "Latest actual": round(latest) if np.isfinite(latest) else np.nan,
            "No-event baseline": round(forecast) if np.isfinite(forecast) else np.nan,
            "Low (80%)": round(result["low"]) if np.isfinite(result["low"]) else np.nan,
            "High (80%)": round(result["high"]) if np.isfinite(result["high"]) else np.nan,
            "Expected WoW": f"{change:+.1f}%" if np.isfinite(change) else "—",
            "Current phase": current_phase or "—",
            "Method": result["method"],
        })
    return pd.DataFrame(rows)
