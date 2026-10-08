from __future__ import annotations

import math
import numpy as np
import pandas as pd

from utils import CREATIONS_COLUMN, DATE_COLUMN

DEFAULT_INTL_DAILY = {
    "micro": 800.0,
    "medium": 2000.0,
    "viral": 3000.0,
}


def validate_benchmarks(micro: float, medium: float, viral: float) -> dict[str, float]:
    values = [float(micro), float(medium), float(viral)]
    if any(v <= 0 for v in values):
        raise ValueError("Benchmarks must be greater than 0.")
    if not (values[0] < values[1] < values[2]):
        raise ValueError("Benchmarks must follow Micro < Medium < Viral.")
    return {"micro": values[0], "medium": values[1], "viral": values[2]}


def weekly_benchmarks(daily: dict[str, float]) -> dict[str, float]:
    return {key: value * 7.0 for key, value in daily.items()}


def _tier_from_daily(value: float, daily: dict[str, float]) -> str:
    if not np.isfinite(value):
        return "Insufficient data"
    if value >= daily["viral"]:
        return "Viral"
    if value >= daily["medium"]:
        return "Medium Trend"
    if value >= daily["micro"]:
        return "Micro Trend"
    return "Below Trend"


def classify_trend_level(
    df: pd.DataFrame,
    daily_benchmarks: dict[str, float],
    persistence_weeks: int = 1,
) -> dict:
    """Classify trend scale independently from lifecycle phase.

    Weekly creations are converted to an estimated daily average by dividing by 7.
    A tier is confirmed only if the threshold is maintained for the requested number
    of consecutive weeks. The peak tier is retained as historical context while the
    current tier reflects the latest week.
    """
    data = df.sort_values(DATE_COLUMN).copy()
    creations = pd.to_numeric(data[CREATIONS_COLUMN], errors="coerce").fillna(0).clip(lower=0)
    daily = creations / 7.0
    tiers = daily.map(lambda x: _tier_from_daily(float(x), daily_benchmarks))

    rank = {"Below Trend": 0, "Micro Trend": 1, "Medium Trend": 2, "Viral": 3}
    labels = ["Below Trend", "Micro Trend", "Medium Trend", "Viral"]

    confirmed_peak_rank = 0
    confirmed_peak_week = None
    for tier_rank in range(1, 4):
        threshold_label = labels[tier_rank]
        mask = tiers.map(rank).ge(tier_rank).astype(int)
        rolling = mask.rolling(max(1, int(persistence_weeks))).sum()
        qualifying = rolling[rolling >= max(1, int(persistence_weeks))]
        if not qualifying.empty:
            confirmed_peak_rank = tier_rank
            confirmed_peak_week = pd.Timestamp(data.loc[qualifying.index[-1], DATE_COLUMN])

    peak_idx = int(daily.idxmax()) if len(daily) else 0
    peak_daily = float(daily.loc[peak_idx]) if len(daily) else float("nan")
    latest_daily = float(daily.iloc[-1]) if len(daily) else float("nan")
    latest_tier = _tier_from_daily(latest_daily, daily_benchmarks)
    raw_peak_tier = _tier_from_daily(peak_daily, daily_benchmarks)
    confirmed_peak_tier = labels[confirmed_peak_rank]

    return {
        "current_tier": latest_tier,
        "peak_tier": confirmed_peak_tier,
        "raw_peak_tier": raw_peak_tier,
        "latest_daily_creations": latest_daily,
        "peak_daily_creations": peak_daily,
        "peak_week": pd.Timestamp(data.loc[peak_idx, DATE_COLUMN]) if len(data) else None,
        "confirmed_week": confirmed_peak_week,
        "persistence_weeks": int(persistence_weeks),
        "daily_benchmarks": daily_benchmarks,
        "weekly_benchmarks": weekly_benchmarks(daily_benchmarks),
    }


def tier_for_segment(values: np.ndarray, daily_benchmarks: dict[str, float]) -> str:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "Below Trend"
    # Segment badges describe the highest scale reached inside that lifecycle segment.
    # Persistence is handled separately by the overall trend assessment.
    representative_weekly = float(np.nanmax(values))
    return _tier_from_daily(representative_weekly / 7.0, daily_benchmarks)


def detect_conversion_signal(
    df: pd.DataFrame,
    stream_metric: str,
    trend_result: dict,
    minimum_trend: str = "Micro Trend",
) -> dict:
    """Detect a likely sound-bite pattern using scale-adjusted uplift.

    The comparison is based on log uplift from an early baseline, so creations and
    streams can be compared despite having different units. A warning is returned
    only when creations reach at least the minimum trend threshold and streaming
    uplift is materially weaker.
    """
    data = df.sort_values(DATE_COLUMN).copy()
    c = pd.to_numeric(data[CREATIONS_COLUMN], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    s = pd.to_numeric(data[stream_metric], errors="coerce").fillna(0).clip(lower=0).to_numpy(float)
    n = len(data)
    if n < 5 or np.nanmax(c) <= 0:
        return {"status": "Insufficient data", "is_sound_bite": False, "note": "At least five weekly observations are recommended.", "conversion_score": None}

    baseline_n = min(3, max(2, n // 4))
    c_base = float(np.nanmedian(c[:baseline_n]))
    s_base = float(np.nanmedian(s[:baseline_n]))
    c_peak = float(np.nanpercentile(c, 90))
    s_peak = float(np.nanpercentile(s, 90))

    c_uplift = math.log1p(c_peak) - math.log1p(c_base)
    s_uplift = math.log1p(s_peak) - math.log1p(s_base)
    conversion_score = float(s_uplift / c_uplift) if c_uplift > 1e-9 else None

    rank = {"Below Trend": 0, "Micro Trend": 1, "Medium Trend": 2, "Viral": 3}
    reaches_trend = rank.get(trend_result.get("peak_tier", "Below Trend"), 0) >= rank[minimum_trend]

    recent_n = min(4, n)
    x = np.arange(recent_n, dtype=float)
    recent_c_slope = float(np.polyfit(x, np.log1p(c[-recent_n:]), 1)[0]) if recent_n >= 2 else 0.0
    recent_s_slope = float(np.polyfit(x, np.log1p(s[-recent_n:]), 1)[0]) if recent_n >= 2 else 0.0

    weak_conversion = conversion_score is not None and conversion_score < 0.38
    recent_divergence = recent_c_slope > 0.10 and recent_s_slope <= 0.03
    is_sound_bite = bool(reaches_trend and c_uplift >= 1.0 and (weak_conversion or recent_divergence))

    if is_sound_bite:
        status = "Sound bite risk"
        note = "Creations have broken out, but streaming uplift is materially weaker. The sound may be used as a short-form content device without equivalent full-song conversion."
    elif reaches_trend and conversion_score is not None and conversion_score >= 0.75:
        status = "Strong conversion"
        note = "Streaming uplift is keeping pace with the creation breakout."
    elif reaches_trend:
        status = "Watch conversion"
        note = "The trend has meaningful creation scale, but streaming conversion is not yet clearly strong."
    else:
        status = "Below trend benchmark"
        note = "Creation scale has not yet reached the selected Micro Trend benchmark."

    return {
        "status": status,
        "is_sound_bite": is_sound_bite,
        "note": note,
        "conversion_score": conversion_score,
        "creation_log_uplift": c_uplift,
        "stream_log_uplift": s_uplift,
        "recent_creation_slope": recent_c_slope,
        "recent_stream_slope": recent_s_slope,
    }
