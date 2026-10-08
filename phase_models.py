from __future__ import annotations

import numpy as np
import pandas as pd

from trend_analysis import tier_for_segment
from utils import CREATIONS_COLUMN, DATE_COLUMN

MODEL_LABELS = {
    "Current heuristic": "Current heuristic",
    "Penalized change-point": "Penalized change-point",
    "Binary segmentation": "Binary segmentation",
}

PHASE_STYLES = {
    "emerging": {"fill": "rgba(206,238,194,0.64)", "badge_bg": "#D8D1E8", "badge_fg": "#111111"},
    "growth": {"fill": "rgba(238,222,247,0.54)", "badge_bg": "#6F2DA8", "badge_fg": "#FFFFFF"},
    "steady": {"fill": "rgba(245,232,194,0.50)", "badge_bg": "#B77A16", "badge_fg": "#FFFFFF"},
    "declining": {"fill": "rgba(191,230,247,0.70)", "badge_bg": "#8978C8", "badge_fg": "#FFFFFF"},
}


def _safe_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    med = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = np.nanstd(values)
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return np.nan_to_num((values - med) / scale)


def build_signal(df: pd.DataFrame, stream_metric: str) -> np.ndarray:
    c = pd.to_numeric(df[CREATIONS_COLUMN], errors="coerce").fillna(0).clip(lower=0)
    s = pd.to_numeric(df[stream_metric], errors="coerce").fillna(0).clip(lower=0)
    lc = np.log1p(c.to_numpy(float))
    ls = np.log1p(s.to_numpy(float))
    dc = np.r_[0.0, np.diff(lc)]
    ds = np.r_[0.0, np.diff(ls)]
    d2c = np.r_[0.0, np.diff(dc)]
    return np.column_stack([_safe_z(lc), 0.85 * _safe_z(ls), 0.80 * _safe_z(dc), 0.50 * _safe_z(ds), 0.35 * _safe_z(d2c)])


def _sanitize_breaks(breaks: list[int], n: int, min_size: int = 2, max_breaks: int = 3) -> list[int]:
    cleaned: list[int] = []
    last = 0
    for point in sorted(set(int(x) for x in breaks if 0 < int(x) < n)):
        if point - last >= min_size and n - point >= min_size:
            cleaned.append(point)
            last = point
    return cleaned[:max_breaks]


def heuristic_breaks(df: pd.DataFrame) -> list[int]:
    values = pd.to_numeric(df[CREATIONS_COLUMN], errors="coerce").fillna(0).to_numpy(float)
    n = len(values)
    if n < 6 or np.nanmax(values) <= 0:
        return []
    logv = np.log1p(values)
    growth = np.r_[0.0, np.diff(logv)]
    peak = int(np.nanargmax(values))

    pre_peak = growth[: peak + 1]
    rapid = int(np.argmax(pre_peak)) if len(pre_peak) else 1
    rapid = max(2, rapid - 1)

    # Decline starts only after a confirmed peak/downturn, not merely peak + 1.
    decline = None
    for idx in range(max(peak + 1, rapid + 2), n - 1):
        if growth[idx] < -0.08 and growth[idx + 1] <= 0.03:
            decline = idx
            break
    if decline is None and peak < n - 2:
        decline = peak + 1

    plateau = None
    if peak - rapid >= 4:
        for idx in range(rapid + 2, peak + 1):
            window = growth[max(1, idx - 1): min(n, idx + 2)]
            if len(window) >= 2 and abs(float(np.nanmean(window))) < 0.08:
                plateau = idx
                break

    points = [rapid]
    if plateau is not None and decline is not None and rapid + 2 <= plateau <= decline - 2:
        points.append(plateau)
    if decline is not None:
        points.append(decline)
    return _sanitize_breaks(points, n)


def _segment_cost(signal: np.ndarray, left: int, right: int) -> float:
    block = signal[left:right]
    if len(block) <= 1:
        return 0.0
    center = block.mean(axis=0, keepdims=True)
    return float(np.square(block - center).sum())


def penalized_breaks(df: pd.DataFrame, stream_metric: str, sensitivity: str) -> list[int]:
    n = len(df)
    if n < 7:
        return heuristic_breaks(df)
    signal = build_signal(df, stream_metric)
    min_size = 2
    max_breaks = 1 if sensitivity == "Conservative" else (2 if sensitivity == "Balanced" else 3)
    multiplier = {"Conservative": 1.9, "Balanced": 1.02, "Sensitive": 0.57}[sensitivity]
    penalty = multiplier * signal.shape[1] * np.log(max(n, 2))

    candidates: list[tuple[float, list[int]]] = [(_segment_cost(signal, 0, n), [])]
    valid_points = range(min_size, n - min_size + 1)
    for b1 in valid_points:
        candidates.append((_segment_cost(signal, 0, b1) + _segment_cost(signal, b1, n) + penalty, [b1]))
    if max_breaks >= 2:
        for b1 in range(min_size, n - 2 * min_size + 1):
            for b2 in range(b1 + min_size, n - min_size + 1):
                score = _segment_cost(signal, 0, b1) + _segment_cost(signal, b1, b2) + _segment_cost(signal, b2, n) + 2 * penalty
                candidates.append((score, [b1, b2]))
    if max_breaks >= 3 and n >= 9:
        for b1 in range(min_size, n - 3 * min_size + 1):
            for b2 in range(b1 + min_size, n - 2 * min_size + 1):
                for b3 in range(b2 + min_size, n - min_size + 1):
                    score = (_segment_cost(signal, 0, b1) + _segment_cost(signal, b1, b2)
                             + _segment_cost(signal, b2, b3) + _segment_cost(signal, b3, n) + 3 * penalty)
                    candidates.append((score, [b1, b2, b3]))
    best = min(candidates, key=lambda item: item[0])[1]
    return _sanitize_breaks(best, n, max_breaks=max_breaks)


def binary_segmentation_breaks(df: pd.DataFrame, stream_metric: str, sensitivity: str) -> list[int]:
    n = len(df)
    if n < 7:
        return heuristic_breaks(df)
    signal = build_signal(df, stream_metric)
    min_size = 2
    target = {"Conservative": 1, "Balanced": 2, "Sensitive": 3}[sensitivity]
    segments = [(0, n)]
    breaks: list[int] = []
    for _ in range(target):
        best_gain, best_point, best_segment = 0.0, None, None
        for left, right in segments:
            parent = _segment_cost(signal, left, right)
            for point in range(left + min_size, right - min_size + 1):
                gain = parent - _segment_cost(signal, left, point) - _segment_cost(signal, point, right)
                if gain > best_gain:
                    best_gain, best_point, best_segment = gain, point, (left, right)
        if best_point is None:
            break
        segments.remove(best_segment)
        segments.extend([(best_segment[0], best_point), (best_point, best_segment[1])])
        breaks.append(best_point)
    return _sanitize_breaks(breaks, n, max_breaks=target)


def _segment_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = np.log1p(np.clip(values.astype(float), 0, None))
    return float(np.polyfit(x, y, 1)[0])


def _phase_type(c_values: np.ndarray, s_values: np.ndarray, peak: float, global_peak_index: int, left: int, right: int) -> str:
    c_slope = _segment_slope(c_values)
    s_slope = _segment_slope(s_values)
    level = float(np.nanmean(c_values) / peak) if peak > 0 else 0.0
    contains_peak = left <= global_peak_index < right

    if right <= global_peak_index and level < 0.32 and c_slope < 0.16:
        return "emerging"
    if c_slope > 0.10 or (s_slope > 0.10 and c_slope > -0.02):
        return "growth"
    if contains_peak or (level >= 0.62 and abs(c_slope) <= 0.11):
        return "steady"
    if c_slope < -0.075 and right > global_peak_index:
        return "declining"
    if right > global_peak_index and level < 0.72:
        return "declining"
    return "steady" if level >= 0.45 else "emerging"


def _phase_top(phase_type: str) -> str:
    return {"emerging": "Emerging / Steady", "growth": "Rapid Growth", "steady": "Peak / Steady", "declining": "Declining"}[phase_type]


def _phase_badge(phase_type: str, trend_tier: str) -> str:
    if phase_type == "emerging":
        return "Bubbling" if trend_tier == "Below Trend" else trend_tier
    if phase_type in {"growth", "steady"}:
        return trend_tier
    return "Post-Peak"


def _confidence(c_values: np.ndarray, phase_type: str, peak: float) -> int:
    slope = _segment_slope(c_values)
    level = float(np.nanmean(c_values) / peak) if peak > 0 else 0.0
    if phase_type == "growth":
        score = 0.54 + min(0.34, max(0.0, slope) * 0.70)
    elif phase_type == "declining":
        score = 0.54 + min(0.34, max(0.0, -slope) * 0.78)
    elif phase_type == "steady":
        score = 0.58 + min(0.28, level * 0.26) + min(0.06, max(0.0, 0.10 - abs(slope)))
    else:
        score = 0.58 + min(0.25, max(0.0, 0.40 - level) * 0.40)
    return int(np.clip(round(score * 100), 55, 95))


def make_phase_result(
    df: pd.DataFrame,
    stream_metric: str,
    method: str,
    sensitivity: str = "Balanced",
    daily_benchmarks: dict[str, float] | None = None,
) -> dict:
    data = df.sort_values(DATE_COLUMN).reset_index(drop=True).copy()
    n = len(data)
    if daily_benchmarks is None:
        daily_benchmarks = {"micro": 800.0, "medium": 2000.0, "viral": 3000.0}
    if n < 4:
        return {"method": method, "segments": [], "break_indices": [], "fallback": True}

    if method == "Penalized change-point":
        breaks = penalized_breaks(data, stream_metric, sensitivity)
    elif method == "Binary segmentation":
        breaks = binary_segmentation_breaks(data, stream_metric, sensitivity)
    else:
        breaks = heuristic_breaks(data)

    breaks = _sanitize_breaks(breaks, n, max_breaks=3)
    c_all = pd.to_numeric(data[CREATIONS_COLUMN], errors="coerce").fillna(0).to_numpy(float)
    s_all = pd.to_numeric(data[stream_metric], errors="coerce").fillna(0).to_numpy(float)
    peak = float(np.max(c_all)) if len(c_all) else 0.0
    peak_index = int(np.argmax(c_all)) if len(c_all) else 0

    # Business safeguard: a sustained post-peak fall should create a Declining
    # boundary even when the statistical model groups the plateau and tail together.
    logc = np.log1p(np.clip(c_all, 0, None))
    growth = np.r_[0.0, np.diff(logc)]
    decline_index = None
    for idx in range(peak_index + 1, n - 1):
        if growth[idx] < -0.08 and growth[idx + 1] <= 0.03:
            decline_index = idx
            break
    if decline_index is not None and decline_index >= 2 and n - decline_index >= 2:
        if all(abs(decline_index - point) >= 2 for point in breaks):
            if len(breaks) < 3:
                breaks.append(decline_index)
            else:
                # Replace the least useful interior point, preserving the earliest breakout.
                replaceable = [point for point in breaks if point > peak_index]
                if replaceable:
                    breaks.remove(replaceable[-1])
                    breaks.append(decline_index)
        breaks = _sanitize_breaks(breaks, n, max_breaks=3)

    boundaries = [0] + breaks + [n]

    segments = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        c_vals = c_all[left:right]
        s_vals = s_all[left:right]
        phase_type = _phase_type(c_vals, s_vals, peak, peak_index, left, right)
        trend_tier = tier_for_segment(c_vals, daily_benchmarks)
        segments.append({
            "label": _phase_top(phase_type),
            "phase_type": phase_type,
            "top": _phase_top(phase_type),
            "badge": _phase_badge(phase_type, trend_tier),
            "trend_tier": trend_tier,
            "start": data.loc[left, DATE_COLUMN],
            "end": data.loc[right - 1, DATE_COLUMN],
            "start_index": left,
            "end_index": right - 1,
            "confidence": _confidence(c_vals, phase_type, peak),
            "creation_slope": _segment_slope(c_vals),
            "stream_slope": _segment_slope(s_vals),
            "avg_daily_creations": float(np.nanmean(c_vals) / 7.0) if len(c_vals) else 0.0,
        })

    # Prevent illogical order reversals such as Declining -> Growth in the tail.
    seen_decline = False
    for segment in segments:
        if seen_decline and segment["phase_type"] == "growth":
            segment["phase_type"] = "steady"
            segment["top"] = "Post-Peak / Steady"
            segment["label"] = segment["top"]
            segment["badge"] = segment["trend_tier"]
        if segment["phase_type"] == "declining":
            seen_decline = True

    return {"method": method, "segments": segments, "break_indices": breaks, "fallback": False}


def phase_table(result: dict) -> pd.DataFrame:
    rows = []
    for segment in result.get("segments", []):
        rows.append({
            "Lifecycle phase": segment["top"],
            "Trend level": segment["trend_tier"],
            "Start": pd.Timestamp(segment["start"]).date(),
            "End": pd.Timestamp(segment["end"]).date(),
            "Avg daily creations": round(segment["avg_daily_creations"]),
            "Confidence": f"{segment['confidence']}%",
            "Creation slope": round(segment["creation_slope"], 3),
            "Stream slope": round(segment["stream_slope"], 3),
        })
    return pd.DataFrame(rows)
