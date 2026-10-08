# IG / TikTok → Streaming Report

A report-focused adaptation of the original Creations vs Streams dashboard. The old app remains in `legacy_app.py`; the default `app.py` compares Instagram, TikTok and streaming without legacy conversion scores, viral thresholds or forecasts.

## Run

Python 3.10+ recommended.

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The bundled Honeybee files let you try the app immediately. They are the original uploaded example, not new study evidence.

## Suggested report workflow

1. Upload social and streaming CSVs. Original exports work: `Artist`, `Track`, `Region`, `period_end`, Instagram/TikTok Creations and Views, plus your chosen stream metric. Column mapping supports other headers. Multiple files per uploader must use the same schema; no overlapping duplicate entity/date rows.
2. Select matching artist, track and region. Weekly observations must have aligned period-end dates, including across sources. Daily and weekly files must be reconciled before import. Names are matched exactly; stable identifiers can be mapped to the identity fields. Avoid combining differently scoped global and country exports.
3. Select chart dates, lines and baseline dates. Start with IG Creations + TikTok Creations + Streams, baseline index 100. Repeat for Views. Raw mode uses a right axis for streams and a shared social axis; views and creations have very different scales.
4. Customize title, subtitle, colors, dimensions, labels and trailing mean. Optionally select a campaign/social spike period end.
5. Download PNG for slides, SVG for editable vector charts, or standalone interactive HTML. Export chart/source CSVs and report settings JSON for your audit. Settings JSON is documentation; it is not an import preset.
6. Compare lag curves at the same lag and inspect paired observation counts. Positive lag pairs social now with streams later. First differences are on by default. At least six paired values are required. Do not select the highest correlation and call it proof of conversion.
7. Use before/after event tables to describe changes against a flat pre-event average. Windows are calculated on full selected-track data, independent of the chart filter. An incomplete window or zero baseline suppresses percent change. Event windows for weekly data are weekly periods, not daily precision.

## Data treatment

- Missing observations remain missing; no zero filling or interpolation. Missing reporting dates are inserted on the selected cadence so lag means elapsed daily/weekly periods.
- Duplicates, invalid dates, negative counts and malformed numeric values stop import.
- Cumulative mode differences every mapped metric within artist/track/region. First values remain unavailable and declining counters are rejected. Do not use this checkbox on a file mixing cumulative and period counts; convert mixed metrics before uploading.
- Baseline index = value / observed baseline mean × 100. Percent change = (value / observed baseline mean − 1) × 100. Baseline may fall outside chart dates; observation counts and means are visible in the audit panel. Missing baseline values are excluded from the mean.
- Trailing mean requires a complete window, and applies only to chart data. Lag/event calculations use unsmoothed observations.
- Streams from different DSPs may overlap; the app uses one chosen source metric and never adds partner totals to individual DSP totals.
- IG and TikTok metrics may differ in counting definitions. Compare the same metric type and scope; reach is not unique listeners.

## Interpretation

Frame findings as **observed streaming response / association**, not conversion rates or causal platform contribution. This app does not control for playlists, paid spend, release age or concurrent campaigns. Shared trends and concurrent platform activity can affect the correlation. Compare multiple tracks and both successful and weak-response events before drawing broader conclusions.

PNG/SVG exports use Matplotlib and do not require Chrome. Interactive HTML embeds Plotly and works offline. The static export uses the same selected data, colors and scales, with a presentation-friendly layout; its rendering differs slightly from the interactive chart.
