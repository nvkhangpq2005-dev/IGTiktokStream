from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from charts import aligned_chart, lag_chart, main_chart
from forecast import next_week_prediction
from phase_models import MODEL_LABELS, make_phase_result, phase_table
from trend_analysis import (
    DEFAULT_INTL_DAILY,
    classify_trend_level,
    detect_conversion_signal,
    validate_benchmarks,
)
from utils import (
    CREATIONS_COLUMN, DATE_COLUMN, available_stream_metrics, best_lag,
    calculate_lag_correlations, compact_number, filter_entity, pct_change,
    prepare_data, read_uploaded_csv, trim_leading_empty,
)

st.set_page_config(page_title="Creations vs Streams", page_icon="🎵", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
:root{--pink:#DE83B6;--orange:#F16A2D;--purple:#6F35AE;--ink:#171821;--muted:#687083;--line:#E7E9EF}
html,body,[class*="css"]{font-family:Inter,Arial,sans-serif}.block-container{max-width:1580px;padding-top:1rem;padding-bottom:2.5rem}
[data-testid="stSidebar"]{border-right:1px solid var(--line);background:linear-gradient(180deg,#FBFBFD,#F7F8FB)}
[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:16px;padding:15px 17px;box-shadow:0 5px 18px rgba(28,31,45,.05)}
[data-testid="stMetricLabel"]{font-weight:750;color:#687083;text-transform:uppercase;letter-spacing:.04em;font-size:.76rem}
[data-testid="stMetricValue"]{font-weight:850;color:var(--ink)}
div[data-testid="stPlotlyChart"]{background:#fff;border:1px solid var(--line);border-radius:18px;padding:8px 10px;box-shadow:0 6px 22px rgba(28,31,45,.045)}
.hero{display:flex;align-items:center;gap:14px;margin:4px 0 15px}.hero-icon{width:48px;height:48px;border-radius:50%;display:grid;place-items:center;background:linear-gradient(135deg,#FFE5F3,#F3DDFB);font-size:24px}.hero-title{font-size:2rem;font-weight:850;color:var(--ink);line-height:1.05}.hero-sub{color:var(--muted);margin-top:5px}
.section-title{font-weight:800;color:var(--ink);font-size:1.02rem;margin:.35rem 0 .55rem}.help-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-top:8px}.help-card{border:1px solid var(--line);border-radius:14px;padding:13px 15px;background:#fff;font-size:.87rem;line-height:1.45;color:#4E5668}.help-card b{display:block;color:#222733;margin-bottom:4px}.green{background:#F1FAEE}.purple{background:#F7F0FC}.blue{background:#EEF8FC}.yellow{background:#FFF8E8}.small{color:#737A8B;font-size:.84rem;line-height:1.45}
.note-card{border:1px solid #F0C9AA;background:#FFF7F0;border-radius:14px;padding:14px 16px;color:#6E3C18;margin:8px 0 14px}.ok-card{border:1px solid #BFE3CC;background:#F2FBF5;border-radius:14px;padding:14px 16px;color:#245B35;margin:8px 0 14px}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:14px;overflow:hidden}.stButton>button,.stDownloadButton>button{border-radius:10px;font-weight:700}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="hero-icon">♫</div><div><div class="hero-title">Creations vs Streams</div><div class="hero-sub">Lifecycle phase, trend scale, conversion signal, lag analysis and no-event forecasting</div></div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="section-title">1. Upload files</div>', unsafe_allow_html=True)
    creations_file = st.file_uploader("Creations CSV", type=["csv"])
    streams_file = st.file_uploader("Streams CSV", type=["csv"])
    with st.expander("Sample format"):
        sample_dir = Path(__file__).parent / "sample_data"
        st.download_button("Sample Creations", (sample_dir / "honeybee_Creations.csv").read_bytes(), "honeybee_Creations.csv", "text/csv")
        st.download_button("Sample Streams", (sample_dir / "honeybee_Streams.csv").read_bytes(), "honeybee_Streams.csv", "text/csv")

if creations_file is None or streams_file is None:
    st.info("Upload both CSV files in the sidebar to begin.")
    st.stop()

try:
    creations_raw = read_uploaded_csv(creations_file)
    streams_raw = read_uploaded_csv(streams_file)
except Exception as exc:
    st.error(f"Không đọc được file: {exc}")
    st.stop()

stream_metrics = available_stream_metrics(streams_raw)
if not stream_metrics:
    st.error("Không tìm thấy stream metric hợp lệ trong Streams file.")
    st.stop()

with st.sidebar:
    st.markdown('<div class="section-title">2. Select track</div>', unsafe_allow_html=True)
    stream_metric = st.selectbox("Stream metric", stream_metrics)

try:
    merged = prepare_data(creations_raw, streams_raw, stream_metric)
except Exception as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    artists = sorted(merged["Artist"].dropna().astype(str).unique())
    artist = st.selectbox("Artist", artists)
    tracks = sorted(merged.loc[merged["Artist"].astype(str).eq(artist), "Track"].dropna().astype(str).unique())
    track = st.selectbox("Track", tracks)
    regions = sorted(merged.loc[merged["Artist"].astype(str).eq(artist) & merged["Track"].astype(str).eq(track), "Region"].dropna().astype(str).unique())
    region = st.selectbox("Region", regions)

entity = trim_leading_empty(filter_entity(merged, artist, track, region), stream_metric)
if entity.empty:
    st.warning("Không có dữ liệu cho lựa chọn hiện tại.")
    st.stop()

min_date, max_date = entity[DATE_COLUMN].min().date(), entity[DATE_COLUMN].max().date()
with st.sidebar:
    st.markdown('<div class="section-title">3. Market benchmark</div>', unsafe_allow_html=True)
    market_type = st.radio("Track type", ["International", "Domestic"], horizontal=True)
    if market_type == "International":
        daily_benchmarks = DEFAULT_INTL_DAILY.copy()
        st.caption("INTL daily benchmark: Micro 800 · Medium 2,000 · Viral 3,000 creations/day. App multiplies by 7 for weekly data.")
    else:
        st.caption("Enter your Domestic daily creations benchmarks. Weekly thresholds are calculated automatically × 7.")
        domestic_micro = st.number_input("Micro Trend / day", min_value=1, value=300, step=50)
        domestic_medium = st.number_input("Medium Trend / day", min_value=2, value=800, step=50)
        domestic_viral = st.number_input("Viral / day", min_value=3, value=1500, step=100)
        try:
            daily_benchmarks = validate_benchmarks(domestic_micro, domestic_medium, domestic_viral)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

    persistence_weeks = st.slider("Tier persistence", 1, 3, 1, help="Number of consecutive weeks required to confirm the peak trend tier.")

    st.markdown('<div class="section-title">4. Duration</div>', unsafe_allow_html=True)
    preset = st.radio("Quick range", ["4W", "8W", "12W", "26W", "All"], index=1, horizontal=True)
    weeks_map = {"4W": 4, "8W": 8, "12W": 12, "26W": 26}
    preset_start = min_date if preset == "All" else max(min_date, (pd.Timestamp(max_date) - pd.DateOffset(weeks=weeks_map[preset] - 1)).date())
    custom = st.date_input("Custom date range", value=(preset_start, max_date), min_value=min_date, max_value=max_date)
    if not isinstance(custom, tuple) or len(custom) != 2:
        st.stop()
    start_date, end_date = custom

    st.markdown('<div class="section-title">5. Phase detection</div>', unsafe_allow_html=True)
    phase_method = st.selectbox("Method", list(MODEL_LABELS), index=1,
        help="Lifecycle phase is separate from trend level. Models detect structural changes; business labels describe Emerging, Growth, Steady and Declining.")
    sensitivity = st.select_slider("Model sensitivity", ["Conservative", "Balanced", "Sensitive"], value="Balanced",
        disabled=phase_method == "Current heuristic")

    st.markdown('<div class="section-title">6. Display options</div>', unsafe_allow_html=True)
    show_phases = st.toggle("Show phase background", True)
    show_labels = st.toggle("Show point labels", True)
    show_slider = st.toggle("Show range slider", False)

    st.markdown('<div class="section-title">7. Lag settings</div>', unsafe_allow_html=True)
    max_allowed_lag = max(0, min(12, len(entity) - 4))
    max_lag = st.slider("Maximum lag to test", 0, max_allowed_lag, min(3, max_allowed_lag))
    st.caption("Lag +1 = creations tuần này so với streams tuần sau.")

filtered = entity[entity[DATE_COLUMN].between(pd.Timestamp(start_date), pd.Timestamp(end_date))].copy()
if filtered.empty:
    st.warning("Không có dữ liệu trong khoảng đã chọn.")
    st.stop()

trend_result = classify_trend_level(filtered, daily_benchmarks, persistence_weeks)
conversion = detect_conversion_signal(filtered, stream_metric, trend_result)
phase_result = make_phase_result(filtered, stream_metric, phase_method, sensitivity, daily_benchmarks)
current_phase = phase_result["segments"][-1]["top"] if phase_result.get("segments") else None
lag_df = calculate_lag_correlations(filtered, stream_metric, max_lag)
lag, corr, n_obs = best_lag(lag_df)
latest_creations = filtered[CREATIONS_COLUMN].dropna().iloc[-1] if filtered[CREATIONS_COLUMN].notna().any() else None
latest_streams = filtered[stream_metric].dropna().iloc[-1] if filtered[stream_metric].notna().any() else None
cg, sg = pct_change(filtered[CREATIONS_COLUMN]), pct_change(filtered[stream_metric])

st.markdown(f'<div class="hero"><div><div class="hero-title" style="font-size:1.85rem">{artist} – {track}</div><div class="hero-sub">{region} · {market_type} · {stream_metric}</div></div></div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Latest creations", compact_number(latest_creations), f"{cg:+.1f}% WoW" if cg is not None else None)
m2.metric("Latest streams", compact_number(latest_streams), f"{sg:+.1f}% WoW" if sg is not None else None)
m3.metric("Lifecycle phase", current_phase or "Insufficient data", phase_method)
m4.metric("Trend level", trend_result["current_tier"], f"Peak: {trend_result['peak_tier']}")
m5.metric("Conversion signal", conversion["status"], f"Score: {conversion['conversion_score']:.2f}" if conversion["conversion_score"] is not None else None)

if conversion["is_sound_bite"]:
    st.markdown(f'<div class="note-card"><b>⚠ Sound bite risk</b><br>{conversion["note"]}</div>', unsafe_allow_html=True)
elif conversion["status"] == "Strong conversion":
    st.markdown(f'<div class="ok-card"><b>✓ Strong creation-to-stream conversion</b><br>{conversion["note"]}</div>', unsafe_allow_html=True)
else:
    st.info(conversion["note"])

fig = main_chart(filtered, stream_metric, show_labels, show_slider, show_phases,
                 "Weekly TikTok Creations vs Streams", phase_result)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

bench_col, summary_col = st.columns([1, 1.4])
with bench_col:
    st.markdown('<div class="section-title">Trend benchmarks</div>', unsafe_allow_html=True)
    benchmark_df = pd.DataFrame({
        "Trend level": ["Micro Trend", "Medium Trend", "Viral"],
        "Daily creations": [daily_benchmarks["micro"], daily_benchmarks["medium"], daily_benchmarks["viral"]],
        "Approx. weekly creations": [daily_benchmarks["micro"] * 7, daily_benchmarks["medium"] * 7, daily_benchmarks["viral"] * 7],
    })
    st.dataframe(benchmark_df, use_container_width=True, hide_index=True,
                 column_config={"Daily creations": st.column_config.NumberColumn(format="%,.0f"), "Approx. weekly creations": st.column_config.NumberColumn(format="%,.0f")})
with summary_col:
    st.markdown('<div class="section-title">Trend assessment</div>', unsafe_allow_html=True)
    assessment_df = pd.DataFrame([
        {"Measure": "Latest estimated daily creations", "Value": round(trend_result["latest_daily_creations"])},
        {"Measure": "Peak estimated daily creations", "Value": round(trend_result["peak_daily_creations"])},
        {"Measure": "Current trend level", "Value": trend_result["current_tier"]},
        {"Measure": "Confirmed peak trend level", "Value": trend_result["peak_tier"]},
        {"Measure": "Conversion status", "Value": conversion["status"]},
    ])
    st.dataframe(assessment_df, use_container_width=True, hide_index=True)

st.markdown('<div class="section-title">Detected lifecycle phases</div>', unsafe_allow_html=True)
phase_df = phase_table(phase_result)
if not phase_df.empty:
    st.dataframe(phase_df, use_container_width=True, hide_index=True,
                 column_config={"Avg daily creations": st.column_config.NumberColumn(format="%,.0f")})
else:
    st.info("Chưa đủ điểm dữ liệu để chia phase đáng tin cậy.")

c1, c2 = st.columns([1, 1.35])
with c1:
    if lag_df["correlation"].notna().any():
        st.plotly_chart(lag_chart(lag_df, lag, corr), use_container_width=True, config={"displaylogo": False})
    else:
        st.info("Chưa đủ dữ liệu để tính lag correlation.")
with c2:
    if lag is not None:
        st.plotly_chart(aligned_chart(filtered, stream_metric, lag), use_container_width=True, config={"displaylogo": False})

st.markdown('<div class="section-title">Next-week no-event baseline</div>', unsafe_allow_html=True)
st.caption("Prediction assumes no campaign push, major playlist addition, creator seeding, release event, media event or external shock. It is a baseline, not a causal forecast.")
prediction_df = next_week_prediction(filtered, stream_metric, current_phase)
st.dataframe(
    prediction_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Latest actual": st.column_config.NumberColumn(format="%,d"),
        "No-event baseline": st.column_config.NumberColumn(format="%,d"),
        "Low (80%)": st.column_config.NumberColumn(format="%,d"),
        "High (80%)": st.column_config.NumberColumn(format="%,d"),
    },
)

st.markdown('<div class="section-title">How to read this dashboard</div>', unsafe_allow_html=True)
st.markdown("""
<div class="help-grid">
<div class="help-card green"><b>Lifecycle phase</b>Emerging, Rapid Growth, Peak / Steady and Declining describe how the time series is moving. They do not automatically mean Viral.</div>
<div class="help-card purple"><b>Trend level</b>Micro, Medium and Viral are scale labels based on estimated daily creations. Weekly data is divided by seven.</div>
<div class="help-card blue"><b>Sound bite risk</b>Flagged when creations break out but scale-adjusted streaming uplift remains materially weaker.</div>
<div class="help-card yellow"><b>No-event baseline</b>Damped recent trend plus EWMA. The 80% range reflects recent residual variation, not event risk.</div>
</div>
""", unsafe_allow_html=True)

with st.expander("View & download merged data"):
    show = filtered[["Artist", "Track", "Region", DATE_COLUMN, CREATIONS_COLUMN, stream_metric]].copy()
    show[DATE_COLUMN] = show[DATE_COLUMN].dt.strftime("%Y-%m-%d")
    st.dataframe(show, use_container_width=True, hide_index=True)
    st.download_button("Download filtered CSV", show.to_csv(index=False).encode("utf-8-sig"), f"{track}_filtered.csv", "text/csv")

st.markdown('<div class="small">Trend benchmarks are directional business rules. Weekly creations are converted to estimated daily creations by dividing by seven, so partial weeks and reporting delays may introduce error. Phase boundaries are model-assisted and should be manually reviewed before external reporting.</div>', unsafe_allow_html=True)
