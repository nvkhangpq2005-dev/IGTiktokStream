from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from phase_models import PHASE_STYLES
from utils import CREATIONS_COLUMN, DATE_COLUMN, compact_number, min_max_index

CREATION_COLOR = "#DE83B6"
STREAM_COLOR = "#F16A2D"
PURPLE = "#6F35AE"
GRID = "#E7E9EF"


def _segment_x_bounds(df: pd.DataFrame, segment: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = df[DATE_COLUMN].sort_values().reset_index(drop=True)
    start = pd.Timestamp(segment["start"])
    end = pd.Timestamp(segment["end"])
    if len(dates) > 1:
        half = (dates.diff().dropna().median() or pd.Timedelta(days=7)) / 2
    else:
        half = pd.Timedelta(days=3.5)
    return start - half, end + half


def main_chart(df: pd.DataFrame, stream_metric: str, show_labels: bool, show_slider: bool,
               show_phases: bool, title: str, phase_result: dict | None = None) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    custom = np.column_stack([df[CREATIONS_COLUMN], df[stream_metric]])
    mode = "lines+markers+text" if show_labels else "lines+markers"

    if show_phases and phase_result:
        for segment in phase_result.get("segments", []):
            style = PHASE_STYLES.get(segment.get("phase_type", "emerging"), PHASE_STYLES["emerging"])
            x0, x1 = _segment_x_bounds(df, segment)
            fig.add_shape(type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1, fillcolor=style["fill"], opacity=1, line_width=0, layer="below")
            midpoint = pd.Timestamp(segment["start"]) + (pd.Timestamp(segment["end"]) - pd.Timestamp(segment["start"])) / 2
            fig.add_annotation(x=midpoint, y=0.94, yref="paper", text=f"<b>{segment.get('top', segment.get('label', 'Phase'))}</b>", showarrow=False,
                               font=dict(size=13, color="#1F2028"))
            fig.add_annotation(x=midpoint, y=0.865, yref="paper", text=f"<b>{segment.get('badge', '')}</b>", showarrow=False,
                               bgcolor=style["badge_bg"], borderpad=3, font=dict(size=12, color=style["badge_fg"]))

    fig.add_trace(go.Scatter(
        x=df[DATE_COLUMN], y=df[CREATIONS_COLUMN], name="Weekly TikTok Creations", mode=mode,
        line=dict(color=CREATION_COLOR, width=4), marker=dict(size=9, color=CREATION_COLOR, line=dict(color="white", width=1.4)),
        text=df[CREATIONS_COLUMN].map(compact_number) if show_labels else None, textposition="top center",
        textfont=dict(color=CREATION_COLOR, size=11), customdata=custom,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Creations: %{customdata[0]:,.0f}<br>Streams: %{customdata[1]:,.0f}<extra></extra>"
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df[DATE_COLUMN], y=df[stream_metric], name=stream_metric, mode=mode,
        line=dict(color=STREAM_COLOR, width=4), marker=dict(size=9, color=STREAM_COLOR, line=dict(color="white", width=1.4)),
        text=df[stream_metric].map(compact_number) if show_labels else None, textposition="bottom center",
        textfont=dict(color=STREAM_COLOR, size=11), customdata=custom,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Creations: %{customdata[0]:,.0f}<br>Streams: %{customdata[1]:,.0f}<extra></extra>"
    ), secondary_y=True)

    fig.update_layout(
        title=dict(text=title, x=0.01, y=0.98, xanchor="left", yanchor="top", font=dict(size=22, color="#171821")),
        height=630, paper_bgcolor="white", plot_bgcolor="white", hovermode="x unified",
        margin=dict(l=45, r=45, t=110, b=90 if show_slider else 70),
        legend=dict(orientation="h", yanchor="bottom", y=1.015, xanchor="left", x=0.01, font=dict(size=12)),
        font=dict(family="Arial, sans-serif", color="#4A5060"),
    )
    fig.update_xaxes(showgrid=False, tickformat="%d %b", showline=True, linecolor="#D0D0D5",
                     rangeslider=dict(visible=show_slider, thickness=0.08, bgcolor="#F2F3F7", bordercolor="#D8DAE2", borderwidth=1))
    fig.update_yaxes(title_text="Weekly TikTok Creations", tickformat="~s", rangemode="tozero", gridcolor=GRID,
                     zeroline=False, secondary_y=False)
    fig.update_yaxes(title_text=stream_metric, tickformat="~s", rangemode="tozero", showgrid=False, zeroline=False,
                     secondary_y=True)
    return fig


def lag_chart(lag_df: pd.DataFrame, selected_lag: int | None, selected_corr: float | None) -> go.Figure:
    data = lag_df.dropna(subset=["correlation"]).copy()
    fig = go.Figure(go.Scatter(x=data["lag_weeks"], y=data["correlation"], mode="lines+markers+text",
                               line=dict(color=PURPLE, width=3), marker=dict(size=10, color=PURPLE),
                               text=data["correlation"].map(lambda x: f"{x:.2f}"), textposition="top center"))
    if selected_lag is not None and selected_corr is not None:
        fig.add_trace(go.Scatter(x=[selected_lag], y=[selected_corr], mode="markers",
                                 marker=dict(size=18, color=STREAM_COLOR, line=dict(color="white", width=3)), showlegend=False))
    labels = ["Same week" if x == 0 else f"+{x} week{'s' if x > 1 else ''}" for x in data["lag_weeks"]]
    fig.update_layout(title=dict(text="Lag correlation", x=0.01), height=410, margin=dict(l=35, r=25, t=65, b=70),
                      paper_bgcolor="white", plot_bgcolor="white", showlegend=False)
    fig.update_xaxes(title="Streams measured after creation week", tickmode="array", tickvals=data["lag_weeks"], ticktext=labels, showgrid=False)
    fig.update_yaxes(title="Pearson correlation (r)", range=[-1.05, 1.05], gridcolor=GRID, zeroline=True)
    return fig


def aligned_chart(df: pd.DataFrame, stream_metric: str, lag_weeks: int) -> go.Figure:
    aligned = df[[DATE_COLUMN, CREATIONS_COLUMN, stream_metric]].copy()
    aligned["aligned_streams"] = aligned[stream_metric].shift(-lag_weeks)
    aligned["creation_index"] = min_max_index(aligned[CREATIONS_COLUMN])
    aligned["stream_index"] = min_max_index(aligned["aligned_streams"])
    aligned = aligned.dropna(subset=["creation_index", "stream_index"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=aligned[DATE_COLUMN], y=aligned["creation_index"], name="Creations Index", mode="lines+markers",
                             line=dict(color=CREATION_COLOR, width=4), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=aligned[DATE_COLUMN], y=aligned["stream_index"], name=f"Streams Index (shifted {lag_weeks}W)", mode="lines+markers",
                             line=dict(color=STREAM_COLOR, width=4), marker=dict(size=8)))
    fig.update_layout(title=dict(text=f"Creation-to-stream alignment (best lag = {lag_weeks} week{'s' if lag_weeks != 1 else ''})", x=0.01),
                      height=410, margin=dict(l=35, r=25, t=65, b=70), paper_bgcolor="white", plot_bgcolor="white",
                      hovermode="x unified", legend=dict(orientation="h", y=-0.20, x=0.5, xanchor="center"))
    fig.update_xaxes(tickformat="%d %b", showgrid=False, showline=True, linecolor="#D0D0D5")
    fig.update_yaxes(title="Indexed value (0–100)", range=[0, 110], gridcolor=GRID, zeroline=False)
    return fig
