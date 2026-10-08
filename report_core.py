from __future__ import annotations
import io
import numpy as np
import pandas as pd
from utils import clean_numeric
KEYS = ['Artist', 'Track', 'Region', 'Date']
SOCIAL = ['Instagram Creations', 'Instagram Views', 'TikTok Creations', 'TikTok Views']

def normalize(raw, mapping, metrics, cumulative=False):
    out = pd.DataFrame(index=raw.index)
    for key in KEYS:
        col = mapping.get(key)
        out[key] = raw[col].astype(str).str.strip() if col else ('Unspecified' if key != 'Date' else None)
    out['Date'] = pd.to_datetime(out['Date'], errors='coerce').dt.normalize()
    if out['Date'].isna().any():
        raise ValueError('Invalid/missing dates. Fix the source file before importing.')
    for target, source in metrics.items():
        if source:
            out[target] = clean_numeric(raw[source])
            invalid = raw[source].astype(str).str.strip().isin(['', 'nan', 'None', '-'])
            if (out[target].isna() & ~invalid).any():
                raise ValueError(f'{source}: non-numeric values found.')
            if out[target].dropna().lt(0).any():
                raise ValueError(f'{source}: negative counts found.')
    if out.duplicated(KEYS).any():
        raise ValueError('Duplicate artist/track/region/date rows. Aggregate or remove duplicates before importing.')
    out = out.sort_values(KEYS)
    if cumulative:
        cols = list(metrics)
        cols = [c for c in cols if c in out]
        out[cols] = out.groupby(KEYS[:3], dropna=False)[cols].diff()
        if out[cols].lt(0).any().any():
            raise ValueError('Cumulative counter decreases: check resets before importing.')
    return out

def merge_sources(social, streams):
    return social.merge(streams, on=KEYS, how='outer', validate='one_to_one').sort_values('Date')

def regularize(df, days):
    work = df.set_index('Date').sort_index()
    if len(work) > 1 and ((work.index - work.index[0]).days % days != 0).any():
        raise ValueError('Dates are not aligned to the selected daily/weekly cadence. Align reporting periods first.')
    return work.reindex(pd.date_range(work.index.min(), work.index.max(), freq=f'{days}D')).rename_axis('Date').reset_index()

def transform(df, cols, mode, baseline, rolling):
    out = df[['Date'] + cols].copy()
    means = baseline[cols].mean()
    if mode == 'Baseline index (100)':
        out[cols] = out[cols].div(means.replace(0, np.nan)) * 100
    elif mode == 'Change vs baseline (%)':
        out[cols] = (out[cols].div(means.replace(0, np.nan)) - 1) * 100
    if rolling > 1:
        out[cols] = out[cols].rolling(rolling, min_periods=rolling).mean()
    return out

def lag_table(df, cols, max_lag, change):
    values = df[cols + ['Streams']].copy()
    if change:
        values = values.diff()
    rows = []
    for col in cols:
        for lag in range(max_lag + 1):
            pair = pd.concat([values[col], values.Streams.shift(-lag)], axis=1).dropna()
            valid = len(pair) >= 6 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1
            rows.append({'Metric': col, 'Lag (periods)': lag, 'Pearson r': pair.iloc[:, 0].corr(pair.iloc[:, 1]) if valid else np.nan, 'Paired observations': len(pair)})
    return pd.DataFrame(rows)

def event_response(df, event, pre, post, days):
    event = pd.Timestamp(event)
    before = df[df.Date.between(event - pd.Timedelta(days=pre*days), event - pd.Timedelta(days=days))]
    after = df[df.Date.between(event, event + pd.Timedelta(days=(post-1)*days))]
    rows = []
    for col in [c for c in df if c not in KEYS]:
        b, a = before[col].mean(), after[col].mean()
        complete = before[col].count() == pre and after[col].count() == post
        rows.append({'Metric': col, 'Pre mean': b, 'Post mean': a, 'Change (%)': (a/b-1)*100 if complete and b > 0 else np.nan, 'Pre observations': before[col].count(), 'Post observations': after[col].count(), 'Complete window': complete})
    return pd.DataFrame(rows)

def static_image(df, cols, colors, title, subtitle, ylabel, width, height, event, labels, raw, fmt):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    fig, ax = plt.subplots(figsize=(width/150, height/150), dpi=150)
    right = ax.twinx() if raw and 'Streams' in cols and len(cols) > 1 else ax
    for col in cols:
        target = right if col == 'Streams' else ax
        target.plot(df.Date, df[col], label=col, color=colors[col], linewidth=2.2, marker='o' if labels else None, markersize=3)
        if labels:
            for date, value in zip(df.Date, df[col]):
                if pd.notna(value):
                    target.annotate(f'{value:,.0f}', (date, value), xytext=(0,5), textcoords='offset points', fontsize=7, color=colors[col])
    if event:
        ax.axvline(pd.Timestamp(event), color='#777777', linestyle='--', linewidth=1)
    ax.set_ylabel(ylabel)
    if right is not ax:
        right.set_ylabel('Streams')
        right.yaxis.set_major_formatter(ticker.EngFormatter())
        if len([c for c in cols if c != 'Streams']) > 1:
            ax.set_ylabel('Social counts (views / creations)')
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    ax.grid(axis='y', alpha=.18)
    ax.spines[['top']].set_visible(False)
    handles, names = ax.get_legend_handles_labels()
    if right is not ax:
        h, n = right.get_legend_handles_labels(); handles += h; names += n
    fig.legend(handles, names, loc='lower center', ncol=min(3,len(cols)), frameon=False, fontsize=8)
    fig.suptitle(title, x=.08, ha='left', fontsize=15, weight='bold')
    ax.set_title(subtitle, loc='left', fontsize=9, color='#666666', pad=14)
    fig.autofmt_xdate()
    fig.tight_layout(rect=(0,.08,1,.92))
    buffer = io.BytesIO(); fig.savefig(buffer, format=fmt, facecolor='white'); plt.close(fig)
    return buffer.getvalue()
