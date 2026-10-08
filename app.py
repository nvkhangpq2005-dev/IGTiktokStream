from pathlib import Path
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from utils import read_uploaded_csv
from report_core import SOCIAL, normalize, merge_sources, regularize, transform, lag_table, event_response, static_image
st.set_page_config(page_title='IG / TikTok → Streaming Report', page_icon='🎵', layout='wide')
st.title('IG / TikTok → Streaming Report')
st.caption('Customize charts to compare social activity and subsequent streaming response.')

DEFAULT_COLORS = {'Instagram Creations':'#C13584','Instagram Views':'#E879B5','TikTok Creations':'#009C9A','TikTok Views':'#55BDBA','Streams':'#6741D9'}
def picker(label, columns, defaults, key, required=False):
    options = columns if required else ['—'] + columns
    default = next((c for c in defaults if c in columns), options[0])
    result = st.selectbox(label, options, index=options.index(default), key=key)
    return None if result == '—' else result

def import_source(raw, kind):
    with st.expander(f'{kind}: column mapping', expanded=False):
        columns = list(raw.columns)
        mapping = {k: picker(k,columns,defaults,f'{kind}_{k}',k=='Date') for k,defaults in {'Date':['period_end','Date','date'],'Artist':['Artist','artist'],'Track':['Track','track','Song','song'],'Region':['Region','Country','country']}.items()}
        st.caption('Missing identity fields become “Unspecified”. Both files must use matching artist, track and market names. Map stable IDs to Artist/Track if names differ.')
        if kind == 'Social':
            metrics = {m:picker(m,columns,[m,m.replace('Instagram','IG')],f'social_{m}') for m in SOCIAL}
        else:
            col = picker('Stream metric',columns,['All Partner Streams (Non-UGC)','Spotify Streams','Streams'], 'stream_metric',True)
            metrics = {'Streams': col}
        cumulative = st.checkbox('Counts are cumulative (convert to period increments)', key=f'{kind}_cumulative')
        if cumulative:
            st.caption('First observation per track is unavailable; counter resets are rejected. Supply cumulative totals consistently for every selected metric.')
    return normalize(raw,mapping,metrics,cumulative), metrics

with st.sidebar:
    st.header('1 · Data')
    social_files = st.file_uploader('Social CSVs · IG + TikTok', type='csv', accept_multiple_files=True)
    stream_files = st.file_uploader('Streams CSVs',type='csv',accept_multiple_files=True)
    demo = st.checkbox('Use bundled Honeybee example', value=not(social_files or stream_files))
    cadence = st.selectbox('Source reporting cadence',['Weekly','Daily'])
    st.caption('Use matching cadence and period end dates for both sources. No automatic weekly/daily conversion.')
    for name in ['honeybee_Creations.csv','honeybee_Streams.csv']:
        st.download_button(f'Download {name}',(Path(__file__).parent/'sample_data'/name).read_bytes(),name)
if not demo and (not social_files or not stream_files):
    st.info('Upload social and streams CSVs. Multiple files are supported when their schemas match.'); st.stop()
try:
    if demo:
        base = Path(__file__).parent/'sample_data'
        social_raw = pd.read_csv(base/'honeybee_Creations.csv')
        stream_raw = pd.read_csv(base/'honeybee_Streams.csv')
    else:
        social_raw = pd.concat([read_uploaded_csv(f) for f in social_files],ignore_index=True)
        stream_raw = pd.concat([read_uploaded_csv(f) for f in stream_files],ignore_index=True)
    social, social_mapping = import_source(social_raw,'Social')
    streams, stream_mapping = import_source(stream_raw,'Streams')
    merged = merge_sources(social,streams)
except Exception as exc:
    st.error(str(exc)); st.stop()
with st.sidebar:
    st.header('2 · Track & market')
    artist = st.selectbox('Artist', sorted(merged.Artist.unique()))
    track = st.selectbox('Track', sorted(merged.loc[merged.Artist.eq(artist),'Track'].unique()))
    entity = merged[merged.Artist.eq(artist)&merged.Track.eq(track)]
    region = st.selectbox('Region', sorted(entity.Region.unique()))
entity = entity[entity.Region.eq(region)].copy()
days = 7 if cadence == 'Weekly' else 1
try:
    entity = regularize(entity,days)
except ValueError as exc:
    st.error(str(exc)); st.stop()
metrics = [c for c in SOCIAL+['Streams'] if c in entity and entity[c].notna().any()]
if not metrics:
    st.error('No numeric observations for this selection.'); st.stop()
lo, hi = entity.Date.min().date(),entity.Date.max().date()
with st.sidebar:
    dates = st.date_input('Chart date range',(lo,hi),min_value=lo,max_value=hi)
    if len(dates)!=2: st.stop()
    st.header('3 · Chart settings')
    selected = st.multiselect('Lines to display',metrics, default=[c for c in ['Instagram Creations','TikTok Creations','Streams'] if c in metrics])
    mode = st.selectbox('Scale',['Baseline index (100)','Raw counts','Change vs baseline (%)'])
    baseline_dates = st.date_input('Baseline date range',(max(lo,(pd.Timestamp(hi)-pd.Timedelta(days=3*days)).date()),hi),min_value=lo,max_value=hi)
    if len(baseline_dates)!=2: st.stop()
    rolling = st.slider('Trailing mean · periods',1,8,1)
    labels = st.checkbox('Point labels',False)
    title = st.text_input('Chart title',f'{track} | Social activity & streaming')
    subtitle = st.text_input('Subtitle',f'{artist} · {region} · {cadence} · {stream_mapping["Streams"]}')
    width = st.number_input('Export width (px)',800,3200,1600,100)
    height = st.number_input('Export height (px)',400,1800,800,100)
    colors = {m:st.color_picker(m,DEFAULT_COLORS[m]) for m in selected}
    st.header('4 · Event & lag')
    use_event = st.checkbox('Show event / campaign marker')
    event = st.date_input('Event period end',lo,min_value=lo,max_value=hi) if use_event else None
    event_label = st.text_input('Event label','Campaign / social spike') if use_event else ''
    max_lag = st.slider('Maximum lag · periods',0,14,4)
    changes = st.checkbox('Correlate period changes',True,help='First differences reduce shared level trends. Positive lag pairs social now with streams later.')
filtered = entity[entity.Date.between(pd.Timestamp(dates[0]),pd.Timestamp(dates[1]))].copy()
baseline = entity[entity.Date.between(pd.Timestamp(baseline_dates[0]),pd.Timestamp(baseline_dates[1]))]
if not selected or filtered.empty:
    st.info('Select at least one metric and a date range with observations.'); st.stop()
if mode != 'Raw counts':
    bad = [c for c in selected if pd.isna(baseline[c].mean()) or baseline[c].mean()==0]
    if bad:
        st.error('Baseline has no positive mean for: '+', '.join(bad)); st.stop()
missing = filtered[metrics].isna().sum()
if missing.any():
    st.warning('Missing observations stay as gaps: '+', '.join(f'{c}: {n}' for c,n in missing.items() if n))
if demo: st.info('Bundled example from the uploaded dashboard. Replace with your study data before reporting.')
chart = transform(filtered,selected,mode,baseline,rolling)
raw = mode == 'Raw counts'
ylabel = 'Count' if raw else mode
if raw and any('Views' in c for c in selected) and any('Creations' in c for c in selected):
    st.info('Views and creations share the social axis in raw mode. Select one social metric type or use baseline index for clearer comparison.')
fig = make_subplots(specs=[[{'secondary_y':True}]])
for col in selected:
    fig.add_trace(go.Scatter(x=chart.Date,y=chart[col],name=col,line={'color':colors[col],'width':3},connectgaps=False,mode='lines+markers+text' if labels else 'lines',text=[f'{v:,.0f}' if pd.notna(v) else '' for v in chart[col]] if labels else None,textposition='top center'),secondary_y=raw and col=='Streams' and len(selected)>1)
fig.update_layout(title={'text':title},template='plotly_white',height=height,legend={'orientation':'h','y':-0.22},margin={'t':80,'b':100},hovermode='x unified')
fig.update_yaxes(title_text=ylabel,secondary_y=False)
fig.update_yaxes(title_text='Streams' if raw else '',visible=raw and 'Streams' in selected and len(selected)>1,secondary_y=True)
if event:
    fig.add_vline(x=pd.Timestamp(event).timestamp()*1000,line_dash='dash',line_color='#777777')
    fig.add_annotation(x=pd.Timestamp(event),y=1,yref='paper',text=event_label,showarrow=False)
st.caption(subtitle)
st.plotly_chart(fig,use_container_width=True,config={'displaylogo':False,'toImageButtonOptions':{'format':'png','filename':'social_streaming_report','width':width,'height':height,'scale':2}})
st.caption(f'Baseline: {baseline_dates[0]}–{baseline_dates[1]} · trailing mean: {rolling} {cadence.lower()} periods. Missing dates are preserved. Event marker refers to a reporting period end.')

st.subheader('Export for report')
for fmt in ['png','svg']:
    payload = static_image(chart,selected,colors,title,subtitle,ylabel,width,height,event,labels,raw,fmt)
    st.download_button(f'Download {fmt.upper()}',payload,f'social_streaming_report.{fmt}', 'image/png' if fmt=='png' else 'image/svg+xml')
st.download_button('Download interactive HTML',fig.to_html(include_plotlyjs=True).encode(),'social_streaming_report.html','text/html')
st.download_button('Download chart data',chart.to_csv(index=False).encode('utf-8-sig'),'chart_data.csv','text/csv')
settings = {'artist':artist,'track':track,'region':region,'source_cadence':cadence,'dates':[str(x) for x in dates],'baseline':[str(x) for x in baseline_dates],'metrics':selected,'scale':mode,'rolling_periods':rolling,'colors':colors,'title':title,'subtitle':subtitle,'width':width,'height':height,'event':str(event) if event else None,'event_label':event_label,'max_lag':max_lag,'first_differences':changes,'social_mapping':social_mapping,'stream_mapping':stream_mapping}
st.download_button('Download report settings',json.dumps(settings,indent=2).encode(),'report_settings.json','application/json')

st.subheader('IG vs TikTok · lag comparison')
st.caption(f'Positive lag = social in this period vs streams later. Unit: {cadence.lower()} periods. Uses unsmoothed source observations within the chart date range; minimum 6 paired observations. Correlation is descriptive and does not establish conversion or platform attribution.')
social_cols = [c for c in selected if c in SOCIAL]
if 'Streams' in metrics and social_cols:
    lag = lag_table(filtered,social_cols,max_lag,changes)
    lagfig = go.Figure()
    for col in social_cols:
        part = lag[lag.Metric.eq(col)]
        lagfig.add_trace(go.Scatter(x=part['Lag (periods)'],y=part['Pearson r'],mode='lines+markers',name=col,line={'color':colors[col]}))
    lagfig.update_layout(template='plotly_white',yaxis={'title':'Pearson r','range':[-1,1]},xaxis_title=f'Lag ({cadence.lower()} periods)')
    st.plotly_chart(lagfig,use_container_width=True)
    st.dataframe(lag,hide_index=True)
    st.download_button('Download lag table',lag.to_csv(index=False).encode(),'lag_comparison.csv','text/csv')
else: st.info('Map streams and select at least one social metric to compare lags.')
if event:
    st.subheader('Before / after event')
    a,b = st.columns(2)
    pre = a.number_input('Baseline periods before event',1,26,4)
    post = b.number_input('Response periods from event',1,26,2)
    response = event_response(entity,event,pre,post,days)
    st.caption('Computed on unsmoothed full selected-track data. Percent change is withheld for incomplete windows or zero baseline. A flat pre-event mean is a descriptive benchmark; changes can also reflect campaigns, playlists or other events.')
    st.dataframe(response,hide_index=True)
    st.download_button('Download event comparison',response.to_csv(index=False).encode(),'event_response.csv','text/csv')
with st.expander('Source observations & baseline audit'):
    st.dataframe(filtered,hide_index=True)
    st.dataframe(baseline[metrics].agg(['count','mean']).T)
    st.download_button('Download source observations',filtered.to_csv(index=False).encode(),'source_observations.csv','text/csv')
