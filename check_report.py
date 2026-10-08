from pathlib import Path
import pandas as pd
import numpy as np
from report_core import normalize, merge_sources, regularize, transform, lag_table, event_response, static_image, SOCIAL
base = Path(__file__).parent/'sample_data'
mapping = {'Artist':'Artist','Track':'Track','Region':'Region','Date':'period_end'}
social = normalize(pd.read_csv(base/'honeybee_Creations.csv'),mapping,{c:c for c in SOCIAL})
streams = normalize(pd.read_csv(base/'honeybee_Streams.csv'),mapping,{'Streams':'Spotify Streams'})
data = regularize(merge_sources(social,streams),7)
cols = SOCIAL+['Streams']
chart = transform(data,cols,'Baseline index (100)',data.iloc[-4:],1)
for c in cols:
    assert np.isclose(chart[c].iloc[-4:].mean(),100)
colors = dict(zip(cols,['#C13584','#E879B5','#009C9A','#55BDBA','#6741D9']))
for fmt in ['png','svg']:
    payload = static_image(chart,cols,colors,'Honeybee | Social & streaming','Original bundled example','Baseline index (100)',1600,800,None,False,False,fmt)
    assert len(payload)>1000
    if fmt=='png': (Path(__file__).parent/'preview.png').write_bytes(payload)
# Missing periods must not become adjacent lag observations.
gap = pd.DataFrame({'Date':pd.to_datetime(['2026-01-01','2026-01-03']), 'Streams':[10,30], 'TikTok Views':[1,3]})
filled = regularize(gap,1)
assert len(filled)==3 and pd.isna(filled.Streams.iloc[1])
# A known 2-day delayed response has r=1 at lag 2.
rng = np.random.default_rng(42)
x = rng.normal(size=30)
lagdata = pd.DataFrame({'TikTok Views':x,'Streams':np.r_[np.nan,np.nan,x[:-2]]})
result = lag_table(lagdata,['TikTok Views'],4,False)
assert np.isclose(result.loc[result['Lag (periods)'].eq(2),'Pearson r'].iloc[0],1)
# Cumulative counters: first unavailable, following increments exact.
raw = pd.DataFrame({'date':['2026-01-01','2026-01-02'],'count':[10,25]})
n = normalize(raw,{'Date':'date'},{'Streams':'count'},True)
assert pd.isna(n.Streams.iloc[0]) and n.Streams.iloc[1]==15
try:
    normalize(pd.concat([raw,raw]),{'Date':'date'},{'Streams':'count'})
    raise AssertionError('Duplicates accepted')
except ValueError: pass
response = event_response(filled,'2026-01-02',1,2,1)
assert response['Change (%)'].isna().all()
print('PASS: original data, baseline index, PNG/SVG, missing periods, lag direction, cumulative increments, duplicate rejection, incomplete event windows')
