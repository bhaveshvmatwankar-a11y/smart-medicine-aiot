import sqlite3, re, random
from datetime import datetime
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

DB='medicine_aiot.db'

def init_db():
    c=sqlite3.connect(DB); cur=c.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS medicine_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        simulation_day INTEGER NOT NULL,
        medicine_set INTEGER NOT NULL,
        scheduled_time TEXT NOT NULL,
        status TEXT NOT NULL,
        response_delay INTEGER NOT NULL,
        adherence INTEGER NOT NULL,
        created_at TEXT NOT NULL)''')
    c.commit(); c.close()

def get_events():
    c=sqlite3.connect(DB)
    df=pd.read_sql_query('SELECT * FROM medicine_events ORDER BY simulation_day, medicine_set',c)
    c.close(); return df

def insert_events(events, day=None):
    c=sqlite3.connect(DB); cur=c.cursor()
    if day is None: day=cur.execute('SELECT COALESCE(MAX(simulation_day),0) FROM medicine_events').fetchone()[0]+1
    now=datetime.now().isoformat(timespec='seconds')
    for e in events:
        cur.execute('INSERT INTO medicine_events (simulation_day,medicine_set,scheduled_time,status,response_delay,adherence,created_at) VALUES (?,?,?,?,?,?,?)',
                    (day,e['set'],e['time'],e['status'],e['delay'],int(e['status']=='TAKEN'),now))
    c.commit(); c.close(); return day

def parse_tinkercad(text):
    return [{'set':int(a),'time':b,'status':c,'delay':int(d)} for a,b,c,d in re.findall(r'EVENT,(\d+),(\d{2}:\d{2}),(TAKEN|DELAYED|MISSED),(\d+)',text.upper())]

def generate_training_days(days=60,seed=42):
    random.seed(seed); rows=[]; times=['08:00','12:00','18:00','22:00']; bias={1:.12,2:.20,3:.30,4:.16}; prev={1:0,2:0,3:0,4:0}
    for day in range(1,days+1):
        for s in range(1,5):
            pm=bias[s]+(.12 if prev[s] else 0); pdly=.25+(.10 if prev[s] else 0); r=random.random()
            if r<pm: status,delay='MISSED',0
            elif r<pm+pdly: status,delay='DELAYED',random.randint(1,2)
            else: status,delay='TAKEN',0
            rows.append({'day':day,'set':s,'time':times[s-1],'status':status,'delay':delay}); prev[s]=int(status!='TAKEN')
    return pd.DataFrame(rows)

def make_features(df):
    x=df.copy(); x['hour']=x.time.str[:2].astype(int); x['prev_miss']=0; x['prev_delay']=0; x['rolling_7d_adherence']=0.; x['miss_count_7d']=0
    for s in range(1,5):
        idx=x.set==s; vals=x.loc[idx].sort_values('day'); prev=vals.status.shift(1)
        x.loc[vals.index,'prev_miss']=(prev=='MISSED').astype(int); x.loc[vals.index,'prev_delay']=(prev=='DELAYED').astype(int)
        ad=(vals.status=='TAKEN').astype(int).shift(1); x.loc[vals.index,'rolling_7d_adherence']=ad.rolling(7,min_periods=1).mean().fillna(0)
        ms=(vals.status=='MISSED').astype(int).shift(1); x.loc[vals.index,'miss_count_7d']=ms.rolling(7,min_periods=1).sum().fillna(0)
    return x

FEATURES=['set','hour','prev_miss','prev_delay','rolling_7d_adherence','miss_count_7d']
@st.cache_resource
def train_model():
    d=generate_training_days(80); f=make_features(d); y=(f.status!='TAKEN').astype(int)
    m=RandomForestClassifier(n_estimators=150,max_depth=7,random_state=42); m.fit(f[FEATURES],y); return m

def risk(history,model,s):
    h=history[history.medicine_set==s].sort_values('simulation_day'); last=h.iloc[-1] if not h.empty else None; recent=h.tail(7)
    pm=int(last.status=='MISSED') if last is not None else 0; pdly=int(last.status=='DELAYED') if last is not None else 0
    ad=float(recent.adherence.mean()) if len(recent) else 1.; miss=int((recent.status=='MISSED').sum())
    hour=[8,12,18,22][s-1]
    row=pd.DataFrame([{'set':s,'hour':hour,'prev_miss':pm,'prev_delay':pdly,'rolling_7d_adherence':ad,'miss_count_7d':miss}])
    return float(model.predict_proba(row[FEATURES])[0][1])

st.set_page_config(page_title='Smart Medicine Box AIoT',layout='wide'); init_db()
st.title('💊 Smart Medicine Box — AIoT Dashboard')
st.caption('Tinkercad simulation → SQLite → Random Forest → risk dashboard')

with st.sidebar:
    st.header('Demo Controls')
    if st.button('Generate 30-Day Demo Dataset',use_container_width=True):
        c=sqlite3.connect(DB); c.execute('DELETE FROM medicine_events'); c.commit(); c.close()
        d=generate_training_days(30)
        for day in range(1,31):
            q=d[d.day==day]; insert_events([{'set':int(r['set']),'time':r['time'],'status':r['status'],'delay':int(r['delay'])} for _,r in q.iterrows()],day)
        st.rerun()
    st.subheader('Import Actual Tinkercad Run')
    raw=st.text_area('Paste complete Serial Monitor output',height=170)
    if st.button('Save Tinkercad Run to SQLite',use_container_width=True):
        ev=parse_tinkercad(raw)
        if not ev: st.error('No EVENT records found.')
        elif 'DAY_COMPLETE' not in raw.upper(): st.warning('DAY_COMPLETE is missing.')
        else: insert_events(ev); st.success(f'Saved {len(ev)} events.'); st.rerun()
    if st.button('Clear Database',use_container_width=True):
        c=sqlite3.connect(DB); c.execute('DELETE FROM medicine_events'); c.commit(); c.close(); st.rerun()

events=get_events()
if events.empty:
    st.info('Generate the 30-day demo dataset or import a Tinkercad run from the sidebar.'); st.stop()

total=len(events); taken=int((events.status=='TAKEN').sum()); delayed=int((events.status=='DELAYED').sum()); missed=int((events.status=='MISSED').sum())
c1,c2,c3,c4,c5=st.columns(5); c1.metric('Doses',total); c2.metric('Taken',taken); c3.metric('Delayed',delayed); c4.metric('Missed',missed); c5.metric('On-time adherence',f'{100*taken/total:.1f}%')

st.subheader('Adherence Trend')
daily=events.groupby('simulation_day').agg(doses=('id','count'),taken=('adherence','sum')).reset_index(); daily['adherence_pct']=100*daily.taken/daily.doses
st.line_chart(daily.set_index('simulation_day')['adherence_pct'])

col1,col2=st.columns(2)
with col1:
    st.subheader('Status by Medicine Set'); st.bar_chart(pd.crosstab(events.medicine_set,events.status))
with col2:
    st.subheader('Average Response Delay'); st.bar_chart(events.groupby('medicine_set').response_delay.mean())

st.subheader('🤖 AI Risk Prediction')
st.write('Random Forest predicts the risk that the next dose will be delayed or missed. This prototype is a demonstration, not a medical diagnostic system.')
model=train_model(); rows=[]
for s in range(1,5):
    p=risk(events,model,s); level='LOW' if p<.35 else ('MEDIUM' if p<.65 else 'HIGH')
    rows.append({'Medicine Set':s,'Next Reminder':['08:00','12:00','18:00','22:00'][s-1],'Predicted Non-Adherence Risk':f'{100*p:.1f}%','Risk Level':level})
riskdf=pd.DataFrame(rows); st.dataframe(riskdf,use_container_width=True,hide_index=True)
if (riskdf['Risk Level']=='HIGH').any(): st.warning('AI flag: elevated predicted non-adherence risk in at least one reminder slot.')
else: st.success('No high-risk reminder slots in this simulation.')

st.subheader('SQLite Event Data'); st.dataframe(events.tail(40),use_container_width=True,hide_index=True)
st.caption('Training data is synthetic/simulated for exhibition demonstration. Real anonymized data would be needed for clinical validation.')
