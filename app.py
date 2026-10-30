from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from core import CANDIDATES, MONTHS, score_batch

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='TowerScope | Site status',page_icon='📡',layout='wide')
st.markdown('''<style>
.stApp {background:linear-gradient(135deg,#f5f9ff 0%,#eaf2ff 48%,#ffffff 100%);color:#173255;}
[data-testid="stSidebar"] {background:#f0f6ff;border-right:1px solid #d6e5fa;}
[data-testid="stMetric"] {background:white;border:1px solid #dce7f5;border-radius:16px;padding:20px;box-shadow:0 5px 18px #194b8010;}
[data-testid="stMetricLabel"] {color:#486484;}
[data-testid="stMetricValue"] {color:#154fb1;}
.block-container {padding-top:2.5rem;padding-bottom:3rem;max-width:1440px;}
h1,h2,h3 {color:#153b70;letter-spacing:-.025em;}
.hero {background:linear-gradient(110deg,#103b7c,#2165bc);padding:32px;border-radius:22px;color:white;margin-bottom:24px;}
.hero h1 {color:white;margin:0;font-size:2.3rem;}
.hero p {color:#dceaff;margin:10px 0 0;}
.eyebrow {font-size:.78rem;letter-spacing:.18em;font-weight:700;color:#bed9ff;}
.stButton>button {border-radius:10px;}
</style>''',unsafe_allow_html=True)
px.defaults.template='plotly_white'
px.defaults.color_discrete_sequence=['#2361bd','#36a4ce','#91baf2','#164174','#a6dce9']

@st.cache_data
def read_csv(path):
    return pd.read_csv(path)

@st.cache_data
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

@st.cache_resource
def load_pipeline(path):
    return joblib.load(path)

def get_json(filename):
    p=ROOT/'results'/filename
    try:return read_json(str(p))
    except (OSError,ValueError) as e:
        st.error(f'Cannot read results/{filename}. Restore it from the ZIP or rerun the corrected notebook. {e}')
        st.stop()

def table_file(relative):
    try:return read_csv(str(ROOT/relative))
    except (OSError,ValueError,pd.errors.ParserError) as e:
        st.error(f'Cannot read {relative}. Restore it from the ZIP or rerun the corrected notebook. {e}')
        st.stop()

st.sidebar.markdown('## 📡 TowerScope')
st.sidebar.caption('SYNTHETIC TELECOM ANALYSIS')
page=st.sidebar.radio('Navigation',['Overview','Dataset explorer','Model comparison','Model reports','ANN learning curves','Site status scoring'])
st.sidebar.divider()
st.sidebar.info('Retrospective site status only. This dataset does not establish future failure risk.')
meta=get_json('metadata.json')
st.markdown('<div class="hero"><div class="eyebrow">TOWERSCOPE / ANALYSIS STUDIO</div><h1>Understand recorded site status.</h1><p>Verified notebook outputs · Synthetic telecom data · Transparent evaluation</p></div>',unsafe_allow_html=True)
st.caption('12 folds · Towers kept separate between train and test · Original model configurations')

if page=='Overview':
    st.subheader('A clear view of the supplied dataset')
    a,b,c,d=st.columns(4)
    a.metric('Records',f"{meta['rows']:,}")
    b.metric('Unique towers',f"{meta['towers']:,}")
    c.metric('Recorded down',f"{meta['class_counts']['1']/meta['rows']:.1%}")
    d.metric('Missing cells',meta['missing_cells'])
    st.write('These are dataset descriptors. Model results below come from out-of-fold predictions for all records.')
    metrics=table_file('results/metrics.csv')
    fig=px.bar(metrics,x='model',y='accuracy',color='model',text_auto='.1%',labels={'accuracy':'OOF accuracy','model':'Model'})
    fig.add_hline(y=meta['class_counts']['0']/meta['rows'],line_dash='dot',annotation_text='Always predict class 0 reference')
    fig.update_layout(showlegend=False,yaxis_range=[0,1],height=410)
    st.plotly_chart(fig,width='stretch')
    st.info('Accuracy must be read alongside site-down recall and precision. A constrained model may predict mostly the majority class.')
    st.markdown('**Scope:** repair duration and technician assignment are observed inputs. Scores describe recorded status and must not be interpreted as advance warnings.')

elif page=='Dataset explorer':
    df=table_file('data/records.csv.gz')
    st.subheader('Explore the records')
    a,b=st.columns(2)
    regions=a.multiselect('Regions',sorted(df.Region.unique()),default=sorted(df.Region.unique()))
    months=b.multiselect('Months',[m for m in MONTHS if m in set(df.Month)],default=MONTHS)
    filtered=df[df.Region.isin(regions)&df.Month.isin(months)]
    st.caption('Filters affect this page only. Saved model metrics always cover the complete OOF evaluation.')
    if filtered.empty:
        st.info('No rows match these filters. Select at least one region and month.');st.stop()
    a,b,c=st.columns(3)
    a.metric('Filtered rows',f'{len(filtered):,}')
    b.metric('Recorded down',f'{filtered.is_site_down.mean():.1%}')
    c.metric('Mean repair_time',f'{filtered.repair_time.mean():.2f}')
    st.caption('repair_time units were not documented in the source notebook.')
    summary=filtered.groupby('Region',as_index=False).agg(records=('is_site_down','size'),down_rate=('is_site_down','mean'))
    fig=px.bar(summary,x='Region',y='down_rate',color='Region',text_auto='.1%',hover_data=['records'])
    fig.update_layout(showlegend=False,yaxis_tickformat='.0%',yaxis_range=[0,1])
    st.plotly_chart(fig,width='stretch')
    st.dataframe(filtered.head(500),hide_index=True,width='stretch')
    st.caption('Preview limited to 500 rows; download contains every filtered row.')
    st.download_button('Download filtered records',filtered.to_csv(index=False).encode('utf-8'),'filtered_records.csv','text/csv')

elif page=='Model comparison':
    st.subheader('All existing model configurations')
    st.caption('Global OOF results. No dataset filters apply. Class 1 = recorded site-down.')
    metrics=table_file('results/metrics.csv')
    measure=st.selectbox('Compare metric',['accuracy','precision','recall','f1'])
    fig=px.bar(metrics,x='model',y=measure,color='model',text_auto='.1%')
    fig.update_layout(showlegend=False,yaxis_range=[0,1],height=440)
    st.plotly_chart(fig,width='stretch')
    st.dataframe(metrics.style.format({c:'{:.4f}' for c in metrics if c!='model'}),hide_index=True,width='stretch')
    st.caption('accuracy = pooled OOF accuracy; mean_fold_accuracy = unweighted mean of 12 folds. std_fold_accuracy is descriptive, not a confidence interval.')
    folds=table_file('results/fold_metrics.csv')
    st.plotly_chart(px.line(folds,x='fold',y='accuracy',color='model',markers=True,range_y=[0,1]),width='stretch')

elif page=='Model reports':
    st.subheader('Reports built from every held-out row')
    reports=get_json('reports.json');confusions=get_json('confusions.json')
    if not reports:st.warning('No model reports are available. Rerun the notebook.');st.stop()
    name=st.selectbox('Model report',list(reports))
    report=reports.get(name);cm=confusions.get(name)
    if not isinstance(report,dict) or cm is None:
        st.warning('This report or its confusion matrix is missing. Restore results or rerun the notebook.');st.stop()
    a,b=st.columns([1,1])
    with a:
        fig=px.imshow(cm,x=['Predicted 0','Predicted 1'],y=['Actual 0','Actual 1'],text_auto=True,color_continuous_scale='Blues',aspect='auto')
        fig.update_layout(coloraxis_showscale=False,height=360)
        st.plotly_chart(fig,width='stretch')
    with b:
        rows={k:v for k,v in report.items() if isinstance(v,dict)}
        st.dataframe(pd.DataFrame(rows).T,width='stretch')
        st.metric('Pooled OOF accuracy',f"{report['accuracy']:.2%}")
    st.caption('Rows are actual labels; columns are predictions. Every record contributes once, using a model that did not train on its tower.')

elif page=='ANN learning curves':
    st.subheader('Training diagnostics across 12 folds')
    histories=get_json('ann_histories.json')
    mode=st.selectbox('Curve scope',['Mean of all folds']+[f'Fold {h["fold"]}' for h in histories])
    if mode=='Mean of all folds':
        train=np.mean([h['loss'] for h in histories],axis=0)
        val=np.mean([h['val_loss'] for h in histories],axis=0)
    else:
        h=next(h for h in histories if f'Fold {h["fold"]}'==mode)
        train=h['loss'];val=h['val_loss']
    fig=go.Figure()
    fig.add_scatter(x=list(range(1,6)),y=train,name='Training loss',mode='lines+markers',line_color='#2563eb')
    fig.add_scatter(x=list(range(1,6)),y=val,name='Held-out fold loss',mode='lines+markers',line_color='#0891b2')
    fig.update_layout(template='plotly_white',xaxis_title='Epoch',yaxis_title='Binary cross-entropy',height=430)
    st.plotly_chart(fig,width='stretch')
    st.info('Five fixed epochs. Held-out loss is a diagnostic only; no early stopping or tuning used these curves. This is not a single model trained across all folds.')

elif page=='Site status scoring':
    st.subheader('Score an observed record')
    st.warning('Retrospective use only: requires observed repair_time and assigned technician. The score is uncalibrated and is not a future failure probability.')
    st.caption('Uses the original balanced 200-tree RF configuration, refitted on all data. Score bands 0.4/0.7 are illustrative, not validated intervention thresholds.')
    try:pipeline=load_pipeline(str(ROOT/'results'/'site_status_pipeline.joblib'))
    except Exception as e:
        st.error(f'Cannot load the model. Restore the packaged model and install requirements.txt. {e}');st.stop()
    mode=st.radio('Input mode',['Single record','Batch CSV'],horizontal=True)
    if mode=='Single record':
        combos=pd.DataFrame(meta['region_depot_tech'],columns=['Region','Depot','techs'])
        a,b,c=st.columns(3)
        region=a.selectbox('Region',sorted(combos.Region.unique()))
        subset=combos[combos.Region==region]
        depot=b.selectbox('Depot',sorted(subset.Depot.unique()))
        tech=c.selectbox('Technician ID',sorted(subset.loc[subset.Depot==depot,'techs'],key=int))
        a,b,c=st.columns(3)
        month=a.selectbox('Month',MONTHS)
        import calendar
        day=b.number_input('Day of month',min_value=1,max_value=calendar.monthrange(2023,MONTHS.index(month)+1)[1],value=1)
        lo,hi=meta['repair_time_range']
        repair=c.number_input('Observed repair_time',min_value=float(lo),max_value=float(hi),value=3.0,format='%.4f')
        a,b=st.columns(2)
        network=a.selectbox('Network type',meta['categories']['Network_type'])
        vendor=b.selectbox('Equipment vendor',meta['categories']['equipment_vendor'])
        st.caption('All original candidate inputs are accepted; the saved selector retains only training-selected features: '+', '.join(meta['final_selected']))
        if st.button('Score recorded status',type='primary'):
            record=pd.DataFrame([{'Region':region,'Depot':depot,'techs':tech,'Month':month,'Day':day,'repair_time':repair,'Network_type':network,'equipment_vendor':vendor}])
            try:
                out=score_batch(pipeline,record,meta).iloc[0]
                a,b,c=st.columns(3)
                a.metric('Uncalibrated site-down score',f"{out.site_down_score:.1%}")
                b.metric('Predicted recorded status',str(out.predicted_status))
                c.metric('Illustrative score band',out.score_band)
            except ValueError as e:st.error(str(e))
    else:
        st.write('Use raw names and IDs, not ordinal codes. Required columns:')
        st.code(', '.join(CANDIDATES))
        sample=ROOT/'data'/'example_batch.csv'
        if sample.exists():st.download_button('Download example CSV',sample.read_bytes(),'example_batch.csv','text/csv')
        upload=st.file_uploader('Upload records (CSV, UTF-8)',type=['csv'])
        if upload is not None:
            try:
                incoming=pd.read_csv(upload,encoding='utf-8')
                st.dataframe(incoming.head(10),hide_index=True)
                if st.button('Generate status report',type='primary'):
                    result=score_batch(pipeline,incoming,meta)
                    st.dataframe(result.head(100),hide_index=True)
                    st.download_button('Download scored records',result.to_csv(index=False).encode('utf-8'),'scored_records.csv','text/csv')
            except (ValueError,UnicodeError,pd.errors.ParserError) as e:st.error(f'Please correct the CSV: {e}')

st.divider()
st.caption('TowerScope · Local research dashboard · Synthetic data · Recorded status, not future failure forecasting')
