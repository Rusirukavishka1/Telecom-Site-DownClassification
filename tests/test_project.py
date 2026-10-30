"""End-to-end artifact, input-validation and Streamlit interaction checks."""
from pathlib import Path
import sys,json,tempfile,shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score
from streamlit.testing.v1 import AppTest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core import normalize,score_batch
meta=json.loads((ROOT/'results/metadata.json').read_text(encoding='utf-8'))
pipeline=joblib.load(ROOT/'results/site_status_pipeline.joblib')
sample=pd.read_csv(ROOT/'data/example_batch.csv')
scored=score_batch(pipeline,sample,meta)
np.testing.assert_allclose(scored.site_down_score,pipeline.predict_proba(normalize(sample))[:,1])
for edit in ['missing','null','unknown','negative','fractional','calendar','combo','range','empty']:
    x=sample.copy()
    if edit=='missing':x=x.drop(columns='Depot')
    elif edit=='null':x.loc[0,'repair_time']=np.nan
    elif edit=='unknown':x.loc[0,'Region']='unknown'
    elif edit=='negative':x.loc[0,'repair_time']=-1
    elif edit=='fractional':x['Day']=x.Day.astype(float);x.loc[0,'Day']=1.5
    elif edit=='calendar':x.loc[0,'Month']='february';x.loc[0,'Day']=31
    elif edit=='combo':x.loc[0,'Depot']='Hambantota' # row 0 is western/Malabe
    elif edit=='range':x.loc[0,'repair_time']=1000
    else:x=x.iloc[:0]
    try:score_batch(pipeline,x,meta)
    except ValueError:pass
    else:raise AssertionError('Invalid input accepted: '+edit)
print('Input validation: 9 invalid cases and valid batch passed.')
oof=pd.read_csv(ROOT/'results/oof_predictions.csv.gz')
assert len(oof)==meta['rows'] and oof.row_id.is_unique
assert oof.groupby('tower_id').fold.nunique().max()==1
assert oof.fold.nunique()==12
metrics=pd.read_csv(ROOT/'results/metrics.csv')
cms=json.loads((ROOT/'results/confusions.json').read_text())
for _,r in metrics.iterrows():
    p=oof[r.model]
    assert p.notna().all() and p.between(0,1).all()
    assert abs(accuracy_score(oof.actual,p>.5)-r.accuracy)<1e-12
    assert confusion_matrix(oof.actual,p>.5).tolist()==cms[r.model]
    for field,fn in [('precision',precision_score),('recall',recall_score),('f1',f1_score)]:
        assert abs(fn(oof.actual,p>.5,zero_division=0)-r[field])<1e-12
    fold_acc=np.array([accuracy_score(g.actual,g[r.model]>.5) for _,g in oof.groupby('fold')])
    assert abs(fold_acc.mean()-r.mean_fold_accuracy)<1e-12
    assert abs(fold_acc.std(ddof=1)-r.std_fold_accuracy)<1e-12
print('Every exported metric/confusion matrix matches all OOF rows; towers do not cross folds.')
app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=60).run()
assert not app.exception,list(app.exception)
pages=list(app.sidebar.radio[0].options)
for page in pages:
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception,(page,list(app.exception))
    if page=='Dataset explorer':
        app.multiselect[0].set_value(['western']).run()
        assert not app.exception
        assert app.metric[0].value=='81,913'
        app.multiselect[0].set_value([]).run()
        assert app.info and not app.exception
    elif page=='Model comparison':
        for option in ['precision','recall','f1']:
            app.selectbox[0].set_value(option).run();assert not app.exception
    elif page=='Model reports':
        for name in app.selectbox[0].options:
            app.selectbox[0].set_value(name).run();assert not app.exception
    elif page=='ANN learning curves':
        app.selectbox[0].set_value('Fold 12').run();assert not app.exception
    elif page=='Site status scoring':
        app.button[0].click().run();assert not app.exception
        assert len(app.metric)==3
        app.radio[0].set_value('Batch CSV').run();assert not app.exception
print('All six pages, filters, empty selection, metric/report selectors, ANN fold and single scoring passed.')
# Separate temporary app directory for error paths; no shipped files are mutated.
with tempfile.TemporaryDirectory() as td:
    temp=Path(td)
    shutil.copy(ROOT/'app.py',temp/'app.py')
    shutil.copy(ROOT/'core.py',temp/'core.py')
    (temp/'results').mkdir()
    err=AppTest.from_file(str(temp/'app.py'),default_timeout=60).run()
    assert err.error and not err.exception
    shutil.copy(ROOT/'results/metadata.json',temp/'results/metadata.json')
    shutil.copy(ROOT/'results/metrics.csv',temp/'results/metrics.csv')
    (temp/'results/reports.json').write_text('{"Missing section": {}}',encoding='utf-8')
    (temp/'results/confusions.json').write_text('{}',encoding='utf-8')
    err=AppTest.from_file(str(temp/'app.py'),default_timeout=60).run()
    err.sidebar.radio[0].set_value('Model reports').run()
    assert err.warning and not err.exception
    err.sidebar.radio[0].set_value('Site status scoring').run()
    assert err.error and not err.exception
    err.sidebar.radio[0].set_value('Dataset explorer').run()
    assert err.error and not err.exception
print('Missing metadata/model/dataset and absent report section handled without uncaught exceptions.')
print('ALL TESTS PASSED')
