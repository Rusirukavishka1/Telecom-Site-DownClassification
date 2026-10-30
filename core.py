"""Shared preprocessing and strict inference validation. No UI dependencies."""
import calendar
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

CATEGORICAL = ['Region', 'Depot', 'Network_type', 'Month', 'equipment_vendor', 'Day', 'techs']
CANDIDATES = CATEGORICAL + ['repair_time']
MONTHS = [m.lower() for m in calendar.month_name[1:]]

def normalize(frame):
    missing = sorted(set(CANDIDATES) - set(frame.columns))
    if missing:
        raise ValueError('Missing required columns: ' + ', '.join(missing))
    x = frame[CANDIDATES].copy()
    for c in CATEGORICAL:
        x[c] = x[c].map(lambda v: str(v).strip() if pd.notna(v) else np.nan)
    x['Region'] = x['Region'].str.lower().replace({'nothern':'northern'})
    x['Month'] = x['Month'].str.lower()
    for c in ['Day','techs']:
        v = pd.to_numeric(x[c], errors='raise')
        if ((v.dropna() % 1) != 0).any():
            raise ValueError(c + ' must contain integer values.')
        x[c] = v.map(lambda a: str(int(a)) if pd.notna(a) else np.nan)
    x['repair_time'] = pd.to_numeric(x['repair_time'], errors='raise')
    if np.isinf(x['repair_time']).any() or (x['repair_time'].dropna() < 0).any():
        raise ValueError('repair_time must be finite and nonnegative.')
    invalid_months = set(x['Month'].dropna()) - set(MONTHS)
    if invalid_months:
        raise ValueError('Unknown month names: ' + ', '.join(sorted(invalid_months)))
    for m, d in zip(x['Month'], x['Day']):
        if pd.notna(m) and pd.notna(d):
            # Dataset has 365 days; year is unspecified, so February 29 is rejected.
            if not 1 <= int(d) <= calendar.monthrange(2023, MONTHS.index(m)+1)[1]:
                raise ValueError('Invalid month/day combination (non-leap calendar).')
    return x

class AssociationSelector(BaseEstimator, TransformerMixin):
    """Original p<.05 association rule, fitted on training rows only."""
    def __init__(self, threshold=0.05):
        self.threshold = threshold
    def fit(self, X, y):
        self.p_values_ = {}
        for c in X:
            valid = X[c].notna()
            if c == 'repair_time':
                p = pointbiserialr(np.asarray(y)[valid], X.loc[valid,c]).pvalue
            else:
                table = pd.crosstab(X.loc[valid,c], np.asarray(y)[valid])
                p = chi2_contingency(table)[1] if min(table.shape)>1 else 1.0
            self.p_values_[c] = float(p)
        self.selected_ = [c for c in X if self.p_values_[c] < self.threshold]
        if not self.selected_:
            raise ValueError('No training-fold predictors pass the existing association rule.')
        return self
    def transform(self, X):
        return X[self.selected_].copy()

def preprocessing():
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encode', OneHotEncoder(handle_unknown='ignore', dtype=np.float32))])
    return Pipeline([('select', AssociationSelector()), ('encode', ColumnTransformer([
        ('cat', categorical, make_column_selector(dtype_include=object)),
        ('num', numeric, make_column_selector(dtype_include=np.number))], sparse_threshold=1.0))])

def score_batch(pipeline, frame, metadata):
    if frame.empty:
        raise ValueError('The CSV has no data rows.')
    if len(frame)>200000:
        raise ValueError('Use at most 200,000 rows per upload.')
    x = normalize(frame)
    if x.isna().any().any():
        raise ValueError('Prediction inputs must not contain missing values.')
    for c, values in metadata['categories'].items():
        unknown = set(x[c])-set(values)
        if unknown:
            raise ValueError(f'Unknown {c} values: {sorted(unknown)[:5]}')
    combos=set(tuple(a) for a in metadata['region_depot_tech'])
    if any(t not in combos for t in x[['Region','Depot','techs']].itertuples(index=False,name=None)):
        raise ValueError('Region/Depot/techs combination was not observed in the supplied dataset.')
    low,high=metadata['repair_time_range']
    if not x['repair_time'].between(low,high).all():
        raise ValueError(f'repair_time must be within the observed range {low:.6f}–{high:.6f}.')
    p=pipeline.predict_proba(x)[:,1]
    result=frame.copy()
    result['site_down_score']=p
    result['predicted_status']=(p>0.5).astype(int)
    result['score_band']=np.where(p>=.7,'High',np.where(p>=.4,'Medium','Low'))
    return result
