"""Feature engineering, model comparison (RF vs GB vs baseline), training and prediction."""
import os, re, time
import numpy as np, pandas as pd, joblib
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from ingest import fetch_ts, key, TRAIN_DIR
from logger import update_train_log, update_predict_log

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
VERSION = '0.1'
MODELS = {
    'rf': (RandomForestRegressor(random_state=42), {'rf__n_estimators': [100, 200], 'rf__max_depth': [None, 10]}),
    'gb': (GradientBoostingRegressor(random_state=42), {'gb__n_estimators': [100, 200], 'gb__learning_rate': [0.05, 0.1]}),
}


def engineer_features(ts, training=True):
    """For each day t: revenue in previous 7/14/28/70 days, same 30-day window last year, recent activity.
    Target: total revenue over the 30 days starting at t."""
    s = ts.set_index('date')
    rev = s['revenue']
    X = pd.DataFrame(index=s.index)
    for d in (7, 14, 28, 70):
        X[f'prev_{d}'] = rev.rolling(d, min_periods=1).sum().shift(1)
    y = rev[::-1].rolling(30).sum()[::-1]
    X['prev_year'] = y.shift(365)
    X['recent_invoices'] = s['unique_invoices'].rolling(30, min_periods=1).mean().shift(1)
    X['recent_views'] = s['total_views'].rolling(30, min_periods=1).mean().shift(1)
    X = X.fillna(0)
    if training:
        X, y = X[y.notna()], y[y.notna()]
    return X, y.values


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def compare(ts, test=False):
    """Fit each model on the first 75% of days, score on the last 25% against a naive baseline."""
    X, y = engineer_features(ts)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, shuffle=False)
    preds, fitted = {'baseline': X_te['prev_28'].values * 30 / 28}, {}
    for name, (est, grid) in MODELS.items():
        if test:
            grid = {k: v[:1] for k, v in grid.items()}
        gs = GridSearchCV(Pipeline([('scale', StandardScaler()), (name, est)]), grid, cv=TimeSeriesSplit(3),
                          scoring='neg_root_mean_squared_error', n_jobs=-1).fit(X_tr, y_tr)
        fitted[name], preds[name] = gs.best_estimator_, gs.predict(X_te)
    return dict(X=X, y=y, dates=X_te.index, y_test=y_te, preds=preds, fitted=fitted,
                scores={k: rmse(y_te, p) for k, p in preds.items()})


def model_train(data_dir=TRAIN_DIR, test=False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    prefix = 'test' if test else 'sl'
    data = fetch_ts(data_dir)
    if test:
        data = {k: data[k] for k in ['all'] + [k for k in data if k != 'all'][:1]}
    for country, ts in data.items():
        t0 = time.time()
        r = compare(ts, test)
        best = min(MODELS, key=lambda m: r['scores'][m])
        model = clone(r['fitted'][best]).fit(r['X'], r['y'])  # refit on all data
        joblib.dump(model, os.path.join(MODEL_DIR, f'{prefix}-{country}-{VERSION}.joblib'))
        period = f"{r['X'].index.min().date()}:{r['X'].index.max().date()}"
        update_train_log(country, period, r['scores'], best, round(time.time() - t0, 2), VERSION, test)


def model_load(test=False):
    prefix = 'test' if test else 'sl'
    pat = re.compile(rf'^{prefix}-(.+)-{re.escape(VERSION)}\.joblib$')
    files = [f for f in os.listdir(MODEL_DIR) if pat.match(f)] if os.path.isdir(MODEL_DIR) else []
    if not files:
        raise FileNotFoundError('no models found, call /train first')
    return {pat.match(f).group(1): joblib.load(os.path.join(MODEL_DIR, f)) for f in files}


def model_predict(country, date, data_dir=TRAIN_DIR, test=False, models=None):
    t0 = time.time()
    country = key(country)
    models = models or model_load(test)
    if country not in models:
        raise ValueError(f"unknown country '{country}', options: {sorted(models)}")
    target = pd.Timestamp(date)
    X, _ = engineer_features(fetch_ts(data_dir)[country], training=False)
    if target not in X.index:
        raise ValueError(f'date must be between {X.index.min().date()} and {X.index.max().date()}')
    y_pred = float(models[country].predict(X.loc[[target]])[0])
    update_predict_log(country, target.date(), y_pred, round(time.time() - t0, 3), VERSION, test)
    return {'country': country, 'date': str(target.date()), 'y_pred': y_pred}


if __name__ == '__main__':
    model_train()
    print(model_predict('all', '2018-01-05'))
