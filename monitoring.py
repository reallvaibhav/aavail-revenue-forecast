"""Drift + performance check: compare production data to training data and score the model on it."""
import numpy as np
from scipy import stats
from ingest import fetch_ts, TRAIN_DIR, PROD_DIR
from model import engineer_features, model_load, rmse


def monitor(country='all', train_dir=TRAIN_DIR, prod_dir=PROD_DIR, test=False):
    _, y_train = engineer_features(fetch_ts(train_dir)[country])
    X_prod, y_prod = engineer_features(fetch_ts(prod_dir)[country])
    pred = model_load(test)[country].predict(X_prod)
    ks = stats.ks_2samp(y_train, y_prod)
    return {'country': country,
            'train_target_mean': float(np.mean(y_train)), 'prod_target_mean': float(np.mean(y_prod)),
            'ks_stat': float(ks.statistic), 'ks_pvalue': float(ks.pvalue), 'drift': bool(ks.pvalue < 0.05),
            'prod_rmse': rmse(y_prod, pred),
            'prod_mape': float(np.mean(np.abs(y_prod - pred) / np.maximum(y_prod, 1)))}


if __name__ == '__main__':
    print(monitor())
