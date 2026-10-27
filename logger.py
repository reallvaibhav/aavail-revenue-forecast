"""CSV logging of every train and predict call. Test runs write to separate test-*.log files."""
import os, csv, time, uuid
from datetime import date

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')


def _write(name, header, row, test):
    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"{'test-' if test else ''}{name}-{date.today():%Y-%m}.log")
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
    return path


def update_train_log(tag, period, scores, best, runtime, version, test=False):
    return _write('train', ['id', 'timestamp', 'tag', 'period', 'rmse_scores', 'best_model', 'model_version', 'runtime'],
                  [uuid.uuid4(), time.time(), tag, period, scores, best, version, runtime], test)


def update_predict_log(country, target_date, y_pred, runtime, version, test=False):
    return _write('predict', ['id', 'timestamp', 'country', 'target_date', 'y_pred', 'model_version', 'runtime'],
                  [uuid.uuid4(), time.time(), country, target_date, y_pred, version, runtime], test)
