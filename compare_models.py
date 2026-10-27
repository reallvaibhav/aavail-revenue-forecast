"""Plot holdout predictions of each model against the naive baseline -> reports/model_comparison.png"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ingest import fetch_ts, TRAIN_DIR
from model import compare

country = 'all'
r = compare(fetch_ts(TRAIN_DIR)[country])
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(r['dates'], r['y_test'], 'k', lw=2, label='actual')
for name, p in r['preds'].items():
    ax.plot(r['dates'], p, ls='--' if name == 'baseline' else '-', label=f"{name} (RMSE {r['scores'][name]:,.0f})")
ax.set(title=f'30-day revenue forecast, holdout ({country})', ylabel='revenue next 30 days')
ax.legend()
os.makedirs('reports', exist_ok=True)
fig.savefig(os.path.join('reports', 'model_comparison.png'), dpi=120, bbox_inches='tight')
print(r['scores'])
