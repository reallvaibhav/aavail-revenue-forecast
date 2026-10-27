"""Load AAVAIL invoice JSON files, clean them and build daily revenue time series per country."""
import os, re, shutil
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.environ.get('AAVAIL_TRAIN_DIR', os.path.join(BASE, 'data', 'cs-train'))
PROD_DIR = os.environ.get('AAVAIL_PROD_DIR', os.path.join(BASE, 'data', 'cs-production'))
# column names differ between monthly files
RENAME = {'StreamID': 'stream_id', 'TimesViewed': 'times_viewed', 'total_price': 'price'}


def key(country):
    return re.sub(r'\s+', '_', str(country).strip().lower())


def fetch_data(data_dir):
    files = sorted(f for f in os.listdir(data_dir) if f.endswith('.json'))
    if not files:
        raise FileNotFoundError(f'no json files in {data_dir}')
    df = pd.concat([pd.read_json(os.path.join(data_dir, f)).rename(columns=RENAME) for f in files], ignore_index=True)
    df['invoice'] = df['invoice'].astype(str).str.replace(r'\D+', '', regex=True)
    df['invoice_date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
    df = df.dropna(subset=['invoice_date', 'price'])
    df = df[df['price'] >= 0]
    return df.sort_values('invoice_date').reset_index(drop=True)


def convert_to_ts(df, country=None):
    days = pd.date_range(df['invoice_date'].min(), df['invoice_date'].max(), freq='D')
    if country:
        df = df[df['country'] == country]
    g = df.groupby('invoice_date')
    ts = pd.DataFrame({'purchases': g.size(),
                       'unique_invoices': g['invoice'].nunique(),
                       'unique_streams': g['stream_id'].nunique(),
                       'total_views': g['times_viewed'].sum(),
                       'revenue': g['price'].sum()}).reindex(days, fill_value=0)
    ts.index.name = 'date'
    return ts.reset_index()


def fetch_ts(data_dir, clean=False, n_countries=10):
    """Return {'all': ts, '<country>': ts, ...} for the top countries by revenue (cached as csv)."""
    ts_dir = os.path.join(data_dir, 'ts-data')
    if clean and os.path.isdir(ts_dir):
        shutil.rmtree(ts_dir)
    if os.path.isdir(ts_dir) and os.listdir(ts_dir):
        return {f[3:-4]: pd.read_csv(os.path.join(ts_dir, f), parse_dates=['date']) for f in os.listdir(ts_dir)}
    df = fetch_data(data_dir)
    out = {'all': convert_to_ts(df)}
    for c in df.groupby('country')['price'].sum().nlargest(n_countries).index:
        out[key(c)] = convert_to_ts(df, c)
    os.makedirs(ts_dir, exist_ok=True)
    for k, v in out.items():
        v.to_csv(os.path.join(ts_dir, f'ts-{k}.csv'), index=False)
    return out


if __name__ == '__main__':
    for k, v in fetch_ts(TRAIN_DIR, clean=True).items():
        print(f'{k:20s} days={len(v)} revenue={v.revenue.sum():,.0f}')
