# AAVAIL Revenue Forecasting: 30-Day Revenue Prediction Service

This service forecasts AAVAIL's total revenue for the 30 days that start on a given date. It can forecast for one of the top 10 countries or for all countries combined.

## Layout
| File | Purpose |
|---|---|
| `ingest.py` | Loads the invoice JSON files, cleans them (column names, invoice ids, bad dates, negative prices) and builds daily time series per country, cached in `data/<dir>/ts-data` |
| `notebooks/eda.ipynb` | EDA: revenue over time, top countries, weekly pattern, correlations |
| `model.py` | Builds lag features and compares **Random Forest** and **Gradient Boosting** against a naive baseline. The best model is saved for each country |
| `compare_models.py` | Plots the models against the baseline on the holdout period, saved to `reports/model_comparison.png` |
| `monitoring.py` | Drift check (KS test on training vs production targets) plus model RMSE/MAPE on production data |
| `logger.py` | Logs every train and predict call to `logs/`. Test runs write to separate `test-*.log` files |
| `app.py` | Flask API: `/train`, `/predict`, `/logs`, `/logs/<file>`, `/monitor` |
| `run-tests.py` | Runs the API, model and logging tests in `unittests/`. Tests use `test-*` models and logs |

## Data
Download the course data and place it like this: `data/cs-train/*.json` and `data/cs-production/*.json`.
You can point to other folders with the `AAVAIL_TRAIN_DIR` and `AAVAIL_PROD_DIR` environment variables.

## Run
```bash
pip install -r requirements.txt
python ingest.py            # build time series
python model.py             # train production models
python compare_models.py    # model vs baseline chart
python run-tests.py         # all unit tests
python app.py               # API on :8080
```

## API
```bash
curl -X POST localhost:8080/train
curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d '{"country":"united_kingdom","date":"2018-01-05"}'
curl localhost:8080/logs
curl "localhost:8080/monitor?country=all"
```
Add `"mode":"test"` to a request to use the test models and logs.

## Docker
```bash
docker build -t aavail-forecast .
docker run -p 8080:8080 -v "$PWD/data:/app/data" aavail-forecast
```
