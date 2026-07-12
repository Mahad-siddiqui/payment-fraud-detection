# Payment Fraud Detection Platform

This project is upgraded from a single script into a full-stack fraud monitoring platform:

1. Efficient Python backend API with Isolation Forest and feature engineering.
2. Modern React dashboard for monitoring fraud risk and transaction patterns.
3. Backward-compatible script runner for generating analysis plots.

## Project Structure

- backend/: FastAPI service and ML pipeline.
- frontend/: React + Vite dashboard.
- Code.py: CLI runner that trains and writes plots to ResultsPlots/.
- ResultsPlots/: Output charts from CLI run.

## Dataset

Place your dataset at the repository root as financial_anomaly_data.csv.

Required columns:

- Timestamp
- Amount
- TransactionType

If no CSV file is found, the backend and CLI automatically generate realistic demo data.

## Backend Setup

From project root:

1. Install dependencies:

python -m pip install -r backend/requirements.txt

2. Start API server:

uvicorn backend.app.main:app --reload --port 8000

API base URL:

http://127.0.0.1:8000

Main endpoints:

- GET /api/health
- GET /api/summary
- GET /api/transactions?limit=120&fraud_only=true
- GET /api/charts/trend
- GET /api/charts/distribution
- POST /api/reload

## Frontend Setup

1. Install dependencies:

cd frontend
npm install

2. Start dashboard:

npm run dev

Dashboard URL:

http://127.0.0.1:5173

## Legacy CLI Mode

Run the legacy script for quick analysis and chart export:

python Code.py

Charts are saved in ResultsPlots/ as Plot-1.png to Plot-5.png.

## Model Improvements Included

1. Better features: amount, transaction hour, day-of-week, weekend flag, transaction type encoding.
2. Robust missing-value handling with median imputation.
3. Standardized feature scaling for stable model behavior.
4. Risk scoring normalized to a 0-100 range for UI readability.
5. Cached service model loading in backend for faster API response.

## Notes

1. This is an unsupervised anomaly detector; labels represent anomaly likelihood, not confirmed fraud.
2. Tune contamination via POST /api/reload with contamination in range 0.001 to 0.2.
