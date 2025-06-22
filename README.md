# py_iforest

This project provides a simple Isolation Forest service that follows the
protocol described in the provided OpenAPI specification.  The API is
implemented using **FastAPI** and exposes endpoints for training and
prediction.

## Setup

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the server

```bash
uvicorn ocsvm_server.main:app --reload --port 8000
```

The service exposes the following endpoints:

- `GET /health` – Check service status.
- `POST /train` – Start a training session.
- `POST /train/{ordinal}` – Send discharges for training.
- `POST /predict` – Predict if a discharge is `Normal` or `Anomaly`.

The implementation keeps everything in memory and trains a scikit-learn
`IsolationForest` once all discharges have been received.  Each `Discharge`
may include an optional `anomalyTime` field when the sample represents a
disruption.  During training these anomalous discharges are ignored so
only normal data is used to fit the model.
