import time
import uuid
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from sklearn.ensemble import IsolationForest

from models.schemas import (
    StartTrainingRequest,
    StartTrainingResponse,
    Discharge,
    DischargeAck,
    PredictionResponse,
    HealthCheckResponse,
    TrainingResponse,
    TrainingMetrics,
)

app = FastAPI()

MODEL_NAME = "iforest"
start_time = time.time()
last_training = None

# Global state for training session
current_training_id = None
expected_discharges = 0
received_discharges: List[Discharge] = []
model: IsolationForest | None = None

@app.get("/health", response_model=HealthCheckResponse)
def health() -> HealthCheckResponse:
    uptime = time.time() - start_time
    return HealthCheckResponse(
        name=MODEL_NAME,
        uptime=uptime,
        lastTraining=last_training or "",
    )

@app.post("/train", response_model=StartTrainingResponse, status_code=200)
def start_training(req: StartTrainingRequest):
    global expected_discharges, received_discharges, current_training_id
    print(f"Starting training with request: {req}")
    if current_training_id is not None:
        raise HTTPException(status_code=503, detail="Training already in progress")
    expected_discharges = req.totalDischarges
    received_discharges = []
    current_training_id = str(uuid.uuid4())
    return StartTrainingResponse(expectedDischarges=expected_discharges)

@app.post("/train/{ordinal}", response_model=DischargeAck)
def push_discharge(ordinal: int, discharge: Discharge):
    global received_discharges
    if current_training_id is None:
        raise HTTPException(status_code=503, detail="No active training session")
    if ordinal != len(received_discharges) + 1:
        raise HTTPException(status_code=400, detail="Unexpected ordinal")
    received_discharges.append(discharge)
    ack = DischargeAck(ordinal=ordinal, totalDischarges=expected_discharges)
    if len(received_discharges) == expected_discharges:
        print(f"Received all {expected_discharges} discharges, starting training...")
        _train_model()
    return ack

def _extract_windowed_features(values, window_size=64) -> List[List[float]]:
    """Extract features from windowed signal data"""
    features: List[List[float]] = []
    if len(values) < window_size:
        window = values
        mean_val = np.mean(window)
        fft = np.fft.fft(window)
        psd = np.sum(np.abs(fft) ** 2) / len(window)
        features.append([mean_val, psd])
        return features

    for i in range(0, len(values) - window_size + 1, window_size):
        window = values[i:i + window_size]

        # Feature 1: Mean value
        mean_val = np.mean(window)

        # Feature 2: Power spectral density (sum of squared FFT coefficients)
        fft = np.fft.fft(window)
        psd = np.sum(np.abs(fft) ** 2) / len(window)

        features.append([mean_val, psd])

    return features

def _train_model():
    global model, last_training, current_training_id
    start = time.time()
    normal = [d for d in received_discharges if d.anomalyTime is None]

    # Extract windowed features from all signals of normal discharges
    X: List[List[float]] = []
    for d in normal:
        discharge_values: List[float] = []
        for signal in d.signals:
            discharge_values.extend(signal.values)

        features = _extract_windowed_features(discharge_values, window_size=64)
        X.extend(features)

    if len(X) == 0:
        model = None
        current_training_id = None
        return

    X_array = np.array(X)
    model = IsolationForest(random_state=42, contamination='auto').fit(X_array)
    end = time.time()
    last_training = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    # TODO: Calculate real metrics based on validation set if available
    # TODO: Send webhook to orchestrator
    metrics = TrainingMetrics(accuracy=1.0, loss=0.0, f1Score=1.0)
    TrainingResponse(
        status="SUCCESS",
        message="training completed",
        trainingId=current_training_id,
        metrics=metrics,
        executionTimeMs=(end - start) * 1000,
    )
    current_training_id = None

@app.post("/predict", response_model=PredictionResponse)
def predict(discharge: Discharge):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained")
    start = time.time()
    
    # Extract values from all signals for this discharge
    discharge_values = []
    for signal in discharge.signals:
        discharge_values.extend(signal.values)
    
    # Extract windowed features using the same method as training
    features = _extract_windowed_features(discharge_values, window_size=64)
    if len(features) == 0:
        raise HTTPException(status_code=400, detail="Not enough data for prediction")

    X = np.array(features)
    scores = model.decision_function(X)
    preds = model.predict(X)
    score = float(np.mean(scores))
    pred = 1 if np.mean(preds) >= 0 else -1
    prediction = "Normal" if pred == 1 else "Anomaly"
    # simple sigmoid to map score to confidence 0-1
    confidence = float(1 / (1 + np.exp(-score)))
    end = time.time()
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        executionTimeMs=(end - start) * 1000,
        model=MODEL_NAME,
    )

if __name__ == "__main__":
    print("Starting Isolation Forest server...")
    import uvicorn
    print("Uvicorn version:", uvicorn.__version__)
    uvicorn.run(app, host="0.0.0.0", port=8006)
