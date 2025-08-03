import time
import uuid
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
import scipy
from sklearn.ensemble import IsolationForest
import pickle
import os
import logging

from models.schemas import (
    StartTrainingRequest,
    StartTrainingResponse,
    Discharge,
    DischargeAck,
    PredictionResponse,
    HealthCheckResponse,
    TrainingResponse,
    TrainingMetrics,
    WindowProperties,
    WINDOW_SIZE,
    MODEL_NAME,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "iforest_model.pkl"
SAMPLING_TIME = 2e-3  # 2ms

app = FastAPI()
model = None
start_time = time.time()
last_training = None

# Global state for training session
current_training_id = None
expected_discharges = 0
received_discharges: List[Discharge] = []
model: IsolationForest | None = None

# Load model if exists
if os.path.exists(MODEL_PATH):
    try:
        model = pickle.load(open(MODEL_PATH, "rb"))
        logger.info("Existing model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")


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

def _extract_windowed_features(values, window_size=WINDOW_SIZE, overlap=0) -> List[List[float]]:
    """Extract richer features from windowed signal data with overlap"""
    features: List[List[float]] = []
    if len(values) < window_size:
        window = np.array(values, dtype=np.float64)
        features.append(_compute_features(window))
        return features

    # Use overlapping windows
    for i in range(0, len(values) - window_size + 1, window_size - overlap):
        window = np.array(values[i:i + window_size], dtype=np.float64)
        features.append(_compute_features(window))

    return features

def _compute_features(window: np.ndarray) -> List[float]:
    """Compute a rich set of features from a window of signal data"""
    # Statistical features
    mean_val = float(np.mean(window))
    std_val = float(np.std(window))
    
    # Min, max and range
    min_val = float(np.min(window))
    max_val = float(np.max(window))
    range_val = max_val - min_val

    # slope and 2nd derivative
    diff = np.diff(window) if len(window) > 1 else np.array([0.0])
    abs_second_derivate = np.abs(np.diff(diff)) if len(diff) > 1 else np.array([0.0])
    
    # Frequency domain features
    fft = np.fft.fft(window)
    fft_abs = np.abs(fft)
    psd = float(np.log1p(np.sum(fft_abs ** 2) / len(window)))
    
    # Dominant frequency
    freqs = np.fft.fftfreq(len(window))
    dom_freq_idx = np.argmax(fft_abs[1:len(freqs)//2]) + 1
    dom_freq = float(np.abs(freqs[dom_freq_idx]))
    dom_power = float(np.log1p(fft_abs[dom_freq_idx]))

    # Trend and variation
    if len(window) >= 2:
        diff = np.diff(window)
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff) if len(diff) > 1 else 0)
    else:
        mean_diff = std_diff = 0.0
    
    return [mean_val, 
            std_val, 
            min_val, 
            max_val, 
            range_val,
            float(np.mean(diff) * 1/SAMPLING_TIME),
            float(np.mean(abs_second_derivate) * 1/(SAMPLING_TIME**2)),
            float(np.max(abs_second_derivate) * 1/(SAMPLING_TIME**2)),
            float(np.min(abs_second_derivate) * 1/(SAMPLING_TIME**2)),
            psd, dom_freq, dom_power, mean_diff, std_diff]

def _process_discharge(discharge: Discharge) -> np.ndarray:
    """Process a discharge consistently for both training and prediction"""
    all_features = []
    
    # Process each signal separately
    for signal in discharge.signals:
        features = _extract_windowed_features(signal.values)
        
        # Add signal identifier (0-6) as a feature to maintain signal identity
        for i, feat in enumerate(features):
            signal_features = feat.copy()
            # Add temporal position as a feature
            position = i / max(1, len(features))
            signal_features.append(position)
            all_features.append(signal_features)
    
    return np.array(all_features)

def _train_model():
    global model, last_training, current_training_id
    start = time.time()
    normal = [d for d in received_discharges if d.anomalyTime is None]

    # Extract windowed features from all signals of normal discharges
    X: List[List[float]] = []
    for d in normal:
        discharge_features = _process_discharge(d)
        X.extend(discharge_features)

    if len(X) == 0:
        model = None
        current_training_id = None
        return

    X_array = np.array(X)
    model = IsolationForest(random_state=42, contamination='auto').fit(X_array)
    end = time.time()
    last_training = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    
    # Save the trained model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model trained and saved to {MODEL_PATH}")

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
    
    X = _process_discharge(discharge)

    if len(X) == 0:
        raise HTTPException(status_code=400, detail="Not enough data for prediction")

    scores = model.decision_function(X)
    preds = model.predict(X)
    score = float(np.mean(scores))
    pred = 1 if np.mean(preds) >= 0 else -1
    prediction = "Normal" if pred == 1 else "Anomaly"
    # simple sigmoid to map score to confidence 0-1
    confidence = float(1 / (1 + np.exp(-score)))
    end = time.time()

    windows = [WindowProperties(
        featureValues=[float(val) if not np.isnan(val) and not np.isinf(val) else -5.0 for val in list(X[i])],
        prediction=prediction,
        justification=scores[i] if scores[i] is not None else -1.0,
    ) for i in range(len(X))]

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        executionTimeMs=(end - start) * 1000,
        windowSize=WINDOW_SIZE,
        windows=windows,
        model=MODEL_NAME,
    )

if __name__ == "__main__":
    print("Starting Isolation Forest server...")
    import uvicorn
    print("Uvicorn version:", uvicorn.__version__)
    uvicorn.run(app, host="0.0.0.0", port=8006)
