import time
import uuid
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from sklearn.svm import OneClassSVM
import pickle
import os

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
    MODEL_NAME,
    WINDOW_SIZE,
)

app = FastAPI()

start_time = time.time()
last_training = None

# Global state for training session
current_training_id = None
expected_discharges = 0
received_discharges: List[Discharge] = []
max_signal_values = {} # Stored at training for normalization, used at prediction
model: OneClassSVM | None = None

def get_discharge_id(filename: str) -> int:
    """Extract the discharge ID from the filename."""
    return int(filename.split("_")[2])

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
        print("Training completed.")
    return ack

def filter_and_normalize_received_discharges(training: bool = False):
    """Normalize the received discharges to ensure consistent data format."""
    # Search max value for each signal
    global max_signal_values, received_discharges
    received_discharges = [d for d in received_discharges if d.anomalyTime is None]
    if training:
        max_signal_values = {}
        for discharge in received_discharges:
            for signal in discharge.signals:
                signal_id = get_discharge_id(signal.filename)
                if signal_id not in max_signal_values:
                    max_signal_values[signal_id] = float('-inf')
                max_signal_values[signal_id] = max(max_signal_values[signal_id], max(signal.values))
    else:
        # During prediction, normalize using the stored max values
        if not max_signal_values:
            raise ValueError("Max signal values not set for normalization")

    for discharge in received_discharges:
        for signal in discharge.signals:
            signal_id = get_discharge_id(signal.filename)
            if signal_id in max_signal_values:
                # Normalize values to range [0, 1]
                # signal index is signal_id - 1
                discharge.signals[signal_id - 1] = [
                    float(v) / max_signal_values[signal_id] if max_signal_values[signal_id] > 0 else 0.0
                    for v in signal.values
                ]
            else:
                print(f"Warning: No max value found for signal '{signal.filename}', initializing with zeros")

def _extract_windowed_features(values, window_size=WINDOW_SIZE) -> List[List[float]]:
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


def _safe_screen_samples(
    X: np.ndarray, nu: float, gamma: str | float
) -> tuple[np.ndarray, int, float]:
    """Apply simple Safe Sample Screening before OCSVM training.

    Parameters
    ----------
    X: np.ndarray
        Training samples.
    nu: float
        Nu parameter used by the OCSVM.
    gamma: str | float
        Gamma parameter for the kernel.

    Returns
    -------
    tuple(np.ndarray, int, float)
        Filtered samples, number of removed windows and the screening time in
        seconds.
    """

    screen_start = time.time()
    n_samples = X.shape[0]
    removed = 0

    print(f"[SSS] Starting Safe Sample Screening with {n_samples} samples")

    if nu >= 0.6:
        print("[SSS] Warning: nu >= 0.6, screening disabled")
        return X, removed, 0.0

    if n_samples < 500:
        print(f"[SSS] Warning: only {n_samples} windows - skipping screening")
        return X, removed, 0.0

    # Train a small preliminary model on a random subset to estimate the
    # decision boundary.  This is a lightweight approximation used solely to
    # determine which samples are safely inside the boundary.
    subset_size = min(max(int(0.1 * n_samples), 1), 2000)
    subset_idx = np.random.choice(n_samples, subset_size, replace=False)

    elapsed = time.time() - screen_start
    print(f"[SSS] Training preliminary model on {subset_size} samples. Elapsed: {elapsed:.2f} seconds")

    prelim = OneClassSVM(nu=nu, gamma=gamma).fit(X[subset_idx])

    # Decision function values for all samples using the preliminary model
    f_values = prelim.decision_function(X)

    print(f"[SSS] Decision function computed for {n_samples} samples. Elapsed: {time.time() - screen_start:.2f} seconds")
    # Heuristic threshold: points with large positive score are far inside the
    # decision region and will have alpha=0.  They can be removed without
    # affecting the final boundary.
    threshold = float(np.quantile(f_values, 0.9))
    keep_mask = f_values <= threshold
    removed = int(np.sum(~keep_mask))

    print(f"[SSS] {removed} samples removed based on threshold {threshold:.2f}. Elapsed: {time.time() - screen_start:.2f} seconds")
    
    if removed > 0:
        pct = removed / n_samples * 100.0
        print(f"[SSS] Filtered {removed} of {n_samples} windows ({pct:.2f}%)")
        X = X[keep_mask]

    screen_end = time.time()
    return X, removed, screen_end - screen_start

def _train_model():
    global model, last_training, current_training_id
    start = time.time()

    # Normalize received discharges
    filter_and_normalize_received_discharges(training=True)

    # Extract windowed features from all signals of normal discharges
    X: List[List[float]] = []
    for d in received_discharges:
        discharge_values: List[float] = []
        for signal in d.signals:
            discharge_values.extend(signal)
        features = _extract_windowed_features(discharge_values, window_size=WINDOW_SIZE)
        X.extend(features)

    print(f"Extracted {len(X)} feature windows from {len(received_discharges)} received discharges")

    if len(X) == 0:
        model = None
        current_training_id = None
        return

    X_array = np.array(X)

    # Parameters for the model
    nu_val = 0.5
    gamma_val: str | float = "auto"

    # Safe Sample Screening before training
    X_array, removed, screen_time = _safe_screen_samples(X_array, nu_val, gamma_val)
    print(f"Screened {removed} samples in {screen_time:.2f} seconds")

    model = OneClassSVM(nu=nu_val, gamma=gamma_val).fit(X_array)
    end = time.time()
    last_training = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    print(f"Model trained in {end - start:.2f} seconds")
    # TODO: Calculate real metrics based on validation set if available
    # TODO: Send webhook to orchestrator
    # save the model
    with open(f"{MODEL_NAME}.pkl", "wb") as f:
        pickle.dump(model, f)
    # save the max signal values for normalization during prediction
    with open(f"{MODEL_NAME}_max_values.pkl", "wb") as f:
        pickle.dump(max_signal_values, f)
    metrics = TrainingMetrics(accuracy=1.0, loss=0.0, f1Score=1.0)
    TrainingResponse(
        status="SUCCESS",
        message="training completed",
        trainingId=current_training_id,
        metrics=metrics,
        executionTimeMs=(end - start + screen_time) * 1000,
    )
    current_training_id = None

@app.post("/predict", response_model=PredictionResponse)
def predict(discharge: Discharge):
    global received_discharges, model

    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained")
    start = time.time()
    
    received_discharges = [discharge]  # Reset received discharges to only this one for prediction
    filter_and_normalize_received_discharges(training=False)

    # Extract values from all signals for this discharge
    discharge_values = []
    for signal in discharge.signals:
        discharge_values.extend(signal)
    
    # Extract windowed features using the same method as training
    features = _extract_windowed_features(discharge_values, window_size=WINDOW_SIZE)
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

    windows = [
        WindowProperties(
            featureValues=[float(v) if not np.isnan(v) and not np.isinf(v) else -5.0 for v in list(X[i])],
            prediction=prediction,
            justification=scores[i],
        )
        for i in range(len(X))
    ]
    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        executionTimeMs=(end - start) * 1000,
        model=MODEL_NAME,
        windowSize=WINDOW_SIZE,
        windows=windows,
    )

if __name__ == "__main__":
    print("Starting OCSVM server...")
    import uvicorn
    # Load the model and max signal values if they exist
    try:
        with open(f"{MODEL_NAME}.pkl", "rb") as f:
            model = pickle.load(f)
            print("Model loaded successfully.")
        with open(f"{MODEL_NAME}_max_values.pkl", "rb") as f:
            max_signal_values = pickle.load(f)
            print("Max signal values loaded successfully.")
    except FileNotFoundError:
        print("No pre-trained model found.")
    uvicorn.run(app, host="0.0.0.0", port=8001)
