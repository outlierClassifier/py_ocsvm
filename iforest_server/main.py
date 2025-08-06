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
    num_tendency = 3 # Use the past 3 values to compute tendency

    if len(values) < window_size * num_tendency:
        window = np.array(values, dtype=np.float64)
        features.append(_compute_features(window))
        return features

    # Use overlapping windows
    for i in range(0, len(values) - window_size + 1, window_size - overlap):
        # skip the firsts num_tendency windows
        if i < num_tendency * (window_size - overlap):
            continue

        window = np.array(values[i:i + window_size], dtype=np.float64)
        features.append(_compute_features(window))
        # Append the last num_tendency slope values as features.
        initial_pos = i - num_tendency * (window_size - overlap)
        end_pos = i
        for j in range(num_tendency):
            window = np.array(values[initial_pos + j * (window_size - overlap):end_pos + j * (window_size - overlap)], dtype=np.float64)
            if len(window) < window_size:
                logger.warning(f"Window size {len(window)} is less than expected {window_size}, filling with NaN")
                features[-1].append(float('nan'))
            slope = np.diff(window)
            mean_slope = np.mean(slope) if len(slope) > 0 else 0.0
            features[-1].append(float(mean_slope))
    
    logger.info(f"Extracted {len(features)} windowed features from signal data")
    
    return features

def _compute_features(window: np.ndarray) -> List[float]:
    """Compute a rich set of features from a window of signal data. Applies log transformation to all values"""
    # Statistical features
    mean_val = float(np.abs(np.mean(window)))
    std_window = np.std(window)
    std_val = float(np.log1p(std_window)) if std_window > 0 else 0.0

    # Min, max and range
    min_val = float(np.log1p(np.abs(np.min(window))))
    max_val = float(np.log1p(np.abs(np.max(window))))
    range_val = float(np.log1p(np.abs(max_val - min_val)))

    # slope and 2nd derivative
    diff = np.diff(window) if len(window) > 1 else np.array([0.0])
    std_diff = np.std(diff) # Aux var to avoid repeated computation
    log_std_diff = float(np.log1p(std_diff)) if len(diff) > 1 and std_diff > 0 else 0.0
    abs_second_derivate = np.abs(np.diff(diff)) if len(diff) > 1 and std_diff > 0 else np.array([0.0])

    # Frequency domain features
    fft = np.fft.fft(window)
    fft_abs = np.log1p(np.abs(fft))
    psd = float(np.log1p(np.sum(fft_abs ** 2) / len(window)))
    
    # Dominant frequency
    freqs = np.fft.fftfreq(len(window))
    dom_freq_idx = np.argmax(fft_abs[1:len(freqs)//2]) + 1
    dom_freq = float(np.log1p(np.abs(freqs[dom_freq_idx]))) if np.abs(freqs[dom_freq_idx]) > 0 else 0.0
    dom_power = float(np.log1p(fft_abs[dom_freq_idx])) if fft_abs[dom_freq_idx] > 0 else 0.0

    return [mean_val, 
            std_val, 
            min_val, 
            max_val, 
            range_val,
            float(np.log1p(np.mean(diff))) if np.mean(diff) > 0 else 0.0,
            float(np.log1p(np.mean(abs_second_derivate))) if np.mean(abs_second_derivate) > 0 else 0.0,
            float(np.log1p(np.max(abs_second_derivate))) if np.max(abs_second_derivate) > 0 else 0.0,
            float(np.log1p(np.min(abs_second_derivate))) if np.min(abs_second_derivate) > 0 else 0.0,
            psd, dom_freq, dom_power, log_std_diff]

def _process_discharge(discharge: Discharge) -> np.ndarray:
    """Process a discharge consistently for both training and prediction"""
    all_features = []
    num_tendency = 3  # Past 3 values to compute tendency

    signal_lengths = [len(signal.values) for signal in discharge.signals]
    min_signal_length = min(signal_lengths)
    
    overlap = 0  # Adjust if overlap is needed
    n_windows = max(1, (min_signal_length - WINDOW_SIZE) // (WINDOW_SIZE - overlap) + 1)
    
    min_start_idx = num_tendency * (WINDOW_SIZE - overlap)
    
    # For each window position
    for w_idx in range(n_windows):
        start = w_idx * (WINDOW_SIZE - overlap)
        end = start + WINDOW_SIZE
        
        if start < min_start_idx or end > min_signal_length:
            continue

        window_features = []
        for signal in discharge.signals:
            window = np.array(signal.values[start:end], dtype=np.float64)
            signal_features = _compute_features(window)
            window_features.extend(signal_features)
            
            # Add the last num_tendency slope values as features
            initial_pos = start - num_tendency * (WINDOW_SIZE - overlap)
            end_pos = start
            for i in range(num_tendency):
                slope_window = np.array(signal.values[initial_pos + i * (WINDOW_SIZE - overlap):end_pos + i * (WINDOW_SIZE - overlap)], dtype=np.float64)
                if len(slope_window) < WINDOW_SIZE:
                    logger.warning(f"Window size {len(slope_window)} is less than expected {WINDOW_SIZE}, filling with zero")
                    window_features.append(0.0)
                else:
                    slope = np.diff(slope_window)
                    mean_slope = np.mean(slope) if len(slope) > 0 else 0.0
                    window_features.append(float(mean_slope))

        mean_vals = [float(np.mean(sig.values[start:end])) for sig in discharge.signals]

        # Inter signal features.
        A_MINOR = 0.95 / 2.0  # Minor radius of the JET tokamak in meters. Used in the Greenwald limit.
        rad_power_ratio        = mean_vals[5] / mean_vals[6] if mean_vals[6] != 0.0 else 0.0
        greenwald_density_frac = mean_vals[3] / (mean_vals[0] / (np.pi * A_MINOR**2)) if mean_vals[0] != 0.0 else 0.0
        locked_mode_norm       = mean_vals[1] / mean_vals[0] if mean_vals[0] != 0.0 else 0.0
        inner_induct_norm      = mean_vals[2] / mean_vals[0] if mean_vals[0] != 0.0 else 0.0
        beta_loss              = abs(mean_vals[5]) / mean_vals[6] if mean_vals[6] != 0.0 else 0.0

        inter_features = [
            rad_power_ratio, greenwald_density_frac,
            locked_mode_norm, inner_induct_norm, beta_loss
        ]
        # Add the concatenated features for this window
        all_features.append(window_features)
        # Uncomment to add inter-signal features (consistent for all entries)
        all_features[-1].extend(inter_features)
    
    if not all_features:
        return np.array([])
        
    # Check for consistency in feature dimensions
    feature_lengths = [len(features) for features in all_features]
    if len(set(feature_lengths)) > 1:
        logger.warning(f"Inconsistent feature lengths: {feature_lengths}")
        # Pad shorter feature lists to match the maximum length
        max_length = max(feature_lengths)
        for i, features in enumerate(all_features):
            if len(features) < max_length:
                all_features[i] = features + [0.0] * (max_length - len(features))
    
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
    model = IsolationForest(random_state=42, 
                            contamination=0.001,
                            n_estimators=250,
                            max_samples=1.0).fit(X_array)
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

    raw_scores = model.decision_function(X)
    scores = 1.0 / (1.0 + np.exp(raw_scores))
    preds = model.predict(X)
    score = float(np.quantile(scores, 0.1))
    pred = 1 if np.mean(preds) >= 0 else -1
    prediction = "Normal" if pred == 1 else "Anomaly"
    # simple sigmoid to map score to confidence 0-1
    confidence = float(np.quantile(scores, 0.9))
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
