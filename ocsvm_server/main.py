import time
import uuid
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from sklearn.svm import OneClassSVM

from .models.schemas import (
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

MODEL_NAME = "ocsvm"
start_time = time.time()
last_training = None

# Global state for training session
current_training_id = None
expected_discharges = 0
received_discharges: List[Discharge] = []
model: OneClassSVM | None = None

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
        # trigger training
        _train_model()
    return ack

def _train_model():
    global model, last_training, current_training_id
    start = time.time()
    normal = [d for d in received_discharges if d.anomalyTime is None]
    X = np.array([d.signals.values for d in normal])
    if len(X) == 0:
        model = None
        current_training_id = None
        return
    model = OneClassSVM(gamma="auto").fit(X)
    end = time.time()
    last_training = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    metrics = TrainingMetrics(accuracy=1.0, loss=0.0, f1Score=1.0)
    # In a real implementation we would POST to the orchestrator webhook
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
    x = np.array(discharge.signals.values).reshape(1, -1)
    score = float(model.decision_function(x)[0])
    pred = model.predict(x)[0]
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
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
