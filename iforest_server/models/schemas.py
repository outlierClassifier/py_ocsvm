from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Any

MODEL_NAME = "iforest"
WINDOW_SIZE = 32

class Signal(BaseModel):
    filename: str
    values: List[float]

class Discharge(BaseModel):
    id: str
    signals: List[Signal]
    times: List[float]
    length: int
    anomalyTime: Optional[float] = None

    @field_validator("signals", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> List[Signal] | Any:
        if isinstance(v, dict):
            return [v]
        return v

class StartTrainingRequest(BaseModel):
    totalDischarges: int = Field(..., ge=1)
    timeoutSeconds: int = Field(..., ge=1)

class StartTrainingResponse(BaseModel):
    expectedDischarges: int

class DischargeAck(BaseModel):
    ordinal: int
    totalDischarges: int

class TrainingMetrics(BaseModel):
    accuracy: float
    loss: float
    f1Score: float

class TrainingResponse(BaseModel):
    status: str
    message: str
    trainingId: str
    metrics: TrainingMetrics
    executionTimeMs: float

class WindowProperties(BaseModel):
    featureValues: List[float] = Field(..., min_items=1)
    prediction: str = Field(..., pattern=r'^(Anomaly|Normal)$')
    justification: float

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    executionTimeMs: float
    model: str
    windowSize: int = WINDOW_SIZE
    windows: List[WindowProperties]

class HealthCheckResponse(BaseModel):
    name: str = MODEL_NAME
    uptime: float
    lastTraining: str
