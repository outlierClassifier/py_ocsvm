from pydantic import BaseModel, Field
from typing import List, Optional

class Signal(BaseModel):
    filename: str
    values: List[float]

class Discharge(BaseModel):
    id: str
    signals: List[Signal]
    times: List[float]
    length: int
    anomalyTime: Optional[float] = None

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

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    executionTimeMs: float
    model: str

class HealthCheckResponse(BaseModel):
    name: str
    uptime: float
    lastTraining: str
