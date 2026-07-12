from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ReloadRequest(BaseModel):
    contamination: float = Field(default=0.01, ge=0.001, le=0.2)


class SummaryResponse(BaseModel):
    total_transactions: int
    potential_fraud_count: int
    potential_fraud_rate: float
    avg_amount: float
    median_amount: float
    p95_risk_score: float
    data_source: str
    trained_at: str


class ChartPoint(BaseModel):
    x: str
    y: float


class HistogramResponse(BaseModel):
    bins: list[float]
    counts: list[int]


class TypeRiskResponse(BaseModel):
    TransactionType: str
    transaction_count: int
    avg_risk: float
    fraud_count: int
    fraud_rate: float
