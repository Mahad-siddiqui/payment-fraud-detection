from functools import lru_cache
from io import StringIO
from io import BytesIO

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from backend.app.config import DEFAULT_DATASET_PATH
from backend.app.schemas import (
    ChartPoint,
    HistogramResponse,
    ReloadRequest,
    SummaryResponse,
    TypeRiskResponse,
)
from backend.app.services.fraud_service import FraudDetectionService

router = APIRouter(prefix="/api", tags=["fraud"])


@lru_cache(maxsize=1)
def get_service() -> FraudDetectionService:
    service = FraudDetectionService(dataset_path=DEFAULT_DATASET_PATH)
    service.load_and_train(contamination=0.01)
    return service


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/reload", response_model=SummaryResponse)
def reload_model(payload: ReloadRequest) -> dict:
    service = get_service()
    service.load_and_train(contamination=payload.contamination)
    return service.get_summary()


@router.get("/summary", response_model=SummaryResponse)
def summary() -> dict:
    return get_service().get_summary()


@router.get("/transactions")
def transactions(
    limit: int = 200,
    fraud_only: bool = False,
    tx_type: str | None = None,
    min_risk: float = 0.0,
) -> list[dict]:
    return get_service().get_transactions(
        limit=limit,
        fraud_only=fraud_only,
        tx_type=tx_type,
        min_risk=min_risk,
    )


@router.get("/meta/transaction-types")
def transaction_types() -> list[str]:
    return get_service().get_transaction_types()


@router.get("/charts/type-risk", response_model=list[TypeRiskResponse])
def chart_type_risk() -> list[dict]:
    return get_service().get_type_risk_stats()


@router.get("/insights")
def insights() -> list[str]:
    return get_service().get_executive_insights()


@router.get("/live-feed")
def live_feed(limit: int = 25) -> list[dict]:
    return get_service().get_live_feed(limit=limit)


@router.post("/simulate")
def simulate(append_to_dataset: bool = True) -> dict:
    return get_service().simulate_transaction(append_to_dataset=append_to_dataset)


@router.get("/charts/trend", response_model=list[ChartPoint])
def chart_trend() -> list[dict]:
    return get_service().get_monthly_fraud_trend()


@router.get("/charts/distribution", response_model=HistogramResponse)
def chart_distribution(bins: int = 30) -> dict:
    return get_service().get_histogram(bins=bins)


@router.post("/upload", response_model=SummaryResponse)
async def upload_csv(
    file: UploadFile = File(...),
    contamination: float = Form(0.01),
) -> dict:
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    try:
        raw = await file.read()
        decoded = raw.decode("utf-8-sig")
        df = pd.read_csv(StringIO(decoded))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {exc}") from exc

    service = get_service()
    try:
        service.train_with_dataframe(df=df, contamination=contamination)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Training failed: {exc}") from exc

    return service.get_summary()


@router.get("/report/excel")
def report_excel() -> StreamingResponse:
    report_bytes = get_service().build_excel_report_bytes()
    content = BytesIO(report_bytes)

    return StreamingResponse(
        content,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=fraud_intelligence_report.xlsx"},
    )
