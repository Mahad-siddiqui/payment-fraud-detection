from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routers.fraud import router as fraud_router

app = FastAPI(title="Payment Fraud Detection API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(fraud_router)
