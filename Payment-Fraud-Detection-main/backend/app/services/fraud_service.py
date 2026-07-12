from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


@dataclass
class FraudArtifacts:
    dataframe: pd.DataFrame
    encoded_columns: list[str]
    contamination: float
    data_source: str
    trained_at: str


class FraudDetectionService:
    def __init__(self, dataset_path: Path, random_state: int = 42) -> None:
        self.dataset_path = dataset_path
        self.random_state = random_state
        self.model: IsolationForest | None = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.artifacts: FraudArtifacts | None = None
        self.data_source = "demo"
        self.trained_at = ""

    def load_and_train(self, contamination: float = 0.01) -> None:
        df, source_name = self._load_data()
        self._train_from_dataframe(df=df, contamination=contamination, source_name=source_name)

    def train_with_dataframe(self, df: pd.DataFrame, contamination: float = 0.01) -> None:
        self._train_from_dataframe(df=df, contamination=contamination, source_name="uploaded-csv")

    def _train_from_dataframe(self, df: pd.DataFrame, contamination: float, source_name: str) -> None:
        prepared_df, features = self._prepare_features(df)

        x_imputed = self.imputer.fit_transform(features)
        x_scaled = self.scaler.fit_transform(x_imputed)

        self.model = IsolationForest(
            n_estimators=300,
            contamination=contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(x_scaled)

        predictions = self.model.predict(x_scaled)
        anomaly_scores = self.model.decision_function(x_scaled)

        # Convert anomaly scores to an intuitive 0-100 risk score.
        anomaly_index = -anomaly_scores
        risk_scores = 100 * (anomaly_index - anomaly_index.min()) / (
            anomaly_index.max() - anomaly_index.min() + 1e-9
        )

        prepared_df["Fraudulent"] = predictions
        prepared_df["AnomalyScore"] = anomaly_scores
        prepared_df["RiskScore"] = risk_scores.round(2)

        self.data_source = source_name
        self.trained_at = datetime.now(timezone.utc).isoformat()

        self.artifacts = FraudArtifacts(
            dataframe=prepared_df,
            encoded_columns=list(features.columns),
            contamination=contamination,
            data_source=self.data_source,
            trained_at=self.trained_at,
        )

    def get_summary(self) -> dict:
        df = self._ensure_df()
        fraud_count = int((df["Fraudulent"] == -1).sum())
        total = int(len(df))

        return {
            "total_transactions": total,
            "potential_fraud_count": fraud_count,
            "potential_fraud_rate": round((fraud_count / max(total, 1)) * 100, 2),
            "avg_amount": round(float(df["Amount"].mean()), 2),
            "median_amount": round(float(df["Amount"].median()), 2),
            "p95_risk_score": round(float(df["RiskScore"].quantile(0.95)), 2),
            "data_source": self.data_source,
            "trained_at": self.trained_at,
        }

    def get_transactions(
        self,
        limit: int = 200,
        fraud_only: bool = False,
        tx_type: str | None = None,
        min_risk: float = 0.0,
    ) -> list[dict]:
        df = self._ensure_df().copy()
        if fraud_only:
            df = df[df["Fraudulent"] == -1]
        if tx_type:
            df = df[df["TransactionType"] == tx_type]
        df = df[df["RiskScore"] >= min_risk]

        cols = [
            "Timestamp",
            "TransactionType",
            "Amount",
            "TransactionHour",
            "DayOfWeek",
            "Fraudulent",
            "RiskScore",
            "AnomalyScore",
        ]
        available_cols = [c for c in cols if c in df.columns]
        data = (
            df.sort_values("RiskScore", ascending=False)
            .head(limit)[available_cols]
            .assign(Timestamp=lambda d: d["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
            .to_dict(orient="records")
        )
        return data

    def get_transaction_types(self) -> list[str]:
        df = self._ensure_df()
        return sorted(df["TransactionType"].dropna().astype(str).unique().tolist())

    def get_type_risk_stats(self) -> list[dict]:
        df = self._ensure_df()
        grouped = (
            df.groupby("TransactionType")
            .agg(
                transaction_count=("Amount", "count"),
                avg_risk=("RiskScore", "mean"),
                fraud_count=("Fraudulent", lambda x: int((x == -1).sum())),
            )
            .reset_index()
        )
        grouped["fraud_rate"] = (grouped["fraud_count"] / grouped["transaction_count"]) * 100
        grouped["avg_risk"] = grouped["avg_risk"].round(2)
        grouped["fraud_rate"] = grouped["fraud_rate"].round(2)
        return grouped.sort_values("avg_risk", ascending=False).to_dict(orient="records")

    def get_live_feed(self, limit: int = 25) -> list[dict]:
        df = self._ensure_df().copy()
        cols = [
            "Timestamp",
            "TransactionType",
            "Amount",
            "RiskScore",
            "Fraudulent",
            "AnomalyScore",
        ]
        out = (
            df.sort_values("Timestamp", ascending=False)
            .head(limit)[cols]
            .assign(Timestamp=lambda d: d["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S"))
            .to_dict(orient="records")
        )
        return out

    def get_executive_insights(self) -> list[str]:
        df = self._ensure_df()
        type_stats = self.get_type_risk_stats()

        top_risk_type = type_stats[0]["TransactionType"] if type_stats else "N/A"
        high_risk_rate = float((df["RiskScore"] >= 80).mean() * 100)
        night_risk_rate = float((df[df["TransactionHour"].between(0, 5)]["Fraudulent"] == -1).mean() * 100)
        night_risk_rate = 0.0 if np.isnan(night_risk_rate) else night_risk_rate

        insights = [
            f"Highest average risk observed in '{top_risk_type}' transactions.",
            f"{high_risk_rate:.2f}% of transactions are in critical risk band (RiskScore >= 80).",
            f"Fraud propensity during 00:00-05:59 window is {night_risk_rate:.2f}%.",
            "Use transaction type and minimum-risk filters to isolate explainable fraud segments.",
        ]
        return insights

    def simulate_transaction(self, append_to_dataset: bool = True) -> dict:
        df = self._ensure_df()
        rng = np.random.default_rng()

        transaction_types = self.get_transaction_types()
        if not transaction_types:
            transaction_types = ["Online", "POS", "Transfer", "ATM"]

        tx_type = rng.choice(transaction_types)
        timestamp = pd.Timestamp.utcnow().tz_localize(None)

        # Bias to produce occasional high-risk events for demo impact.
        if rng.random() < 0.15:
            amount = float(rng.lognormal(mean=5.1, sigma=0.5))
        else:
            amount = float(rng.lognormal(mean=2.8, sigma=0.6))

        tx = pd.DataFrame(
            {
                "Timestamp": [timestamp],
                "Amount": [amount],
                "TransactionType": [tx_type],
            }
        )

        prepared_tx, features_tx = self._prepare_features(tx)
        features_tx = features_tx.reindex(columns=self.artifacts.encoded_columns, fill_value=0.0)

        x_imputed = self.imputer.transform(features_tx)
        x_scaled = self.scaler.transform(x_imputed)

        pred = int(self.model.predict(x_scaled)[0])
        anomaly = float(self.model.decision_function(x_scaled)[0])

        anomaly_index_existing = -df["AnomalyScore"]
        min_ai = float(anomaly_index_existing.min())
        max_ai = float(anomaly_index_existing.max())
        current_ai = -anomaly
        risk = float(100 * (current_ai - min_ai) / (max_ai - min_ai + 1e-9))
        risk = float(np.clip(risk, 0, 100))

        prepared_tx["Fraudulent"] = pred
        prepared_tx["AnomalyScore"] = anomaly
        prepared_tx["RiskScore"] = round(risk, 2)

        row = prepared_tx.iloc[0].to_dict()
        row["Timestamp"] = prepared_tx.iloc[0]["Timestamp"].strftime("%Y-%m-%d %H:%M:%S")

        if append_to_dataset:
            self.artifacts.dataframe = pd.concat([df, prepared_tx], ignore_index=True)

        return row

    def build_excel_report_bytes(self) -> bytes:
        summary = pd.DataFrame([self.get_summary()])
        top_risk = pd.DataFrame(self.get_transactions(limit=200, fraud_only=False))
        type_risk = pd.DataFrame(self.get_type_risk_stats())
        trend = pd.DataFrame(self.get_monthly_fraud_trend())

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            summary.to_excel(writer, sheet_name="Summary", index=False)
            top_risk.to_excel(writer, sheet_name="TopTransactions", index=False)
            type_risk.to_excel(writer, sheet_name="TypeRisk", index=False)
            trend.to_excel(writer, sheet_name="Trend", index=False)

        output.seek(0)
        return output.getvalue()

    def get_monthly_fraud_trend(self) -> list[dict]:
        df = self._ensure_df()
        monthly = (
            df[df["Fraudulent"] == -1]
            .groupby(df["Timestamp"].dt.to_period("M"))
            .size()
            .rename("count")
            .reset_index()
        )
        monthly["x"] = monthly["Timestamp"].astype(str)
        monthly["y"] = monthly["count"].astype(float)
        return monthly[["x", "y"]].to_dict(orient="records")

    def get_histogram(self, bins: int = 30) -> dict:
        df = self._ensure_df()
        counts, edges = np.histogram(df["AnomalyScore"], bins=bins)
        return {
            "bins": [round(float(v), 6) for v in edges.tolist()],
            "counts": [int(v) for v in counts.tolist()],
        }

    def _ensure_df(self) -> pd.DataFrame:
        if self.artifacts is None:
            raise RuntimeError("Model is not trained. Call load_and_train first.")
        return self.artifacts.dataframe

    def _load_data(self) -> tuple[pd.DataFrame, str]:
        if self.dataset_path.exists():
            df = pd.read_csv(self.dataset_path)
            source_name = "financial_anomaly_data.csv"
        else:
            df = self._make_demo_data()
            source_name = "generated-demo-data"

        required = {"Amount", "Timestamp", "TransactionType"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {', '.join(sorted(missing))}")

        return df, source_name

    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        prepared = df.copy()
        prepared["Timestamp"] = pd.to_datetime(prepared["Timestamp"], errors="coerce")
        prepared = prepared.dropna(subset=["Timestamp"]).reset_index(drop=True)
        prepared["TransactionType"] = prepared["TransactionType"].fillna("Unknown")

        prepared["TransactionHour"] = prepared["Timestamp"].dt.hour
        prepared["DayOfWeek"] = prepared["Timestamp"].dt.dayofweek
        prepared["IsWeekend"] = prepared["DayOfWeek"].isin([5, 6]).astype(int)

        features = prepared[["Amount", "TransactionHour", "DayOfWeek", "IsWeekend", "TransactionType"]].copy()
        features = pd.get_dummies(features, columns=["TransactionType"], dtype=float)

        return prepared, features

    def _make_demo_data(self, n_rows: int = 30000) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        timestamps = pd.date_range("2023-01-01", "2024-01-01", periods=n_rows)
        transaction_types = rng.choice(
            ["Online", "POS", "Transfer", "ATM"],
            size=n_rows,
            p=[0.4, 0.35, 0.2, 0.05],
        )

        amounts = rng.lognormal(mean=2.8, sigma=0.55, size=n_rows)
        is_large_outlier = rng.random(n_rows) < 0.015
        amounts[is_large_outlier] = rng.lognormal(mean=5.0, sigma=0.45, size=is_large_outlier.sum())

        data = pd.DataFrame(
            {
                "Timestamp": timestamps,
                "Amount": amounts,
                "TransactionType": transaction_types,
            }
        )

        missing_mask = rng.random(n_rows) < 0.01
        data.loc[missing_mask, "Amount"] = np.nan
        return data
