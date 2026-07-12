
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from backend.app.services.fraud_service import FraudDetectionService


def save_plots(df, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=df,
        x="Timestamp",
        y="AnomalyScore",
        hue="Fraudulent",
        palette={-1: "#c1121f", 1: "#2a9d8f"},
        s=16,
        alpha=0.7,
    )
    plt.title("Anomaly Scores Over Time")
    plt.tight_layout()
    plt.savefig(output_dir / "Plot-1.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(df["AnomalyScore"], bins=40, kde=True, color="#457b9d")
    plt.title("Distribution of Anomaly Scores")
    plt.tight_layout()
    plt.savefig(output_dir / "Plot-2.png", dpi=120)
    plt.close()

    df_plot = df.copy()
    df_plot["ClassLabel"] = df_plot["Fraudulent"].map({1: "Normal", -1: "Potential Fraud"})

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=df_plot,
        x="ClassLabel",
        y="AnomalyScore",
        hue="ClassLabel",
        legend=False,
        palette={"Normal": "#2a9d8f", "Potential Fraud": "#c1121f"},
    )
    plt.title("Box Plot of Anomaly Scores")
    plt.tight_layout()
    plt.savefig(output_dir / "Plot-3.png", dpi=120)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.violinplot(
        data=df_plot,
        x="ClassLabel",
        y="AnomalyScore",
        hue="ClassLabel",
        legend=False,
        palette={"Normal": "#2a9d8f", "Potential Fraud": "#c1121f"},
        inner="quartile",
    )
    plt.title("Violin Plot of Anomaly Scores")
    plt.tight_layout()
    plt.savefig(output_dir / "Plot-4.png", dpi=120)
    plt.close()

    pair_plot = sns.pairplot(
        df[["Amount", "AnomalyScore", "RiskScore", "Fraudulent"]],
        hue="Fraudulent",
        palette={-1: "#c1121f", 1: "#2a9d8f"},
        corner=True,
    )
    pair_plot.figure.suptitle("Pair Plot of Risk Variables", y=1.02)
    pair_plot.figure.savefig(output_dir / "Plot-5.png", dpi=120)
    plt.close(pair_plot.figure)


def main() -> None:
    repo_dir = Path(__file__).resolve().parent
    dataset_path = repo_dir / "financial_anomaly_data.csv"
    output_dir = repo_dir / "ResultsPlots"

    service = FraudDetectionService(dataset_path=dataset_path)
    service.load_and_train(contamination=0.01)

    summary = service.get_summary()
    df = service.artifacts.dataframe

    print("Model summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    save_plots(df, output_dir)
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

