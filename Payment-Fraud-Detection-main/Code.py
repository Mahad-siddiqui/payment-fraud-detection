
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", category=UserWarning)


def make_demo_data(n_rows: int = 20000, random_state: int = 42) -> pd.DataFrame:
	"""Create synthetic transactions when no CSV is available."""
	rng = np.random.default_rng(random_state)
	timestamps = pd.date_range("2023-01-01", "2023-12-01", periods=n_rows)
	transaction_types = rng.choice(
		["Online", "POS", "Transfer", "ATM"], size=n_rows, p=[0.35, 0.4, 0.2, 0.05]
	)

	# Most transactions are small; a tiny portion are very large and anomalous.
	normal_amounts = rng.lognormal(mean=2.6, sigma=0.45, size=n_rows)
	anomaly_mask = rng.random(n_rows) < 0.01
	normal_amounts[anomaly_mask] = rng.lognormal(mean=5.2, sigma=0.4, size=anomaly_mask.sum())

	df_local = pd.DataFrame(
		{
			"Timestamp": timestamps,
			"Amount": normal_amounts,
			"TransactionType": transaction_types,
		}
	)

	# Inject a few missing values to exercise imputation flow.
	missing_mask = rng.random(n_rows) < 0.01
	df_local.loc[missing_mask, "Amount"] = np.nan
	return df_local


def load_data(base_dir: Path) -> pd.DataFrame:
	csv_path = base_dir / "financial_anomaly_data.csv"
	if csv_path.exists():
		print(f"Loading dataset: {csv_path}")
		return pd.read_csv(csv_path)

	print("financial_anomaly_data.csv not found. Using generated demo data.")
	return make_demo_data()


def main() -> None:
	base_dir = Path(__file__).resolve().parent
	output_dir = base_dir / "ResultsPlots"
	output_dir.mkdir(exist_ok=True)

	df = load_data(base_dir)
	if "Timestamp" in df.columns:
		df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

	features = ["Amount"]
	print("Missing values before imputation:")
	print(df[features].isnull().sum())

	imputer = SimpleImputer(strategy="mean")
	df[features] = imputer.fit_transform(df[features])
	df[features] = (df[features] - df[features].mean()) / df[features].std(ddof=0)

	model = IsolationForest(contamination=0.01, random_state=42)
	model.fit(df[features])

	df["Fraudulent"] = model.predict(df[features])
	df["AnomalyScore"] = model.decision_function(df[features])

	potential_fraud = df[df["Fraudulent"] == -1]
	print(f"Potential fraudulent transactions: {len(potential_fraud)}")

	if "Timestamp" in df.columns and "TransactionType" in df.columns:
		df["TransactionHour"] = df["Timestamp"].dt.hour
		june_data = df[(df["Timestamp"] >= "2023-06-01") & (df["Timestamp"] < "2023-07-01")]
		monthly_spend = june_data.groupby("TransactionType")["Amount"].sum()
		print("June spend by transaction type:")
		print(monthly_spend)

	train, test = train_test_split(df, test_size=0.2, random_state=42)
	model.fit(train[features])
	predictions = model.predict(test[features])
	print("Classification report (using model labels as target):")
	print(classification_report(test["Fraudulent"], predictions))

	# Plot 1: anomaly scores over time
	plt.figure(figsize=(12, 6))
	sns.scatterplot(
		x=df["Timestamp"],
		y=df["AnomalyScore"],
		hue=df["Fraudulent"],
		palette={-1: "red", 1: "green"},
		s=25,
	)
	plt.title("Anomaly Scores Over Time")
	plt.xlabel("Timestamp")
	plt.ylabel("Anomaly Score")
	plt.xticks(rotation=45)
	plt.tight_layout()
	plt.savefig(output_dir / "Plot-1.png", dpi=120)
	plt.close()

	# Plot 2: distribution
	plt.figure(figsize=(10, 6))
	sns.histplot(df["AnomalyScore"], bins=50, kde=True, color="skyblue")
	plt.title("Distribution of Anomaly Scores")
	plt.xlabel("Anomaly Score")
	plt.ylabel("Frequency")
	plt.tight_layout()
	plt.savefig(output_dir / "Plot-2.png", dpi=120)
	plt.close()

	# Plot 3: box plot
	df_plot = df.copy()
	df_plot["ClassLabel"] = df_plot["Fraudulent"].map({1: "Normal", -1: "Potential Fraud"})
	plt.figure(figsize=(8, 6))
	sns.boxplot(
		data=df_plot,
		x="ClassLabel",
		y="AnomalyScore",
		order=["Normal", "Potential Fraud"],
		palette={"Normal": "red", "Potential Fraud": "green"},
	)
	plt.title("Box Plot of Anomaly Scores for Normal and Fraudulent Transactions")
	plt.xlabel("Transaction Class")
	plt.ylabel("Anomaly Score")
	plt.tight_layout()
	plt.savefig(output_dir / "Plot-3.png", dpi=120)
	plt.close()

	# Plot 4: violin plot
	plt.figure(figsize=(10, 6))
	sns.violinplot(
		data=df_plot,
		x="ClassLabel",
		y="AnomalyScore",
		order=["Normal", "Potential Fraud"],
		palette={"Normal": "red", "Potential Fraud": "green"},
		inner="quartile",
	)
	plt.title("Violin Plot of Anomaly Scores for Normal and Fraudulent Transactions")
	plt.xlabel("Transaction Class")
	plt.ylabel("Anomaly Score")
	plt.tight_layout()
	plt.savefig(output_dir / "Plot-4.png", dpi=120)
	plt.close()

	# Plot 5: pair plot of relevant variables
	pair_plot_cols = ["Amount", "AnomalyScore", "Fraudulent"]
	pair_plot = sns.pairplot(df[pair_plot_cols], hue="Fraudulent", palette={-1: "red", 1: "green"})
	pair_plot.figure.suptitle("Pair Plot of Relevant Variables", y=1.02)
	pair_plot.figure.savefig(output_dir / "Plot-5.png", dpi=120)
	plt.close(pair_plot.figure)

	print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
	main()

