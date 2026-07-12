import { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  fetchDistribution,
  fetchInsights,
  fetchLiveFeed,
  fetchSummary,
  fetchTransactionTypes,
  fetchTransactions,
  fetchTrend,
  fetchTypeRisk,
  downloadExcelReport,
  retrain,
  simulateTransaction,
  uploadDataset,
} from "./api";
import KpiCard from "./components/KpiCard";

function formatMoney(value) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  }).format(value || 0);
}

export default function App() {
  const [summary, setSummary] = useState(null);
  const [trend, setTrend] = useState([]);
  const [distribution, setDistribution] = useState([]);
  const [typeRisk, setTypeRisk] = useState([]);
  const [insights, setInsights] = useState([]);
  const [liveFeed, setLiveFeed] = useState([]);
  const [transactions, setTransactions] = useState([]);
  const [transactionTypes, setTransactionTypes] = useState([]);
  const [fraudOnly, setFraudOnly] = useState(true);
  const [selectedType, setSelectedType] = useState("");
  const [minRisk, setMinRisk] = useState(0);
  const [contamination, setContamination] = useState(0.012);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [liveMode, setLiveMode] = useState(false);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(true);
  const [retraining, setRetraining] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");

  const load = async () => {
    setLoading(true);
    const [
      summaryData,
      trendData,
      distributionData,
      txData,
      types,
      typeRiskData,
      insightsData,
      feedData,
    ] = await Promise.all([
      fetchSummary(),
      fetchTrend(),
      fetchDistribution(),
      fetchTransactions(120, fraudOnly, selectedType, minRisk),
      fetchTransactionTypes(),
      fetchTypeRisk(),
      fetchInsights(),
      fetchLiveFeed(20),
    ]);

    const histData = distributionData.counts.map((count, index) => ({
      x: distributionData.bins[index],
      y: count,
    }));

    setSummary(summaryData);
    setTrend(trendData);
    setDistribution(histData);
    setTransactions(txData);
    setTransactionTypes(types);
    setTypeRisk(typeRiskData);
    setInsights(insightsData);
    setLiveFeed(feedData);
    setLoading(false);
  };

  useEffect(() => {
    load();
  }, [fraudOnly, selectedType, minRisk]);

  useEffect(() => {
    if (!autoRefresh) {
      return undefined;
    }

    const timerId = window.setInterval(() => {
      load();
    }, 10000);

    return () => window.clearInterval(timerId);
  }, [autoRefresh, fraudOnly, selectedType, minRisk]);

  useEffect(() => {
    if (!liveMode) {
      return undefined;
    }

    const timerId = window.setInterval(async () => {
      await simulateTransaction(true);
      await load();
      setStatusMessage("Live event ingested and model metrics refreshed.");
    }, 4500);

    return () => window.clearInterval(timerId);
  }, [liveMode, fraudOnly, selectedType, minRisk]);

  const topRisk = useMemo(() => {
    if (!transactions.length) {
      return [];
    }
    return transactions.slice(0, 12);
  }, [transactions]);

  const handleRetrain = async () => {
    setRetraining(true);
    await retrain(contamination);
    await load();
    setStatusMessage("Model retrained successfully.");
    setRetraining(false);
  };

  const handleUpload = async () => {
    if (!file) {
      setStatusMessage("Please select a CSV file first.");
      return;
    }

    setUploading(true);
    try {
      await uploadDataset(file, contamination);
      await load();
      setStatusMessage("Dataset uploaded and model retrained.");
      setFile(null);
    } catch (error) {
      setStatusMessage(error?.response?.data?.detail || "Upload failed. Check CSV format.");
    } finally {
      setUploading(false);
    }
  };

  if (loading || !summary) {
    return <div className="loading">Loading fraud intelligence dashboard...</div>;
  }

  return (
    <div className="page">
      <header className="hero">
        <div>
          <p className="eyebrow">Payment Intelligence Console</p>
          <h1>Fraud Detection Dashboard</h1>
          <p className="subtitle">
            Isolation Forest powered anomaly monitoring with explainable risk scoring.
          </p>
          <p className="subtitle meta-line">
            Source: {summary.data_source} | Trained at: {summary.trained_at}
          </p>
        </div>
        <div className="hero-actions">
          <button onClick={handleRetrain} disabled={retraining}>
            {retraining ? "Retraining..." : "Retrain Model"}
          </button>
          <label className="toggle">
            <input
              type="checkbox"
              checked={fraudOnly}
              onChange={(e) => setFraudOnly(e.target.checked)}
            />
            Show only potential fraud
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={liveMode}
              onChange={(e) => setLiveMode(e.target.checked)}
            />
            Live Simulation Mode
          </label>
        </div>
      </header>

      <section className="panel control-panel">
        <h2>Live Model Controls</h2>
        <div className="controls-grid">
          <label>
            Contamination: {contamination.toFixed(3)}
            <input
              type="range"
              min="0.001"
              max="0.08"
              step="0.001"
              value={contamination}
              onChange={(e) => setContamination(Number(e.target.value))}
            />
          </label>

          <label>
            Transaction Type
            <select value={selectedType} onChange={(e) => setSelectedType(e.target.value)}>
              <option value="">All Types</option>
              {transactionTypes.map((type) => (
                <option key={type} value={type}>
                  {type}
                </option>
              ))}
            </select>
          </label>

          <label>
            Minimum Risk: {minRisk}
            <input
              type="range"
              min="0"
              max="100"
              step="1"
              value={minRisk}
              onChange={(e) => setMinRisk(Number(e.target.value))}
            />
          </label>

          <label className="toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh every 10s
          </label>
        </div>

        <div className="upload-row">
          <input
            type="file"
            accept=".csv"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
          <button onClick={handleUpload} disabled={uploading}>
            {uploading ? "Uploading..." : "Upload CSV and Retrain"}
          </button>
          <button onClick={load}>Refresh Now</button>
          <button onClick={downloadExcelReport}>Download Excel Intelligence Report</button>
        </div>
        {statusMessage ? <p className="status-msg">{statusMessage}</p> : null}
      </section>

      <section className="panel">
        <h2>Executive Insights</h2>
        <div className="insight-grid">
          {insights.map((insight, idx) => (
            <article className="insight-card" key={`insight-${idx}`}>
              {insight}
            </article>
          ))}
        </div>
      </section>

      <section className="kpi-grid">
        <KpiCard label="Total Transactions" value={summary.total_transactions} />
        <KpiCard
          label="Potential Fraud"
          value={summary.potential_fraud_count}
          tone="risk"
        />
        <KpiCard
          label="Fraud Rate"
          value={`${summary.potential_fraud_rate}%`}
          tone="risk"
        />
        <KpiCard label="Average Amount" value={formatMoney(summary.avg_amount)} />
        <KpiCard label="Median Amount" value={formatMoney(summary.median_amount)} />
        <KpiCard
          label="P95 Risk Score"
          value={summary.p95_risk_score}
          tone="accent"
        />
      </section>

      <section className="chart-grid">
        <article className="panel">
          <h2>Monthly Fraud Trend</h2>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey="y"
                stroke="#e63946"
                strokeWidth={3}
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </article>

        <article className="panel">
          <h2>Anomaly Score Distribution</h2>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={distribution}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" />
              <YAxis />
              <Tooltip />
              <Area
                type="monotone"
                dataKey="y"
                stroke="#1d3557"
                fill="url(#riskGradient)"
                strokeWidth={2}
              />
              <defs>
                <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ff7f51" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#ffbe0b" stopOpacity={0.1} />
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </article>

        <article className="panel">
          <h2>Risk by Transaction Type</h2>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={typeRisk}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="TransactionType" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="avg_risk" name="Avg Risk" fill="#ff7f51" />
              <Bar dataKey="fraud_rate" name="Fraud Rate %" fill="#1d3557" />
            </BarChart>
          </ResponsiveContainer>
        </article>
      </section>

      <section className="panel">
        <h2>Top Risk Transactions</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Type</th>
                <th>Amount</th>
                <th>Risk Score</th>
                <th>Flag</th>
              </tr>
            </thead>
            <tbody>
              {topRisk.map((row, index) => (
                <tr key={`${row.Timestamp}-${index}`}>
                  <td>{row.Timestamp}</td>
                  <td>{row.TransactionType}</td>
                  <td>{formatMoney(row.Amount)}</td>
                  <td>{row.RiskScore}</td>
                  <td>
                    <span className={row.Fraudulent === -1 ? "badge danger" : "badge safe"}>
                      {row.Fraudulent === -1 ? "Potential Fraud" : "Normal"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <h2>Live Stream Feed</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Type</th>
                <th>Amount</th>
                <th>Risk Score</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {liveFeed.map((row, index) => (
                <tr key={`${row.Timestamp}-${index}`}>
                  <td>{row.Timestamp}</td>
                  <td>{row.TransactionType}</td>
                  <td>{formatMoney(row.Amount)}</td>
                  <td>{row.RiskScore}</td>
                  <td>
                    <span className={row.Fraudulent === -1 ? "badge danger" : "badge safe"}>
                      {row.Fraudulent === -1 ? "Flagged" : "Normal"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
