import axios from "axios";

const API_BASE = "http://127.0.0.1:8000/api";

export async function fetchSummary() {
  const { data } = await axios.get(`${API_BASE}/summary`);
  return data;
}

export async function fetchTrend() {
  const { data } = await axios.get(`${API_BASE}/charts/trend`);
  return data;
}

export async function fetchDistribution() {
  const { data } = await axios.get(`${API_BASE}/charts/distribution`, {
    params: { bins: 32 },
  });
  return data;
}

export async function fetchTransactions(
  limit = 100,
  fraudOnly = false,
  txType = "",
  minRisk = 0
) {
  const { data } = await axios.get(`${API_BASE}/transactions`, {
    params: {
      limit,
      fraud_only: fraudOnly,
      tx_type: txType || undefined,
      min_risk: minRisk,
    },
  });
  return data;
}

export async function fetchTransactionTypes() {
  const { data } = await axios.get(`${API_BASE}/meta/transaction-types`);
  return data;
}

export async function fetchTypeRisk() {
  const { data } = await axios.get(`${API_BASE}/charts/type-risk`);
  return data;
}

export async function fetchInsights() {
  const { data } = await axios.get(`${API_BASE}/insights`);
  return data;
}

export async function fetchLiveFeed(limit = 25) {
  const { data } = await axios.get(`${API_BASE}/live-feed`, { params: { limit } });
  return data;
}

export async function simulateTransaction(appendToDataset = true) {
  const { data } = await axios.post(`${API_BASE}/simulate`, null, {
    params: { append_to_dataset: appendToDataset },
  });
  return data;
}

export async function retrain(contamination = 0.01) {
  const { data } = await axios.post(`${API_BASE}/reload`, { contamination });
  return data;
}

export async function uploadDataset(file, contamination = 0.01) {
  const form = new FormData();
  form.append("file", file);
  form.append("contamination", String(contamination));

  const { data } = await axios.post(`${API_BASE}/upload`, form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function downloadExcelReport() {
  const response = await axios.get(`${API_BASE}/report/excel`, { responseType: "blob" });
  const blob = new Blob([response.data], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });

  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "fraud_intelligence_report.xlsx";
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}
