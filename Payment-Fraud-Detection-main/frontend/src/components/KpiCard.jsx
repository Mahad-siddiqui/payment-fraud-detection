export default function KpiCard({ label, value, tone = "default" }) {
  return (
    <div className={`kpi-card tone-${tone}`}>
      <p className="kpi-label">{label}</p>
      <h3 className="kpi-value">{value}</h3>
    </div>
  );
}
