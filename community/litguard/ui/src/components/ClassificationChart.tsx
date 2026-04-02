import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import type { Metrics } from "../hooks/useMetrics";

interface Props {
  metrics: Metrics;
}

const COLORS = ["#f43f5e", "#10b981"];

export default function ClassificationChart({ metrics }: Props) {
  const data = [
    { name: "Injection", value: metrics.injection_count },
    { name: "Benign", value: metrics.benign_count },
  ];

  const total = metrics.injection_count + metrics.benign_count;
  const injectionPct = total > 0 ? ((metrics.injection_count / total) * 100).toFixed(1) : "0";
  const benignPct = total > 0 ? ((metrics.benign_count / total) * 100).toFixed(1) : "0";

  return (
    <div className="card p-6 h-full">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-[15px] font-semibold text-[var(--text-primary)]">
            Classification Distribution
          </h3>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            {total.toLocaleString()} total classifications
          </p>
        </div>
      </div>

      {total === 0 ? (
        <div className="flex flex-col items-center justify-center h-52 text-[var(--text-muted)]">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mb-3 opacity-40">
            <circle cx="12" cy="12" r="10"/><path d="M8 12h8"/>
          </svg>
          <p className="text-sm">No classifications yet</p>
        </div>
      ) : (
        <div className="flex items-center gap-6">
          <div className="flex-1">
            <ResponsiveContainer width="100%" height={200}>
              <PieChart>
                <Pie
                  data={data}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={4}
                  dataKey="value"
                  strokeWidth={0}
                >
                  {data.map((_, i) => (
                    <Cell key={i} fill={COLORS[i]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(17, 24, 39, 0.95)",
                    border: "1px solid var(--border-light)",
                    borderRadius: "10px",
                    boxShadow: "0 8px 32px rgba(0,0,0,0.3)",
                    padding: "8px 14px",
                    fontFamily: "Inter",
                    fontSize: "13px",
                    color: "var(--text-primary)",
                  }}
                  itemStyle={{ color: "var(--text-secondary)" }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-4 min-w-[140px]">
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-[#f43f5e] shadow-sm shadow-rose-500/30" />
              <div>
                <p className="text-sm font-semibold text-[var(--text-primary)]">
                  {metrics.injection_count.toLocaleString()}
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  Injection ({injectionPct}%)
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <div className="w-3 h-3 rounded-full bg-[#10b981] shadow-sm shadow-emerald-500/30" />
              <div>
                <p className="text-sm font-semibold text-[var(--text-primary)]">
                  {metrics.benign_count.toLocaleString()}
                </p>
                <p className="text-xs text-[var(--text-muted)]">
                  Benign ({benignPct}%)
                </p>
              </div>
            </div>
            <div className="pt-2 border-t border-[var(--border)]">
              <p className="text-xs text-[var(--text-muted)]">
                Detection Rate
              </p>
              <p className="text-lg font-bold text-[var(--text-primary)]">
                {injectionPct}%
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
