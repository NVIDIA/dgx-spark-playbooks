import type { Metrics } from "../hooks/useMetrics";

interface Props {
  metrics: Metrics;
}

interface StatCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: React.ReactNode;
  accent?: string;
}

function StatCard({ title, value, unit, icon, accent = "indigo" }: StatCardProps) {
  const accentMap: Record<string, string> = {
    indigo: "from-indigo-500/10 to-transparent border-indigo-500/10",
    emerald: "from-emerald-500/10 to-transparent border-emerald-500/10",
    amber: "from-amber-500/10 to-transparent border-amber-500/10",
    violet: "from-violet-500/10 to-transparent border-violet-500/10",
  };
  const iconBgMap: Record<string, string> = {
    indigo: "bg-indigo-500/10 text-indigo-400",
    emerald: "bg-emerald-500/10 text-emerald-400",
    amber: "bg-amber-500/10 text-amber-400",
    violet: "bg-violet-500/10 text-violet-400",
  };

  return (
    <div className={`card p-6 bg-gradient-to-br ${accentMap[accent]}`}>
      <div className="flex items-start justify-between mb-4">
        <span className="text-[13px] font-medium text-[var(--text-secondary)]">{title}</span>
        <div className={`w-8 h-8 rounded-lg ${iconBgMap[accent]} flex items-center justify-center`}>
          {icon}
        </div>
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-3xl font-bold tracking-tight text-[var(--text-primary)]">
          {value}
        </span>
        {unit && (
          <span className="text-sm font-medium text-[var(--text-muted)]">{unit}</span>
        )}
      </div>
    </div>
  );
}

export default function MetricsPanel({ metrics }: Props) {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-5">
      <StatCard
        title="Throughput"
        value={metrics.requests_per_second}
        unit="req/s"
        accent="indigo"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
          </svg>
        }
      />
      <StatCard
        title="Avg Latency"
        value={metrics.avg_latency_ms}
        unit="ms"
        accent="amber"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
          </svg>
        }
      />
      <StatCard
        title="Total Requests"
        value={metrics.total_requests.toLocaleString()}
        accent="emerald"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
          </svg>
        }
      />
      <StatCard
        title="GPU Utilization"
        value={metrics.gpu_utilization === "N/A" ? "N/A" : metrics.gpu_utilization}
        unit={metrics.gpu_utilization === "N/A" ? undefined : "%"}
        accent="violet"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>
            <line x1="9" y1="2" x2="9" y2="4"/><line x1="15" y1="2" x2="15" y2="4"/>
            <line x1="9" y1="20" x2="9" y2="22"/><line x1="15" y1="20" x2="15" y2="22"/>
            <line x1="20" y1="9" x2="22" y2="9"/><line x1="20" y1="15" x2="22" y2="15"/>
            <line x1="2" y1="9" x2="4" y2="9"/><line x1="2" y1="15" x2="4" y2="15"/>
          </svg>
        }
      />
    </div>
  );
}
