import type { Metrics } from "../hooks/useMetrics";

interface Props {
  metrics: Metrics;
}

export default function ModelStatus({ metrics }: Props) {
  const models = metrics.models_loaded || [];

  return (
    <div className="card p-6 h-full">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-[15px] font-semibold text-[var(--text-primary)]">
            Active Models
          </h3>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            {models.length} model{models.length !== 1 ? "s" : ""} deployed
          </p>
        </div>
      </div>

      {models.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-52 text-[var(--text-muted)]">
          <p className="text-sm">No models loaded</p>
        </div>
      ) : (
        <div className="space-y-3">
          {models.map((m) => (
            <div
              key={m.name}
              className="group rounded-xl border border-[var(--border)] bg-[var(--bg-primary)]/50 p-4 hover:border-[var(--border-light)] transition-colors"
            >
              <div className="flex items-start justify-between mb-2">
                <span className="text-sm font-semibold text-[var(--text-primary)]">
                  {m.name}
                </span>
                <span className="inline-flex items-center gap-1.5 text-[11px] font-medium px-2.5 py-1 rounded-full bg-[var(--success-bg)] text-[var(--success)] border border-[var(--success)]/15">
                  <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)]" />
                  Running
                </span>
              </div>

              <p className="text-xs text-[var(--text-muted)] font-mono mb-3 break-all leading-relaxed">
                {m.hf_model}
              </p>

              <div className="flex items-center gap-3">
                <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)] bg-[var(--bg-card)] px-2.5 py-1 rounded-md border border-[var(--border)]">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="4" y="4" width="16" height="16" rx="2"/>
                    <rect x="9" y="9" width="6" height="6"/>
                  </svg>
                  {m.device}
                </div>
                <div className="flex items-center gap-1.5 text-xs text-[var(--text-secondary)] bg-[var(--bg-card)] px-2.5 py-1 rounded-md border border-[var(--border)]">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <rect x="2" y="7" width="20" height="14" rx="2" ry="2"/>
                    <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>
                  </svg>
                  Batch {m.batch_size}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
