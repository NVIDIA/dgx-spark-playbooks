import type { HistoryRecord } from "../hooks/useMetrics";

interface Props {
  history: HistoryRecord[];
}

export default function RequestsTable({ history }: Props) {
  const sorted = [...history].reverse();

  return (
    <div className="card overflow-hidden">
      <div className="px-6 py-5 border-b border-[var(--border)] flex items-center justify-between">
        <div>
          <h3 className="text-[15px] font-semibold text-[var(--text-primary)]">
            Recent Requests
          </h3>
          <p className="text-xs text-[var(--text-muted)] mt-0.5">
            Last {sorted.length} classification{sorted.length !== 1 ? "s" : ""}
          </p>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-[var(--border)]">
              <th className="text-left px-6 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                Timestamp
              </th>
              <th className="text-left px-6 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                Input
              </th>
              <th className="text-left px-6 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                Verdict
              </th>
              <th className="text-right px-6 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                Confidence
              </th>
              <th className="text-right px-6 py-3 text-[11px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
                Latency
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border)]/50">
            {sorted.length === 0 ? (
              <tr>
                <td colSpan={5} className="px-6 py-16 text-center">
                  <div className="flex flex-col items-center text-[var(--text-muted)]">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="mb-3 opacity-40">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                      <polyline points="14 2 14 8 20 8"/>
                    </svg>
                    <p className="text-sm">No requests yet</p>
                    <p className="text-xs mt-1 text-[var(--text-muted)]">
                      Send a request to /v1/chat/completions to see results
                    </p>
                  </div>
                </td>
              </tr>
            ) : (
              sorted.slice(0, 50).map((r, i) => (
                <tr
                  key={i}
                  className="group hover:bg-[var(--bg-card-hover)]/50 transition-colors"
                >
                  <td className="px-6 py-3.5 whitespace-nowrap">
                    <span className="text-xs font-mono text-[var(--text-muted)]">
                      {new Date(r.timestamp * 1000).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                        second: "2-digit",
                      })}
                    </span>
                  </td>
                  <td className="px-6 py-3.5 max-w-md">
                    <p
                      className="text-sm text-[var(--text-secondary)] truncate group-hover:text-[var(--text-primary)] transition-colors"
                      title={r.input_preview}
                    >
                      {r.input_preview}
                    </p>
                  </td>
                  <td className="px-6 py-3.5">
                    {r.label === "injection" ? (
                      <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide px-2.5 py-1 rounded-md bg-[var(--danger-bg)] text-[var(--danger)] border border-[var(--danger)]/15">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
                          <path d="M12 2L1 21h22L12 2zm0 4l7.53 13H4.47L12 6z"/>
                        </svg>
                        Injection
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wide px-2.5 py-1 rounded-md bg-[var(--success-bg)] text-[var(--success)] border border-[var(--success)]/15">
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                          <polyline points="20 6 9 17 4 12"/>
                        </svg>
                        Benign
                      </span>
                    )}
                  </td>
                  <td className="px-6 py-3.5 text-right">
                    <span className="text-sm font-mono font-medium text-[var(--text-primary)]">
                      {(r.score * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-3.5 text-right">
                    <span className="text-sm font-mono text-[var(--text-muted)]">
                      {r.latency_ms.toFixed(0)}
                      <span className="text-[10px] ml-0.5">ms</span>
                    </span>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
