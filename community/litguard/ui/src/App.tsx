import { useMetrics } from "./hooks/useMetrics";
import MetricsPanel from "./components/MetricsPanel";
import ClassificationChart from "./components/ClassificationChart";
import RequestsTable from "./components/RequestsTable";
import ModelStatus from "./components/ModelStatus";

export default function App() {
  const { metrics, history, error } = useMetrics(2000);

  return (
    <div className="min-h-screen">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-[var(--border)] bg-[var(--bg-primary)]/80 backdrop-blur-xl">
        <div className="max-w-[1400px] mx-auto px-8 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Logo mark */}
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-semibold tracking-tight text-[var(--text-primary)]">
                LitGuard
              </h1>
              <p className="text-xs text-[var(--text-muted)] -mt-0.5">
                Prompt Injection Detection
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            {error && (
              <div className="flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full bg-[var(--danger-bg)] text-[var(--danger)] border border-[var(--danger)]/20">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--danger)] animate-pulse" />
                Disconnected
              </div>
            )}
            {!error && metrics && (
              <div className="flex items-center gap-2 text-xs font-medium px-3 py-1.5 rounded-full bg-[var(--success-bg)] text-[var(--success)] border border-[var(--success)]/20">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)]" />
                Live
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Main content */}
      <main className="max-w-[1400px] mx-auto px-8 py-8">
        {error && !metrics ? (
          <div className="flex flex-col items-center justify-center py-32">
            <div className="w-16 h-16 rounded-2xl bg-[var(--danger-bg)] flex items-center justify-center mb-6">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--danger)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10"/>
                <line x1="15" y1="9" x2="9" y2="15"/>
                <line x1="9" y1="9" x2="15" y2="15"/>
              </svg>
            </div>
            <p className="text-[var(--text-primary)] text-lg font-medium mb-2">Cannot connect to backend</p>
            <p className="text-[var(--text-muted)] text-sm">{error}</p>
          </div>
        ) : metrics ? (
          <div className="space-y-8">
            <MetricsPanel metrics={metrics} />

            <div className="grid grid-cols-1 lg:grid-cols-5 gap-8">
              <div className="lg:col-span-3">
                <ClassificationChart metrics={metrics} />
              </div>
              <div className="lg:col-span-2">
                <ModelStatus metrics={metrics} />
              </div>
            </div>

            <RequestsTable history={history} />
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-32">
            <div className="w-8 h-8 border-2 border-[var(--accent)] border-t-transparent rounded-full animate-spin mb-4" />
            <p className="text-[var(--text-muted)] text-sm">Connecting to server...</p>
          </div>
        )}
      </main>
    </div>
  );
}
