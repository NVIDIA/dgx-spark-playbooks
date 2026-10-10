import { useState, useEffect, useCallback } from "react";

const API_URL = import.meta.env.VITE_API_URL || "";

interface ModelInfo {
  name: string;
  hf_model: string;
  device: string;
  batch_size: number;
}

export interface Metrics {
  total_requests: number;
  requests_per_second: number;
  avg_latency_ms: number;
  injection_count: number;
  benign_count: number;
  gpu_utilization: string;
  models_loaded: ModelInfo[];
}

export interface HistoryRecord {
  timestamp: number;
  input_preview: string;
  model: string;
  label: string;
  score: number;
  latency_ms: number;
}

export function useMetrics(pollInterval = 2000) {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [history, setHistory] = useState<HistoryRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [metricsRes, historyRes] = await Promise.all([
        fetch(`${API_URL}/metrics`),
        fetch(`${API_URL}/api/history`),
      ]);
      if (metricsRes.ok) {
        setMetrics(await metricsRes.json());
      }
      if (historyRes.ok) {
        const data = await historyRes.json();
        setHistory(data.history || []);
      }
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Connection failed");
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, pollInterval);
    return () => clearInterval(id);
  }, [fetchData, pollInterval]);

  return { metrics, history, error };
}
