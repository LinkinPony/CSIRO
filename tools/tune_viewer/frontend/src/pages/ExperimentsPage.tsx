import { Link } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";

import { getHealth, listExperiments, type Health } from "../api/endpoints";
import type { ExperimentSummary } from "../api/types";

export default function ExperimentsPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [filter, setFilter] = useState<string>("");
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((d) => !cancelled && setHealth(d))
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e || "Unknown error"));
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const refresh = async () => {
      try {
        const exps = await listExperiments();
        if (!cancelled) {
          setExperiments(exps);
          setError("");
        }
      } catch (e: unknown) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e || "Unknown error"));
        }
      }
    };

    void refresh();
    const pollSeconds = health?.poll_seconds ?? 8;
    timer = window.setInterval(() => void refresh(), Math.max(2, pollSeconds) * 1000);

    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
  }, [health?.poll_seconds]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return experiments;
    return experiments.filter((e) => e.name.toLowerCase().includes(q));
  }, [experiments, filter]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="card">
        <div style={{ fontWeight: 700 }}>Backend</div>
        {health ? (
          <div className="mono" style={{ marginTop: 8 }}>
            <div>results_root: {health.results_root}</div>
            <div>repo_root: {health.repo_root}</div>
            <div>poll_seconds: {health.poll_seconds}</div>
          </div>
        ) : (
          <div className="muted" style={{ marginTop: 8 }}>
            Loading backend status…
          </div>
        )}
        {error ? (
          <div className="mono" style={{ marginTop: 8, color: "crimson" }}>
            {error}
          </div>
        ) : null}
      </div>

      <div className="row">
        <input
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          placeholder="Filter experiments (e.g. tune-vitdet)"
          style={{ minWidth: 320 }}
        />
        <div className="muted" style={{ alignSelf: "center" }}>
          {filtered.length} / {experiments.length}
        </div>
      </div>

      <div className="grid">
        {filtered.map((e) => {
          const updatedMs = e.last_update_ns ? Math.floor(e.last_update_ns / 1e6) : null;
          const updated = updatedMs ? new Date(updatedMs).toLocaleString() : "—";
          return (
            <Link key={e.name} to={`/experiments/${encodeURIComponent(e.name)}`}>
              <div className="card">
                <div style={{ fontWeight: 700 }}>{e.name}</div>
                <div className="muted" style={{ marginTop: 4 }}>
                  {e.metric} / {e.mode}
                  {e.tune_config_file ? ` (${e.tune_config_file})` : ""}
                </div>
                <div className="row" style={{ marginTop: 8 }}>
                  <span className="pill">trials: {e.n_trials}</span>
                  {Object.entries(e.counts || {})
                    .filter(([, v]) => v)
                    .map(([k, v]) => (
                      <span key={k} className="pill">
                        {k}: {v}
                      </span>
                    ))}
                </div>
                <div className="mono" style={{ marginTop: 10 }}>
                  <div>best: {e.best == null ? "—" : e.best.toFixed(6)}</div>
                  <div>best_trial: {e.best_trial_id ?? "—"}</div>
                  <div>updated: {updated}</div>
                </div>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}

