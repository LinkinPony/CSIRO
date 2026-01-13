import { Link, useParams } from "react-router-dom";
import { useEffect, useMemo, useState } from "react";

import { getHealth, listTrials } from "../api/endpoints";
import type { TrialSummary } from "../api/types";

export default function ExperimentPage() {
  const { expName } = useParams();
  const exp = expName ? decodeURIComponent(expName) : "";

  const [pollSeconds, setPollSeconds] = useState<number>(8);
  const [trials, setTrials] = useState<TrialSummary[]>([]);
  const [error, setError] = useState<string>("");
  const [filter, setFilter] = useState<string>("");
  const [status, setStatus] = useState<string>("ALL");
  const [sortKey, setSortKey] = useState<string>("best");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((h) => {
        if (!cancelled) setPollSeconds(h.poll_seconds ?? 8);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const refresh = async () => {
      if (!exp) return;
      try {
        const rows = await listTrials(exp);
        if (!cancelled) {
          setTrials(rows);
          setError("");
        }
      } catch (e: unknown) {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e || "Unknown error"));
        }
      }
    };

    void refresh();
    timer = window.setInterval(() => void refresh(), Math.max(2, pollSeconds) * 1000);

    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
  }, [exp, pollSeconds]);

  const metric = trials[0]?.metric ?? "—";
  const mode = trials[0]?.mode ?? "—";

  const paramColumns = useMemo(() => {
    const freq = new Map<string, number>();
    for (const t of trials) {
      for (const k of Object.keys(t.params || {})) {
        freq.set(k, (freq.get(k) ?? 0) + 1);
      }
    }
    return Array.from(freq.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 8)
      .map(([k]) => k);
  }, [trials]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    return trials.filter((t) => {
      if (status !== "ALL" && t.status !== status) return false;
      if (!q) return true;
      if (t.trial_id.toLowerCase().includes(q)) return true;
      if (t.trial_dirname.toLowerCase().includes(q)) return true;
      // param search (key or value)
      for (const [k, v] of Object.entries(t.params || {})) {
        const vv = String(v);
        if (k.toLowerCase().includes(q) || vv.toLowerCase().includes(q)) return true;
      }
      return false;
    });
  }, [trials, filter, status]);

  const sorted = useMemo(() => {
    const dirMul = sortDir === "asc" ? 1 : -1;
    const getVal = (t: TrialSummary): number | string | null => {
      const vr2 = t.pinned_metrics?.["val_r2"];
      const vloss5d = t.pinned_metrics?.["val_loss_5d_weighted"];
      const tloss5d = t.pinned_metrics?.["train_loss_5d_weighted"];
      if (sortKey === "trial_id") return t.trial_id;
      if (sortKey === "status") return t.status;
      if (sortKey === "best") return t.best;
      if (sortKey === "last") return t.last;
      if (sortKey === "best_epoch") return t.best_epoch;
      if (sortKey === "last_epoch") return t.last_epoch;
      if (sortKey === "val_r2_best") return vr2?.best ?? null;
      if (sortKey === "val_r2_last") return vr2?.last ?? null;
      if (sortKey === "train_loss_5d_weighted_best") return tloss5d?.best ?? null;
      if (sortKey === "train_loss_5d_weighted_last") return tloss5d?.last ?? null;
      if (sortKey === "val_loss_5d_weighted_best") return vloss5d?.best ?? null;
      if (sortKey === "val_loss_5d_weighted_last") return vloss5d?.last ?? null;
      if (sortKey === "time_total_s") return t.time_total_s;
      return (t as any)[sortKey] ?? null;
    };
    const rows = [...filtered];
    rows.sort((a, b) => {
      const va = getVal(a);
      const vb = getVal(b);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "number" && typeof vb === "number") {
        return (va - vb) * dirMul;
      }
      return String(va).localeCompare(String(vb)) * dirMul;
    });
    return rows;
  }, [filtered, sortDir, sortKey]);

  const toggleSort = (key: string) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      if (key === "trial_id" || key === "status") {
        setSortDir("asc");
      } else if (key.startsWith("val_loss_5d_weighted_") || key.startsWith("train_loss_5d_weighted_")) {
        setSortDir("asc");
      } else {
        setSortDir("desc");
      }
    }
  };

  const fmt = (v: unknown): string => {
    if (v == null) return "—";
    if (typeof v === "number") return Number.isFinite(v) ? v.toFixed(6) : String(v);
    if (Array.isArray(v)) return `[${v.map((x) => String(x)).join(", ")}]`;
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
  };

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="card">
        <div style={{ fontWeight: 700 }}>Experiment</div>
        <div className="mono" style={{ marginTop: 8 }}>
          <div>{exp}</div>
          <div>
            metric/mode: {metric} / {mode}
          </div>
          <div>trials: {trials.length}</div>
        </div>
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
          placeholder="Search trial_id / dirname / params"
          style={{ minWidth: 340 }}
        />
        <select value={status} onChange={(e) => setStatus(e.target.value)}>
          <option value="ALL">All statuses</option>
          <option value="RUNNING">RUNNING</option>
          <option value="TERMINATED">TERMINATED</option>
          <option value="ERROR">ERROR</option>
          <option value="NO_DATA">NO_DATA</option>
        </select>
        <div className="muted" style={{ alignSelf: "center" }}>
          showing {sorted.length} / {trials.length}
        </div>
      </div>

      <div className="card" style={{ overflowX: "auto" }}>
        <table className="table">
          <thead>
            <tr>
              <th>
                <button onClick={() => toggleSort("trial_id")}>trial_id</button>
              </th>
              <th>
                <button onClick={() => toggleSort("status")}>status</button>
              </th>
              <th>
                <button onClick={() => toggleSort("best")}>best</button>
              </th>
              <th>
                <button onClick={() => toggleSort("best_epoch")}>best_epoch</button>
              </th>
              <th>
                <button onClick={() => toggleSort("last")}>last</button>
              </th>
              <th>
                <button onClick={() => toggleSort("last_epoch")}>last_epoch</button>
              </th>
              <th>
                <button onClick={() => toggleSort("val_r2_best")}>val_r2(best)</button>
              </th>
              <th>
                <button onClick={() => toggleSort("val_r2_last")}>val_r2(last)</button>
              </th>
              <th>
                <button onClick={() => toggleSort("train_loss_5d_weighted_best")}>train_loss_5d_weighted(best)</button>
              </th>
              <th>
                <button onClick={() => toggleSort("train_loss_5d_weighted_last")}>train_loss_5d_weighted(last)</button>
              </th>
              <th>
                <button onClick={() => toggleSort("val_loss_5d_weighted_best")}>val_loss_5d_weighted(best)</button>
              </th>
              <th>
                <button onClick={() => toggleSort("val_loss_5d_weighted_last")}>val_loss_5d_weighted(last)</button>
              </th>
              <th>
                <button onClick={() => toggleSort("time_total_s")}>time_total_s</button>
              </th>
              {paramColumns.map((k) => (
                <th key={k} title={k}>
                  <span className="mono">{k}</span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sorted.map((t) => {
              const vr2 = t.pinned_metrics?.["val_r2"];
              const vloss5d = t.pinned_metrics?.["val_loss_5d_weighted"];
              const tloss5d = t.pinned_metrics?.["train_loss_5d_weighted"];
              return (
                <tr key={t.trial_dirname}>
                <td className="mono">
                  <Link
                    to={`/experiments/${encodeURIComponent(exp)}/trials/${encodeURIComponent(
                      t.trial_dirname,
                    )}`}
                  >
                    {t.trial_id}
                  </Link>
                </td>
                <td>
                  <span className="pill">{t.status}</span>
                </td>
                <td className="mono">{t.best == null ? "—" : t.best.toFixed(6)}</td>
                <td className="mono">{t.best_epoch ?? "—"}</td>
                <td className="mono">{t.last == null ? "—" : t.last.toFixed(6)}</td>
                <td className="mono">{t.last_epoch ?? "—"}</td>
                <td className="mono">{vr2?.best == null ? "—" : vr2.best.toFixed(6)}</td>
                <td className="mono">{vr2?.last == null ? "—" : vr2.last.toFixed(6)}</td>
                <td className="mono">{tloss5d?.best == null ? "—" : tloss5d.best.toFixed(6)}</td>
                <td className="mono">{tloss5d?.last == null ? "—" : tloss5d.last.toFixed(6)}</td>
                <td className="mono">{vloss5d?.best == null ? "—" : vloss5d.best.toFixed(6)}</td>
                <td className="mono">{vloss5d?.last == null ? "—" : vloss5d.last.toFixed(6)}</td>
                <td className="mono">
                  {t.time_total_s == null ? "—" : t.time_total_s.toFixed(1)}
                </td>
                {paramColumns.map((k) => (
                  <td key={k} className="mono">
                    {fmt((t.params || {})[k])}
                  </td>
                ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

