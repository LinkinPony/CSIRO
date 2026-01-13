import yaml from "js-yaml";
import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  getHealth,
  getLightningMetrics,
  getTrialTrainYaml,
  getTrialTimeseries,
  listTrialFiles,
  readTrialFile,
} from "../api/endpoints";
import type { TrialFileEntry } from "../api/types";

function emaSmooth(xs: Array<{ x: number; y: number }>, alpha: number): Array<{ x: number; y: number }> {
  const a = Math.min(0.99, Math.max(0, alpha));
  if (!xs.length || a <= 0) return xs;
  const out: Array<{ x: number; y: number }> = [];
  let s = xs[0].y;
  out.push({ x: xs[0].x, y: s });
  for (let i = 1; i < xs.length; i++) {
    s = a * xs[i].y + (1 - a) * s;
    out.push({ x: xs[i].x, y: s });
  }
  return out;
}

type XY = { x: number; y: number };

const DEFAULT_TS_EXTRA_METRICS = ["train_loss_5d_weighted", "val_loss_5d_weighted"] as const;

function toNumber(x: unknown): number | null {
  if (typeof x === "number") return Number.isFinite(x) ? x : null;
  if (typeof x === "string") {
    const s = x.trim();
    if (!s) return null;
    const n = Number(s);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

function quantileSorted(sorted: number[], q: number): number {
  if (!sorted.length) return NaN;
  const qq = Math.min(1, Math.max(0, q));
  const pos = (sorted.length - 1) * qq;
  const base = Math.floor(pos);
  const rest = pos - base;
  const v0 = sorted[base] ?? sorted[0];
  const v1 = sorted[base + 1];
  if (typeof v1 !== "number") return v0;
  return v0 + rest * (v1 - v0);
}

function computeIqrBounds(ys: number[], iqrK: number): { lo: number; hi: number } | null {
  if (ys.length < 4) return null;
  const s = [...ys].sort((a, b) => a - b);
  const q1 = quantileSorted(s, 0.25);
  const q3 = quantileSorted(s, 0.75);
  const iqr = q3 - q1;
  if (!Number.isFinite(iqr) || iqr <= 0) return null;
  const k = Math.max(0, iqrK);
  return { lo: q1 - k * iqr, hi: q3 + k * iqr };
}

function robustFilterSeries(
  pts: XY[],
  opts?: { hardAbs?: number; iqrK?: number; padFrac?: number },
): { points: XY[]; yDomain: [number, number] | null; dropped: number } {
  const hardAbs = opts?.hardAbs ?? 1e8; // also catches sentinel-like defaults (e.g. -1e9)
  const iqrK = opts?.iqrK ?? 6;
  const padFrac = opts?.padFrac ?? 0.05;

  const ysForBounds: number[] = [];
  for (const p of pts) {
    if (!Number.isFinite(p.y)) continue;
    if (Math.abs(p.y) > hardAbs) continue;
    ysForBounds.push(p.y);
  }
  if (!ysForBounds.length) return { points: [], yDomain: null, dropped: pts.length };

  const bounds = computeIqrBounds(ysForBounds, iqrK);

  let dropped = 0;
  const out: XY[] = [];
  for (const p of pts) {
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) {
      dropped++;
      continue;
    }
    if (Math.abs(p.y) > hardAbs) {
      dropped++;
      continue;
    }
    if (bounds && (p.y < bounds.lo || p.y > bounds.hi)) {
      dropped++;
      continue;
    }
    out.push(p);
  }

  // If the IQR bounds were too aggressive (rare), fall back to hard-abs filtering only.
  if (!out.length && pts.length) {
    dropped = 0;
    for (const p of pts) {
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) {
        dropped++;
        continue;
      }
      if (Math.abs(p.y) > hardAbs) {
        dropped++;
        continue;
      }
      out.push(p);
    }
  }

  if (!out.length) return { points: [], yDomain: null, dropped: pts.length };

  let yMin = out[0].y;
  let yMax = out[0].y;
  for (let i = 1; i < out.length; i++) {
    const y = out[i].y;
    if (y < yMin) yMin = y;
    if (y > yMax) yMax = y;
  }
  const span0 = yMax - yMin;
  const span = span0 > 0 ? span0 : Math.max(1e-3, Math.abs(yMin) * 0.1);
  const pad = span * padFrac;
  const yDomain: [number, number] = [yMin - pad, yMax + pad];
  return { points: out, yDomain, dropped };
}

function formatTick(v: unknown): string {
  const n = toNumber(v);
  if (n == null) return String(v ?? "");
  const a = Math.abs(n);
  if (a === 0) return "0";
  if (a < 1e-3 || a >= 1e4) return n.toExponential(2);
  if (a < 1) return n.toFixed(4).replace(/0+$/, "").replace(/\.$/, "");
  if (a < 10) return n.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
  if (a < 100) return n.toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
  if (a < 1000) return n.toFixed(1).replace(/0+$/, "").replace(/\.$/, "");
  return n.toFixed(0);
}

export default function TrialPage() {
  const { expName, trialDir } = useParams();
  const exp = expName ? decodeURIComponent(expName) : "";
  const td = trialDir ? decodeURIComponent(trialDir) : "";

  const [pollSeconds, setPollSeconds] = useState<number>(8);
  const [error, setError] = useState<string>("");

  const [availableMetrics, setAvailableMetrics] = useState<string[]>([]);
  const [selectedMetric, setSelectedMetric] = useState<string>("val_r2");
  const [points, setPoints] = useState<Array<Record<string, unknown>>>([]);
  const [params, setParams] = useState<Record<string, unknown>>({});

  // Lightning (TensorBoard-like) scalars
  const [ltAvail, setLtAvail] = useState<string[]>([]);
  const [ltSelected, setLtSelected] = useState<string[]>([
    "train_loss",
    "train_loss_5d_weighted",
    "val_loss",
    "val_loss_5d_weighted",
    "val_r2",
  ]);
  const [ltPoints, setLtPoints] = useState<Array<Record<string, unknown>>>([]);
  const [ltXKey, setLtXKey] = useState<"step" | "epoch">("step");
  const [ltSmooth, setLtSmooth] = useState<number>(0.0);
  const [ltFilter, setLtFilter] = useState<string>("");
  const [ltCsvRel, setLtCsvRel] = useState<string>("");

  const [files, setFiles] = useState<TrialFileEntry[]>([]);
  const [trainYamlRaw, setTrainYamlRaw] = useState<string>("");
  const [trainYamlSource, setTrainYamlSource] = useState<string>("");
  const [trainYamlInferred, setTrainYamlInferred] = useState<boolean>(false);
  const [trainYamlAppliedParamsCount, setTrainYamlAppliedParamsCount] = useState<number>(0);
  const [trainLogTail, setTrainLogTail] = useState<string>("");
  const [openFilePath, setOpenFilePath] = useState<string>("");
  const [openFileContent, setOpenFileContent] = useState<string>("");

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
      if (!exp || !td) return;
      try {
        const metricsReq = Array.from(
          new Set([selectedMetric || "val_r2", ...DEFAULT_TS_EXTRA_METRICS].filter((m) => String(m || "").trim())),
        );
        const resp = await getTrialTimeseries(exp, td, metricsReq);
        if (cancelled) return;
        setParams(resp.params || {});
        setAvailableMetrics(resp.available_metrics || []);
        setPoints(resp.points || []);
        setError("");

        // If selected metric isn't available, fall back to the first available.
        if (
          selectedMetric &&
          resp.available_metrics &&
          resp.available_metrics.length > 0 &&
          !resp.available_metrics.includes(selectedMetric)
        ) {
          setSelectedMetric(resp.available_metrics[0]);
        }
      } catch (e: unknown) {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e || "Unknown error"));
      }
    };

    void refresh();
    timer = window.setInterval(() => void refresh(), Math.max(2, pollSeconds) * 1000);
    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
  }, [exp, td, pollSeconds, selectedMetric]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;

    const refresh = async () => {
      if (!exp || !td) return;
      try {
        const resp = await getLightningMetrics(exp, td, ltSelected, 10000);
        if (cancelled) return;
        setLtAvail(resp.available_columns || []);
        setLtPoints(resp.points || []);
        setLtCsvRel(resp.csv_relpath || "");

        // Keep selection valid.
        if (resp.available_columns && resp.available_columns.length) {
          const keep = ltSelected.filter((t) => resp.available_columns.includes(t));
          if (!keep.length) {
            setLtSelected(resp.requested_columns?.length ? resp.requested_columns : resp.available_columns.slice(0, 3));
          } else if (keep.length !== ltSelected.length) {
            setLtSelected(keep);
          }
        }
      } catch {
        if (!cancelled) {
          setLtAvail([]);
          setLtPoints([]);
          setLtCsvRel("");
        }
      }
    };

    void refresh();
    timer = window.setInterval(() => void refresh(), Math.max(2, pollSeconds) * 1000);
    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
  }, [exp, td, pollSeconds, ltSelected]);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      if (!exp || !td) return;
      try {
        const f = await listTrialFiles(exp, td);
        if (!cancelled) setFiles(f);
      } catch {
        // ignore
      }
    };
    void refresh();
    return () => {
      cancelled = true;
    };
  }, [exp, td]);

  useEffect(() => {
    let cancelled = false;
    const refresh = async () => {
      if (!exp || !td) return;
      try {
        const r = await getTrialTrainYaml(exp, td);
        if (cancelled) return;
        setTrainYamlRaw(r.yaml || "");
        const srcParts: string[] = [];
        if (r.source_kind) srcParts.push(r.source_kind);
        if (r.source_trial_dirname && r.source_trial_dirname !== td) srcParts.push(r.source_trial_dirname);
        if (r.source_relpath) srcParts.push(r.source_relpath);
        setTrainYamlSource(srcParts.length ? srcParts.join(" :: ") : "");
        setTrainYamlInferred(Boolean(r.inferred));
        setTrainYamlAppliedParamsCount(Number(r.applied_params_count ?? 0));
      } catch {
        if (cancelled) return;
        setTrainYamlRaw("");
        setTrainYamlSource("");
        setTrainYamlInferred(false);
        setTrainYamlAppliedParamsCount(0);
      }
    };
    void refresh();
    return () => {
      cancelled = true;
    };
  }, [exp, td]);

  useEffect(() => {
    let cancelled = false;
    let timer: number | undefined;
    const refresh = async () => {
      if (!exp || !td) return;
      try {
        const r = await readTrialFile(exp, td, "run/logs/train.log", { tailLines: 2000 });
        if (!cancelled) setTrainLogTail(r.content || "");
      } catch {
        if (!cancelled) setTrainLogTail("");
      }
    };
    void refresh();
    timer = window.setInterval(() => void refresh(), Math.max(5, pollSeconds) * 1000);
    return () => {
      cancelled = true;
      if (timer) window.clearInterval(timer);
    };
  }, [exp, td, pollSeconds]);

  const parsedTrainYaml = useMemo(() => {
    if (!trainYamlRaw.trim()) return null;
    try {
      return yaml.load(trainYamlRaw);
    } catch {
      return null;
    }
  }, [trainYamlRaw]);

  const xKey = useMemo(() => {
    const hasIter = points.some((p) => typeof p.training_iteration === "number");
    if (hasIter) return "training_iteration";
    const hasEpoch = points.some((p) => typeof p.epoch === "number");
    return hasEpoch ? "epoch" : "row_idx";
  }, [points]);

  const tsCharts = useMemo(() => {
    const metrics = Array.from(
      new Set([selectedMetric || "val_r2", ...DEFAULT_TS_EXTRA_METRICS].filter((m) => String(m || "").trim())),
    ) as string[];

    const out: Record<
      string,
      { data: Array<Record<string, unknown>>; yDomain: [number, number] | null; dropped: number }
    > = {};

    for (const m of metrics) {
      const series: XY[] = [];
      for (const p of points) {
        const x = toNumber((p as any)?.[xKey]);
        const y = toNumber((p as any)?.[m]);
        if (x == null || y == null) continue;
        series.push({ x, y });
      }
      series.sort((a, b) => a.x - b.x);
      const filtered = robustFilterSeries(series, { hardAbs: 1e8, iqrK: 6, padFrac: 0.05 });
      const data = filtered.points.map((pt) => ({ [xKey]: pt.x, [m]: pt.y } as Record<string, unknown>));
      out[m] = { data, yDomain: filtered.yDomain, dropped: filtered.dropped };
    }
    return out;
  }, [points, xKey, selectedMetric]);

  const mainMetric = selectedMetric || "val_r2";
  const mainChart = tsCharts[mainMetric] ?? { data: [], yDomain: null, dropped: 0 };
  const extraMetrics = DEFAULT_TS_EXTRA_METRICS.filter((m) => m !== mainMetric);

  const openFile = async (relPath: string) => {
    if (!exp || !td) return;
    setOpenFilePath(relPath);
    setOpenFileContent("Loading…");
    try {
      const r = await readTrialFile(exp, td, relPath, { tailLines: relPath.endsWith(".log") ? 4000 : undefined });
      setOpenFileContent(r.content || "");
    } catch (e: unknown) {
      setOpenFileContent(e instanceof Error ? e.message : String(e || "Failed to read file"));
    }
  };

  const copyTrainYaml = async () => {
    if (!trainYamlRaw.trim()) return;
    try {
      await navigator.clipboard.writeText(trainYamlRaw);
    } catch {
      // Fallback for older browsers / permissions.
      const ta = document.createElement("textarea");
      ta.value = trainYamlRaw;
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
  };

  const downloadTrainYaml = () => {
    if (!trainYamlRaw.trim()) return;
    const safeName = (s: string) => s.replace(/[^a-zA-Z0-9._-]+/g, "_").slice(0, 160);
    const filename = `${safeName(exp || "exp")}__${safeName(td || "trial")}__train.yaml`;
    const blob = new Blob([trainYamlRaw], { type: "text/yaml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const ltGroups = useMemo(() => {
    const q = ltFilter.trim().toLowerCase();
    const cols = q ? ltAvail.filter((c) => c.toLowerCase().includes(q)) : ltAvail;
    const groupKey = (c: string): string => {
      if (c.includes("_")) return c.split("_")[0];
      if (c.includes("-")) return c.split("-")[0];
      return "other";
    };
    const groups = new Map<string, string[]>();
    for (const c of cols) {
      const g = groupKey(c);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g)!.push(c);
    }
    return Array.from(groups.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  }, [ltAvail, ltFilter]);

  const ltSeries = useMemo(() => {
    const xKey = ltXKey;
    const series: Record<string, Array<{ x: number; y: number }>> = {};
    for (const tag of ltSelected) series[tag] = [];
    for (const p of ltPoints) {
      const xRaw = p[xKey];
      const x = typeof xRaw === "number" ? xRaw : typeof xRaw === "string" ? Number(xRaw) : null;
      if (x == null || !Number.isFinite(x)) continue;
      for (const tag of ltSelected) {
        const yRaw = p[tag];
        const y = typeof yRaw === "number" ? yRaw : typeof yRaw === "string" ? Number(yRaw) : null;
        if (y == null || !Number.isFinite(y)) continue;
        series[tag].push({ x, y });
      }
    }
    // Ensure monotonic x for charts.
    for (const tag of Object.keys(series)) {
      series[tag].sort((a, b) => a.x - b.x);
    }
    return series;
  }, [ltPoints, ltSelected, ltXKey]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="card">
        <div style={{ fontWeight: 700 }}>Trial</div>
        <div className="mono" style={{ marginTop: 8 }}>
          <div>experiment: {exp}</div>
          <div>trial_dir: {td}</div>
        </div>
        {error ? (
          <div className="mono" style={{ marginTop: 8, color: "crimson" }}>
            {error}
          </div>
        ) : null}
      </div>

      <div className="card">
        <div className="row" style={{ alignItems: "center" }}>
          <div style={{ fontWeight: 700 }}>Scalars (TensorBoard-like)</div>
          <div className="muted" style={{ alignSelf: "center" }}>
            source: {ltCsvRel ? <span className="mono">{ltCsvRel}</span> : "metrics.csv not found"}
          </div>
          <div style={{ flex: 1 }} />
          <span className="pill">tags: {ltAvail.length}</span>
          <span className="pill">selected: {ltSelected.length}</span>
          <select value={ltXKey} onChange={(e) => setLtXKey(e.target.value as any)}>
            <option value="step">x=step</option>
            <option value="epoch">x=epoch</option>
          </select>
          <label className="mono" style={{ display: "flex", gap: 8, alignItems: "center" }}>
            smooth
            <input
              type="range"
              min={0}
              max={0.95}
              step={0.05}
              value={ltSmooth}
              onChange={(e) => setLtSmooth(Number(e.target.value))}
            />
            {ltSmooth.toFixed(2)}
          </label>
        </div>

        <div className="row" style={{ marginTop: 10, alignItems: "center" }}>
          <input
            value={ltFilter}
            onChange={(e) => setLtFilter(e.target.value)}
            placeholder="Search tags (e.g. val_loss, train_loss, lr-AdamW)"
            style={{ minWidth: 360 }}
          />
          <button
            onClick={() =>
              setLtSelected(
                ["train_loss", "train_loss_5d_weighted", "val_loss", "val_loss_5d_weighted", "val_r2"].filter((t) =>
                  ltAvail.includes(t),
                ),
              )
            }
            style={{ padding: "6px 10px", borderRadius: 8, cursor: "pointer" }}
          >
            Default tags
          </button>
          <button
            onClick={() => setLtSelected(ltAvail.filter((t) => t.startsWith("val_")).slice(0, 12))}
            style={{ padding: "6px 10px", borderRadius: 8, cursor: "pointer" }}
          >
            Top val_*
          </button>
          <button
            onClick={() => setLtSelected(ltAvail.filter((t) => t.startsWith("train_")).slice(0, 12))}
            style={{ padding: "6px 10px", borderRadius: 8, cursor: "pointer" }}
          >
            Top train_*
          </button>
        </div>

        <div className="grid" style={{ marginTop: 10 }}>
          <div className="card" style={{ maxHeight: 360, overflow: "auto" }}>
            <div style={{ fontWeight: 700 }}>Tags</div>
            {ltGroups.length ? (
              ltGroups.map(([g, cols]) => (
                <div key={g} style={{ marginTop: 10 }}>
                  <div className="muted mono">{g}</div>
                  {cols.slice(0, 80).map((c) => {
                    const checked = ltSelected.includes(c);
                    return (
                      <label key={c} className="mono" style={{ display: "flex", gap: 8, marginTop: 6 }}>
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setLtSelected([...ltSelected, c]);
                            } else {
                              setLtSelected(ltSelected.filter((x) => x !== c));
                            }
                          }}
                        />
                        <span title={c} style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {c}
                        </span>
                      </label>
                    );
                  })}
                </div>
              ))
            ) : (
              <div className="muted" style={{ marginTop: 8 }}>
                No tags found for this trial.
              </div>
            )}
          </div>

          <div className="card">
            <div style={{ fontWeight: 700 }}>Charts</div>
            {ltSelected.length ? (
              <div className="grid" style={{ marginTop: 10 }}>
                {ltSelected.slice(0, 12).map((tag) => {
                  const raw = ltSeries[tag] || [];
                  const filtered = robustFilterSeries(raw, { hardAbs: 1e8, iqrK: 6, padFrac: 0.05 });
                  const sm = emaSmooth(filtered.points, ltSmooth);
                  const data = filtered.points.map((p, i) => ({
                    x: p.x,
                    y: p.y,
                    y_smooth: sm[i]?.y ?? p.y,
                  }));
                  return (
                    <div key={tag} className="card">
                      <div className="mono" title={tag} style={{ fontWeight: 700, overflow: "hidden" }}>
                        {tag}
                      </div>
                      <div className="muted" style={{ marginTop: 4 }}>
                        points: {data.length}
                        {filtered.dropped ? (
                          <span className="mono" style={{ marginLeft: 8 }}>
                            (clipped: {filtered.dropped})
                          </span>
                        ) : null}
                      </div>
                      <div style={{ height: 220, marginTop: 8 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="x" type="number" />
                            <YAxis domain={filtered.yDomain ?? ["auto", "auto"]} tickFormatter={formatTick} />
                            <Tooltip />
                            <Legend />
                            <Line
                              type="monotone"
                              dataKey="y"
                              name="raw"
                              dot={false}
                              strokeWidth={1}
                              stroke="#8884d8"
                              isAnimationActive={false}
                              connectNulls
                            />
                            {ltSmooth > 0 ? (
                              <Line
                                type="monotone"
                                dataKey="y_smooth"
                                name="smooth"
                                dot={false}
                                strokeWidth={2}
                                stroke="#82ca9d"
                                isAnimationActive={false}
                                connectNulls
                              />
                            ) : null}
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div className="muted" style={{ marginTop: 8 }}>
                Select tags on the left.
              </div>
            )}
            {ltSelected.length > 12 ? (
              <div className="muted" style={{ marginTop: 8 }}>
                Showing first 12 selected charts (to keep the page responsive).
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <div className="card">
        <div className="row" style={{ alignItems: "center" }}>
          <div style={{ fontWeight: 700 }}>Metrics</div>
          <div style={{ flex: 1 }} />
          <select value={selectedMetric} onChange={(e) => setSelectedMetric(e.target.value)}>
            {availableMetrics.length ? null : <option value="val_r2">val_r2</option>}
            {availableMetrics.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <span className="pill">
            x: <span className="mono">{xKey}</span>
          </span>
          <span className="pill">points: {points.length}</span>
        </div>

        <div className="muted" style={{ marginTop: 6 }}>
          {mainChart.dropped ? <span className="mono">clipped outliers: {mainChart.dropped}</span> : <span className="mono">—</span>}
        </div>

        <div style={{ height: 360, marginTop: 10 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={mainChart.data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={xKey} type="number" />
              <YAxis domain={mainChart.yDomain ?? ["auto", "auto"]} tickFormatter={formatTick} />
              <Tooltip />
              <Line
                type="monotone"
                dataKey={mainMetric}
                dot={false}
                strokeWidth={2}
                stroke="#8884d8"
                isAnimationActive={false}
                connectNulls
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {extraMetrics.length ? (
          <div className="grid" style={{ marginTop: 12 }}>
            {extraMetrics.map((m) => {
              const ch = tsCharts[m];
              if (!ch || !ch.data.length) return null;
              const stroke = m.startsWith("val_") ? "#82ca9d" : "#8884d8";
              return (
                <div key={m} className="card">
                  <div className="mono" title={m} style={{ fontWeight: 700, overflow: "hidden" }}>
                    {m}
                  </div>
                  <div className="muted" style={{ marginTop: 4 }}>
                    points: {ch.data.length}
                    {ch.dropped ? (
                      <span className="mono" style={{ marginLeft: 8 }}>
                        (clipped: {ch.dropped})
                      </span>
                    ) : null}
                  </div>
                  <div style={{ height: 220, marginTop: 8 }}>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={ch.data}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey={xKey} type="number" />
                        <YAxis domain={ch.yDomain ?? ["auto", "auto"]} tickFormatter={formatTick} />
                        <Tooltip />
                        <Line
                          type="monotone"
                          dataKey={m}
                          dot={false}
                          strokeWidth={2}
                          stroke={stroke}
                          isAnimationActive={false}
                          connectNulls
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              );
            })}
          </div>
        ) : null}
      </div>

      <div className="grid">
        <div className="card">
          <div style={{ fontWeight: 700 }}>Params (params.json)</div>
          <pre className="mono" style={{ margin: 0, marginTop: 8, whiteSpace: "pre-wrap" }}>
            {JSON.stringify(params, null, 2)}
          </pre>
        </div>

        <div className="card">
            <div className="row" style={{ alignItems: "center" }}>
              <div style={{ fontWeight: 700 }}>Resolved train config (train.yaml)</div>
              <div style={{ flex: 1 }} />
              {trainYamlInferred ? <span className="pill">inferred</span> : null}
              {trainYamlAppliedParamsCount ? (
                <span className="pill">params_applied: {trainYamlAppliedParamsCount}</span>
              ) : null}
              <button
                onClick={() => void copyTrainYaml()}
                disabled={!trainYamlRaw.trim()}
                style={{ padding: "6px 10px", borderRadius: 8, cursor: "pointer" }}
                title="Copy to clipboard"
              >
                Copy
              </button>
              <button
                onClick={downloadTrainYaml}
                disabled={!trainYamlRaw.trim()}
                style={{ padding: "6px 10px", borderRadius: 8, cursor: "pointer" }}
                title="Download as train.yaml"
              >
                Download
              </button>
            </div>

            {trainYamlSource ? (
              <div className="muted" style={{ marginTop: 6 }}>
                source: <span className="mono">{trainYamlSource}</span>
              </div>
            ) : null}

            {trainYamlRaw ? (
            <>
              <div className="muted" style={{ marginTop: 6 }}>
                Raw YAML (truncated by backend max bytes if huge)
              </div>
              <pre className="mono" style={{ margin: 0, marginTop: 8, whiteSpace: "pre-wrap" }}>
                {trainYamlRaw}
              </pre>
              {parsedTrainYaml != null ? (
                <>
                  <div className="muted" style={{ marginTop: 12 }}>
                    Parsed view (JSON)
                  </div>
                  <pre className="mono" style={{ margin: 0, marginTop: 8, whiteSpace: "pre-wrap" }}>
                    {JSON.stringify(parsedTrainYaml, null, 2)}
                  </pre>
                </>
              ) : null}
            </>
          ) : (
            <div className="muted" style={{ marginTop: 8 }}>
              train.yaml not found for this trial.
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div style={{ fontWeight: 700 }}>Train log tail (run/logs/train.log)</div>
        {trainLogTail ? (
          <pre className="mono" style={{ margin: 0, marginTop: 8, whiteSpace: "pre-wrap" }}>
            {trainLogTail}
          </pre>
        ) : (
          <div className="muted" style={{ marginTop: 8 }}>
            train.log not found (or empty).
          </div>
        )}
      </div>

      <div className="grid">
        <div className="card" style={{ overflowX: "auto" }}>
          <div style={{ fontWeight: 700 }}>Artifacts / files</div>
          <table className="table" style={{ marginTop: 8 }}>
            <thead>
              <tr>
                <th>path</th>
                <th>size</th>
                <th>mtime</th>
              </tr>
            </thead>
            <tbody>
              {files
                .filter((f) => !f.is_dir)
                .slice(0, 200)
                .map((f) => (
                  <tr key={f.path}>
                    <td className="mono">
                      <button
                        onClick={() => void openFile(f.path)}
                        style={{ font: "inherit", background: "none", border: 0, padding: 0, cursor: "pointer" }}
                        title="Open"
                      >
                        {f.path}
                      </button>
                    </td>
                    <td className="mono">{f.size}</td>
                    <td className="mono">{new Date(Math.floor(f.mtime_ns / 1e6)).toLocaleString()}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          {files.length > 200 ? (
            <div className="muted" style={{ marginTop: 8 }}>
              Showing first 200 files (backend limits traversal).
            </div>
          ) : null}
        </div>

        <div className="card">
          <div style={{ fontWeight: 700 }}>Open file</div>
          <div className="mono muted" style={{ marginTop: 6 }}>
            {openFilePath || "—"}
          </div>
          <pre className="mono" style={{ margin: 0, marginTop: 8, whiteSpace: "pre-wrap" }}>
            {openFileContent || "Select a file from the list."}
          </pre>
        </div>
      </div>
    </div>
  );
}

