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
  const [ltSelected, setLtSelected] = useState<string[]>(["train_loss", "val_loss", "val_r2"]);
  const [ltPoints, setLtPoints] = useState<Array<Record<string, unknown>>>([]);
  const [ltXKey, setLtXKey] = useState<"step" | "epoch">("step");
  const [ltSmooth, setLtSmooth] = useState<number>(0.0);
  const [ltFilter, setLtFilter] = useState<string>("");
  const [ltCsvRel, setLtCsvRel] = useState<string>("");

  const [files, setFiles] = useState<TrialFileEntry[]>([]);
  const [trainYamlRaw, setTrainYamlRaw] = useState<string>("");
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
        const resp = await getTrialTimeseries(exp, td, [selectedMetric || "val_r2"]);
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
        const r = await readTrialFile(exp, td, "run/logs/train.yaml");
        if (!cancelled) setTrainYamlRaw(r.content || "");
      } catch {
        if (!cancelled) setTrainYamlRaw("");
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

  const yKeys = useMemo(() => {
    // Plot the selected metric if present; otherwise plot nothing.
    if (!selectedMetric) return [];
    return [selectedMetric];
  }, [selectedMetric]);

  const chartData = useMemo(() => {
    return points.map((p) => p as any);
  }, [points]);

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
            onClick={() => setLtSelected(["train_loss", "val_loss", "val_r2"].filter((t) => ltAvail.includes(t)))}
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
                  const sm = emaSmooth(raw, ltSmooth);
                  const data = raw.map((p, i) => ({
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
                      </div>
                      <div style={{ height: 220, marginTop: 8 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart data={data}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="x" />
                            <YAxis />
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

        <div style={{ height: 360, marginTop: 10 }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey={xKey} />
              <YAxis />
              <Tooltip />
              <Legend />
              {yKeys.map((k, idx) => (
                <Line
                  key={k}
                  type="monotone"
                  dataKey={k}
                  dot={false}
                  strokeWidth={2}
                  stroke={idx === 0 ? "#8884d8" : "#82ca9d"}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid">
        <div className="card">
          <div style={{ fontWeight: 700 }}>Params (params.json)</div>
          <pre className="mono" style={{ margin: 0, marginTop: 8, whiteSpace: "pre-wrap" }}>
            {JSON.stringify(params, null, 2)}
          </pre>
        </div>

        <div className="card">
          <div style={{ fontWeight: 700 }}>Resolved train config (run/logs/train.yaml)</div>
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

