import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { listExperiments, listTrials } from "../api/endpoints";
import type { ExperimentSummary, TrialSummary } from "../api/types";

type EffectRow =
  | {
      param: string;
      kind: "categorical";
      n_unique: number;
      score: number;
      best_value: string;
      worst_value: string;
      mean_best: number;
      mean_worst: number;
      delta_mean: number;
      n_best: number;
    }
  | {
      param: string;
      kind: "numeric";
      n_unique: number;
      score: number;
      spearman: number | null;
      x_log10: boolean;
      x_min: number | null;
      x_max: number | null;
    };

function median(xs: number[]): number | null {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  if (s.length % 2 === 1) return s[mid];
  return (s[mid - 1] + s[mid]) / 2;
}

function mean(xs: number[]): number | null {
  if (!xs.length) return null;
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function rank(xs: number[]): number[] {
  const sorted = xs
    .map((v, i) => ({ v, i }))
    .sort((a, b) => (a.v < b.v ? -1 : a.v > b.v ? 1 : 0));
  const out = new Array(xs.length).fill(0);
  for (let i = 0; i < sorted.length; ) {
    let j = i + 1;
    while (j < sorted.length && sorted[j].v === sorted[i].v) j++;
    const avgRank = (i + 1 + j) / 2; // ranks are 1-based
    for (let k = i; k < j; k++) out[sorted[k].i] = avgRank;
    i = j;
  }
  return out;
}

function pearson(x: number[], y: number[]): number | null {
  if (x.length !== y.length || x.length < 2) return null;
  const mx = mean(x);
  const my = mean(y);
  if (mx == null || my == null) return null;
  let cov = 0;
  let vx = 0;
  let vy = 0;
  for (let i = 0; i < x.length; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    cov += dx * dy;
    vx += dx * dx;
    vy += dy * dy;
  }
  if (vx <= 0 || vy <= 0) return null;
  return cov / Math.sqrt(vx * vy);
}

function spearman(x: number[], y: number[]): number | null {
  const rx = rank(x);
  const ry = rank(y);
  return pearson(rx, ry);
}

function normalizeParamValue(v: unknown): { kind: "numeric" | "categorical"; value: number | string } | null {
  if (v == null) return null;
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return null;
    return { kind: "numeric", value: v };
  }
  if (typeof v === "string" || typeof v === "boolean") {
    return { kind: "categorical", value: String(v) };
  }
  if (Array.isArray(v)) {
    return { kind: "categorical", value: JSON.stringify(v) };
  }
  if (typeof v === "object") {
    try {
      return { kind: "categorical", value: JSON.stringify(v) };
    } catch {
      return { kind: "categorical", value: String(v) };
    }
  }
  return { kind: "categorical", value: String(v) };
}

export default function AnalysisPage() {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedExp, setSelectedExp] = useState<string>("");
  const [trials, setTrials] = useState<TrialSummary[]>([]);
  const [error, setError] = useState<string>("");
  const [paramFilter, setParamFilter] = useState<string>("");
  const [selectedParam, setSelectedParam] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    listExperiments()
      .then((exps) => {
        if (cancelled) return;
        setExperiments(exps);
        if (!selectedExp && exps.length) setSelectedExp(exps[0].name);
      })
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!selectedExp) return;
    listTrials(selectedExp)
      .then((rows) => {
        if (cancelled) return;
        setTrials(rows);
        setError("");
      })
      .catch((e: unknown) => !cancelled && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
    };
  }, [selectedExp]);

  const selectedExpMeta = useMemo(() => experiments.find((e) => e.name === selectedExp) ?? null, [
    experiments,
    selectedExp,
  ]);

  const effects = useMemo(() => {
    if (!trials.length) return [] as EffectRow[];
    const mode = (trials[0]?.mode ?? "max").toLowerCase();
    const yRaw = trials
      .map((t) => ({ t, y: t.best }))
      .filter((r) => typeof r.y === "number" && Number.isFinite(r.y)) as Array<{
      t: TrialSummary;
      y: number;
    }>;
    if (!yRaw.length) return [] as EffectRow[];

    const yEff = yRaw.map((r) => (mode === "min" ? -r.y : r.y));

    const keys = new Set<string>();
    for (const r of yRaw) {
      for (const k of Object.keys(r.t.params || {})) keys.add(k);
    }

    const rows: EffectRow[] = [];
    for (const k of Array.from(keys)) {
      // Collect param values aligned with y.
      const vals: Array<{ kind: "numeric" | "categorical"; value: number | string }> = [];
      const ys: number[] = [];
      for (let i = 0; i < yRaw.length; i++) {
        const norm = normalizeParamValue((yRaw[i].t.params || {})[k]);
        if (!norm) continue;
        vals.push(norm);
        ys.push(yEff[i]);
      }
      if (vals.length < 3) continue;

      const nUnique = new Set(vals.map((v) => `${v.kind}:${String(v.value)}`)).size;

      const isNumeric = vals.every((v) => v.kind === "numeric");
      if (isNumeric) {
        const x0 = vals.map((v) => Number(v.value));
        const allPos = x0.every((v) => v > 0);
        const xEff = allPos ? x0.map((v) => Math.log10(v)) : x0;
        const corr = spearman(xEff, ys);
        const score = corr == null ? 0 : Math.abs(corr);
        rows.push({
          param: k,
          kind: "numeric",
          n_unique: nUnique,
          score,
          spearman: corr,
          x_log10: allPos,
          x_min: x0.length ? Math.min(...x0) : null,
          x_max: x0.length ? Math.max(...x0) : null,
        });
      } else {
        // categorical: group by value
        const groups = new Map<string, number[]>();
        for (let i = 0; i < vals.length; i++) {
          const key = `${vals[i].kind}:${String(vals[i].value)}`;
          if (!groups.has(key)) groups.set(key, []);
          groups.get(key)!.push(ys[i]);
        }
        const stats = Array.from(groups.entries()).map(([valKey, ysG]) => {
          const m = mean(ysG) ?? 0;
          return {
            value: valKey.replace(/^categorical:/, "").replace(/^numeric:/, ""),
            count: ysG.length,
            mean: m,
            median: median(ysG) ?? m,
            min: Math.min(...ysG),
            max: Math.max(...ysG),
          };
        });
        stats.sort((a, b) => (mode === "min" ? a.mean - b.mean : b.mean - a.mean));
        if (!stats.length) continue;
        const best = stats[0];
        const worst = stats[stats.length - 1];
        const delta = best.mean - worst.mean; // note: ys already oriented (yEff)
        const score = Math.abs(delta);
        rows.push({
          param: k,
          kind: "categorical",
          n_unique: nUnique,
          score,
          best_value: best.value,
          worst_value: worst.value,
          mean_best: best.mean,
          mean_worst: worst.mean,
          delta_mean: delta,
          n_best: best.count,
        });
      }
    }

    rows.sort((a, b) => b.score - a.score);

    const q = paramFilter.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter((r) => r.param.toLowerCase().includes(q));
  }, [paramFilter, trials]);

  const selectedEffect = useMemo(() => effects.find((e) => e.param === selectedParam) ?? null, [
    effects,
    selectedParam,
  ]);

  const plotData = useMemo(() => {
    if (!selectedEffect) return null;
    const mode = (trials[0]?.mode ?? "max").toLowerCase();
    const yRaw = trials
      .map((t) => ({ t, y: t.best }))
      .filter((r) => typeof r.y === "number" && Number.isFinite(r.y)) as Array<{
      t: TrialSummary;
      y: number;
    }>;
    if (!yRaw.length) return null;

    if (selectedEffect.kind === "numeric") {
      const pts: Array<{ x: number; y: number }> = [];
      for (const r of yRaw) {
        const norm = normalizeParamValue((r.t.params || {})[selectedEffect.param]);
        if (!norm || norm.kind !== "numeric") continue;
        pts.push({ x: Number(norm.value), y: mode === "min" ? -r.y : r.y });
      }
      return { kind: "numeric" as const, points: pts };
    }

    // categorical
    const groups = new Map<string, number[]>();
    for (const r of yRaw) {
      const norm = normalizeParamValue((r.t.params || {})[selectedEffect.param]);
      if (!norm) continue;
      const key = `${norm.kind}:${String(norm.value)}`.replace(/^categorical:/, "").replace(/^numeric:/, "");
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(mode === "min" ? -r.y : r.y);
    }
    const bars = Array.from(groups.entries()).map(([value, ys]) => ({
      value,
      count: ys.length,
      mean: mean(ys) ?? 0,
    }));
    bars.sort((a, b) => b.mean - a.mean);
    return { kind: "categorical" as const, bars };
  }, [selectedEffect, trials]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div className="card">
        <div style={{ fontWeight: 700 }}>Cross-experiment (objective best)</div>
        <div className="muted" style={{ marginTop: 6 }}>
          Uses each experiment’s configured metric/mode (from `conf/tune*.yaml` when available).
        </div>
        <div style={{ overflowX: "auto", marginTop: 8 }}>
          <table className="table">
            <thead>
              <tr>
                <th>experiment</th>
                <th>metric</th>
                <th>mode</th>
                <th>trials</th>
                <th>best</th>
                <th>best_trial</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.name}>
                  <td className="mono">{e.name}</td>
                  <td className="mono">{e.metric}</td>
                  <td className="mono">{e.mode}</td>
                  <td className="mono">{e.n_trials}</td>
                  <td className="mono">{e.best == null ? "—" : e.best.toFixed(6)}</td>
                  <td className="mono">{e.best_trial_id ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <div style={{ fontWeight: 700 }}>Hyperparameter effects</div>
        <div className="row" style={{ marginTop: 10, alignItems: "center" }}>
          <select value={selectedExp} onChange={(e) => setSelectedExp(e.target.value)}>
            {experiments.map((e) => (
              <option key={e.name} value={e.name}>
                {e.name}
              </option>
            ))}
          </select>
          <input
            value={paramFilter}
            onChange={(e) => setParamFilter(e.target.value)}
            placeholder="Filter params (e.g. optimizer.lr)"
            style={{ minWidth: 320 }}
          />
          <div className="muted" style={{ alignSelf: "center" }}>
            {selectedExpMeta ? `${selectedExpMeta.metric} / ${selectedExpMeta.mode}` : ""}
          </div>
          <div className="muted" style={{ alignSelf: "center" }}>
            trials: {trials.length}, params: {effects.length}
          </div>
        </div>

        {error ? (
          <div className="mono" style={{ marginTop: 8, color: "crimson" }}>
            {error}
          </div>
        ) : null}

        <div className="grid" style={{ marginTop: 12 }}>
          <div className="card" style={{ overflowX: "auto" }}>
            <div style={{ fontWeight: 700 }}>Ranked effects</div>
            <div className="muted" style={{ marginTop: 6 }}>
              categorical: Δ(mean), numeric: |Spearman| (objective-oriented)
            </div>
            <table className="table" style={{ marginTop: 8 }}>
              <thead>
                <tr>
                  <th>param</th>
                  <th>kind</th>
                  <th>score</th>
                  <th>details</th>
                </tr>
              </thead>
              <tbody>
                {effects.slice(0, 80).map((r) => (
                  <tr
                    key={r.param}
                    style={{
                      cursor: "pointer",
                      background:
                        selectedParam === r.param
                          ? "color-mix(in oklab, currentColor 10%, transparent)"
                          : "transparent",
                    }}
                    onClick={() => setSelectedParam(r.param)}
                  >
                    <td className="mono">{r.param}</td>
                    <td className="mono">{r.kind}</td>
                    <td className="mono">{r.score.toFixed(4)}</td>
                    <td className="mono">
                      {r.kind === "numeric"
                        ? `spearman=${r.spearman == null ? "—" : r.spearman.toFixed(4)}`
                        : `Δmean=${r.delta_mean.toFixed(4)} (best=${r.best_value})`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {effects.length > 80 ? (
              <div className="muted" style={{ marginTop: 8 }}>
                Showing top 80.
              </div>
            ) : null}
          </div>

          <div className="card">
            <div style={{ fontWeight: 700 }}>Selected param</div>
            <div className="mono muted" style={{ marginTop: 6 }}>
              {selectedEffect ? selectedEffect.param : "—"}
            </div>

            {plotData?.kind === "numeric" ? (
              <div style={{ height: 360, marginTop: 12 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="x" name="x" />
                    <YAxis dataKey="y" name="y" />
                    <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                    <Legend />
                    <Scatter name={selectedEffect?.param ?? "x"} data={plotData.points} fill="#8884d8" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
            ) : null}

            {plotData?.kind === "categorical" ? (
              <div style={{ height: 360, marginTop: 12 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={plotData.bars}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="value" hide />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="mean" name="mean(objective)" fill="#82ca9d" />
                  </BarChart>
                </ResponsiveContainer>
                <div className="muted" style={{ marginTop: 8 }}>
                  (X-axis labels hidden; hover bars for value)
                </div>
              </div>
            ) : null}

            {!plotData ? (
              <div className="muted" style={{ marginTop: 12 }}>
                Click a param row to see a plot.
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}

