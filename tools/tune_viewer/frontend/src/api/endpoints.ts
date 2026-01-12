import { fetchJson } from "./client";
import type {
  ExperimentSummary,
  LightningMetricsResponse,
  TrialFileEntry,
  TrialFileResponse,
  TrialSummary,
  TrialTimeseriesResponse,
} from "./types";

export type Health = {
  ok: boolean;
  results_root: string;
  repo_root: string;
  poll_seconds: number;
};

export async function getHealth(): Promise<Health> {
  return await fetchJson<Health>("/api/health");
}

export async function listExperiments(): Promise<ExperimentSummary[]> {
  return await fetchJson<ExperimentSummary[]>("/api/experiments");
}

export async function listTrials(expName: string): Promise<TrialSummary[]> {
  return await fetchJson<TrialSummary[]>(
    `/api/experiments/${encodeURIComponent(expName)}/trials`,
  );
}

export async function getTrialTimeseries(
  expName: string,
  trialDirname: string,
  metrics: string[],
): Promise<TrialTimeseriesResponse> {
  const metricParam = metrics.length ? metrics.join(",") : "val_r2";
  const path = `/api/experiments/${encodeURIComponent(expName)}/trials/${encodeURIComponent(
    trialDirname,
  )}/timeseries?metrics=${encodeURIComponent(metricParam)}`;
  return await fetchJson<TrialTimeseriesResponse>(path);
}

export async function listTrialFiles(
  expName: string,
  trialDirname: string,
): Promise<TrialFileEntry[]> {
  return await fetchJson<TrialFileEntry[]>(
    `/api/experiments/${encodeURIComponent(expName)}/trials/${encodeURIComponent(
      trialDirname,
    )}/files`,
  );
}

export async function readTrialFile(
  expName: string,
  trialDirname: string,
  relPath: string,
  opts?: { tailLines?: number },
): Promise<TrialFileResponse> {
  const params = new URLSearchParams({ path: relPath });
  if (opts?.tailLines) params.set("tail_lines", String(opts.tailLines));
  return await fetchJson<TrialFileResponse>(
    `/api/experiments/${encodeURIComponent(expName)}/trials/${encodeURIComponent(
      trialDirname,
    )}/file?${params.toString()}`,
  );
}

export async function getLightningMetrics(
  expName: string,
  trialDirname: string,
  columns: string[],
  maxPoints?: number,
): Promise<LightningMetricsResponse> {
  const colParam = columns.length ? columns.join(",") : "";
  const params = new URLSearchParams();
  if (colParam) params.set("columns", colParam);
  if (maxPoints) params.set("max_points", String(maxPoints));
  const q = params.toString();
  return await fetchJson<LightningMetricsResponse>(
    `/api/experiments/${encodeURIComponent(expName)}/trials/${encodeURIComponent(
      trialDirname,
    )}/lightning/metrics${q ? `?${q}` : ""}`,
  );
}

