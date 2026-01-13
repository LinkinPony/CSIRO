export type ExperimentSummary = {
  name: string;
  path: string;
  n_trials: number;
  metric: string;
  mode: "min" | "max" | string;
  min_epoch_for_best: number;
  counts: Record<string, number>;
  best: number | null;
  best_trial_id: string | null;
  best_trial_dirname: string | null;
  pinned_best?: Record<string, ExperimentPinnedBest>;
  last_update_ns: number | null;
  now_s: number;
  tune_config_file: string | null;
};

export type ExperimentPinnedBest = {
  best: number | null;
  best_trial_id: string | null;
  best_trial_dirname: string | null;
};

export type MetricSummary = {
  metric: string;
  mode: "min" | "max" | string;
  min_epoch_for_best: number;
  n_metric_records: number;
  best: number | null;
  best_epoch: number | null;
  best_training_iteration: number | null;
  last: number | null;
  last_epoch: number | null;
  last_training_iteration: number | null;
};

export type TrialSummary = {
  exp_name: string;
  trial_dirname: string;
  trial_dir: string;
  trial_id: string;
  status: string;
  n_records: number;
  n_metric_records: number;
  metric: string;
  mode: "min" | "max" | string;
  min_epoch_for_best: number;
  best: number | null;
  best_epoch: number | null;
  best_training_iteration: number | null;
  last: number | null;
  last_epoch: number | null;
  last_training_iteration: number | null;
  time_total_s: number | null;
  result_mtime_ns: number | null;
  result_size: number | null;
  params: Record<string, unknown>;
  pinned_metrics?: Record<string, MetricSummary>;
};

export type TrialTimeseriesResponse = {
  exp_name: string;
  trial_dirname: string;
  trial_dir: string;
  params: Record<string, unknown>;
  available_metrics: string[];
  points: Array<Record<string, unknown>>;
};

export type TrialFileEntry = {
  path: string;
  name: string;
  is_dir: boolean;
  size: number;
  mtime_ns: number;
};

export type TrialFileResponse = {
  path: string;
  abs_path: string;
  size: number | null;
  mtime_ns: number | null;
  max_bytes: number;
  tail_lines: number | null;
  content: string;
};

export type TrainYamlResponse = {
  exp_name: string;
  trial_dirname: string;
  trial_dir: string;
  yaml: string;
  inferred: boolean;
  source_kind: string | null;
  source_trial_dirname: string | null;
  source_relpath: string | null;
  applied_params: boolean;
  applied_params_count: number;
};

export type LightningMetricsResponse = {
  exp_name: string;
  trial_dirname: string;
  trial_dir: string;
  csv_relpath: string | null;
  available_columns: string[];
  requested_columns: string[];
  max_points: number;
  points: Array<Record<string, unknown>>;
};

