#!/usr/bin/env python3
"""Analyze partial or complete TPU v5e CCE matrix results."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "results" / "v5e-cce-matrix" / "results"
DEFAULT_ANALYSIS_DIR = SCRIPT_DIR / "results" / "v5e-cce-matrix" / "analysis"


def read_jsonl(path: Path) -> pd.DataFrame:
  rows: list[dict[str, Any]] = []
  with path.open() as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return pd.DataFrame(rows)


def read_input(path: Path) -> pd.DataFrame:
  if not path.exists():
    return pd.DataFrame()
  if path.suffix == ".csv":
    return pd.read_csv(path)
  return read_jsonl(path)


def discover_inputs(args: argparse.Namespace) -> list[Path]:
  if args.input:
    return [path.expanduser().resolve() for path in args.input]
  results_dir = args.results_dir.expanduser().resolve()
  merged = results_dir / "all_results.jsonl"
  if merged.exists():
    return [merged]
  return sorted(results_dir.glob("shard_*.jsonl"))


def coerce_bool(series: pd.Series) -> pd.Series:
  if series.dtype == bool:
    return series
  return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def numeric(frame: pd.DataFrame, columns: list[str]) -> None:
  for column in columns:
    if column in frame.columns:
      frame[column] = pd.to_numeric(frame[column], errors="coerce")


def normalize_rows(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return pd.DataFrame(
        columns=[
            "ok",
            "cce_enabled",
            "steady_state_mean_step_time_sec",
            "xla_planned_hbm_gib_per_chip",
            "cce_loop_count",
            "chunk_configuration",
            "shape",
            "mesh_configuration",
        ]
    )
  rows = rows.copy()
  if "steady_state_mean_step_time_sec" not in rows and "mean_step_time_sec_excl_first" in rows:
    rows["steady_state_mean_step_time_sec"] = rows["mean_step_time_sec_excl_first"]
  if "xla_planned_hbm_gib_per_chip" not in rows and "xla_train_step_gib_per_chip" in rows:
    rows["xla_planned_hbm_gib_per_chip"] = rows["xla_train_step_gib_per_chip"]
  if "shape" not in rows and {"global_batch_size", "sequence_length"} <= set(rows.columns):
    rows["shape"] = rows.apply(
        lambda row: f"b{int(row['global_batch_size'])}/L{int(row['sequence_length'])}",
        axis=1,
    )
  if "mesh_configuration" not in rows and {"fsdp_degree", "tp_degree"} <= set(rows.columns):
    rows["mesh_configuration"] = rows.apply(
        lambda row: f"fsdp={int(row['fsdp_degree'])},tp={int(row['tp_degree'])}",
        axis=1,
    )
  if "cce_enabled" not in rows and "loss_impl" in rows:
    rows["cce_enabled"] = rows["loss_impl"].astype(str).eq("cce")
  rows["cce_enabled"] = coerce_bool(
      rows.get("cce_enabled", pd.Series(False, index=rows.index))
  )
  status = rows["status"] if "status" in rows else pd.Series("", index=rows.index)
  rows["ok"] = status.astype(str).str.lower().isin({"success", "ok"})
  numeric(
      rows,
      [
          "experiment_id",
          "repeat_index",
          "global_batch_size",
          "sequence_length",
          "fsdp_degree",
          "tp_degree",
          "token_chunk",
          "vocab_chunk",
          "token_loop_count",
          "vocab_loop_count",
          "cce_loop_count",
          "steady_state_mean_step_time_sec",
          "median_step_time_sec",
          "p95_step_time_sec",
          "tokens_per_sec",
          "xla_planned_hbm_gib_per_chip",
          "runtime_hbm_peak_gb",
          "vocab_size",
      ],
  )
  if "chunk_configuration" not in rows:
    rows["chunk_configuration"] = rows.apply(
        lambda row: (
            f"{int(row['token_chunk'])}/{int(row['vocab_chunk'])}"
            if bool(row["cce_enabled"])
            and pd.notna(row.get("token_chunk"))
            and pd.notna(row.get("vocab_chunk"))
            else "default_ce"
        ),
        axis=1,
    )
  return rows


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def baseline_key() -> list[str]:
  return [
      "model_name",
      "global_batch_size",
      "sequence_length",
      "shape",
      "mesh_configuration",
      "fsdp_degree",
      "tp_degree",
  ]


def build_fastest(rows: pd.DataFrame) -> pd.DataFrame:
  ok = rows[rows["ok"] & rows["steady_state_mean_step_time_sec"].notna()].copy()
  if ok.empty:
    return pd.DataFrame()
  return (
      ok.sort_values("steady_state_mean_step_time_sec")
      .groupby(["mesh_configuration", "shape"], as_index=False)
      .head(1)
      .sort_values(["mesh_configuration", "shape"])
  )


def build_cce_vs_default(rows: pd.DataFrame) -> pd.DataFrame:
  ok = rows[rows["ok"] & rows["steady_state_mean_step_time_sec"].notna()].copy()
  if ok.empty:
    return pd.DataFrame()
  key = baseline_key()
  defaults = ok[~ok["cce_enabled"]].copy()
  cce = ok[ok["cce_enabled"]].copy()
  if defaults.empty or cce.empty:
    return pd.DataFrame()
  default_summary = (
      defaults.groupby(key, dropna=False)
      .agg(
          default_step_time_sec=("steady_state_mean_step_time_sec", "median"),
          default_tokens_per_sec=("tokens_per_sec", "median"),
          default_xla_hbm_gib=("xla_planned_hbm_gib_per_chip", "median"),
          default_runtime_hbm_gb=("runtime_hbm_peak_gb", "median"),
      )
      .reset_index()
  )
  matched = cce.merge(default_summary, on=key, how="left")
  matched = matched[matched["default_step_time_sec"].notna()].copy()
  if matched.empty:
    return matched
  matched["cce_to_default_step_time_ratio"] = (
      matched["steady_state_mean_step_time_sec"] / matched["default_step_time_sec"]
  )
  matched["speedup_vs_default_x"] = (
      matched["default_step_time_sec"] / matched["steady_state_mean_step_time_sec"]
  )
  matched["xla_hbm_saving_gib"] = (
      matched["default_xla_hbm_gib"] - matched["xla_planned_hbm_gib_per_chip"]
  )
  matched["xla_hbm_saving_pct"] = (
      matched["xla_hbm_saving_gib"] / matched["default_xla_hbm_gib"] * 100.0
  )
  matched["runtime_hbm_saving_gb"] = (
      matched["default_runtime_hbm_gb"] - matched["runtime_hbm_peak_gb"]
  )
  return matched.sort_values([
      "mesh_configuration",
      "shape",
      "cce_to_default_step_time_ratio",
  ])


def build_outliers(matched: pd.DataFrame) -> pd.DataFrame:
  if matched.empty:
    return pd.DataFrame()
  rows = []
  for threshold in [2, 5, 10]:
    part = matched[matched["cce_to_default_step_time_ratio"] > threshold].copy()
    part["threshold"] = threshold
    rows.append(part)
  if not rows:
    return pd.DataFrame()
  return pd.concat(rows, ignore_index=True, sort=False).sort_values(
      ["threshold", "cce_to_default_step_time_ratio"],
      ascending=[True, False],
  )


def build_recovery_report(
    matched: pd.DataFrame,
    *,
    min_recovery_x: float,
    max_hbm_increase_pct: float,
) -> pd.DataFrame:
  if matched.empty:
    return pd.DataFrame()
  rows = []
  group_cols = baseline_key()
  for _, group in matched.groupby(group_cols, dropna=False):
    group = group[
        group["steady_state_mean_step_time_sec"].notna()
        & group["cce_loop_count"].notna()
    ].copy()
    if len(group) < 2:
      continue
    for _, slow in group.iterrows():
      candidates = group[group["cce_loop_count"] < slow["cce_loop_count"]]
      for _, fast in candidates.iterrows():
        if fast["steady_state_mean_step_time_sec"] <= 0:
          continue
        recovery_x = (
            slow["steady_state_mean_step_time_sec"]
            / fast["steady_state_mean_step_time_sec"]
        )
        if recovery_x < min_recovery_x:
          continue
        hbm_increase_pct = math.nan
        if pd.notna(slow["xla_planned_hbm_gib_per_chip"]) and pd.notna(
            fast["xla_planned_hbm_gib_per_chip"]
        ):
          hbm_increase_pct = (
              (
                  fast["xla_planned_hbm_gib_per_chip"]
                  - slow["xla_planned_hbm_gib_per_chip"]
              )
              / slow["xla_planned_hbm_gib_per_chip"]
              * 100.0
          )
          if hbm_increase_pct > max_hbm_increase_pct:
            continue
        rows.append({
            "model_name": slow.get("model_name"),
            "mesh_configuration": slow.get("mesh_configuration"),
            "shape": slow.get("shape"),
            "slow_chunk": slow.get("chunk_configuration"),
            "fast_chunk": fast.get("chunk_configuration"),
            "slow_cce_loop_count": slow.get("cce_loop_count"),
            "fast_cce_loop_count": fast.get("cce_loop_count"),
            "slow_step_time_sec": slow.get("steady_state_mean_step_time_sec"),
            "fast_step_time_sec": fast.get("steady_state_mean_step_time_sec"),
            "recovery_x": recovery_x,
            "slow_xla_hbm_gib": slow.get("xla_planned_hbm_gib_per_chip"),
            "fast_xla_hbm_gib": fast.get("xla_planned_hbm_gib_per_chip"),
            "hbm_increase_pct": hbm_increase_pct,
            "slow_default_ratio": slow.get("cce_to_default_step_time_ratio"),
            "fast_default_ratio": fast.get("cce_to_default_step_time_ratio"),
        })
  return pd.DataFrame(rows).sort_values("recovery_x", ascending=False)


def add_loss_logits_proxy(matched: pd.DataFrame) -> pd.DataFrame:
  if matched.empty:
    return matched
  matched = matched.copy()
  dtype_bytes = 2.0
  matched["loss_logits_proxy_gib"] = (
      matched["global_batch_size"]
      * matched["sequence_length"]
      * matched["vocab_size"].fillna(262144)
      * dtype_bytes
      / (1024**3)
  )
  return matched


def corr(frame: pd.DataFrame, left: str, right: str) -> float | None:
  if frame.empty or left not in frame or right not in frame:
    return None
  part = frame[[left, right]].dropna()
  if len(part) < 3:
    return None
  return float(part[left].corr(part[right]))


def build_hypothesis_report(rows: pd.DataFrame, matched: pd.DataFrame) -> tuple[pd.DataFrame, str]:
  matched = add_loss_logits_proxy(matched)
  memory_corr = corr(matched, "loss_logits_proxy_gib", "xla_hbm_saving_gib")
  loop_time_corr = corr(matched, "cce_loop_count", "cce_to_default_step_time_ratio")
  mesh_summary = pd.DataFrame()
  if not matched.empty:
    mesh_summary = (
        matched.groupby(["mesh_configuration"], dropna=False)
        .agg(
            median_cce_to_default_ratio=("cce_to_default_step_time_ratio", "median"),
            p95_cce_to_default_ratio=("cce_to_default_step_time_ratio", lambda s: s.quantile(0.95)),
            median_xla_hbm_saving_pct=("xla_hbm_saving_pct", "median"),
            max_cce_loop_count=("cce_loop_count", "max"),
            rows=("experiment_id", "count"),
        )
        .reset_index()
      )
  tests = pd.DataFrame([
      {
          "hypothesis_component": "memory benefit vs loss-logits proxy",
          "metric": "pearson_corr(loss_logits_proxy_gib, xla_hbm_saving_gib)",
          "value": memory_corr,
          "interpretation": (
              "stronger positive values support shape/loss-logits dominance"
              if memory_corr is not None
              else "insufficient matched rows"
          ),
      },
      {
          "hypothesis_component": "throughput failure vs chunk-loop granularity",
          "metric": "pearson_corr(cce_loop_count, cce_to_default_step_time_ratio)",
          "value": loop_time_corr,
          "interpretation": (
              "positive values support loop granularity as a slowdown factor"
              if loop_time_corr is not None
              else "insufficient matched rows"
          ),
      },
      {
          "hypothesis_component": "mesh interaction",
          "metric": "mesh_count_with_matched_rows",
          "value": int(mesh_summary["mesh_configuration"].nunique())
          if not mesh_summary.empty
          else 0,
          "interpretation": "compare median/p95 ratios by mesh in mesh_hypothesis_summary.csv",
      },
  ])
  support = "inconclusive"
  if memory_corr is not None and loop_time_corr is not None:
    support = "partially supported"
    if memory_corr > 0.5 and loop_time_corr > 0.3 and not mesh_summary.empty:
      support = "supported by current partial data"
  lines = [
      "# TPU v5e CCE Hypothesis Report",
      "",
      'Hypothesis: "CCE memory benefit is mostly governed by loss-logits dominance, while CCE throughput failure is governed by the interaction between chunk-loop granularity and FSDP/TP mesh layout."',
      "",
      f"Current status: {support}.",
      "",
      f"- Total rows read: {len(rows)}",
      f"- Successful matched CCE/default rows: {len(matched)}",
      f"- Memory correlation: {memory_corr if memory_corr is not None else 'insufficient data'}",
      f"- Loop/time correlation: {loop_time_corr if loop_time_corr is not None else 'insufficient data'}",
      "",
      "Use `cce_vs_default.csv`, `mesh_hypothesis_summary.csv`, and `recovery_report.csv` for the detailed evidence.",
  ]
  return tests, "\n".join(lines) + "\n"


def plot_step_vs_loop(rows: pd.DataFrame, path: Path) -> None:
  cce = rows[
      rows["ok"]
      & rows["cce_enabled"]
      & rows["cce_loop_count"].notna()
      & rows["steady_state_mean_step_time_sec"].notna()
  ].copy()
  if cce.empty:
    return
  fig, ax = plt.subplots(figsize=(8, 5))
  for mesh, group in cce.groupby("mesh_configuration"):
    ax.scatter(
        group["cce_loop_count"],
        group["steady_state_mean_step_time_sec"],
        label=mesh,
        alpha=0.8,
        s=46,
    )
  ax.set_xscale("log", base=2)
  ax.set_xlabel("CCE loop count")
  ax.set_ylabel("Mean measured step time (sec)")
  ax.set_title("Step time vs CCE loop count")
  ax.grid(True, alpha=0.25)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path, dpi=170)
  plt.close(fig)


def plot_chunk_by_mesh(rows: pd.DataFrame, path: Path) -> None:
  cce = rows[
      rows["ok"]
      & rows["cce_enabled"]
      & rows["steady_state_mean_step_time_sec"].notna()
  ].copy()
  if cce.empty:
    return
  summary = (
      cce.groupby(["mesh_configuration", "chunk_configuration"], dropna=False)
      .agg(step_time_sec=("steady_state_mean_step_time_sec", "median"))
      .reset_index()
  )
  chunk_order = sorted(
      summary["chunk_configuration"].unique(),
      key=lambda value: tuple(int(x) for x in str(value).split("/") if x.isdigit()),
  )
  fig, ax = plt.subplots(figsize=(9, 5))
  for mesh, group in summary.groupby("mesh_configuration"):
    group = group.set_index("chunk_configuration").reindex(chunk_order)
    ax.plot(
        range(len(chunk_order)),
        group["step_time_sec"],
        marker="o",
        linewidth=2,
        label=mesh,
    )
  ax.set_xticks(range(len(chunk_order)))
  ax.set_xticklabels(chunk_order, rotation=35, ha="right")
  ax.set_xlabel("token_chunk / vocab_chunk")
  ax.set_ylabel("Median measured step time (sec)")
  ax.set_title("Step time vs chunk configuration by mesh")
  ax.grid(True, axis="y", alpha=0.25)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path, dpi=170)
  plt.close(fig)


def plot_memory_saving(matched: pd.DataFrame, path: Path) -> None:
  if matched.empty or "xla_hbm_saving_pct" not in matched:
    return
  part = matched[matched["xla_hbm_saving_pct"].notna()].copy()
  if part.empty:
    return
  summary = (
      part.groupby(["shape", "mesh_configuration"], dropna=False)
      .agg(saving_pct=("xla_hbm_saving_pct", "median"))
      .reset_index()
  )
  shapes = sorted(summary["shape"].unique())
  fig, ax = plt.subplots(figsize=(8, 5))
  for mesh, group in summary.groupby("mesh_configuration"):
    group = group.set_index("shape").reindex(shapes)
    ax.plot(
        range(len(shapes)),
        group["saving_pct"],
        marker="o",
        linewidth=2,
        label=mesh,
    )
  ax.set_xticks(range(len(shapes)))
  ax.set_xticklabels(shapes)
  ax.set_xlabel("Shape")
  ax.set_ylabel("Median XLA HBM saving vs Default CE (%)")
  ax.set_title("Memory saving vs shape")
  ax.grid(True, axis="y", alpha=0.25)
  ax.legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path, dpi=170)
  plt.close(fig)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=Path, action="append", default=[])
  parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
  parser.add_argument("--outdir", type=Path, default=DEFAULT_ANALYSIS_DIR)
  parser.add_argument("--min-recovery-x", type=float, default=2.0)
  parser.add_argument("--max-hbm-increase-pct", type=float, default=10.0)
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  outdir = args.outdir.expanduser().resolve()
  outdir.mkdir(parents=True, exist_ok=True)
  inputs = discover_inputs(args)
  frames = [read_input(path) for path in inputs]
  rows = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
  rows = normalize_rows(rows)

  write_frame(rows, outdir / "normalized_results.csv")
  fastest = build_fastest(rows)
  matched = build_cce_vs_default(rows)
  outliers = build_outliers(matched)
  recovery = build_recovery_report(
      matched,
      min_recovery_x=args.min_recovery_x,
      max_hbm_increase_pct=args.max_hbm_increase_pct,
  )
  tests, hypothesis_md = build_hypothesis_report(rows, matched)

  write_frame(fastest, outdir / "fastest_configurations.csv")
  write_frame(matched, outdir / "cce_vs_default.csv")
  write_frame(outliers, outdir / "outlier_report.csv")
  write_frame(recovery, outdir / "recovery_report.csv")
  write_frame(tests, outdir / "hypothesis_tests.csv")
  if not matched.empty:
    mesh_summary = (
        matched.groupby(["mesh_configuration"], dropna=False)
        .agg(
            median_cce_to_default_ratio=("cce_to_default_step_time_ratio", "median"),
            p95_cce_to_default_ratio=("cce_to_default_step_time_ratio", lambda s: s.quantile(0.95)),
            median_xla_hbm_saving_pct=("xla_hbm_saving_pct", "median"),
            rows=("experiment_id", "count"),
        )
        .reset_index()
    )
  else:
    mesh_summary = pd.DataFrame()
  write_frame(mesh_summary, outdir / "mesh_hypothesis_summary.csv")
  (outdir / "hypothesis_report.md").write_text(hypothesis_md)

  plot_step_vs_loop(rows, outdir / "step_time_vs_cce_loop_count.png")
  plot_chunk_by_mesh(rows, outdir / "step_time_vs_chunk_by_mesh.png")
  plot_memory_saving(matched, outdir / "memory_saving_vs_shape.png")

  print(f"inputs={len(inputs)}")
  print(f"rows={len(rows)}")
  print(f"outdir={outdir}")


if __name__ == "__main__":
  main()
