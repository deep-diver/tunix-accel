#!/usr/bin/env python3
"""Collect and visualize the Gemma3 270M CCE rerun artifacts.

The TPU workers each emit a small tarball with the same internal layout:

  gemma3-270m-cce-rerun/<suite>/<suite>_results.csv

This script extracts all available tarballs, builds compact CSVs used by the
01-CCE report, and redraws the figures. It is safe to rerun as more artifacts
arrive.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma3_270m_full_cce"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

VARIANT_LABEL = {"default": "Default CE", "cce": "CCE"}
VARIANT_COLOR = {"default": "#555555", "cce": "#159A78"}


def context_label(value: float) -> str:
  value = int(value)
  if value >= 1024:
    return f"{value // 1024}K"
  return f"{value}"


def gib_label(value: float) -> str:
  if value >= 128:
    return f"{int(value)}"
  if value >= 16:
    return f"{value:.0f}"
  return f"{value:.1f}".rstrip("0").rstrip(".")


def read_csvs(pattern: str) -> pd.DataFrame:
  frames = []
  for path in sorted(RAW_DIR.glob(pattern)):
    try:
      frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
      continue
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  return pd.concat(frames, ignore_index=True, sort=False)


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def extract_artifacts() -> None:
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    profile = tarball.name.removeprefix("gemma3-270m-cce-rerun-").removesuffix(
        ".tar.gz"
    )
    target = RAW_DIR / profile
    if target.exists():
      shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(target)], check=True)


def complete_frontier(rows: pd.DataFrame) -> pd.DataFrame:
  rows = rows.copy()
  for col in [
      "batch_size",
      "max_length",
      "lora_rank",
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "runtime_peak_hbm_gb",
  ]:
    if col in rows:
      rows[col] = pd.to_numeric(rows[col], errors="coerce")
  rows["variant_label"] = rows["variant"].map(VARIANT_LABEL).fillna(rows["variant"])
  rows["ok"] = rows["status"].eq("ok")
  return rows


def build_frontier_summary(frontier: pd.DataFrame) -> pd.DataFrame:
  if frontier.empty:
    return pd.DataFrame()
  grouped = []
  for keys, part in frontier.groupby(["suite", "lora_rank", "batch_size", "variant"]):
    ok = part[part["ok"]]
    grouped.append({
        "suite": keys[0],
        "lora_rank": int(keys[1]),
        "batch_size": int(keys[2]),
        "variant": keys[3],
        "max_ok_context": int(ok["max_length"].max()) if not ok.empty else 0,
        "first_failed_context": int(part[~part["ok"]]["max_length"].min())
        if not part[~part["ok"]].empty
        else "",
        "ok_rows": int(ok.shape[0]),
        "failed_rows": int((~part["ok"]).sum()),
      })
  out = pd.DataFrame(grouped)
  return out.sort_values(["suite", "lora_rank", "batch_size", "variant"])


def plot_frontier(frontier: pd.DataFrame, summary: pd.DataFrame) -> None:
  if frontier.empty or summary.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)

  fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.4), constrained_layout=True)
  fig.suptitle(
      "Gemma3 270M LoRA on TPU v5litepod-1: Cut Cross Entropy moves the context frontier",
      fontsize=14,
      fontweight="bold",
  )

  ax = axes[0]
  subset = summary[
      summary["suite"].isin(["frontier_low", "frontier_high"])
      & summary["lora_rank"].eq(16)
  ].copy()
  for variant in ["default", "cce"]:
    part = subset[subset["variant"].eq(variant)].sort_values("batch_size")
    ax.plot(
        part["batch_size"],
        part["max_ok_context"].clip(lower=1),
        marker="o",
        markersize=7,
        linewidth=2.5,
        color=VARIANT_COLOR[variant],
        label=VARIANT_LABEL[variant],
    )
    for _, row in part.iterrows():
      label = "none" if row["max_ok_context"] == 0 else f'{int(row["max_ok_context"]):,}'
      ax.annotate(
          label,
          (row["batch_size"], max(row["max_ok_context"], 1)),
          textcoords="offset points",
          xytext=(0, 8 if variant == "cce" else -17),
          ha="center",
          fontsize=8,
          color=VARIANT_COLOR[variant],
      )
  ax.set_xscale("log", base=2)
  ax.set_yscale("log", base=2)
  ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
  ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
  ax.set_yticks([1, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
  ax.get_yaxis().set_major_formatter(
      FuncFormatter(lambda x, _: "none" if x == 1 else f"{int(x):,}")
  )
  ax.set_xlabel("Batch size")
  ax.set_ylabel("Maximum completed context length")
  ax.grid(True, which="major", color="#E6E6E6", zorder=0)
  ax.legend(frameon=False, loc="lower left")

  ax = axes[1]
  memory = frontier[
      frontier["suite"].isin(["frontier_low", "frontier_high"])
      & frontier["lora_rank"].eq(16)
      & frontier["batch_size"].isin([8, 16, 32])
  ].copy()
  for (variant, batch), part in memory.groupby(["variant", "batch_size"]):
    part = part.sort_values("max_length")
    linestyle = "-" if variant == "cce" else "--"
    marker = "o" if variant == "cce" else "s"
    ax.plot(
        part["max_length"],
        part["xla_train_step_gib_per_chip"],
        linestyle=linestyle,
        marker=marker,
        markersize=5.5,
        linewidth=2,
        color=VARIANT_COLOR[variant],
        alpha=0.95 if variant == "cce" else 0.72,
        label=f"{VARIANT_LABEL[variant]}, b{int(batch)}",
    )
  ax.set_xscale("log", base=2)
  ax.set_yscale("log", base=2)
  ax.set_xticks([256, 512, 1024, 2048, 4096, 8192, 16384, 32768])
  ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, _: context_label(x)))
  ax.set_yticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
  ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, _: gib_label(x)))
  ax.set_xlabel("Context length")
  ax.set_ylabel("XLA planned HBM per chip (GiB)")
  ax.grid(True, which="major", color="#E6E6E6", zorder=0)
  ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_frontier.png", dpi=180)
  plt.close(fig)


def plot_status_heatmap(frontier: pd.DataFrame) -> None:
  if frontier.empty:
    return
  subset = frontier[
      frontier["suite"].isin(["frontier_low", "frontier_high"])
      & frontier["lora_rank"].eq(16)
  ].copy()
  if subset.empty:
    return
  batches = sorted(subset["batch_size"].dropna().unique())
  contexts = sorted(subset["max_length"].dropna().unique())
  fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), constrained_layout=True)
  fig.suptitle(
      "Gemma3 270M CCE pass/fail map on TPU v5litepod-1",
      fontsize=14,
      fontweight="bold",
  )
  for ax, variant in zip(axes, ["default", "cce"], strict=True):
    grid = []
    labels = []
    for batch in batches:
      row = []
      label_row = []
      for context in contexts:
        part = subset[
            subset["variant"].eq(variant)
            & subset["batch_size"].eq(batch)
            & subset["max_length"].eq(context)
        ]
        if part.empty:
          row.append(math.nan)
          label_row.append("")
        else:
          ok = bool(part.iloc[0]["ok"])
          row.append(1 if ok else 0)
          label_row.append("OK" if ok else "OOM")
      grid.append(row)
      labels.append(label_row)
    ax.imshow(grid, cmap=plt.matplotlib.colors.ListedColormap(["#F3B3A8", "#A7D8C5"]), vmin=0, vmax=1)
    ax.set_title(VARIANT_LABEL[variant])
    ax.set_xticks(range(len(contexts)))
    ax.set_xticklabels([context_label(c) for c in contexts], rotation=45, ha="right")
    ax.set_yticks(range(len(batches)))
    ax.set_yticklabels([f"b{int(b)}" for b in batches])
    ax.set_xlabel("Context length")
    ax.set_ylabel("Batch size")
    for y, row in enumerate(labels):
      for x, text in enumerate(row):
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color="#222222")
  fig.savefig(ASSET_DIR / "gemma3_270m_cce_status_heatmap.png", dpi=180)
  plt.close(fig)


def read_history_rows() -> pd.DataFrame:
  rows: list[dict[str, Any]] = []
  for path in sorted(RAW_DIR.glob("quality-*/**/history.csv")):
    if "/unpacked/" in str(path):
      continue
    frame = pd.read_csv(path)
    case_dir = path.parent
    summary_path = case_dir / "case_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
      summary = json.loads(summary_path.read_text())
    for _, item in frame.iterrows():
      rows.append({
          "run": summary.get("suite", case_dir.parent.name),
          "case": summary.get("case", case_dir.name),
          "variant": summary.get("variant", "cce" if "quality-cce" in str(path) else "default"),
          "batch_size": summary.get("batch_size"),
          "max_length": summary.get("max_length"),
          "step": int(item["step"]),
          "loss": float(item["loss"]),
          "step_time_sec": float(item["step_time_sec"]),
          "cumulative_loss_tokens": float(item.get("cumulative_loss_tokens", 0)),
          "cumulative_step_time_sec": float(item.get("cumulative_step_time_sec", 0)),
      })
  return pd.DataFrame(rows)


def plot_quality(history: pd.DataFrame, quality: pd.DataFrame) -> None:
  if history.empty or quality.empty:
    return
  same = quality[
      quality["suite"].isin(["quality_default_b16_l512", "quality_cce_b16_l512"])
  ].copy()
  same["variant_label"] = same["variant"].map(VARIANT_LABEL)
  hist = history[
      history["run"].isin(["quality_default_b16_l512", "quality_cce_b16_l512"])
  ].copy()
  if hist.empty or same.empty:
    return

  fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.8), constrained_layout=True)
  fig.suptitle(
      "Gemma3 270M OPUS100 EN-FR LoRA SFT: same-shape quality parity",
      fontsize=14,
      fontweight="bold",
  )

  ax = axes[0]
  for variant, part in hist.groupby("variant"):
    part = part.sort_values("step")
    color = VARIANT_COLOR[variant]
    label = VARIANT_LABEL[variant]
    ax.plot(part["step"], part["loss"], color=color, alpha=0.18, linewidth=0.7)
    smooth = part["loss"].rolling(75, min_periods=1).mean()
    ax.plot(part["step"], smooth, color=color, linewidth=2.3, label=label)
  ax.set_xlabel("Training step")
  ax.set_ylabel("Train loss")
  ax.grid(True, color="#E6E6E6")
  ax.legend(frameon=False)

  metrics = [
      ("xla_train_step_gib_per_chip", "XLA peak\nGiB/chip", False),
      ("mean_step_time_sec_excl_first", "Mean step\nsec", False),
      ("final_loss", "Final\ntrain loss", False),
      ("eval_loss", "Eval\nloss", False),
      ("bleu", "BLEU\n16 samples", False),
  ]
  ax = axes[1]
  x_positions = []
  heights = []
  colors = []
  tick_labels = []
  text_labels = []
  for idx, (metric, label, higher_is_better) in enumerate(metrics):
    values = []
    for variant in ["default", "cce"]:
      row = same[same["variant"].eq(variant)]
      val = float(row.iloc[0][metric]) if not row.empty and pd.notna(row.iloc[0][metric]) else math.nan
      values.append(val)
    baseline = values[0] if values[0] and not math.isnan(values[0]) else 1.0
    for j, variant in enumerate(["default", "cce"]):
      x = idx * 3 + j
      x_positions.append(x)
      heights.append(values[j] / baseline if baseline else math.nan)
      colors.append(VARIANT_COLOR[variant])
      tick_labels.append(label if j == 0 else "")
      text_labels.append(f"{values[j]:.3g}")
  bars = ax.bar(x_positions, heights, color=colors, width=0.72)
  for bar, text in zip(bars, text_labels, strict=True):
    if not math.isnan(bar.get_height()):
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() + 0.035,
          text,
          ha="center",
          va="bottom",
          fontsize=8,
      )
  ax.axhline(1.0, color="#777777", linewidth=1, linestyle="--")
  ax.set_xticks([i * 3 + 0.5 for i in range(len(metrics))])
  ax.set_xticklabels([m[1] for m in metrics])
  ax.set_ylabel("Normalized to Default CE b16/L512")
  ax.grid(True, axis="y", color="#E6E6E6")
  handles = [
      plt.Line2D([0], [0], color=VARIANT_COLOR["default"], lw=8, label="Default CE"),
      plt.Line2D([0], [0], color=VARIANT_COLOR["cce"], lw=8, label="CCE"),
  ]
  ax.legend(handles=handles, frameon=False, ncol=2, loc="upper right")

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_quality.png", dpi=180)
  plt.close(fig)


def plot_chunk_rank(chunk: pd.DataFrame, rank: pd.DataFrame) -> None:
  if chunk.empty and rank.empty:
    return
  fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8), constrained_layout=True)
  fig.suptitle(
      "Gemma3 270M CCE tuning checks on TPU v5litepod-1",
      fontsize=14,
      fontweight="bold",
  )

  ax = axes[0]
  if not chunk.empty:
    ok = chunk[chunk["status"].eq("ok") & chunk["max_length"].eq(512)].copy()
    if not ok.empty:
      pivot = ok.pivot_table(
          values="mean_step_time_sec_excl_first",
          index="token_chunk",
          columns="vocab_chunk",
          aggfunc="mean",
      ).sort_index().sort_index(axis=1)
      image = ax.imshow(pivot.values, cmap="YlGnBu")
      ax.set_xticks(range(len(pivot.columns)))
      ax.set_xticklabels([context_label(c) for c in pivot.columns], rotation=45, ha="right")
      ax.set_yticks(range(len(pivot.index)))
      ax.set_yticklabels([str(int(i)) for i in pivot.index])
      ax.set_xlabel("Vocab chunk")
      ax.set_ylabel("Token chunk")
      ax.set_title("CCE chunk tuning at b16/L512\nmean step time, seconds")
      for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
          val = pivot.values[y, x]
          ax.text(x, y, f"{val:.3f}", ha="center", va="center", fontsize=8)
      fig.colorbar(image, ax=ax, shrink=0.78)
    else:
      ax.text(0.5, 0.5, "No completed chunk rows yet", ha="center", va="center")
      ax.axis("off")
  else:
    ax.text(0.5, 0.5, "No chunk artifact yet", ha="center", va="center")
    ax.axis("off")

  ax = axes[1]
  if not rank.empty:
    subset = rank[rank["batch_size"].isin([8, 16, 32])].copy()
    matched = subset.pivot_table(
        values="xla_train_step_gib_per_chip",
        index=["lora_rank", "batch_size", "max_length"],
        columns="variant",
        aggfunc="first",
    ).reset_index()
    matched = matched.dropna(subset=["default", "cce"])
    matched["reduction_pct"] = (1 - matched["cce"] / matched["default"]) * 100
    matched["shape"] = matched.apply(
        lambda row: f"b{int(row['batch_size'])}/L{context_label(row['max_length'])}",
        axis=1,
    )
    shape_order = (
        matched[["batch_size", "max_length", "shape"]]
        .drop_duplicates()
        .sort_values(["batch_size", "max_length"])["shape"]
        .tolist()
    )
    x_base = list(range(len(shape_order)))
    width = 0.23
    rank_values = sorted(matched["lora_rank"].dropna().unique())
    for idx, rank_value in enumerate(rank_values):
      part = matched[matched["lora_rank"].eq(rank_value)].set_index("shape")
      heights = [part.loc[shape, "reduction_pct"] if shape in part.index else math.nan for shape in shape_order]
      offset = (idx - (len(rank_values) - 1) / 2) * width
      ax.bar(
          [x + offset for x in x_base],
          heights,
          width=width,
          label=f"rank {int(rank_value)}",
          alpha=0.86,
      )
    ax.axhline(0, color="#888888", linewidth=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels(shape_order, rotation=35, ha="right", fontsize=8)
    ax.set_xlabel("Matched passing shape")
    ax.set_ylabel("XLA memory reduction at matched shapes (%)")
    ax.set_title("Rank sensitivity: reduction tracks logits pressure")
    ax.grid(True, color="#E6E6E6")
    ax.legend(frameon=False)
  else:
    ax.text(0.5, 0.5, "No rank artifact yet", ha="center", va="center")
    ax.axis("off")

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_tuning.png", dpi=180)
  plt.close(fig)


def write_generation_tables(quality: pd.DataFrame) -> None:
  metric_cols = [
      "suite",
      "variant",
      "batch_size",
      "max_length",
      "max_steps",
      "eval_batches",
      "eval_loss",
      "bleu",
      "chrf",
      "final_loss",
      "wall_time_sec",
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
  ]
  available_metric_cols = [col for col in metric_cols if col in quality.columns]
  metrics = quality[available_metric_cols].copy()
  write_frame(metrics, DATA_DIR / "generation_metrics.csv")

  sample_paths = {}
  for path in sorted(RAW_DIR.glob("quality-*/**/unpacked/translations.jsonl")):
    key = "capacity_cce" if "quality-capacity" in str(path) else (
        "cce_b16" if "quality-cce" in str(path) else "default_b16"
    )
    sample_paths[key] = path
  if not sample_paths:
    return

  merged: dict[tuple[str, str], dict[str, Any]] = {}
  for key, path in sample_paths.items():
    for idx, line in enumerate(path.read_text().splitlines()):
      if not line.strip():
        continue
      item = json.loads(line)
      row_key = (item.get("source", ""), item.get("reference", ""))
      row = merged.setdefault(
          row_key,
          {
              "sample_id": idx,
              "source": item.get("source", ""),
              "reference": item.get("reference", ""),
              "prompt_tokens": item.get("prompt_tokens", ""),
          },
      )
      row[f"{key}_prediction"] = item.get("prediction", "")
  out_path = DATA_DIR / "generation_samples.jsonl"
  with out_path.open("w") as f:
    for row in sorted(merged.values(), key=lambda r: int(r["sample_id"])):
      f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_profile_tables(all_results: pd.DataFrame) -> None:
  profile_cols = [
      "suite",
      "case",
      "variant",
      "batch_size",
      "max_length",
      "lora_rank",
      "token_chunk",
      "vocab_chunk",
      "status",
      "failure_type",
      "tpu",
      "chips",
      "xla_train_step_gib_per_chip",
      "runtime_peak_hbm_gb",
      "runtime_hbm_limit_gb",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "wall_time_sec",
      "xla_report",
  ]
  cols = [col for col in profile_cols if col in all_results.columns]
  write_frame(all_results[cols].copy(), DATA_DIR / "profile_summary.csv")

  pressure = all_results[all_results["suite"].eq("pressure_points")].copy()
  write_frame(pressure, DATA_DIR / "pressure_points.csv")

  oom = all_results[all_results["status"].ne("ok")].copy()
  oom_cols = [
      "suite",
      "case",
      "variant",
      "batch_size",
      "max_length",
      "lora_rank",
      "token_chunk",
      "vocab_chunk",
      "failure_type",
      "oom_used_gib",
      "oom_limit_gib",
      "oom_exceeded_gib",
      "xla_train_step_gib_per_chip",
      "xla_report",
      "source_csv",
  ]
  write_frame(oom[[col for col in oom_cols if col in oom.columns]], DATA_DIR / "oom_events.csv")

  rank = complete_frontier(all_results[all_results["suite"].eq("rank_sensitivity")].copy())
  if not rank.empty:
    summary = build_frontier_summary(rank)
    write_frame(summary, DATA_DIR / "rank_frontier_summary.csv")


def build_manifest() -> pd.DataFrame:
  rows = []
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    rows.append({
        "artifact": str(tarball.relative_to(SCRIPT_DIR)),
        "bytes": tarball.stat().st_size,
        "profile": tarball.name.removeprefix("gemma3-270m-cce-rerun-").removesuffix(
            ".tar.gz"
        ),
        "tpu": "v5litepod-1",
        "chips": 1,
        "project": "gcp-ml-172005",
        "zone": "us-west4-a",
        "model": "google/gemma-3-270m-it",
    })
  return pd.DataFrame(rows)


def main() -> None:
  extract_artifacts()
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  ASSET_DIR.mkdir(parents=True, exist_ok=True)

  manifest = build_manifest()
  write_frame(manifest, DATA_DIR / "run_manifest.csv")

  all_results = read_csvs("*/gemma3-270m-cce-rerun/**/*_results.csv")
  if all_results.empty:
    print("No result CSVs found")
    return
  for col in [
      "batch_size",
      "max_length",
      "lora_rank",
      "token_chunk",
      "vocab_chunk",
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "wall_time_sec",
      "final_loss",
      "eval_loss",
      "bleu",
      "chrf",
  ]:
    if col in all_results:
      all_results[col] = pd.to_numeric(all_results[col], errors="coerce")
  write_frame(all_results, DATA_DIR / "all_runs.csv")

  parity = all_results[all_results["suite"].eq("parity_270m_one_step")].copy()
  quality = all_results[all_results["suite"].str.startswith("quality_", na=False)].copy()
  frontier = complete_frontier(
      all_results[all_results["suite"].isin(["frontier_low", "frontier_high"])].copy()
  )
  rank = complete_frontier(
      all_results[all_results["suite"].eq("rank_sensitivity")].copy()
  )
  chunk = complete_frontier(
      all_results[all_results["suite"].isin(["chunk_tuning", "pressure_points"])].copy()
  )
  history = read_history_rows()

  write_frame(parity, DATA_DIR / "parity_summary.csv")
  write_frame(quality, DATA_DIR / "training_summary.csv")
  write_frame(history, DATA_DIR / "training_history.csv")
  write_frame(frontier, DATA_DIR / "frontier_runs.csv")
  write_frame(build_frontier_summary(frontier), DATA_DIR / "frontier_summary.csv")
  write_frame(rank, DATA_DIR / "rank_sensitivity.csv")
  write_frame(chunk, DATA_DIR / "chunk_tuning.csv")
  write_generation_tables(quality)
  write_profile_tables(all_results)

  plot_frontier(frontier, build_frontier_summary(frontier))
  plot_status_heatmap(frontier)
  plot_quality(history, quality)
  plot_chunk_rank(chunk, rank)

  print(f"wrote={DATA_DIR}")
  print(f"figures={ASSET_DIR / 'gemma3_270m_cce_frontier.png'}")


if __name__ == "__main__":
  main()
