#!/usr/bin/env python3
"""Collect Gemma 1B/E2B CCE transfer artifacts and redraw comparison plots."""

from __future__ import annotations

import math
from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma_1b_e2b_cce_transfer"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"

MODEL_ORDER = ["Gemma3 270M", "Gemma3 1B", "Gemma4 E2B"]
MODEL_LABEL = {
    "Gemma3 270M": "Gemma3 270M",
    "Gemma3 1B": "Gemma3 1B",
    "Gemma4 E2B": "Gemma4 E2B",
}
VARIANT_COLOR = {"default": "#555555", "cce": "#159A78"}
VARIANT_LABEL = {"default": "Default CE", "cce": "CCE"}


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def extract_artifacts() -> None:
  if RAW_DIR.exists():
    shutil.rmtree(RAW_DIR)
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    target = RAW_DIR / tarball.name.removesuffix(".tar.gz")
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(target)], check=True)


def numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
  for col in columns:
    if col in frame:
      frame[col] = pd.to_numeric(frame[col], errors="coerce")
  return frame


def add_common_columns(rows: pd.DataFrame) -> pd.DataFrame:
  if rows.empty:
    return rows
  numeric(
      rows,
      [
          "batch_size",
          "max_length",
          "chips",
          "mesh_fsdp",
          "mesh_tp",
          "token_chunk",
          "vocab_chunk",
          "max_steps",
          "xla_train_step_gib_per_chip",
          "mean_step_time_sec_excl_first",
          "valid_tokens_per_sec_excl_first",
          "final_loss",
          "eval_loss",
          "wall_time_sec",
          "steps_recorded",
      ],
  )
  rows["ok"] = rows["status"].eq("ok")
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}"
      if pd.notna(row.get("mesh_fsdp")) and pd.notna(row.get("mesh_tp"))
      else "",
      axis=1,
  )
  rows["variant_label"] = rows["variant"].map(VARIANT_LABEL).fillna(rows["variant"])
  rows["model"] = rows["model"].replace({"Gemma4 E2B": "Gemma4 E2B"})
  rows["model_order"] = rows["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return rows


def read_transfer_results() -> pd.DataFrame:
  frames = []
  for path in sorted(RAW_DIR.glob("**/*_results.csv")):
    frame = pd.read_csv(path)
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  rows = pd.concat(frames, ignore_index=True, sort=False)
  rows = add_common_columns(rows)
  dedupe = [
      "model",
      "suite",
      "case",
      "variant",
      "batch_size",
      "max_length",
      "mesh_fsdp",
      "mesh_tp",
      "token_chunk",
      "vocab_chunk",
  ]
  rows = rows.drop_duplicates(subset=[col for col in dedupe if col in rows], keep="first")
  return rows.sort_values([
      "model_order",
      "mesh_fsdp",
      "mesh_tp",
      "batch_size",
      "max_length",
      "variant",
  ])


def read_270m_frontier() -> pd.DataFrame:
  path = SCRIPT_DIR / "data" / "gemma3_270m_4chip_frontier" / "frontier_runs.csv"
  if not path.exists():
    return pd.DataFrame()
  rows = pd.read_csv(path)
  rows["source_csv"] = str(path.relative_to(SCRIPT_DIR))
  return add_common_columns(rows)


def read_270m_quality() -> tuple[pd.DataFrame, pd.DataFrame]:
  quality_dir = SCRIPT_DIR / "data" / "gemma3_270m_4chip_quality"
  summary_path = quality_dir / "training_summary.csv"
  history_path = quality_dir / "training_history.csv"
  summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
  history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
  if not summary.empty:
    summary["source_csv"] = str(summary_path.relative_to(SCRIPT_DIR))
    summary = add_common_columns(summary)
  if not history.empty:
    history["model"] = "Gemma3 270M"
    history["shape"] = "b16/L512"
  return summary, history


def case_history_path(case: str) -> Path | None:
  matches = sorted(RAW_DIR.glob(f"**/{case}/history.csv"))
  return matches[0] if matches else None


def read_transfer_history(rows: pd.DataFrame) -> pd.DataFrame:
  frames = []
  quality = rows[rows["dataset_mode"].eq("opus100") & rows["ok"]].copy()
  for row in quality.itertuples(index=False):
    path = case_history_path(row.case)
    if path is None:
      continue
    frame = pd.read_csv(path)
    frame["case"] = row.case
    frame["variant"] = row.variant
    frame["variant_label"] = VARIANT_LABEL.get(row.variant, row.variant)
    frame["batch_size"] = row.batch_size
    frame["max_length"] = row.max_length
    frame["model"] = row.model
    frame["shape"] = f"b{int(row.batch_size)}/L{int(row.max_length)}"
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  history = pd.concat(frames, ignore_index=True, sort=False)
  numeric(history, ["step", "loss", "step_time_sec", "cumulative_loss_tokens"])
  return history.sort_values(["model", "variant", "step"])


def build_manifest() -> pd.DataFrame:
  rows = []
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    rows.append({
        "artifact": str(tarball.relative_to(SCRIPT_DIR)),
        "bytes": tarball.stat().st_size,
        "profile": tarball.name.removesuffix(".tar.gz"),
        "tpu": "v5litepod-4",
        "chips": 4,
        "project": "gcp-ml-172005",
        "zone": "us-west4-a",
    })
  return pd.DataFrame(rows)


def build_frontier_summary(rows: pd.DataFrame) -> pd.DataFrame:
  frontier = rows[
      rows["suite"].fillna("").str.contains("fourchip_frontier")
      & rows["mesh"].eq("fsdp4/tp1")
  ].copy()
  out = []
  for keys, part in frontier.groupby(["model", "batch_size", "variant"], dropna=False):
    ok = part[part["ok"]]
    failed = part[~part["ok"]]
    out.append({
        "model": keys[0],
        "batch_size": int(keys[1]),
        "variant": keys[2],
        "max_ok_context": int(ok["max_length"].max()) if not ok.empty else 0,
        "first_failed_context": int(failed["max_length"].min()) if not failed.empty else "",
        "ok_rows": int(ok.shape[0]),
        "failed_rows": int(failed.shape[0]),
        "mesh": "fsdp4/tp1",
        "tpu": "v5litepod-4",
        "chips": 4,
    })
  summary = pd.DataFrame(out)
  if summary.empty:
    return summary
  summary["model_order"] = summary["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return summary.sort_values(["model_order", "batch_size", "variant"]).drop(columns="model_order")


def build_matched_metrics(rows: pd.DataFrame) -> pd.DataFrame:
  ok = rows[rows["ok"]].copy()
  metric_cols = [
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "final_loss",
      "eval_loss",
  ]
  matched = ok.pivot_table(
      index=["model", "suite", "mesh", "batch_size", "max_length"],
      columns="variant",
      values=[col for col in metric_cols if col in ok],
      aggfunc="first",
  )
  if matched.empty:
    return pd.DataFrame()
  matched.columns = ["_".join(col).strip("_") for col in matched.columns.values]
  matched = matched.reset_index()
  if {
      "xla_train_step_gib_per_chip_default",
      "xla_train_step_gib_per_chip_cce",
  } <= set(matched.columns):
    matched["xla_reduction_pct"] = (
        1
        - matched["xla_train_step_gib_per_chip_cce"]
        / matched["xla_train_step_gib_per_chip_default"]
    ) * 100
  if {
      "mean_step_time_sec_excl_first_default",
      "mean_step_time_sec_excl_first_cce",
  } <= set(matched.columns):
    matched["step_time_multiplier_cce_vs_default"] = (
        matched["mean_step_time_sec_excl_first_cce"]
        / matched["mean_step_time_sec_excl_first_default"]
    )
  if {"final_loss_default", "final_loss_cce"} <= set(matched.columns):
    matched["loss_delta_cce_minus_default"] = (
        matched["final_loss_cce"] - matched["final_loss_default"]
    )
  matched["model_order"] = matched["model"].map({name: idx for idx, name in enumerate(MODEL_ORDER)})
  return matched.sort_values(["model_order", "mesh", "batch_size", "max_length"]).drop(
      columns="model_order"
  )


def context_rank(value: float) -> int:
  value = int(value)
  if value <= 0:
    return 0
  return {256: 1, 512: 2, 1024: 3, 2048: 4, 4096: 5, 8192: 6}.get(
      value, int(math.log2(value // 256)) + 1
  )


def context_label(value: float) -> str:
  value = int(value)
  if value <= 0:
    return "none"
  return f"{value // 1024}K" if value >= 1024 else str(value)


def plot_frontier(summary: pd.DataFrame) -> None:
  if summary.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(
      3,
      1,
      figsize=(9.4, 8.0),
      constrained_layout=True,
      sharex=False,
  )
  fig.suptitle(
      "CCE shifts the FSDP-only context frontier across Gemma sizes\n"
      "TPU v5litepod-4, 4 chips, fsdp=4/tp=1, LoRA rank 16",
      fontsize=11.5,
      fontweight="bold",
  )
  batches = [1, 4, 8, 16, 32, 64, 128]
  width = 0.35
  for ax, model in zip(axes, MODEL_ORDER, strict=True):
    part = summary[summary["model"].eq(model)].copy()
    x_base = list(range(len(batches)))
    for idx, variant in enumerate(["default", "cce"]):
      variant_part = part[part["variant"].eq(variant)].set_index("batch_size")
      values = [
          variant_part.loc[batch, "max_ok_context"] if batch in variant_part.index else 0
          for batch in batches
      ]
      heights = [context_rank(value) for value in values]
      bars = ax.bar(
          [x + (-0.5 + idx) * width for x in x_base],
          heights,
          width=width,
          color=VARIANT_COLOR[variant],
          label=VARIANT_LABEL[variant],
      )
      for bar, value, height in zip(bars, values, heights, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.05,
            context_label(value),
            ha="center",
            va="bottom",
            fontsize=6.6,
        )
    ax.set_ylim(-0.2, 6.45)
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(["none", "256", "512", "1K", "2K", "4K", "8K"])
    ax.set_ylabel(f"{MODEL_LABEL[model]}\nmax context")
    ax.set_xticks(x_base)
    ax.set_xticklabels([f"b{batch}" for batch in batches], fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E6E6E6")
    if model == MODEL_ORDER[0]:
      ax.legend(frameon=False, loc="upper right", fontsize=8)
  axes[-1].set_xlabel("Batch size")
  fig.savefig(ASSET_DIR / "gemma_cce_transfer_frontier.png", dpi=190)
  plt.close(fig)


def plot_quality(summary: pd.DataFrame, history: pd.DataFrame) -> None:
  if summary.empty:
    return
  ok = summary[summary["ok"]].copy()
  ok["shape"] = ok.apply(
      lambda row: f"b{int(row['batch_size'])}/L{int(row['max_length'])}", axis=1
  )
  models = [model for model in MODEL_ORDER if model in set(ok["model"])]
  x = list(range(len(models)))
  width = 0.34
  fig, axes = plt.subplots(
      3,
      1,
      figsize=(9.0, 8.8),
      constrained_layout=True,
      gridspec_kw={"height_ratios": [1.05, 1.05, 1.2]},
  )
  fig.suptitle(
      "Same-shape OPUS100 sanity runs: lower planned HBM, similar loss band\n"
      "TPU v5litepod-4, 4 chips, fsdp=4/tp=1, LoRA rank 16, 1,000 steps",
      fontsize=11.2,
      fontweight="bold",
  )
  metrics = [
      ("xla_train_step_gib_per_chip", "XLA planned HBM (GiB/chip)", "{:.2f}"),
      ("mean_step_time_sec_excl_first", "Mean step time (sec)", "{:.3f}"),
  ]
  for ax, (col, ylabel, fmt) in zip(axes[:2], metrics, strict=True):
    for idx, variant in enumerate(["default", "cce"]):
      values = []
      labels = []
      for model in models:
        row = ok[ok["model"].eq(model) & ok["variant"].eq(variant)].iloc[0]
        values.append(row[col])
        labels.append(fmt.format(row[col]))
      bars = ax.bar(
          [i + (-0.5 + idx) * width for i in x],
          values,
          width=width,
          color=VARIANT_COLOR[variant],
          label=VARIANT_LABEL[variant],
      )
      for bar, label in zip(bars, labels, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.02,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{MODEL_LABEL[model]}\n{ok[ok['model'].eq(model)].iloc[0]['shape']}"
        for model in models
    ])
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#E6E6E6")
    ax.legend(frameon=False, fontsize=8, loc="upper right")

  ax = axes[2]
  for model in models:
    part = history[history["model"].eq(model)].copy()
    if part.empty:
      continue
    for variant in ["default", "cce"]:
      line = part[part["variant"].eq(variant)].sort_values("step")
      if line.empty:
        continue
      smooth = line["loss"].rolling(25, min_periods=1).mean()
      ax.plot(
          line["step"],
          smooth,
          color=VARIANT_COLOR[variant],
          linewidth=1.7,
          alpha=0.95 if model == models[-1] else 0.55,
          linestyle="-" if model != "Gemma4 E2B" else "--",
          label=f"{MODEL_LABEL[model]} {VARIANT_LABEL[variant]}",
      )
  ax.set_xlabel("Training step")
  ax.set_ylabel("Smoothed train loss")
  ax.set_title("Loss curves are sanity checks, not translation-quality claims")
  ax.set_axisbelow(True)
  ax.grid(True, color="#E6E6E6")
  ax.legend(frameon=False, ncol=2, fontsize=7.4)
  fig.savefig(ASSET_DIR / "gemma_cce_transfer_quality.png", dpi=190)
  plt.close(fig)


def conservative_and_best(rows: pd.DataFrame) -> pd.DataFrame:
  rows = rows[rows["ok"] & rows["variant"].eq("cce")].copy()
  rows = rows[rows["suite"].fillna("").str.contains("chunk")]
  if rows.empty:
    return pd.DataFrame()
  key_cols = ["model", "mesh", "batch_size", "max_length"]
  out = []
  for keys, part in rows.groupby(key_cols, dropna=False):
    conservative = part[(part["token_chunk"].eq(128)) & (part["vocab_chunk"].eq(8192))]
    best = part.sort_values("mean_step_time_sec_excl_first").head(1)
    if conservative.empty:
      conservative = best
    for label, frame in [("conservative", conservative.head(1)), ("best", best)]:
      row = frame.iloc[0]
      out.append({
          "model": keys[0],
          "mesh": keys[1],
          "batch_size": int(keys[2]),
          "max_length": int(keys[3]),
          "selection": label,
          "token_chunk": int(row["token_chunk"]),
          "vocab_chunk": int(row["vocab_chunk"]),
          "mean_step_time_sec_excl_first": row["mean_step_time_sec_excl_first"],
          "xla_train_step_gib_per_chip": row["xla_train_step_gib_per_chip"],
      })
  return pd.DataFrame(out)


def plot_chunk_and_mesh(rows: pd.DataFrame, chunk_summary: pd.DataFrame) -> None:
  fig, axes = plt.subplots(2, 1, figsize=(9.4, 7.8), constrained_layout=True)
  fig.suptitle(
      "Chunk policy and mesh layout explain the remaining CCE tradeoff\n"
      "TPU v5litepod-4, 4 chips",
      fontsize=11.4,
      fontweight="bold",
  )

  ax = axes[0]
  chunk = conservative_and_best(rows)
  old_chunk = pd.read_csv(SCRIPT_DIR / "data" / "gemma3_270m_4chip_chunk" / "chunk_summary.csv")
  old_chunk = old_chunk[
      old_chunk["suite"].isin([
          "fourchip_chunk_fsdp2_tp2_b16_l512",
          "fourchip_chunk_candidate_fsdp2_tp2",
      ])
      & old_chunk["batch_size"].eq(16)
      & old_chunk["max_length"].eq(512)
  ].copy()
  if not old_chunk.empty:
    old_rows = conservative_and_best(add_common_columns(old_chunk.assign(
        model="Gemma3 270M",
        variant="cce",
        status="ok",
    )))
    chunk = pd.concat([old_rows, chunk], ignore_index=True, sort=False)
  labels = []
  groups = []
  for row in chunk[["model", "batch_size", "max_length", "mesh"]].drop_duplicates().itertuples(index=False):
    group = chunk[
        chunk["model"].eq(row.model)
        & chunk["batch_size"].eq(row.batch_size)
        & chunk["max_length"].eq(row.max_length)
        & chunk["mesh"].eq(row.mesh)
    ]
    if {"conservative", "best"} <= set(group["selection"]):
      groups.append(group)
      labels.append(f"{MODEL_LABEL.get(row.model, row.model)}\n{row.mesh} b{int(row.batch_size)}/L{int(row.max_length)}")
  x = list(range(len(groups)))
  height = 0.34
  for idx, selection in enumerate(["conservative", "best"]):
    values = [group[group["selection"].eq(selection)].iloc[0]["mean_step_time_sec_excl_first"] for group in groups]
    bars = ax.barh(
        [i + (0.5 - idx) * height for i in x],
        values,
        height=height,
        color="#8A8A8A" if selection == "conservative" else "#159A78",
        label="128/8192" if selection == "conservative" else "fastest tested",
    )
    for bar, group in zip(bars, groups, strict=True):
      row = group[group["selection"].eq(selection)].iloc[0]
      ax.text(
          bar.get_width() * 1.01,
          bar.get_y() + bar.get_height() / 2,
          f"{bar.get_width():.2f}s  {int(row['token_chunk'])}/{int(row['vocab_chunk'])//1024}K",
          ha="left",
          va="center",
          fontsize=7.2,
      )
  ax.set_yticks(x)
  ax.set_yticklabels(labels, fontsize=7.4)
  ax.invert_yaxis()
  ax.set_xlabel("Mean step time (sec)")
  ax.set_title("Larger chunks can buy back throughput without raising planned HBM")
  ax.set_axisbelow(True)
  ax.grid(True, axis="x", color="#E6E6E6")
  ax.legend(frameon=False, fontsize=7.4)

  ax = axes[1]
  mesh = rows[
      rows["suite"].fillna("").str.contains("mesh_probe")
      | (
          rows["suite"].fillna("").str.contains("fourchip_frontier")
          & rows["batch_size"].eq(1)
          & rows["max_length"].eq(256)
          & rows["mesh"].isin(["fsdp4/tp1", "fsdp2/tp2"])
      )
  ].copy()
  mesh = mesh[mesh["ok"]].copy()
  mesh["label"] = mesh.apply(
      lambda row: f"{MODEL_LABEL.get(row['model'], row['model'])}\n{row['mesh']} {VARIANT_LABEL.get(row['variant'], row['variant'])}",
      axis=1,
  )
  mesh = mesh.sort_values(["model_order", "mesh_fsdp", "mesh_tp", "variant"])
  bars = ax.barh(
      range(len(mesh)),
      mesh["mean_step_time_sec_excl_first"],
      color=[VARIANT_COLOR.get(value, "#555555") for value in mesh["variant"]],
      height=0.7,
  )
  for bar, row in zip(bars, mesh.itertuples(index=False), strict=True):
    ax.text(
        bar.get_width() * 1.01,
        bar.get_y() + bar.get_height() / 2,
        f"{row.mean_step_time_sec_excl_first:.1f}s",
        ha="left",
        va="center",
        fontsize=7,
    )
  ax.set_yticks(range(len(mesh)))
  ax.set_yticklabels(mesh["label"], fontsize=7.2)
  ax.invert_yaxis()
  ax.set_xlabel("Mean step time at b1/L256 (sec)")
  ax.set_title("TP-heavy meshes are impractical here even when memory fits")
  ax.set_axisbelow(True)
  ax.grid(True, axis="x", color="#E6E6E6")
  fig.savefig(ASSET_DIR / "gemma_cce_transfer_chunk_mesh.png", dpi=190)
  plt.close(fig)


def main() -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  extract_artifacts()
  transfer = read_transfer_results()
  frontier_rows = pd.concat([read_270m_frontier(), transfer], ignore_index=True, sort=False)
  quality_270m, history_270m = read_270m_quality()
  quality_transfer = transfer[transfer["dataset_mode"].eq("opus100")].copy()
  quality_rows = pd.concat([quality_270m, quality_transfer], ignore_index=True, sort=False)
  history = pd.concat([history_270m, read_transfer_history(transfer)], ignore_index=True, sort=False)

  frontier_summary = build_frontier_summary(frontier_rows)
  matched = build_matched_metrics(frontier_rows)
  chunk_rows = transfer[transfer["suite"].fillna("").str.contains("chunk")].copy()
  chunk_summary = conservative_and_best(transfer)

  write_frame(build_manifest(), DATA_DIR / "run_manifest.csv")
  write_frame(transfer, DATA_DIR / "transfer_runs.csv")
  write_frame(frontier_summary, DATA_DIR / "frontier_summary.csv")
  write_frame(matched, DATA_DIR / "matched_metrics.csv")
  write_frame(chunk_rows, DATA_DIR / "chunk_runs.csv")
  write_frame(chunk_summary, DATA_DIR / "chunk_summary.csv")
  write_frame(quality_rows, DATA_DIR / "training_summary.csv")
  write_frame(history, DATA_DIR / "training_history.csv")

  plot_frontier(frontier_summary)
  plot_quality(quality_rows, history)
  plot_chunk_and_mesh(transfer, chunk_summary)

  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma_cce_transfer_frontier.png'}")
  print(f"figure={ASSET_DIR / 'gemma_cce_transfer_quality.png'}")
  print(f"figure={ASSET_DIR / 'gemma_cce_transfer_chunk_mesh.png'}")


if __name__ == "__main__":
  main()
