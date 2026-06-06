#!/usr/bin/env python3
"""Collect Gemma3 270M four-chip CCE chunk-tuning artifacts."""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import matplotlib.pyplot as plt
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "gemma3_270m_4chip_chunk"
ARTIFACT_DIR = DATA_DIR / "raw_artifacts"
RAW_DIR = DATA_DIR / "raw"
ASSET_DIR = SCRIPT_DIR / "assets"


def write_frame(frame: pd.DataFrame, path: Path) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  frame.to_csv(path, index=False)


def extract_artifacts() -> None:
  if RAW_DIR.exists():
    shutil.rmtree(RAW_DIR)
  RAW_DIR.mkdir(parents=True, exist_ok=True)
  for tarball in sorted(ARTIFACT_DIR.glob("*.tar.gz")):
    target = RAW_DIR / tarball.name.removesuffix(".tar.gz")
    if target.exists():
      shutil.rmtree(target)
    target.mkdir(parents=True)
    subprocess.run(["tar", "-xzf", str(tarball), "-C", str(target)], check=True)


def read_results() -> pd.DataFrame:
  frames = []
  for path in sorted(RAW_DIR.glob("**/*_results.csv")):
    frame = pd.read_csv(path)
    frame["source_csv"] = str(path.relative_to(SCRIPT_DIR))
    frames.append(frame)
  if not frames:
    return pd.DataFrame()
  rows = pd.concat(frames, ignore_index=True, sort=False)
  for col in [
      "batch_size",
      "max_length",
      "chips",
      "mesh_fsdp",
      "mesh_tp",
      "token_chunk",
      "vocab_chunk",
      "xla_train_step_gib_per_chip",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "final_loss",
  ]:
    if col in rows:
      rows[col] = pd.to_numeric(rows[col], errors="coerce")
  rows["ok"] = rows["status"].eq("ok")
  rows["mesh"] = rows.apply(
      lambda row: f"fsdp{int(row['mesh_fsdp'])}/tp{int(row['mesh_tp'])}",
      axis=1,
  )
  return rows.sort_values([
      "mesh_fsdp",
      "mesh_tp",
      "batch_size",
      "max_length",
      "token_chunk",
      "vocab_chunk",
  ])


def build_summary(rows: pd.DataFrame) -> pd.DataFrame:
  ok = rows[rows["ok"]].copy()
  if ok.empty:
    return pd.DataFrame()
  summary = ok[[
      "suite",
      "mesh",
      "mesh_fsdp",
      "mesh_tp",
      "batch_size",
      "max_length",
      "max_steps",
      "token_chunk",
      "vocab_chunk",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "xla_train_step_gib_per_chip",
      "final_loss",
      "source_csv",
  ]].copy()
  summary["step_time_rank"] = summary["mean_step_time_sec_excl_first"].rank(
      method="min"
  )
  summary["throughput_rank"] = (
      -summary["valid_tokens_per_sec_excl_first"]
  ).rank(method="min")
  return summary.sort_values([
      "mean_step_time_sec_excl_first",
      "token_chunk",
      "vocab_chunk",
  ])


def plot_chunk_tuning(summary: pd.DataFrame) -> None:
  if summary.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  grid = summary[
      summary["suite"].eq("fourchip_chunk_fsdp2_tp2_b16_l512")
      & summary["max_length"].eq(512)
  ].copy()
  candidate = summary[
      summary["suite"].eq("fourchip_chunk_candidate_fsdp2_tp2")
  ].copy()
  if grid.empty:
    grid = summary[summary["max_length"].eq(512)].copy()
  token_chunks = sorted(grid["token_chunk"].dropna().astype(int).unique())
  vocab_chunks = sorted(grid["vocab_chunk"].dropna().astype(int).unique())
  pivot_time = grid.pivot_table(
      index="token_chunk",
      columns="vocab_chunk",
      values="mean_step_time_sec_excl_first",
      aggfunc="first",
  ).reindex(index=token_chunks, columns=vocab_chunks)

  fig, axes = plt.subplots(
      1,
      2,
      figsize=(10.6, 4.4),
      constrained_layout=True,
  )
  fig.suptitle(
      "Gemma3 270M CCE chunk tuning on TPU v5litepod-4 (4 chips, fsdp=2/tp=2)",
      fontsize=11.5,
      fontweight="bold",
  )

  ax = axes[0]
  image = ax.imshow(pivot_time.values, cmap="YlGnBu_r")
  ax.set_title("Full grid at b16/L512, lower is better")
  ax.set_xlabel("Vocab chunk")
  ax.set_ylabel("Token chunk")
  ax.set_xticks(range(len(vocab_chunks)))
  ax.set_xticklabels([f"{value // 1024}K" for value in vocab_chunks])
  ax.set_yticks(range(len(token_chunks)))
  ax.set_yticklabels([str(value) for value in token_chunks])
  for y, token_chunk in enumerate(token_chunks):
    for x, vocab_chunk in enumerate(vocab_chunks):
      value = pivot_time.loc[token_chunk, vocab_chunk]
      if pd.notna(value):
        ax.text(x, y, f"{value:.2f}s", ha="center", va="center", fontsize=8)
  fig.colorbar(image, ax=ax, shrink=0.85)

  ax = axes[1]
  if not candidate.empty:
    max_value = candidate["mean_step_time_sec_excl_first"].max()
    for max_length, part in candidate.groupby("max_length"):
      part = part.sort_values("vocab_chunk")
      ax.plot(
          part["vocab_chunk"],
          part["mean_step_time_sec_excl_first"],
          marker="o",
          markersize=6,
          linewidth=2.2,
          label=f"L{int(max_length)}",
      )
      for _, row in part.iterrows():
        ax.text(
            row["vocab_chunk"],
            row["mean_step_time_sec_excl_first"] * 1.08,
            f"{row['mean_step_time_sec_excl_first']:.2f}s\n{row['xla_train_step_gib_per_chip']:.2f}GiB",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(candidate["vocab_chunk"].dropna().astype(int).unique()))
    ax.set_xticklabels([
        f"{value // 1024}K"
        for value in sorted(candidate["vocab_chunk"].dropna().astype(int).unique())
    ])
    ax.set_ylabel("Mean step time (seconds)")
    ax.set_xlabel("Vocab chunk, token chunk fixed at 512")
    ax.set_title("Candidate confirm run")
    ax.set_ylim(0, max_value * 1.28)
    ax.legend(frameon=False)
  else:
    best = grid.head(8).sort_values("mean_step_time_sec_excl_first")
    labels = [
        f"tc{int(row.token_chunk)}\nvc{int(row.vocab_chunk) // 1024}K"
        for row in best.itertuples(index=False)
    ]
    bars = ax.bar(
        range(len(best)),
        best["mean_step_time_sec_excl_first"],
        color="#159A78",
        width=0.72,
    )
    for bar, (_, row) in zip(bars, best.iterrows(), strict=True):
      mem = row["xla_train_step_gib_per_chip"]
      ax.text(
          bar.get_x() + bar.get_width() / 2,
          bar.get_height() * 1.02,
          f"{bar.get_height():.2f}s\n{mem:.2f}GiB",
          ha="center",
          va="bottom",
          fontsize=7,
      )
    ax.set_xticks(range(len(best)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Mean step time (seconds)")
    ax.set_title("Fastest passing chunk pairs")
  ax.set_axisbelow(True)
  ax.grid(True, axis="y", color="#E6E6E6")

  fig.savefig(ASSET_DIR / "gemma3_270m_cce_4chip_chunk_tuning.png", dpi=180)
  plt.close(fig)


def build_axis_ablation(summary: pd.DataFrame) -> pd.DataFrame:
  grid = summary[
      summary["suite"].eq("fourchip_chunk_fsdp2_tp2_b16_l512")
      & summary["max_length"].eq(512)
  ].copy()
  if grid.empty:
    return pd.DataFrame()
  slices = []
  token_fixed = grid[grid["token_chunk"].eq(128)].copy()
  token_fixed["axis"] = "vocab_sweep_token128"
  token_fixed["fixed_chunk"] = "token_chunk=128"
  token_fixed["swept_chunk"] = token_fixed["vocab_chunk"]
  slices.append(token_fixed)

  vocab_fixed = grid[grid["vocab_chunk"].eq(8192)].copy()
  vocab_fixed["axis"] = "token_sweep_vocab8192"
  vocab_fixed["fixed_chunk"] = "vocab_chunk=8192"
  vocab_fixed["swept_chunk"] = vocab_fixed["token_chunk"]
  slices.append(vocab_fixed)

  if not slices:
    return pd.DataFrame()
  axis = pd.concat(slices, ignore_index=True, sort=False)
  return axis[[
      "axis",
      "fixed_chunk",
      "swept_chunk",
      "token_chunk",
      "vocab_chunk",
      "mean_step_time_sec_excl_first",
      "valid_tokens_per_sec_excl_first",
      "xla_train_step_gib_per_chip",
      "final_loss",
      "source_csv",
  ]].sort_values(["axis", "swept_chunk"])


def plot_axis_ablation(axis: pd.DataFrame) -> None:
  if axis.empty:
    return
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  fig, axes = plt.subplots(
      1,
      2,
      figsize=(10.0, 4.0),
      constrained_layout=True,
      gridspec_kw={"width_ratios": [1.35, 1.0]},
  )
  fig.suptitle(
      "CCE chunk axis ablation: both axes reduce the same loop product",
      fontsize=11.0,
      fontweight="bold",
  )
  plot_ax = axes[0]
  colors = {
      "vocab_sweep_token128": "#1677A3",
      "token_sweep_vocab8192": "#D97904",
  }
  labels = {
      "vocab_sweep_token128": "Increase vocab chunk, token fixed at 128",
      "token_sweep_vocab8192": "Increase token chunk, vocab fixed at 8192",
  }
  base_step = axis[
      axis["token_chunk"].eq(128) & axis["vocab_chunk"].eq(8192)
  ]["mean_step_time_sec_excl_first"].iloc[0]
  for axis_name in ["vocab_sweep_token128", "token_sweep_vocab8192"]:
    part = axis[axis["axis"].eq(axis_name)].sort_values("swept_chunk").copy()
    part["chunk_multiplier"] = part["swept_chunk"] / part["swept_chunk"].iloc[1]
    plot_ax.plot(
        part["chunk_multiplier"],
        part["mean_step_time_sec_excl_first"],
        marker="o",
        markersize=7,
        linewidth=2.4,
        color=colors[axis_name],
        label=labels[axis_name],
    )
  plot_ax.axhline(base_step, color="#777777", linewidth=1.0, linestyle="--")
  plot_ax.text(
      0.53,
      base_step * 1.03,
      "128 / 8192 baseline",
      ha="left",
      va="bottom",
      fontsize=8,
      color="#555555",
  )
  plot_ax.set_xscale("log", base=2)
  plot_ax.set_xticks([0.5, 1, 2, 4])
  plot_ax.set_xticklabels(["0.5x", "1x", "2x", "4x"])
  plot_ax.set_xlabel("Chunk size multiplier on the swept axis")
  plot_ax.set_ylabel("Mean step time (seconds)")
  plot_ax.set_title("Near-overlap means the bottleneck is chunk granularity")
  plot_ax.grid(True, axis="y", color="#E8E8E8")
  plot_ax.set_axisbelow(True)
  plot_ax.legend(frameon=False, loc="upper right", fontsize=8)
  plot_ax.set_ylim(0, axis["mean_step_time_sec_excl_first"].max() * 1.18)
  table_ax = axes[1]
  table_ax.axis("off")
  table_ax.set_title("What changed?", fontsize=10.0)
  table_rows = [
      ["Baseline", "128 / 8192", "15.39s", "1.0x"],
      ["Vocab 4x", "128 / 32768", "4.11s", "3.75x"],
      ["Token 4x", "512 / 8192", "4.10s", "3.76x"],
      ["Both large", "512 / 65536", "0.83s", "18.5x"],
  ]
  table = table_ax.table(
      cellText=table_rows,
      colLabels=["Case", "token / vocab", "Step", "Speedup"],
      loc="center",
      cellLoc="center",
      colLoc="center",
      colWidths=[0.28, 0.30, 0.18, 0.20],
  )
  table.auto_set_font_size(False)
  table.set_fontsize(8.2)
  table.scale(1.0, 1.45)
  for (row, _), cell in table.get_celld().items():
    cell.set_edgecolor("#DDDDDD")
    if row == 0:
      cell.set_facecolor("#F3F3F3")
      cell.set_text_props(weight="bold")
  table_ax.text(
      0.5,
      0.10,
      "XLA planned HBM was 2.65 GiB/chip for the one-axis rows.",
      transform=table_ax.transAxes,
      ha="center",
      va="center",
      fontsize=8,
      color="#555555",
      wrap=True,
  )
  fig.savefig(ASSET_DIR / "gemma3_270m_cce_4chip_chunk_axis_ablation.png", dpi=180)
  plt.close(fig)


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
        "model": "google/gemma-3-270m-it",
    })
  return pd.DataFrame(rows)


def main() -> None:
  DATA_DIR.mkdir(parents=True, exist_ok=True)
  ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
  extract_artifacts()
  rows = read_results()
  write_frame(build_manifest(), DATA_DIR / "run_manifest.csv")
  if rows.empty:
    print("No four-chip chunk-tuning results found")
    return
  summary = build_summary(rows)
  axis = build_axis_ablation(summary)
  write_frame(rows, DATA_DIR / "chunk_runs.csv")
  write_frame(summary, DATA_DIR / "chunk_summary.csv")
  write_frame(axis, DATA_DIR / "chunk_axis_ablation.csv")
  plot_chunk_tuning(summary)
  plot_axis_ablation(axis)
  print(f"wrote={DATA_DIR}")
  print(f"figure={ASSET_DIR / 'gemma3_270m_cce_4chip_chunk_tuning.png'}")
  print(f"figure={ASSET_DIR / 'gemma3_270m_cce_4chip_chunk_axis_ablation.png'}")


if __name__ == "__main__":
  main()
