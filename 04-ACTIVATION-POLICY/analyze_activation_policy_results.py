#!/usr/bin/env python3
"""Aggregate Gemma3 activation remat/offload benchmark artifacts."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "data" / "raw"
RUNS_ROOT = RAW_ROOT / "activation-policy-04-artifacts" / "runs"
XLA_ROOT = RAW_ROOT / "tmp"
DATA_DIR = ROOT / "data"
ASSET_DIR = ROOT / "assets"

CHIPS = 8
TPU = "v5litepod-8"
MESH = "fsdp=8,tp=1"
MODEL = "gemma3-4b-it"
HBM_LIMIT_GIB_PER_CHIP = 15.75
HBM_LIMIT_GIB_AGG = HBM_LIMIT_GIB_PER_CHIP * CHIPS
BYTES_PER_GIB = 1024**3
MAIN_CASE_IDS = {
    "l2048_default_ce_none",
    "l2048_default_ce_split_offload",
    "l4096_default_ce_none",
    "l4096_default_ce_split_offload",
}


RUN_SPECS = [
    {
        "case_id": "l2048_default_ce_none",
        "context": 2048,
        "ce": "Default CE",
        "policy": "none",
        "status": "ok",
        "run_summary": "context/l2048/none/summary.json",
        "xla_dir": "xla_act_l2048_none",
        "note": "activation policy disabled",
    },
    {
        "case_id": "l2048_default_ce_split_remat",
        "context": 2048,
        "ce": "Default CE",
        "policy": "split_remat",
        "status": "ok",
        "run_summary": "context/l2048/split_remat/summary.json",
        "xla_dir": "xla_act_l2048_split_remat",
        "note": "attention and MLP remat regions",
    },
    {
        "case_id": "l2048_default_ce_split_offload",
        "context": 2048,
        "ce": "Default CE",
        "policy": "split_offload",
        "status": "ok",
        "run_summary": "context/l2048/split_offload/summary.json",
        "xla_dir": "xla_act_l2048_split_offload",
        "note": "split remat plus pinned-host residual offload",
    },
    {
        "case_id": "l4096_default_ce_none",
        "context": 4096,
        "ce": "Default CE",
        "policy": "none",
        "status": "oom",
        "run_summary": None,
        "xla_dir": "xla_act_l4096_none",
        "note": "compile-time HBM OOM",
    },
    {
        "case_id": "l4096_default_ce_split_remat",
        "context": 4096,
        "ce": "Default CE",
        "policy": "split_remat",
        "status": "oom",
        "run_summary": None,
        "xla_dir": "xla_act_l4096_split_remat",
        "note": "compile-time HBM OOM; remat alone was not enough",
    },
    {
        "case_id": "l4096_default_ce_split_offload",
        "context": 4096,
        "ce": "Default CE",
        "policy": "split_offload",
        "status": "unknown",
        "run_summary": "context/l4096/split_offload/summary.json",
        "xla_dir": "xla_act_l4096_split_offload",
        "note": "added after the first artifact pull",
    },
    {
        "case_id": "l4096_cce_none",
        "context": 4096,
        "ce": "CCE",
        "policy": "none",
        "status": "oom",
        "run_summary": None,
        "xla_dir": "xla_act_cce_l4096_none",
        "note": "compile-time HBM OOM after CCE removed vocab-logit pressure",
    },
    {
        "case_id": "l4096_cce_split_remat",
        "context": 4096,
        "ce": "CCE",
        "policy": "split_remat",
        "status": "oom",
        "run_summary": None,
        "xla_dir": "xla_act_cce_l4096_split_remat",
        "note": "compile-time HBM OOM; remat barely moved HBM",
    },
    {
        "case_id": "l4096_cce_split_offload",
        "context": 4096,
        "ce": "CCE",
        "policy": "split_offload",
        "status": "ok",
        "run_summary": "cce_context/l4096/split_offload/summary.json",
        "xla_dir": "xla_act_cce_l4096_split_offload",
        "note": "completed 5-step run",
    },
]


def _read_json(path: Path) -> Any:
  with path.open() as f:
    return json.load(f)


def _summary(path: str | None) -> dict[str, Any]:
  if not path:
    return {}
  summary_path = RUNS_ROOT / path
  if not summary_path.exists():
    return {}
  data = _read_json(summary_path)
  if isinstance(data, list):
    return data[0] if data else {}
  return data


def _history_path(summary_path: str | None) -> Path | None:
  if not summary_path:
    return None
  path = RUNS_ROOT / summary_path
  if not path.exists():
    return None
  return path.with_name("history.csv")


def _steady_step_time(summary_path: str | None) -> float | str:
  path = _history_path(summary_path)
  if path is None or not path.exists():
    return ""
  with path.open(newline="") as f:
    rows = list(csv.DictReader(f))
  values = [
      float(row["step_time_sec"])
      for row in rows
      if int(row["step"]) >= 3 and row.get("step_time_sec")
  ]
  if not values:
    return ""
  return statistics.median(values)


def _parse_total_bytes(report: Path) -> dict[str, int]:
  totals: dict[str, int] = {}
  current_space = ""
  total_re = re.compile(r"Total bytes:\s+(\d+)")
  with report.open(errors="replace") as f:
    for line in f:
      if line.startswith("Memory Space:"):
        current_space = line.split(":", 1)[1].strip().split()[0]
      match = total_re.search(line)
      if match and current_space:
        totals[current_space] = int(match.group(1))
  return totals


def _xla_totals(xla_dir: str) -> dict[str, Any]:
  path = XLA_ROOT / xla_dir
  reports = sorted(path.glob("*train_step*memory-usage-report.txt"))
  if not reports:
    return {
        "xla_report_count": 0,
        "xla_report": "",
        "xla_hbm_gib_per_chip": "",
        "xla_hbm_gib_aggregate": "",
        "xla_vmem_gib_per_chip": "",
        "xla_cmem_gib_per_chip": "",
    }
  parsed = []
  for report in reports:
    totals = _parse_total_bytes(report)
    parsed.append((totals.get("default", 0), report, totals))
  default_bytes, report, totals = max(parsed, key=lambda item: item[0])
  return {
      "xla_report_count": len(reports),
      "xla_report": str(report.relative_to(ROOT)),
      "xla_hbm_gib_per_chip": default_bytes / BYTES_PER_GIB,
      "xla_hbm_gib_aggregate": default_bytes * CHIPS / BYTES_PER_GIB,
      "xla_vmem_gib_per_chip": totals.get("vmem", 0) / BYTES_PER_GIB,
      "xla_cmem_gib_per_chip": totals.get("cmem", 0) / BYTES_PER_GIB,
  }


def _gib(value: Any) -> float | str:
  if value in (None, ""):
    return ""
  return float(value) / BYTES_PER_GIB


def build_rows() -> list[dict[str, Any]]:
  rows: list[dict[str, Any]] = []
  for spec in RUN_SPECS:
    summary = _summary(spec["run_summary"])
    xla = _xla_totals(spec["xla_dir"])
    status = spec["status"]
    if status == "unknown":
      status = "ok" if summary else "oom"
    memory_after = summary.get("memory_after_train", {}).get("aggregate", {})
    row = {
        "case_id": spec["case_id"],
        "model": MODEL,
        "tpu": TPU,
        "chips": CHIPS,
        "mesh": MESH,
        "training_mode": "lora",
        "lora_rank": 16,
        "batch_size": 1,
        "max_length": spec["context"],
        "ce": spec["ce"],
        "activation_policy": spec["policy"],
        "status": status,
        "steps_recorded": summary.get("steps_recorded", ""),
        "xla_hbm_gib_per_chip": xla["xla_hbm_gib_per_chip"],
        "xla_hbm_gib_aggregate": xla["xla_hbm_gib_aggregate"],
        "xla_vmem_gib_per_chip": xla["xla_vmem_gib_per_chip"],
        "xla_cmem_gib_per_chip": xla["xla_cmem_gib_per_chip"],
        "runtime_peak_gib_aggregate": _gib(memory_after.get("peak_bytes_in_use")),
        "runtime_limit_gib_aggregate": _gib(memory_after.get("bytes_limit")),
        "mean_step_time_sec_excl_first": summary.get(
            "mean_step_time_sec_excl_first",
            "",
        ),
        "steady_step_time_sec_steps3_5": _steady_step_time(
            spec["run_summary"],
        ),
        "final_loss": summary.get("final_loss", ""),
        "activation_policy_installed": summary.get(
            "activation_policy_installed",
            spec["policy"] != "none",
        ),
        "ce_disabled": summary.get("ce_disabled", spec["ce"] == "Default CE"),
        "xla_report_count": xla["xla_report_count"],
        "xla_report": xla["xla_report"],
        "note": spec["note"],
    }
    rows.append(row)
  return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  keys = list(rows[0])
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def _fmt_gib(value: float | str) -> str:
  if value == "":
    return ""
  return f"{float(value):.1f}"


def plot_context_before_after_memory(rows: list[dict[str, Any]]) -> None:
  plot_rows = [
      row
      for row in rows
      if row["case_id"] in MAIN_CASE_IDS
      and row["xla_hbm_gib_aggregate"] != ""
  ]
  order = {
      (2048, "none"): 0,
      (2048, "split_offload"): 1,
      (4096, "none"): 2,
      (4096, "split_offload"): 3,
  }
  plot_rows.sort(
      key=lambda row: order[(row["max_length"], row["activation_policy"])]
  )

  x = [0.0, 0.82, 2.15, 2.97]
  values = [float(row["xla_hbm_gib_aggregate"]) for row in plot_rows]
  labels = [
      "Before\nno policy",
      "After\nsplit offload",
      "Before\nno policy",
      "After\nsplit offload",
  ]
  colors = [
      "#4A5568",
      "#2C7A7B",
      "#A23E48",
      "#2C7A7B",
  ]

  fig, ax = plt.subplots(figsize=(10.2, 5.8))
  ax.set_axisbelow(True)
  bars = ax.bar(x, values, width=0.58, color=colors, zorder=3)
  ax.axhline(
      HBM_LIMIT_GIB_AGG,
      color="#333333",
      linestyle=(0, (5, 4)),
      linewidth=1.2,
      zorder=4,
  )
  ax.text(
      3.35,
      HBM_LIMIT_GIB_AGG + 2.0,
      "v5litepod-8 limit\n126 GiB aggregate",
      ha="right",
      va="bottom",
      fontsize=9,
      color="#333333",
      zorder=5,
  )

  for bar, row, value in zip(bars, plot_rows, values, strict=True):
    status = "OOM" if row["status"] == "oom" else "OK"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        value - 5.5,
        f"{value:.1f} GiB\n{status}",
        ha="center",
        va="top",
        fontsize=9,
        color="white",
        fontweight="bold" if status == "OK" else "normal",
        zorder=5,
    )

  savings_2048 = values[0] - values[1]
  savings_4096 = values[2] - values[3]
  ax.annotate(
      f"same L2048\n-{savings_2048:.1f} GiB; both OK",
      xy=((x[0] + x[1]) / 2, max(values[0], values[1]) + 17),
      ha="center",
      va="center",
      fontsize=10,
      color="#374151",
      zorder=5,
      arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "lw": 1.2},
  )
  ax.annotate(
      f"same L4096\n-{savings_4096:.1f} GiB; OOM -> OK",
      xy=((x[2] + x[3]) / 2, max(values[2], values[3]) + 17),
      ha="center",
      va="center",
      fontsize=10,
      color="#374151",
      fontweight="bold",
      zorder=5,
      arrowprops={"arrowstyle": "-", "color": "#9CA3AF", "lw": 1.2},
  )

  ax.text(
      (x[0] + x[1]) / 2,
      -24,
      "Context length 2048",
      ha="center",
      va="top",
      fontsize=11,
      fontweight="bold",
      transform=ax.transData,
  )
  ax.text(
      (x[2] + x[3]) / 2,
      -24,
      "Context length 4096",
      ha="center",
      va="top",
      fontsize=11,
      fontweight="bold",
      transform=ax.transData,
  )

  ax.set_xticks(x, labels)
  ax.set_ylabel("XLA planned HBM, aggregate GiB\n(per-chip train-step peak x 8 chips)")
  ax.set_title("Gemma3 4B LoRA: same-context memory before/after activation offload")
  ax.set_ylim(0, max(values + [HBM_LIMIT_GIB_AGG]) * 1.24)
  ax.grid(axis="y", color="#EAEAEA", linewidth=0.8, zorder=0)
  ax.spines[["top", "right"]].set_visible(False)
  ax.text(
      0.0,
      -0.22,
      "Default CE, batch 1, LoRA rank 16, v5litepod-8. "
      "Dashed line is capacity only; savings are same-context before/after differences.",
      transform=ax.transAxes,
      ha="left",
      va="top",
      fontsize=9,
      color="#4B5563",
      zorder=5,
  )
  fig.subplots_adjust(bottom=0.25, top=0.88)
  fig.savefig(ASSET_DIR / "gemma3_4b_activation_before_after_memory.png", dpi=180)
  plt.close(fig)


def plot_context_hbm_headroom(rows: list[dict[str, Any]]) -> None:
  plot_rows = [
      row
      for row in rows
      if row["case_id"] in MAIN_CASE_IDS
      and row["xla_hbm_gib_aggregate"] != ""
  ]
  order = {
      (2048, "none"): 0,
      (2048, "split_offload"): 1,
      (4096, "none"): 2,
      (4096, "split_offload"): 3,
  }
  plot_rows.sort(
      key=lambda row: order[(row["max_length"], row["activation_policy"])]
  )

  y = [3.2, 2.45, 1.1, 0.35]
  planned = [float(row["xla_hbm_gib_aggregate"]) for row in plot_rows]
  headroom = [HBM_LIMIT_GIB_AGG - value for value in planned]
  labels = [
      "L2048 before\nno policy",
      "L2048 after\nsplit offload",
      "L4096 before\nno policy",
      "L4096 after\nsplit offload",
  ]
  colors = [
      "#4A5568",
      "#2C7A7B",
      "#A23E48",
      "#2C7A7B",
  ]

  fig, ax = plt.subplots(figsize=(10.8, 5.8))
  ax.set_axisbelow(True)
  bars = ax.barh(y, headroom, height=0.48, color=colors, zorder=3)
  ax.axvline(0, color="#111827", linewidth=1.4, zorder=4)
  ax.text(
      1.5,
      3.72,
      "0 = v5litepod-8 HBM limit",
      ha="left",
      va="center",
      fontsize=9,
      color="#111827",
      zorder=5,
  )

  for bar, row, value, planned_value in zip(
      bars,
      plot_rows,
      headroom,
      planned,
      strict=True,
  ):
    status = "OOM" if row["status"] == "oom" else "OK"
    label = f"{value:+.1f} GiB  {status}\nplanned {planned_value:.1f}"
    if value >= 20:
      x_text = value - 2.0
      ha = "right"
      color = "white"
    elif value <= -20:
      x_text = value + 2.0
      ha = "left"
      color = "white"
    elif value >= 0:
      x_text = value + 1.8
      ha = "left"
      color = "#111827"
    else:
      x_text = value - 1.8
      ha = "right"
      color = "#111827"
    ax.text(
        x_text,
        bar.get_y() + bar.get_height() / 2,
        label,
        ha=ha,
        va="center",
        fontsize=9,
        fontweight="bold" if status == "OK" else "normal",
        color=color,
        zorder=5,
    )

  swing_2048 = headroom[1] - headroom[0]
  swing_4096 = headroom[3] - headroom[2]
  ax.text(
      -66,
      3.86,
      f"L2048: both already fit; offload adds {swing_2048:.1f} GiB headroom",
      ha="left",
      va="center",
      fontsize=10,
      color="#374151",
      zorder=5,
  )
  ax.text(
      -66,
      1.73,
      f"L4096: same-context reduction = {swing_4096:.1f} GiB; OOM -> OK",
      ha="left",
      va="center",
      fontsize=10,
      color="#374151",
      fontweight="bold",
      zorder=5,
  )

  ax.set_yticks(y, labels)
  ax.set_xlabel("HBM headroom vs limit, GiB  (limit - XLA planned HBM)")
  ax.set_title("Gemma3 4B LoRA: same-context HBM headroom before/after offload")
  ax.set_xlim(-68, 68)
  ax.set_ylim(-0.25, 4.05)
  ax.grid(axis="x", color="#EAEAEA", linewidth=0.8, zorder=0)
  ax.spines[["top", "right", "left"]].set_visible(False)
  ax.text(
      0.0,
      -0.18,
      "Default CE, batch 1, LoRA rank 16, v5litepod-8, fsdp=8,tp=1. "
      "0 is a capacity threshold, not a cause of larger savings.",
      transform=ax.transAxes,
      ha="left",
      va="top",
      fontsize=9,
      color="#4B5563",
      zorder=5,
  )
  fig.subplots_adjust(bottom=0.22, top=0.86, left=0.22, right=0.96)
  fig.savefig(ASSET_DIR / "gemma3_4b_activation_hbm_headroom.png", dpi=180)
  plt.close(fig)


def plot_l4096_frontier(rows: list[dict[str, Any]]) -> None:
  ASSET_DIR.mkdir(parents=True, exist_ok=True)
  plot_rows = [
      row
      for row in rows
      if row["max_length"] == 4096
      and row["ce"] == "Default CE"
      and row["activation_policy"] in {"none", "split_offload"}
      and row["xla_hbm_gib_aggregate"] != ""
  ]
  order = {
      ("Default CE", "none"): 0,
      ("Default CE", "split_offload"): 1,
  }
  plot_rows.sort(key=lambda row: order.get((row["ce"], row["activation_policy"]), 99))

  labels = []
  values = []
  colors = []
  statuses = []
  for row in plot_rows:
    labels.append(
        "Before\nno policy"
        if row["activation_policy"] == "none"
        else "After\nsplit offload"
    )
    values.append(float(row["xla_hbm_gib_aggregate"]))
    colors.append("#2C7A7B" if row["status"] == "ok" else "#A23E48")
    statuses.append(row["status"])

  fig, ax = plt.subplots(figsize=(7.8, 5.4))
  bars = ax.bar(labels, values, color=colors, width=0.58)
  ax.axhline(
      HBM_LIMIT_GIB_AGG,
      color="#333333",
      linestyle=(0, (5, 4)),
      linewidth=1.2,
  )
  ax.text(
      len(labels) - 0.45,
      HBM_LIMIT_GIB_AGG + 2.0,
      "v5litepod-8 HBM limit\n126 GiB aggregate",
      ha="right",
      va="bottom",
      fontsize=9,
      color="#333333",
  )
  for bar, value, status in zip(bars, values, statuses, strict=True):
    y = bar.get_height()
    label = f"{value:.1f} GiB"
    if status == "oom":
      label += "\nOOM"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y + 2.0,
        label,
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold" if status == "ok" else "normal",
    )
  ax.set_title("Gemma3 4B L4096: before vs after activation offload")
  ax.set_ylabel("XLA planned HBM, aggregate GiB\n(per-chip peak x 8 chips)")
  ax.set_ylim(0, max(values + [HBM_LIMIT_GIB_AGG]) * 1.18)
  ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
  ax.spines[["top", "right"]].set_visible(False)
  fig.tight_layout()
  fig.savefig(ASSET_DIR / "gemma3_4b_l4096_activation_frontier.png", dpi=180)
  plt.close(fig)


def plot_l2048_tradeoff(rows: list[dict[str, Any]]) -> None:
  plot_rows = [
      row
      for row in rows
      if row["max_length"] == 2048
      and row["ce"] == "Default CE"
      and row["status"] == "ok"
      and row["activation_policy"] in {"none", "split_offload"}
      and row["xla_hbm_gib_aggregate"] != ""
  ]
  order = {"none": 0, "split_offload": 1}
  plot_rows.sort(key=lambda row: order[row["activation_policy"]])
  labels = [
      "Before\nno policy"
      if row["activation_policy"] == "none"
      else "After\nsplit offload"
      for row in plot_rows
  ]
  memory = [float(row["xla_hbm_gib_aggregate"]) for row in plot_rows]
  runtime_peak = [
      float(row["runtime_peak_gib_aggregate"])
      if row["runtime_peak_gib_aggregate"] != ""
      else 0.0
      for row in plot_rows
  ]
  step = [
      float(row["steady_step_time_sec_steps3_5"])
      if row["steady_step_time_sec_steps3_5"] != ""
      else 0.0
      for row in plot_rows
  ]

  fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
  ax = axes[0]
  x = range(len(labels))
  bars = ax.bar(x, memory, width=0.56, color=["#4A5568", "#2C7A7B"])
  ax.plot(x, runtime_peak, color="#D97706", marker="o", linewidth=2.0, label="runtime peak")
  for bar, value in zip(bars, memory, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1.2,
        f"{value:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )
  for idx, value in enumerate(runtime_peak):
    ax.text(idx, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=8, color="#9A3412")
  ax.set_xticks(list(x), labels)
  ax.set_ylabel("Aggregate GiB")
  ax.set_title("Memory at L2048")
  ax.legend(frameon=False, fontsize=9)
  ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
  ax.spines[["top", "right"]].set_visible(False)

  ax = axes[1]
  bars = ax.bar(labels, step, color=["#4A5568", "#2C7A7B"], width=0.56)
  for bar, value in zip(bars, step, strict=True):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{value:.3f}s",
        ha="center",
        va="bottom",
        fontsize=9,
    )
  ax.set_ylabel("Seconds / step")
  ax.set_title("Steady step-time proxy\n(median of steps 3-5)")
  ax.set_ylim(0, max(step) * 1.28)
  ax.grid(axis="y", color="#E6E6E6", linewidth=0.8)
  ax.spines[["top", "right"]].set_visible(False)
  fig.suptitle("Gemma3 4B L2048: before vs after activation offload")
  fig.tight_layout()
  fig.savefig(ASSET_DIR / "gemma3_4b_l2048_activation_tradeoff.png", dpi=180)
  plt.close(fig)


def main() -> None:
  rows = build_rows()
  main_rows = [row for row in rows if row["case_id"] in MAIN_CASE_IDS]
  write_csv(DATA_DIR / "gemma3_4b_activation_policy_keypoints.csv", main_rows)
  plot_context_before_after_memory(rows)
  plot_context_hbm_headroom(rows)
  plot_l4096_frontier(rows)
  plot_l2048_tradeoff(rows)
  manifest = {
      "model": MODEL,
      "tpu": TPU,
      "chips": CHIPS,
      "mesh": MESH,
      "hbm_limit_gib_per_chip": HBM_LIMIT_GIB_PER_CHIP,
      "hbm_limit_gib_aggregate": HBM_LIMIT_GIB_AGG,
      "rows": len(main_rows),
      "scope": "Default CE before/after only; CCE composition rows are raw artifacts, not headline comparisons.",
      "figures": [
          "assets/gemma3_4b_activation_hbm_headroom.png",
          "assets/gemma3_4b_activation_before_after_memory.png",
          "assets/gemma3_4b_l4096_activation_frontier.png",
          "assets/gemma3_4b_l2048_activation_tradeoff.png",
      ],
      "parity": "data/gemma3_4b_activation_policy_parity.json",
  }
  (DATA_DIR / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
  main()
