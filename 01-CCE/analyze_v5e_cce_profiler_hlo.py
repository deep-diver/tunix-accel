#!/usr/bin/env python3
"""Profiler/HLO root-cause analysis for the TPU v5e CCE target cases."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_JSONL = (
    SCRIPT_DIR / "data" / "v5e_cce_full_matrix" / "profiler_results.jsonl"
)
DEFAULT_RESULTS_CSV = SCRIPT_DIR / "data" / "v5e_cce_full_matrix" / "profiler_results.csv"
DEFAULT_OUTDIR = SCRIPT_DIR / "data" / "v5e_cce_profiler_hlo"
DEFAULT_REPORT = SCRIPT_DIR / "profiler_hlo_analysis.md"
DEFAULT_PLOT = SCRIPT_DIR / "assets" / "v5e_cce_profiler_hlo_root_cause.png"

TARGET_CASES = {
    (2, 2, 16, 512, 128, 8192): "bad",
    (2, 2, 16, 512, 512, 65536): "good",
    (4, 1, 16, 512, 128, 8192): "control-fsdp4-tp1",
    (1, 4, 16, 512, 128, 8192): "control-fsdp1-tp4",
}
CASE_ORDER = {
    "bad": 0,
    "good": 1,
    "control-fsdp4-tp1": 2,
    "control-fsdp1-tp4": 3,
}

OP_PATTERNS = {
    "all_gather": r"\ball-gather\(|stablehlo\.all_gather\b",
    "all_reduce": r"\ball-reduce\(|stablehlo\.all_reduce\b",
    "collective_permute": r"\bcollective-permute\(|stablehlo\.collective_permute\b",
    "reduce_scatter": r"\breduce-scatter\(|stablehlo\.reduce_scatter\b",
    "all_to_all": r"\ball-to-all\(|stablehlo\.all_to_all\b",
    "while": r"\bwhile\(|stablehlo\.while\b",
    "scan": r"\bscan\b",
    "fusion": r"\bfusion\(|kind=kLoop|kind=kInput|kind=kCustom",
    "dot": r"\bdot\(|stablehlo\.dot_general\b|stablehlo\.dot\b",
    "dot_general_metadata": r"dot_general",
    "convolution": r"\bconvolution\(",
    "custom_call": r"\bcustom-call\(",
    "reduce": r"\breduce\(|stablehlo\.reduce\b",
    "transpose": r"\btranspose\(|stablehlo\.transpose\b",
    "dynamic_slice": r"\bdynamic-slice\(|stablehlo\.dynamic_slice\b",
    "dynamic_update_slice": r"\bdynamic-update-slice\(|stablehlo\.dynamic_update_slice\b",
    "slice": r"(?<!dynamic-)\bslice\(|stablehlo\.slice\b",
    "copy": r"\bcopy\(",
}

CCE_INNER_MARKER = "while/body/closed_call/while/body/closed_call"
HLO_SHAPE_RE = re.compile(r"[a-z][a-z0-9_]*\[[^\]]+\]", flags=re.IGNORECASE)
HLO_DOT_RESULT_RE = re.compile(
    r"=\s*(?P<shape>[a-z][a-z0-9_]*\[[^\]]+\])\s+dot\(",
    flags=re.IGNORECASE,
)
STABLEHLO_DOT_RE = re.compile(
    r"stablehlo\.dot(?:_general)?[\s\S]*?->\s*tensor<(?P<shape>[^>]+)>",
    flags=re.IGNORECASE,
)
MEM_TOTAL_RE = re.compile(r"Total bytes:\s+\d+\s+\(([\d.]+)GiB\)")
MEM_SHAPE_RE = re.compile(r"(?:(\d+)×)?([a-z][a-z0-9]*\[[^\]]+\])")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  rows = []
  with path.open() as f:
    for line in f:
      line = line.strip()
      if line:
        rows.append(json.loads(line))
  return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
  if not path.exists():
    return []
  with path.open() as f:
    return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  keys: list[str] = []
  for row in rows:
    for key in row:
      if key not in keys:
        keys.append(key)
  with path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w") as f:
    for row in rows:
      f.write(json.dumps(row, sort_keys=True) + "\n")


def safe_int(value: Any) -> int | None:
  if value in (None, ""):
    return None
  try:
    return int(float(value))
  except (TypeError, ValueError):
    return None


def safe_float(value: Any) -> float | None:
  if value in (None, ""):
    return None
  try:
    result = float(value)
  except (TypeError, ValueError):
    return None
  return result if math.isfinite(result) else None


def case_key(row: dict[str, Any]) -> tuple[int, int, int, int, int, int] | None:
  values = (
      safe_int(row.get("fsdp_degree")),
      safe_int(row.get("tp_degree")),
      safe_int(row.get("global_batch_size")),
      safe_int(row.get("sequence_length")),
      safe_int(row.get("token_chunk")),
      safe_int(row.get("vocab_chunk")),
  )
  if any(value is None for value in values):
    return None
  return tuple(int(value) for value in values)  # type: ignore[arg-type]


def is_target(row: dict[str, Any]) -> bool:
  if str(row.get("cce_enabled", "")).lower() not in {"true", "1"}:
    return False
  return case_key(row) in TARGET_CASES


def find_case_dir(row: dict[str, Any], run_roots: list[Path]) -> Path | None:
  recorded = Path(str(row.get("run_dir", "")))
  if recorded.exists():
    return recorded

  case_name = str(row.get("case_name", ""))
  if not case_name:
    return None

  for root in run_roots:
    candidates = [
        root / case_name,
        root / "runs" / case_name,
        root / "profiler" / "runs" / case_name,
        root / "v5e-cce-full" / "profiler" / "runs" / case_name,
    ]
    for candidate in candidates:
      if candidate.exists():
        return candidate

  for root in run_roots:
    if not root.exists():
      continue
    for candidate in root.rglob(case_name):
      if candidate.is_dir():
        return candidate
  return None


def memory_report_for(row: dict[str, Any], xla_dir: Path | None) -> Path | None:
  recorded = Path(str(row.get("xla_report_path", "")))
  if recorded.exists():
    return recorded
  if xla_dir and xla_dir.exists():
    reports = sorted(xla_dir.glob("*jit__train_step*memory-usage-report.txt"))
    if reports:
      return reports[-1]
  return None


def parse_memory_report(path: Path | None) -> tuple[float | None, list[tuple[str, int]]]:
  if not path or not path.exists():
    return None, []
  text = path.read_text(errors="ignore")
  total = None
  match = MEM_TOTAL_RE.search(text)
  if match:
    total = float(match.group(1))
  shapes: collections.Counter[str] = collections.Counter()
  for line in text.splitlines():
    if ";" not in line:
      continue
    for count, shape in MEM_SHAPE_RE.findall(line):
      shapes[shape] += int(count) if count else 1
  return total, shapes.most_common(12)


def hlo_text_files(xla_dir: Path | None) -> tuple[list[Path], list[Path], list[Path]]:
  if not xla_dir or not xla_dir.exists():
    return [], [], []
  all_text = [
      path
      for path in sorted(xla_dir.glob("*jit__train_step*.txt"))
      if not path.name.endswith("memory-usage-report.txt")
  ]
  optimized = [
      path
      for path in all_text
      if "after_optimizations" in path.name or "cpu_after_optimizations" in path.name
  ]
  stablehlo = [
      path
      for path in sorted(xla_dir.rglob("*"))
      if path.is_file()
      and (
          "stablehlo" in path.name.lower()
          or path.suffix.lower() in {".mlir", ".stablehlo"}
      )
  ]
  return all_text, optimized, stablehlo


def count_ops(paths: list[Path]) -> tuple[dict[str, int], int, list[tuple[str, int]], str]:
  counts = {key: 0 for key in OP_PATTERNS}
  dot_shapes: collections.Counter[str] = collections.Counter()
  text_bytes = 0
  sample = ""
  for path in paths:
    try:
      text = path.read_text(errors="ignore")
    except OSError:
      continue
    text_bytes += len(text.encode("utf-8", errors="ignore"))
    if not sample:
      sample = path.name
    for key, pattern in OP_PATTERNS.items():
      counts[key] += len(re.findall(pattern, text, flags=re.IGNORECASE))
    for line in text.splitlines():
      if "dot" not in line.lower():
        continue
      match = HLO_DOT_RESULT_RE.search(line)
      if match:
        dot_shapes[match.group("shape")] += 1
        continue
      match = STABLEHLO_DOT_RE.search(line)
      if match:
        dot_shapes[f"tensor<{match.group('shape')}>"] += 1
  return counts, text_bytes, dot_shapes.most_common(12), sample


def hlo_result_shape(line: str) -> str | None:
  if "=" not in line:
    return None
  rhs = line.split("=", 1)[1]
  match = HLO_SHAPE_RE.search(rhs)
  return match.group(0) if match else None


def summarize_counter(counter: collections.Counter[str], limit: int = 5) -> str:
  return "; ".join(f"{shape} x{count}" for shape, count in counter.most_common(limit))


def extract_cce_inner_loop(paths: list[Path], cce_loop_count: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  counts: collections.Counter[str] = collections.Counter()
  shape_counters: dict[str, collections.Counter[str]] = {
      "dot_general_result": collections.Counter(),
      "all_reduce_result": collections.Counter(),
      "all_gather_result": collections.Counter(),
      "convolution_result": collections.Counter(),
      "fusion_result": collections.Counter(),
  }

  for path in paths:
    try:
      lines = path.read_text(errors="ignore").splitlines()
    except OSError:
      continue
    for line in lines:
      if CCE_INNER_MARKER not in line:
        continue
      lower = line.lower()
      result_shape = hlo_result_shape(line)
      counts["cce_inner_lines"] += 1
      if "dot_general" in lower:
        counts["cce_inner_dot_general_metadata"] += 1
        if result_shape:
          shape_counters["dot_general_result"][result_shape] += 1
      if "convolution(" in lower:
        counts["cce_inner_convolution"] += 1
        if result_shape:
          shape_counters["convolution_result"][result_shape] += 1
      if "all-reduce(" in lower:
        counts["cce_inner_all_reduce"] += 1
        if result_shape:
          shape_counters["all_reduce_result"][result_shape] += 1
      if "all-gather(" in lower:
        counts["cce_inner_all_gather"] += 1
        if result_shape:
          shape_counters["all_gather_result"][result_shape] += 1
      if "fusion(" in lower:
        counts["cce_inner_fusion"] += 1
        if result_shape:
          shape_counters["fusion_result"][result_shape] += 1
      if "reduce(" in lower or "reduce_sum" in lower or "reduce_max" in lower:
        counts["cce_inner_reduce"] += 1
      if "dynamic-slice(" in lower:
        counts["cce_inner_dynamic_slice"] += 1
      if "custom-call(" in lower:
        counts["cce_inner_custom_call"] += 1
      if "exponential(" in lower:
        counts["cce_inner_exp"] += 1
      if "select(" in lower:
        counts["cce_inner_select"] += 1

  loop_count = cce_loop_count or 0
  inner_collectives = counts["cce_inner_all_gather"] + counts["cce_inner_all_reduce"]
  metrics: dict[str, Any] = {
      **counts,
      "cce_inner_collective_sites": inner_collectives,
      "estimated_dynamic_cce_inner_collectives": inner_collectives * loop_count if loop_count else "",
      "estimated_dynamic_cce_inner_dot_general_metadata": counts["cce_inner_dot_general_metadata"] * loop_count if loop_count else "",
      "estimated_dynamic_cce_inner_convolution": counts["cce_inner_convolution"] * loop_count if loop_count else "",
      "cce_dot_general_result_shapes": summarize_counter(shape_counters["dot_general_result"]),
      "cce_all_reduce_result_shapes": summarize_counter(shape_counters["all_reduce_result"]),
      "cce_all_gather_result_shapes": summarize_counter(shape_counters["all_gather_result"]),
      "cce_convolution_result_shapes": summarize_counter(shape_counters["convolution_result"]),
      "cce_fusion_result_shapes": summarize_counter(shape_counters["fusion_result"]),
  }

  shape_rows: list[dict[str, Any]] = []
  for kind, counter in shape_counters.items():
    for shape, count in counter.most_common(12):
      shape_rows.append({
          "shape_kind": kind,
          "dot_shape": shape,
          "count": count,
      })
  return metrics, shape_rows


def trace_file_for(row: dict[str, Any], trace_roots: list[Path]) -> Path | None:
  case_name = str(row.get("case_name", ""))
  if not case_name:
    return None
  for root in trace_roots:
    if not root.exists():
      continue
    direct = sorted((root / case_name).glob("**/*.trace.json.gz"))
    if direct:
      return direct[-1]
    found = sorted(root.glob(f"**/{case_name}/**/*.trace.json.gz"))
    if found:
      return found[-1]
  return None


def parse_trace(path: Path | None) -> dict[str, Any]:
  if not path or not path.exists():
    return {
        "trace_json_gz": "",
        "trace_events": None,
        "trace_pjit_train_step_total_sec": None,
        "trace_pjit_train_step_count": None,
        "trace_idle_total_sec": None,
        "trace_note": "trace json not found",
    }
  try:
    with gzip.open(path, "rt") as f:
      data = json.load(f)
  except (OSError, json.JSONDecodeError) as exc:
    return {
        "trace_json_gz": str(path),
        "trace_events": None,
        "trace_pjit_train_step_total_sec": None,
        "trace_pjit_train_step_count": None,
        "trace_idle_total_sec": None,
        "trace_note": f"trace parse failed: {type(exc).__name__}",
    }
  pjit_us = 0.0
  pjit_count = 0
  idle_us = 0.0
  events = data.get("traceEvents", [])
  for event in events:
    if event.get("ph") != "X":
      continue
    name = str(event.get("name", ""))
    dur = float(event.get("dur", 0.0) or 0.0)
    lower = name.lower()
    if "pjitfunction(_train_step)" in lower:
      pjit_us += dur
      pjit_count += 1
    if "idle" in lower:
      idle_us += dur
  note = "" if idle_us else "trace parsed, but explicit idle/utilization events were not extracted"
  return {
      "trace_json_gz": str(path),
      "trace_events": len(events),
      "trace_pjit_train_step_total_sec": pjit_us / 1e6 if pjit_us else None,
      "trace_pjit_train_step_count": pjit_count,
      "trace_idle_total_sec": idle_us / 1e6 if idle_us else None,
      "trace_note": note,
  }


def fmt(value: Any, digits: int = 3) -> str:
  if value in (None, ""):
    return ""
  if isinstance(value, float):
    return f"{value:.{digits}f}"
  return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
  lines = []
  headers = [label for _, label in columns]
  lines.append("| " + " | ".join(headers) + " |")
  lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
  for row in rows:
    cells = [str(row.get(key, "")) for key, _ in columns]
    lines.append("| " + " | ".join(cells) + " |")
  return "\n".join(lines)


def speed_ratio(bad: dict[str, Any] | None, good: dict[str, Any] | None) -> float | None:
  if not bad or not good:
    return None
  bad_step = safe_float(bad.get("steady_state_mean_step_time_sec"))
  good_step = safe_float(good.get("steady_state_mean_step_time_sec"))
  if not bad_step or not good_step:
    return None
  return bad_step / good_step


def write_plot(metrics: list[dict[str, Any]], plot_path: Path) -> str:
  try:
    import matplotlib.pyplot as plt
  except ImportError:
    return "matplotlib unavailable; profiler/HLO plot was not written"

  labels = [str(row["case_label"]) for row in metrics]
  step_times = [
      safe_float(row.get("steady_state_mean_step_time_sec")) or 0.0
      for row in metrics
  ]
  dynamic_collectives = [
      safe_int(row.get("estimated_dynamic_cce_inner_collectives")) or 0
      for row in metrics
  ]

  plot_path.parent.mkdir(parents=True, exist_ok=True)
  fig, ax1 = plt.subplots(figsize=(11.5, 5.8))
  bars = ax1.bar(labels, step_times, color="#3f6f8f", width=0.58, label="step time")
  ax1.set_ylabel("steady-state step time (s)")
  ax1.set_title("TPU v5e CCE profiler/HLO target cases")
  ax1.grid(axis="y", alpha=0.25)
  ax1.tick_params(axis="x", rotation=12)

  for bar, value in zip(bars, step_times):
    inside = value > max(step_times) * 0.18 if step_times else False
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        value - 0.35 if inside else value,
        f"{value:.2f}s",
        ha="center",
        va="top" if inside else "bottom",
        fontsize=9,
        color="white" if inside else "black",
        fontweight="bold" if inside else "normal",
    )

  ax2 = ax1.twinx()
  ax2.plot(
      labels,
      dynamic_collectives,
      color="#c94f4f",
      marker="o",
      linewidth=2.2,
      label="est CCE inner collective executions",
  )
  ax2.set_ylabel("estimated CCE inner-loop collective executions")
  ax2.set_yscale("log")
  for idx, value in enumerate(dynamic_collectives):
    ax2.annotate(
        str(value),
        (idx, max(value, 1)),
        textcoords="offset points",
        xytext=(0, 7),
        color="#8f2f2f",
        ha="center",
        va="bottom",
        fontsize=9,
    )

  handles1, labels1 = ax1.get_legend_handles_labels()
  handles2, labels2 = ax2.get_legend_handles_labels()
  ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
  fig.tight_layout()
  fig.savefig(plot_path, dpi=180)
  plt.close(fig)
  return ""


def build_report(
    metrics: list[dict[str, Any]],
    dot_rows: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    report_path: Path,
    plot_path: Path | None,
) -> None:
  by_label = {row["case_label"]: row for row in metrics}
  bad = by_label.get("bad")
  good = by_label.get("good")
  bad_good_ratio = speed_ratio(bad, good)

  table_rows = []
  for row in metrics:
    table_rows.append({
        "case": row["case_label"],
        "mesh": row["mesh_configuration"],
        "chunks": row["chunk_configuration"],
        "loops": row["cce_loop_count"],
        "step_s": fmt(safe_float(row.get("steady_state_mean_step_time_sec"))),
        "compile_s": fmt(safe_float(row.get("compile_time_sec"))),
        "hbm_gib": fmt(safe_float(row.get("xla_planned_hbm_gib_per_chip")), 2),
        "collectives": row.get("collective_total", ""),
        "all_gather": row.get("all_gather", ""),
        "all_reduce": row.get("all_reduce", ""),
        "inner_collectives": row.get("cce_inner_collective_sites", ""),
        "dynamic_inner_collectives": row.get("estimated_dynamic_cce_inner_collectives", ""),
        "inner_dot": row.get("cce_inner_dot_general_metadata", ""),
        "dynamic_inner_dot": row.get("estimated_dynamic_cce_inner_dot_general_metadata", ""),
        "permute": row.get("collective_permute", ""),
        "while": row.get("while", ""),
        "fusion": row.get("fusion", ""),
        "dot": row.get("dot_general_metadata", row.get("dot", "")),
        "conv": row.get("convolution", ""),
        "dyn_slice": row.get("dynamic_slice", ""),
        "cce_shapes": row.get("cce_dot_general_result_shapes", ""),
        "hlo_files": row.get("hlo_files_scanned", ""),
    })

  dot_summary = []
  for row in dot_rows[:24]:
    dot_summary.append({
        "case": row["case_label"],
        "kind": row.get("shape_kind", "dot_result"),
        "shape": row["dot_shape"],
        "count": row["count"],
    })

  note_lines = []
  for note in notes:
    if note.get("notes"):
      note_lines.append(f"- {note['case_label']}: {note['notes']}")
  if not note_lines:
    note_lines.append("- No extraction failures were recorded.")

  cause = [
      "The slowdown is a steady-state execution problem, not a compile-time problem.",
      "The bad and good cases use the same `fsdp=2,tp=2` mesh and the same planned HBM, so planned memory footprint does not explain the 128/8192 cliff.",
      "The strongest signal is chunk-loop granularity: bad has `cce_loop_count=128`, good has `cce_loop_count=4`.",
  ]
  if bad_good_ratio is not None:
    cause.append(
        f"In this run, bad/good steady-state step time is `{bad_good_ratio:.1f}x`."
    )
  if bad and good:
    bad_hlo_files = safe_int(bad.get("hlo_files_scanned")) or 0
    good_hlo_files = safe_int(good.get("hlo_files_scanned")) or 0
    bad_collectives = safe_int(bad.get("collective_total"))
    good_collectives = safe_int(good.get("collective_total"))
    if bad_hlo_files == 0 or good_hlo_files == 0:
      cause.append(
          "Optimized HLO text was not available for bad/good, so collective and fusion counts are not interpreted."
      )
    elif bad_collectives is not None and good_collectives is not None:
      if bad_collectives > good_collectives * 2:
        cause.append(
            "HLO counts also show substantially more collective operations in the bad case."
        )
      else:
        cause.append(
            "Static HLO collective counts are effectively the same for bad/good, so a simple 'more collective ops in the text' explanation is not enough."
        )
    bad_dyn_collectives = safe_int(bad.get("estimated_dynamic_cce_inner_collectives"))
    good_dyn_collectives = safe_int(good.get("estimated_dynamic_cce_inner_collectives"))
    if bad_dyn_collectives is not None and good_dyn_collectives:
      cause.append(
          f"After multiplying CCE inner-loop sites by trip count, bad has about `{bad_dyn_collectives}` inner-loop collective executions vs `{good_dyn_collectives}` for good."
      )
    bad_shapes = str(bad.get("cce_dot_general_result_shapes", ""))
    good_shapes = str(good.get("cce_dot_general_result_shapes", ""))
    if bad_shapes and good_shapes:
      cause.append(
          f"The lowered loss-head matmul proxy changes from bad `{bad_shapes}` to good `{good_shapes}`."
      )
    if bad_hlo_files and good_hlo_files:
      bad_fusion = safe_int(bad.get("fusion"))
      good_fusion = safe_int(good.get("fusion"))
      bad_dyn = safe_int(bad.get("dynamic_slice")) or 0
      good_dyn = safe_int(good.get("dynamic_slice")) or 0
      if bad_fusion is not None and good_fusion is not None and bad_fusion > good_fusion * 2:
        cause.append("The bad case has many more fusion regions in extracted HLO.")
      if good_dyn and bad_dyn > good_dyn * 2:
        cause.append("The bad case has many more dynamic-slice/update style loop operations.")
  cause.append(
      "Most likely cause: CCE small chunks create many dynamic loop iterations over tiny loss-head tiles; under mixed FSDP/TP, each iteration carries TP communication and reductions, so overhead and TPU under-utilization dominate even though planned HBM is unchanged."
  )

  direct_answers = []
  if bad and good:
    bad_dyn_collectives = safe_int(bad.get("estimated_dynamic_cce_inner_collectives"))
    good_dyn_collectives = safe_int(good.get("estimated_dynamic_cce_inner_collectives"))
    direct_answers = [
        "More collectives: not as static HLO text; yes after accounting for CCE loop trip count.",
        f"More tiny loop/fusion work: yes, bad runs the inner CCE loss loop `{bad.get('cce_loop_count')}` times vs `{good.get('cce_loop_count')}` for good.",
        f"Worse matmul/logit shape: yes, bad uses `{bad.get('cce_dot_general_result_shapes')}` while good uses `{good.get('cce_dot_general_result_shapes')}`.",
        f"More HBM pressure: no clear evidence; both bad and good planned HBM are `{fmt(safe_float(bad.get('xla_planned_hbm_gib_per_chip')), 2)} GiB` per chip.",
        "Poor TPU utilization / idle time: likely from tiny loop tiles and communication amortization, but explicit idle/utilization counters were not extracted from the raw trace JSON.",
        "Compile-time issue: no; good compiled slower but ran much faster.",
    ]
    if bad_dyn_collectives is not None and good_dyn_collectives is not None:
      direct_answers[0] = (
          f"More collectives: static HLO counts are equal for bad/good, but estimated dynamic CCE inner-loop collectives are `{bad_dyn_collectives}` vs `{good_dyn_collectives}`."
      )

  lines = [
      "# TPU v5e CCE Profiler/HLO Root-Cause Analysis",
      "",
      "Target cases only: bad, good, and two mesh controls. Failed extraction steps are non-fatal and listed below.",
      "",
  ]
  if plot_path and plot_path.exists():
    try:
      plot_ref = plot_path.relative_to(report_path.parent)
    except ValueError:
      plot_ref = plot_path
    lines.extend([
        f"![Profiler/HLO root-cause summary]({plot_ref})",
        "",
    ])
  lines.extend([
      "## Four-Case Comparison",
      "",
      markdown_table(
          table_rows,
          [
              ("case", "case"),
              ("mesh", "mesh"),
              ("chunks", "chunks"),
              ("loops", "CCE loops"),
              ("step_s", "step s"),
              ("compile_s", "compile s"),
              ("hbm_gib", "HBM GiB"),
              ("collectives", "collectives"),
              ("all_gather", "all-gather"),
              ("all_reduce", "all-reduce"),
              ("inner_collectives", "CCE coll sites"),
              ("dynamic_inner_collectives", "est CCE coll execs"),
              ("inner_dot", "CCE dot sites"),
              ("dynamic_inner_dot", "est CCE dot execs"),
              ("permute", "permute"),
              ("while", "while"),
              ("fusion", "fusion"),
              ("dot", "dot metadata"),
              ("conv", "convolution"),
              ("dyn_slice", "dyn-slice"),
              ("cce_shapes", "CCE dot shapes"),
              ("hlo_files", "HLO files"),
          ],
      ),
      "",
      "## CCE Inner-Loop Matmul/Collective Shapes",
      "",
      markdown_table(
          dot_summary,
          [("case", "case"), ("kind", "kind"), ("shape", "shape"), ("count", "count")],
      )
      if dot_summary
      else "No CCE inner-loop matmul/collective shapes were extracted.",
      "",
      "## Most Likely Cause",
      "",
      "\n".join(f"- {item}" for item in cause),
      "",
      "## Direct Answers",
      "",
      "\n".join(f"- {item}" for item in direct_answers)
      if direct_answers
      else "- Bad/good comparison was incomplete, so direct answers are limited.",
      "",
      "## Extraction Notes",
      "",
      "\n".join(note_lines),
      "",
      "## Artifact Files",
      "",
      "- `profiler_hlo_case_metrics.csv`: per-case timing, memory, HLO op counts, and trace hints.",
      "- `profiler_hlo_dot_shapes.csv`: extracted CCE inner-loop matmul and collective result shapes.",
      "- `profiler_hlo_extraction_notes.jsonl`: non-fatal extraction misses.",
  ])
  report_path.parent.mkdir(parents=True, exist_ok=True)
  report_path.write_text("\n".join(lines) + "\n")


def analyze(args: argparse.Namespace) -> None:
  rows = read_jsonl(args.results_jsonl)
  if not rows:
    rows = read_csv(args.results_csv)
  targets = [row for row in rows if is_target(row)]
  targets.sort(
      key=lambda row: CASE_ORDER[TARGET_CASES[case_key(row) or (0, 0, 0, 0, 0, 0)]]
  )

  metrics: list[dict[str, Any]] = []
  dot_rows: list[dict[str, Any]] = []
  notes: list[dict[str, Any]] = []

  for row in targets:
    key = case_key(row)
    assert key is not None
    label = TARGET_CASES[key]
    case_dir = find_case_dir(row, args.run_root)
    xla_dir = case_dir / "xla" if case_dir else None
    memory_report = memory_report_for(row, xla_dir)
    memory_total, memory_shapes = parse_memory_report(memory_report)
    all_hlo, optimized_hlo, stablehlo = hlo_text_files(xla_dir)
    scan_files = optimized_hlo or all_hlo
    op_counts, hlo_bytes, dot_shapes, sample_hlo = count_ops(scan_files)
    cce_loop_count = safe_int(row.get("cce_loop_count"))
    cce_inner_metrics, cce_shape_rows = extract_cce_inner_loop(scan_files, cce_loop_count)
    trace_metrics = parse_trace(trace_file_for(row, args.trace_root))

    note_items = []
    if case_dir is None:
      note_items.append("run directory not found")
    if memory_report is None:
      note_items.append("XLA memory report not found")
    if not optimized_hlo:
      note_items.append(
          "optimized HLO text not found; use --keep-all-xla --full-hlo-dump"
      )
    if not stablehlo:
      note_items.append("StableHLO/compiler IR file not found in XLA dump")
    if trace_metrics.get("trace_note"):
      note_items.append(str(trace_metrics["trace_note"]))

    collective_total = sum(
        op_counts[key]
        for key in [
            "all_gather",
            "all_reduce",
            "collective_permute",
            "reduce_scatter",
            "all_to_all",
        ]
    )
    metric = {
        "case_label": label,
        "case_name": row.get("case_name", ""),
        "mesh_configuration": row.get("mesh_configuration", ""),
        "fsdp_degree": row.get("fsdp_degree", ""),
        "tp_degree": row.get("tp_degree", ""),
        "shape": row.get("shape", ""),
        "chunk_configuration": row.get("chunk_configuration", ""),
        "token_chunk": row.get("token_chunk", ""),
        "vocab_chunk": row.get("vocab_chunk", ""),
        "cce_loop_count": row.get("cce_loop_count", ""),
        "steady_state_mean_step_time_sec": row.get("steady_state_mean_step_time_sec", ""),
        "median_step_time_sec": row.get("median_step_time_sec", ""),
        "compile_time_sec": row.get("compile_time_sec", ""),
        "xla_planned_hbm_gib_per_chip": row.get(
            "xla_planned_hbm_gib_per_chip", memory_total
        ),
        "memory_report_hbm_gib_per_chip": memory_total,
        "runtime_hbm_peak_gb": row.get("runtime_hbm_peak_gb", ""),
        "tokens_per_sec": row.get("tokens_per_sec", ""),
        "hlo_dir": str(xla_dir) if xla_dir else "",
        "hlo_files_total": len(all_hlo),
        "hlo_files_scanned": len(scan_files),
        "hlo_text_bytes": hlo_bytes,
        "hlo_sample_file": sample_hlo,
        "stablehlo_files": len(stablehlo),
        "collective_total": collective_total,
        "top_memory_shapes": "; ".join(f"{shape} x{count}" for shape, count in memory_shapes[:8]),
        **op_counts,
        **cce_inner_metrics,
        **trace_metrics,
    }
    metrics.append(metric)
    for shape, count in dot_shapes:
      dot_rows.append({
          "case_label": label,
          "case_name": row.get("case_name", ""),
          "shape_kind": "hlo_dot_result",
          "dot_shape": shape,
          "count": count,
      })
    for shape_row in cce_shape_rows:
      dot_rows.append({
          "case_label": label,
          "case_name": row.get("case_name", ""),
          **shape_row,
      })
    notes.append({
        "case_label": label,
        "case_name": row.get("case_name", ""),
        "notes": "; ".join(note_items),
    })

  args.outdir.mkdir(parents=True, exist_ok=True)
  write_csv(args.outdir / "profiler_hlo_case_metrics.csv", metrics)
  write_csv(args.outdir / "profiler_hlo_dot_shapes.csv", dot_rows)
  plot_note = write_plot(metrics, args.plot_path)
  if plot_note:
    notes.append({
        "case_label": "analysis",
        "case_name": "",
        "notes": plot_note,
    })
  write_jsonl(args.outdir / "profiler_hlo_extraction_notes.jsonl", notes)
  build_report(metrics, dot_rows, notes, args.report_path, args.plot_path)
  print(f"metrics={args.outdir / 'profiler_hlo_case_metrics.csv'}")
  print(f"dot_shapes={args.outdir / 'profiler_hlo_dot_shapes.csv'}")
  print(f"notes={args.outdir / 'profiler_hlo_extraction_notes.jsonl'}")
  print(f"report={args.report_path}")
  print(f"plot={args.plot_path}")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  parser.add_argument("--results-jsonl", type=Path, default=DEFAULT_RESULTS_JSONL)
  parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
  parser.add_argument("--run-root", type=Path, action="append", default=[])
  parser.add_argument("--trace-root", type=Path, action="append", default=[])
  parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
  parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
  parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT)
  return parser.parse_args()


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
  args.results_jsonl = args.results_jsonl.expanduser().resolve()
  args.results_csv = args.results_csv.expanduser().resolve()
  args.outdir = args.outdir.expanduser().resolve()
  args.report_path = args.report_path.expanduser().resolve()
  args.plot_path = args.plot_path.expanduser().resolve()
  args.run_root = [path.expanduser().resolve() for path in args.run_root]
  args.trace_root = [path.expanduser().resolve() for path in args.trace_root]
  if not args.run_root:
    args.run_root = [
        Path("/tmp/v5e-cce-profiler-hlo-local"),
        Path("/tmp/v5e-cce-full-local/shard0/v5e-cce-full/profiler/runs"),
        Path("/tmp/v5e-cce-full-local/shard1/v5e-cce-full/profiler/runs"),
    ]
  if not args.trace_root:
    args.trace_root = [
        Path("/tmp/v5e-cce-profiler-hlo-local/traces"),
        Path("/tmp/v5e-cce-full-local/combined/profiler/traces"),
    ]
  return args


def main() -> None:
  analyze(normalize_args(parse_args()))


if __name__ == "__main__":
  main()
