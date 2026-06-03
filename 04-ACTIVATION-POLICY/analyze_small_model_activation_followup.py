"""Summarize the Gemma3 270M/1B activation-policy follow-up artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "results" / "small-model-splash-activation-ablation"
ASSET_DIR = ROOT / "assets"
DATA_DIR = ROOT / "data"

CONTEXTS = [8192, 16384, 32768]
POLICIES = [("none", "No offload"), ("split_offload", "Split offload")]
CONFIGS = {
    "270m": {
        "label": "Gemma3 270M",
        "model_id": "google/gemma-3-270m-it",
        "tpu_type": "v5litepod-1",
        "chips": 1,
        "fsdp": 1,
        "tp": 1,
    },
    "1b": {
        "label": "Gemma3 1B",
        "model_id": "google/gemma-3-1b-it",
        "tpu_type": "v5litepod-4",
        "chips": 4,
        "fsdp": 4,
        "tp": 1,
    },
}


def _read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(errors="ignore")


def _read_status(path: Path) -> int | None:
    match = re.search(r"RUN_STATUS=(\d+)", _read_text(path))
    return int(match.group(1)) if match else None


def _parse_memory_report(path: Path) -> float | None:
    text = _read_text(path)
    match = re.search(
        r"Memory Space:\s+default[\s\S]*?Total bytes:\s+(\d+)\s+\(([^)]+)GiB\)",
        text,
    )
    if match is None:
        match = re.search(r"Total bytes:\s+(\d+)\s+\(([^)]+)GiB\)", text)
    if match is None:
        return None
    return int(match.group(1)) / (1024**3)


def _parse_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if isinstance(data, list) and data:
        return data[0]
    if isinstance(data, dict):
        return data
    return {}


def _parse_history(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open()))
    if not rows:
        return {}
    row = rows[-1]
    out = {}
    for key in ("loss", "step_time_sec", "valid_tokens", "loss_tokens"):
        try:
            out[key] = float(row[key])
        except (KeyError, TypeError, ValueError):
            pass
    return out


def _parse_log(path: Path) -> dict[str, Any]:
    text = _read_text(path)
    used = re.search(r"Used\s+([0-9.]+)G\s+of\s+([0-9.]+)G", text)
    dense_patterns = (
        "BTNH,BSNH->BTNS",
        "BTNS,BSNH->BTNH",
        "broadcast.475 = f32[1,4,16384,16384]",
        "f32[1,4,32768,32768]",
        "f32[1,16,16384,16384]",
        "f32[1,16,32768,32768]",
    )
    return {
        "resource_exhausted": "RESOURCE_EXHAUSTED" in text,
        "oom_used_gib": float(used.group(1)) if used else None,
        "oom_limit_gib": float(used.group(2)) if used else None,
        "dense_attention_evidence": any(pattern in text for pattern in dense_patterns),
    }


def collect_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for model_key, cfg in CONFIGS.items():
        model_root = RESULT_ROOT / model_key
        for context in CONTEXTS:
            for policy, policy_label in POLICIES:
                run_dir = model_root / f"{model_key}_l{context}_{policy}_fsdp{cfg['fsdp']}"
                status_code = _read_status(run_dir / "status.txt")
                summary = _parse_summary(run_dir / "summary.json")
                history = _parse_history(run_dir / "history.csv")
                log = _parse_log(run_dir / "run.log")
                peak = _parse_memory_report(run_dir / "train_step_memory_report.txt")

                if status_code == 0:
                    status = "ok"
                elif log["resource_exhausted"]:
                    status = "compile_oom"
                else:
                    status = f"exit_{status_code}"

                limit = None
                try:
                    limit = (
                        summary["memory_after_train"]["devices"][0]["bytes_limit"]
                        / (1024**3)
                    )
                except (KeyError, IndexError, TypeError):
                    pass
                if limit is None:
                    limit = log["oom_limit_gib"] or 15.748

                records.append(
                    {
                        "model": model_key,
                        "model_label": cfg["label"],
                        "model_id": cfg["model_id"],
                        "context_length": context,
                        "policy": policy,
                        "policy_label": policy_label,
                        "status": status,
                        "xla_peak_hbm_gib_per_chip": peak,
                        "xla_peak_hbm_gib_aggregate": (
                            peak * cfg["chips"] if peak is not None else None
                        ),
                        "hbm_limit_gib_per_chip": limit,
                        "hbm_limit_gib_aggregate": limit * cfg["chips"],
                        "headroom_gib_per_chip": (
                            limit - peak if peak is not None else None
                        ),
                        "step_time_sec": history.get("step_time_sec")
                        or summary.get("mean_step_time_sec_excl_first"),
                        "final_loss": summary.get("final_loss"),
                        "tpu_type": cfg["tpu_type"],
                        "chips": cfg["chips"],
                        "mesh_fsdp": cfg["fsdp"],
                        "mesh_tp": cfg["tp"],
                        "batch_size": summary.get("batch_size", 1),
                        "lora_rank": summary.get("lora_rank", 16),
                        "ce_enabled": not bool(summary.get("ce_disabled", False))
                        if summary
                        else True,
                        "tiled_mlp_enabled": bool(
                            summary.get("tiled_mlp_enabled", True)
                        )
                        if summary
                        else True,
                        "splash_attention_requested": True,
                        "splash_attention_installed": (
                            bool(summary.get("splash_attention_installed", False))
                            if summary
                            else None
                        ),
                        "activation_policy_installed": (
                            bool(
                                summary.get(
                                    "activation_policy_installed", policy != "none"
                                )
                            )
                            if summary
                            else policy != "none"
                        ),
                        "dense_attention_evidence_in_oom_log": log[
                            "dense_attention_evidence"
                        ],
                        "resource_exhausted": log["resource_exhausted"],
                        "run_dir": str(run_dir),
                    }
                )
    return records


def write_metrics(records: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    for path in (
        RESULT_ROOT / "small_model_splash_activation_metrics.csv",
        DATA_DIR / "gemma3_small_model_activation_followup.csv",
    ):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
    (RESULT_ROOT / "small_model_splash_activation_metrics.json").write_text(
        json.dumps(records, indent=2)
    )


def write_plot(records: list[dict[str, Any]]) -> None:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9.2,
            "figure.dpi": 170,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.65))
    colors = {"none": "#475569", "split_offload": "#0f766e"}
    edge = {"ok": "#111827", "compile_oom": "#dc2626"}
    hatches = {"ok": "", "compile_oom": "///"}
    width = 0.34
    legend_handles = None

    for ax, model_key in zip(axes, ("270m", "1b"), strict=True):
        cfg = CONFIGS[model_key]
        subset = [row for row in records if row["model"] == model_key]
        limit = float(subset[0]["hbm_limit_gib_per_chip"])
        ymax = max(
            max(float(row["xla_peak_hbm_gib_per_chip"]) for row in subset) * 1.18,
            limit * 1.55,
        )
        ax.set_ylim(0, ymax)
        ax.axhspan(limit, ymax, color="#fef2f2", alpha=0.85, zorder=0)
        ax.axhline(
            limit, color="#b91c1c", linestyle=(0, (4, 3)), linewidth=1.25, zorder=1
        )

        xs = np.arange(len(CONTEXTS))
        for idx, (policy, label) in enumerate(POLICIES):
            vals = []
            statuses = []
            for context in CONTEXTS:
                row = next(
                    r
                    for r in subset
                    if r["context_length"] == context and r["policy"] == policy
                )
                vals.append(float(row["xla_peak_hbm_gib_per_chip"]))
                statuses.append(str(row["status"]))
            bars = ax.bar(
                xs + (idx - 0.5) * width,
                vals,
                width,
                label=label,
                color=colors[policy],
                edgecolor=[edge[status] for status in statuses],
                linewidth=1.2,
                hatch=[hatches[status] for status in statuses],
                zorder=3,
            )
            for bar, val, status in zip(bars, vals, statuses, strict=True):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + ymax * 0.016,
                    f"{val:.1f}\n{'OK' if status == 'ok' else 'OOM'}",
                    ha="center",
                    va="bottom",
                    fontsize=8.1,
                    color="#111827",
                )

        ax.set_title(
            f"{cfg['label']} - {cfg['tpu_type']} - {cfg['chips']} "
            f"chip{'s' if cfg['chips'] > 1 else ''}"
        )
        ax.set_xticks(xs, ["8K", "16K", "32K"])
        ax.set_xlabel("Context length")
        ax.set_ylabel("XLA train-step peak HBM\n(GiB per chip)")
        ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if legend_handles is None:
            legend_handles, _ = ax.get_legend_handles_labels()

    limit_handle = Line2D(
        [0],
        [0],
        color="#b91c1c",
        linestyle=(0, (4, 3)),
        linewidth=1.25,
        label="HBM limit (~15.7 GiB/chip)",
    )
    fig.legend(
        [*legend_handles, limit_handle],
        ["No offload", "Split offload", "HBM limit (~15.7 GiB/chip)"],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
    )
    fig.suptitle(
        "Small-model follow-up: activation offload under the requested Splash stack",
        y=1.07,
        fontsize=13.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "Fixed requested stack: CCE + Tiled MLP + Splash Attention, LoRA rank 16, "
        "batch 1. Hatching marks compile-time OOM.\nDense-attention allocations "
        "still appear in long-context OOM logs, so this is retained as "
        "adapter-coverage evidence, not a clean Splash-only proof.",
        ha="center",
        va="top",
        fontsize=8.8,
        color="#374151",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92), w_pad=2.0)
    fig.savefig(
        ASSET_DIR / "gemma3_small_model_activation_followup.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    records = collect_records()
    write_metrics(records)
    write_plot(records)
    print(f"Wrote {len(records)} rows")


if __name__ == "__main__":
    main()
