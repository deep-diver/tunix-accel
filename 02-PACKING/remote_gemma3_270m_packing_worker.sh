#!/usr/bin/env bash
set -euo pipefail

SUITE="${1:-short-throughput}"
ROOT="${ROOT:-$HOME/TUNIX-TRY}"
OUT_BASE="${OUT_BASE:-/tmp/gemma3-270m-packing}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/tunix-packing-270m-py311}"
CONDA_DIR="${CONDA_DIR:-$HOME/miniconda3}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-$HOME/.conda/envs/tunix-packing-270m-py311}"

cd "${ROOT}"
mkdir -p "${OUT_BASE}"

ensure_python() {
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    return
  fi
  if [[ ! -x "${CONDA_DIR}/bin/conda" ]]; then
    sudo apt-get update
    sudo apt-get install -y ca-certificates curl bzip2
    curl -fsSL \
      -o /tmp/miniconda.sh \
      https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "${CONDA_DIR}"
  fi
  "${CONDA_DIR}/bin/conda" tos accept \
    --override-channels \
    --channel https://repo.anaconda.com/pkgs/main || true
  "${CONDA_DIR}/bin/conda" tos accept \
    --override-channels \
    --channel https://repo.anaconda.com/pkgs/r || true
  if [[ ! -x "${CONDA_ENV_DIR}/bin/python" ]]; then
    "${CONDA_DIR}/bin/conda" create -y -p "${CONDA_ENV_DIR}" python=3.11 pip
  fi
  PYTHON_BIN="${CONDA_ENV_DIR}/bin/python"
}

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  ensure_python
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install -r requirements.txt
  python -m pip install pytest
  python -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
else
  # shellcheck disable=SC1091
  [[ -f "${VENV_DIR}/bin/activate" ]] && source "${VENV_DIR}/bin/activate"
fi

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TUNIX_ACCEL_DISABLE_AUTOPATCH=1
export TUNIX_ACCEL_DISABLE_CE=1
export TUNIX_ACCEL_DISABLE_TILED_MLP=1
export TUNIX_ACCEL_DISABLE_ACTIVATION_POLICY=1
export TUNIX_ACCEL_ENABLE_SPLASH_ATTENTION=0
if [[ "${MODEL_ID:-}" == *"gemma-4"* ]]; then
  export TUNIX_ACCEL_ENABLE_GEMMA4_HF_LOADER="${TUNIX_ACCEL_ENABLE_GEMMA4_HF_LOADER:-1}"
  export TUNIX_ACCEL_MODEL_DOWNLOAD_PATH="${TUNIX_ACCEL_MODEL_DOWNLOAD_PATH:-${OUT_BASE}/hf-cache/${MODEL_ID##*/}}"
fi

run_sweep() {
  local model_args=(
    --model-id "${MODEL_ID-google/gemma-3-270m-it}"
    --model-source "${MODEL_SOURCE-gcs}"
    --model-path "${MODEL_PATH-gs://gemma-data/checkpoints/gemma3-270m-it}"
    --model-download-path "${MODEL_DOWNLOAD_PATH:-${TUNIX_ACCEL_MODEL_DOWNLOAD_PATH:-}}"
    --tokenizer-source "${TOKENIZER_SOURCE-sentencepiece}"
    --tokenizer-path "${TOKENIZER_PATH-gs://gemma-data/tokenizers/tokenizer_gemma3.model}"
  )
  if [[ "${ALLOW_DOWNLOAD:-0}" == "1" ]]; then
    model_args+=(--allow-download)
  fi
  python 02-PACKING/run_gemma3_270m_packing_sweep.py "${model_args[@]}" "$@"
}

case "${SUITE}" in
  prepare)
    python 02-PACKING/run_gemma_training_benchmark.py \
      --dataset-mode "${DATASET_MODE:-opus100}" \
      --long-example-policy "${LONG_EXAMPLE_POLICY:-drop}" \
      --num-examples "${NUM_EXAMPLES:-5000}" \
      --variants unpacked,packed \
      --batch-size "${BATCH_SIZE:-16}" \
      --max-length "${MAX_LENGTH:-512}" \
      --max-steps 1 \
      --prepare-only \
      --outdir "${OUT_BASE}/prepare"
    ;;
  short-throughput)
    run_sweep \
      --suite short-throughput \
      --variants unpacked,packed \
      --batch-sizes "${BATCH_SIZES:-8,16,32}" \
      --contexts "${CONTEXTS:-512,1024}" \
      --dataset-mode "${DATASET_MODE:-opus100}" \
      --long-example-policy "${LONG_EXAMPLE_POLICY:-drop}" \
      --num-examples "${NUM_EXAMPLES:-5000}" \
      --max-steps "${MAX_STEPS:-50}" \
      --skip-quality-eval \
      --tpu "${TPU_TYPE:-v5litepod-1}" \
      --chips "${CHIPS:-1}" \
      --mesh-fsdp "${MESH_FSDP:-${CHIPS:-1}}" \
      --mesh-tp "${MESH_TP:-1}" \
      --force \
      --outdir "${OUT_BASE}/short-throughput"
    ;;
  quality-unpacked)
    run_sweep \
      --suite quality-unpacked \
      --variants unpacked \
      --batch-sizes "${BATCH_SIZES:-16}" \
      --contexts "${CONTEXTS:-512}" \
      --dataset-mode "${DATASET_MODE:-opus100}" \
      --long-example-policy "${LONG_EXAMPLE_POLICY:-drop}" \
      --num-examples "${NUM_EXAMPLES:-5000}" \
      --max-steps "${MAX_STEPS:-5000}" \
      --eval-examples "${EVAL_EXAMPLES:-512}" \
      --eval-batches "${EVAL_BATCHES:-32}" \
      --generation-examples "${GENERATION_EXAMPLES:-0}" \
      --tpu "${TPU_TYPE:-v5litepod-1}" \
      --chips "${CHIPS:-1}" \
      --mesh-fsdp "${MESH_FSDP:-${CHIPS:-1}}" \
      --mesh-tp "${MESH_TP:-1}" \
      --force \
      --outdir "${OUT_BASE}/quality-unpacked"
    ;;
  quality-packed)
    run_sweep \
      --suite quality-packed \
      --variants packed \
      --batch-sizes "${BATCH_SIZES:-16}" \
      --contexts "${CONTEXTS:-512}" \
      --dataset-mode "${DATASET_MODE:-opus100}" \
      --long-example-policy "${LONG_EXAMPLE_POLICY:-drop}" \
      --num-examples "${NUM_EXAMPLES:-5000}" \
      --max-steps "${MAX_STEPS:-1000}" \
      --eval-examples "${EVAL_EXAMPLES:-512}" \
      --eval-batches "${EVAL_BATCHES:-32}" \
      --generation-examples "${GENERATION_EXAMPLES:-0}" \
      --tpu "${TPU_TYPE:-v5litepod-1}" \
      --chips "${CHIPS:-1}" \
      --mesh-fsdp "${MESH_FSDP:-${CHIPS:-1}}" \
      --mesh-tp "${MESH_TP:-1}" \
      --force \
      --outdir "${OUT_BASE}/quality-packed"
    ;;
  *)
    echo "Unknown suite: ${SUITE}" >&2
    exit 2
    ;;
esac
