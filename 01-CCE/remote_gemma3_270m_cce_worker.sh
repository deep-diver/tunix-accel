#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-frontier-low}"
ROOT="${ROOT:-$HOME/TUNIX-TRY}"
OUT_BASE="${OUT_BASE:-/tmp/gemma3-270m-cce-rerun}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$HOME/.venvs/tunix-cce270m-py311}"

cd "${ROOT}"
mkdir -p "${OUT_BASE}"

ensure_python() {
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    return
  fi

  sudo apt-get update
  sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.11 python3.11-dev python3.11-venv
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
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

run_sweep() {
  python 01-CCE/run_gemma3_270m_cce_sweep.py "$@"
}

run_mesh_sweep() {
  local suite="$1"
  local fsdp="$2"
  local tp="$3"
  run_sweep \
    --suite "${suite}" \
    --variants default,cce \
    --batch-sizes 16,32,64 \
    --contexts 512,1024,2048 \
    --lora-ranks 16 \
    --dataset-mode synthetic \
    --num-examples 2048 \
    --max-steps 3 \
    --skip-quality-eval \
    --mesh-fsdp "${fsdp}" \
    --mesh-tp "${tp}" \
    --tpu v5litepod-4 \
    --chips 4 \
    --force \
    --outdir "${OUT_BASE}/${suite}"
}

run_mesh_repeat() {
  local suite_prefix="$1"
  local fsdp="$2"
  local tp="$3"
  local repeat="$4"
  run_sweep \
    --suite "${suite_prefix}_repeat${repeat}" \
    --variants default,cce \
    --batch-sizes 16,32 \
    --contexts 512,1024 \
    --lora-ranks 16 \
    --dataset-mode synthetic \
    --num-examples 2048 \
    --max-steps 8 \
    --skip-quality-eval \
    --mesh-fsdp "${fsdp}" \
    --mesh-tp "${tp}" \
    --tpu v5litepod-4 \
    --chips 4 \
    --seed "${repeat}" \
    --force \
    --outdir "${OUT_BASE}/${suite_prefix}_repeat${repeat}"
}

run_mesh_repeat_family() {
  local suite_prefix="$1"
  local fsdp="$2"
  local tp="$3"
  run_mesh_repeat "${suite_prefix}" "${fsdp}" "${tp}" 1
  run_mesh_repeat "${suite_prefix}" "${fsdp}" "${tp}" 2
  run_mesh_repeat "${suite_prefix}" "${fsdp}" "${tp}" 3
}

run_fourchip_frontier() {
  local suite="$1"
  local fsdp="$2"
  local tp="$3"
  run_sweep \
    --suite "${suite}" \
    --variants default,cce \
    --batch-sizes 1,4,8,16,32,64,128 \
    --contexts 256,512,1024,2048,4096 \
    --lora-ranks 16 \
    --dataset-mode synthetic \
    --num-examples 4096 \
    --max-steps 2 \
    --skip-quality-eval \
    --mesh-fsdp "${fsdp}" \
    --mesh-tp "${tp}" \
    --tpu v5litepod-4 \
    --chips 4 \
    --force \
    --outdir "${OUT_BASE}/${suite}"
}

run_outlier_hlo_mesh() {
  local suite="$1"
  local fsdp="$2"
  local tp="$3"
  run_sweep \
    --suite "${suite}" \
    --variants default,cce \
    --batch-sizes 16 \
    --contexts 512,1024 \
    --lora-ranks 16 \
    --dataset-mode synthetic \
    --num-examples 1024 \
    --max-steps 2 \
    --skip-quality-eval \
    --mesh-fsdp "${fsdp}" \
    --mesh-tp "${tp}" \
    --tpu v5litepod-4 \
    --chips 4 \
    --keep-all-xla \
    --force \
    --outdir "${OUT_BASE}/${suite}"
}

run_fourchip_quality() {
  local suite="$1"
  local variant="$2"
  local fsdp="$3"
  local tp="$4"
  run_sweep \
    --suite "${suite}" \
    --variants "${variant}" \
    --batch-sizes 16 \
    --contexts 512 \
    --lora-ranks 16 \
    --dataset-mode opus100 \
    --num-examples 4096 \
    --max-steps 1000 \
    --eval-examples 128 \
    --eval-batches 8 \
    --generation-examples 0 \
    --mesh-fsdp "${fsdp}" \
    --mesh-tp "${tp}" \
    --tpu v5litepod-4 \
    --chips 4 \
    --force \
    --outdir "${OUT_BASE}/${suite}"
}

case "${PROFILE}" in
  parity)
    python -m pytest -q \
      tests/test_chunked_linear_ce.py \
      tests/test_tunix_lora_gradient_parity.py \
      | tee "${OUT_BASE}/parity_pytest.log"

    run_sweep \
      --suite parity_270m_one_step \
      --variants default,cce \
      --batch-sizes 1,4 \
      --contexts 128,512 \
      --lora-ranks 16 \
      --dataset-mode synthetic \
      --num-examples 128 \
      --max-steps 2 \
      --skip-quality-eval \
      --force \
      --outdir "${OUT_BASE}/parity_270m_one_step"
    ;;

  frontier-low)
    run_sweep \
      --suite frontier_low \
      --variants default,cce \
      --batch-sizes 1,2,4,8 \
      --contexts 256,512,1024,2048,4096,8192,16384,32768 \
      --lora-ranks 16 \
      --dataset-mode synthetic \
      --num-examples 2048 \
      --max-steps 2 \
      --skip-quality-eval \
      --force \
      --outdir "${OUT_BASE}/frontier_low"
    ;;

  frontier-high)
    run_sweep \
      --suite frontier_high \
      --variants default,cce \
      --batch-sizes 16,32,64,128 \
      --contexts 256,512,1024,2048,4096,8192,16384,32768 \
      --lora-ranks 16 \
      --dataset-mode synthetic \
      --num-examples 4096 \
      --max-steps 2 \
      --skip-quality-eval \
      --force \
      --outdir "${OUT_BASE}/frontier_high"
    ;;

  rank)
    run_sweep \
      --suite rank_sensitivity \
      --variants default,cce \
      --batch-sizes 8,16,32,64 \
      --contexts 512,1024,2048,4096 \
      --lora-ranks 4,16,64 \
      --dataset-mode synthetic \
      --num-examples 2048 \
      --max-steps 2 \
      --skip-quality-eval \
      --force \
      --outdir "${OUT_BASE}/rank_sensitivity"
    ;;

  chunk)
    run_sweep \
      --suite chunk_tuning \
      --variants cce \
      --batch-sizes 16 \
      --contexts 512,2048,4096 \
      --lora-ranks 16 \
      --token-chunks 64,128,256,512 \
      --vocab-chunks 4096,8192,16384,32768 \
      --dataset-mode synthetic \
      --num-examples 1024 \
      --max-steps 3 \
      --skip-quality-eval \
      --force \
      --outdir "${OUT_BASE}/chunk_tuning"

    run_sweep \
      --suite pressure_points \
      --variants default,cce \
      --batch-sizes 16,32,64 \
      --contexts 512,1024,2048,4096 \
      --lora-ranks 16 \
      --dataset-mode synthetic \
      --num-examples 2048 \
      --max-steps 5 \
      --skip-quality-eval \
      --force \
      --outdir "${OUT_BASE}/pressure_points"
    ;;

  quality-default)
    run_sweep \
      --suite quality_default_b16_l512 \
      --variants default \
      --batch-sizes 16 \
      --contexts 512 \
      --lora-ranks 16 \
      --dataset-mode opus100 \
      --num-examples 8000 \
      --max-steps 5000 \
      --eval-examples 256 \
      --eval-batches 16 \
      --generation-examples 32 \
      --generation-batch-size 8 \
      --force \
      --outdir "${OUT_BASE}/quality_default_b16_l512"
    ;;

  quality-cce)
    run_sweep \
      --suite quality_cce_b16_l512 \
      --variants cce \
      --batch-sizes 16 \
      --contexts 512 \
      --lora-ranks 16 \
      --dataset-mode opus100 \
      --num-examples 8000 \
      --max-steps 5000 \
      --eval-examples 256 \
      --eval-batches 16 \
      --generation-examples 32 \
      --generation-batch-size 8 \
      --force \
      --outdir "${OUT_BASE}/quality_cce_b16_l512"
    ;;

  quality-capacity)
    run_sweep \
      --suite quality_cce_capacity_b64_l512 \
      --variants cce \
      --batch-sizes 64 \
      --contexts 512 \
      --lora-ranks 16 \
      --dataset-mode opus100 \
      --num-examples 8000 \
      --max-steps 1250 \
      --eval-examples 256 \
      --eval-batches 4 \
      --generation-examples 32 \
      --generation-batch-size 8 \
      --force \
      --outdir "${OUT_BASE}/quality_cce_capacity_b64_l512"
    ;;

  mesh-fsdp4)
    run_mesh_sweep mesh_fsdp4_tp1 4 1
    ;;

  mesh-2x2)
    run_mesh_sweep mesh_fsdp2_tp2 2 2
    ;;

  mesh-tp4)
    run_mesh_sweep mesh_fsdp1_tp4 1 4
    ;;

  mesh-2x2-repeat)
    run_mesh_repeat_family mesh_fsdp2_tp2 2 2
    ;;

  mesh-fsdp4-repeat)
    run_mesh_repeat_family mesh_fsdp4_tp1 4 1
    ;;

  mesh-tp4-repeat)
    run_mesh_repeat_family mesh_fsdp1_tp4 1 4
    ;;

  mesh-repeat-rest)
    run_mesh_repeat_family mesh_fsdp4_tp1 4 1
    run_mesh_repeat_family mesh_fsdp1_tp4 1 4
    ;;

  fourchip-frontier-fsdp4)
    run_fourchip_frontier fourchip_frontier_fsdp4_tp1 4 1
    ;;

  fourchip-frontier-2x2)
    run_fourchip_frontier fourchip_frontier_fsdp2_tp2 2 2
    ;;

  fourchip-frontier-tp4)
    run_fourchip_frontier fourchip_frontier_fsdp1_tp4 1 4
    ;;

  outlier-hlo)
    run_outlier_hlo_mesh outlier_hlo_fsdp4_tp1 4 1
    run_outlier_hlo_mesh outlier_hlo_fsdp2_tp2 2 2
    run_outlier_hlo_mesh outlier_hlo_fsdp1_tp4 1 4
    ;;

  fourchip-chunk-2x2)
    run_sweep \
      --suite fourchip_chunk_fsdp2_tp2_b16_l512 \
      --variants cce \
      --batch-sizes 16 \
      --contexts 512 \
      --lora-ranks 16 \
      --token-chunks 64,128,256,512 \
      --vocab-chunks 4096,8192,16384,32768 \
      --dataset-mode synthetic \
      --num-examples 1024 \
      --max-steps 4 \
      --skip-quality-eval \
      --mesh-fsdp 2 \
      --mesh-tp 2 \
      --tpu v5litepod-4 \
      --chips 4 \
      --force \
      --outdir "${OUT_BASE}/fourchip_chunk_fsdp2_tp2_b16_l512"
    ;;

  fourchip-quality-fsdp4-default)
    run_fourchip_quality quality_4chip_fsdp4_default_b16_l512 default 4 1
    ;;

  fourchip-quality-fsdp4-cce)
    run_fourchip_quality quality_4chip_fsdp4_cce_b16_l512 cce 4 1
    ;;

  *)
    echo "Unknown PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac

tar -C "$(dirname "${OUT_BASE}")" -czf "${OUT_BASE}-${PROFILE}.tar.gz" "$(basename "${OUT_BASE}")"
echo "artifact=${OUT_BASE}-${PROFILE}.tar.gz"
