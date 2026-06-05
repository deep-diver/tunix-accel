#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-frontier-low}"
ROOT="${ROOT:-$HOME/TUNIX-TRY}"
OUT_BASE="${OUT_BASE:-/tmp/gemma3-270m-cce-rerun}"

cd "${ROOT}"

if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
  python3 -m pip install --upgrade pip
  python3 -m pip install -r requirements.txt
  python3 -m pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

export PYTHONUNBUFFERED=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

run_sweep() {
  python3 01-CCE/run_gemma3_270m_cce_sweep.py "$@"
}

case "${PROFILE}" in
  parity)
    python3 -m pytest -q \
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

  *)
    echo "Unknown PROFILE=${PROFILE}" >&2
    exit 2
    ;;
esac

tar -C "$(dirname "${OUT_BASE}")" -czf "${OUT_BASE}-${PROFILE}.tar.gz" "$(basename "${OUT_BASE}")"
echo "artifact=${OUT_BASE}-${PROFILE}.tar.gz"
