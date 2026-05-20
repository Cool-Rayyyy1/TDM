#!/usr/bin/env bash
# HPDv2 eval: PixArt teacher (25-step) + TDM student (4-step anchors) + offline HPS v2.1.
# Usage:
#   bash run_hpdv2_eval.sh
# Resume HPS only (images already done):
#   OUTPUT_DIR=/mnt/afs_zhangyunzhe/TDM/evaluation/outputs/20260519_234052 bash run_hpdv2_eval.sh
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
TDM_DIR="${TDM_DIR:-${WORKSPACE_ROOT}/TDM}"
EVAL_DIR="${EVAL_DIR:-${TDM_DIR}/evaluation}"

CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-tdm}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_GPUS="${NUM_GPUS:-2}"

# User path: TDM/pretrained_models -> workspace pretrained_models (symlink if missing)
PRETRAINED_LINK="${TDM_DIR}/pretrained_models"
if [[ ! -e "${PRETRAINED_LINK}" ]]; then
  ln -sfn "${WORKSPACE_ROOT}/pretrained_models" "${PRETRAINED_LINK}"
fi

TEACHER_MODEL="${TEACHER_MODEL:-/mnt/afs_zhangyunzhe/pretrained_models/PixArt-XL-2-512x512}"
STUDENT_MODEL="${STUDENT_MODEL:-/mnt/afs_zhangyunzhe/TDM/outputs/tdm_pixart512_cfg4.5_totalstep900-Huber/checkpoint-12000}"
STUDENT_BASE_MODEL="${STUDENT_BASE_MODEL:-${TEACHER_MODEL}}"
STUDENT_TOTAL_STEPS="${STUDENT_TOTAL_STEPS:-900}"

TEACHER_STEPS="${TEACHER_STEPS:-25}"
STUDENT_STEPS="${STUDENT_STEPS:-4}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
STUDENT_GUIDANCE="${STUDENT_GUIDANCE:-1.0}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
SEED="${SEED:-8888}"

# Set to an existing run folder to resume (auto-skip gen if images/ complete, then HPS only)
# Any ONE of these works:
#   bash run_hpdv2_eval.sh /mnt/afs_zhangyunzhe/TDM/evaluation/outputs/20260519_234052
#   export OUTPUT_DIR=.../20260519_234052 && bash run_hpdv2_eval.sh
#   OUTPUT_DIR=.../20260519_234052 bash run_hpdv2_eval.sh
if [[ $# -ge 1 && "$1" != -* ]]; then
  export OUTPUT_DIR="$1"
  shift
fi
export OUTPUT_DIR="${OUTPUT_DIR:-}"
SCORE_ONLY="${SCORE_ONLY:-0}"
FORCE_GENERATE="${FORCE_GENERATE:-0}"
MAX_PROMPTS_PER_CATEGORY="${MAX_PROMPTS_PER_CATEGORY:-}"
HPS_CKPT="${HPS_CKPT:-${EVAL_DIR}/hpsv2/weights/HPS_v2.1_compressed.pt}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${EVAL_DIR}"

pip install -q -r requirements_eval.txt
python -c "from hps_scorer import ensure_hpsv2_bpe; ensure_hpsv2_bpe()"

EXTRA=()
[[ -n "${OUTPUT_DIR}" ]] && EXTRA+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${MAX_PROMPTS_PER_CATEGORY}" ]] && EXTRA+=(--max_prompts_per_category "${MAX_PROMPTS_PER_CATEGORY}")
[[ -n "${STUDENT_MODEL}" ]] && EXTRA+=(--student_model "${STUDENT_MODEL}")
[[ -n "${STUDENT_BASE_MODEL}" ]] && EXTRA+=(--student_base_model "${STUDENT_BASE_MODEL}")
EXTRA+=(--student_total_steps "${STUDENT_TOTAL_STEPS}")
[[ "${SCORE_ONLY}" == "1" ]] && EXTRA+=(--score_only)
[[ "${FORCE_GENERATE}" == "1" ]] && EXTRA+=(--force_generate)

if [[ -z "${OUTPUT_DIR}" ]]; then
  echo "WARNING: OUTPUT_DIR is not set -> will create a NEW folder evaluation/outputs/<timestamp>/"
  echo "         (A separate line 'OUTPUT_DIR=...' without export is NOT visible to this script.)"
  echo "         Resume an existing run, e.g.:"
  echo "         bash run_hpdv2_eval.sh ${EVAL_DIR}/outputs/20260519_234052"
  echo "         # or: export OUTPUT_DIR=.../20260519_234052 && bash run_hpdv2_eval.sh"
else
  if [[ ! -d "${OUTPUT_DIR}" ]]; then
    echo "ERROR: OUTPUT_DIR does not exist: ${OUTPUT_DIR}"
    exit 1
  fi
  echo "RESUME: using existing OUTPUT_DIR=${OUTPUT_DIR}"
fi

echo "=============================================="
echo "HPDv2 eval  (run from: ${EVAL_DIR})"
echo "  GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"
echo "  teacher: ${TEACHER_MODEL} (${TEACHER_STEPS} steps, cfg=${GUIDANCE_SCALE})"
echo "  student: ${STUDENT_MODEL}"
echo "  resolution: ${HEIGHT}x${WIDTH}"
echo "  student TDM: ${STUDENT_STEPS} steps, total_steps=${STUDENT_TOTAL_STEPS}, cfg=${STUDENT_GUIDANCE} (training-aligned)"
echo "  HPS ckpt: ${HPS_CKPT}"
echo "  output: ${OUTPUT_DIR:-evaluation/outputs/<auto>}"
echo "  resume: auto-skip generation if images complete | SCORE_ONLY=${SCORE_ONLY} FORCE_GENERATE=${FORCE_GENERATE}"
echo "  split: GPU0 -> anime+concept-art, GPU1 -> paintings+photo (2-GPU mode)"
echo "=============================================="

python run_hpdv2_eval.py \
  --num_gpus "${NUM_GPUS}" \
  --teacher_model "${TEACHER_MODEL}" \
  --teacher_steps "${TEACHER_STEPS}" \
  --teacher_guidance "${GUIDANCE_SCALE}" \
  --student_steps "${STUDENT_STEPS}" \
  --student_guidance "${STUDENT_GUIDANCE}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --seed "${SEED}" \
  --hps_checkpoint "${HPS_CKPT}" \
  "${EXTRA[@]}" \
  "$@"
