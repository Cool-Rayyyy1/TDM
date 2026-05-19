#!/usr/bin/env bash
# HPDv2 eval: PixArt teacher (25-step) + SD3 student (4-step) + offline HPS v2.1.
# Usage: bash run_hpdv2_eval.sh
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

TEACHER_MODEL="${TEACHER_MODEL:-${TDM_DIR}/pretrained_models/PixArt-XL-2-1024-MS/PixArt-XL-2-1024-MS}"
# Finetuned SD3 (global finetune, no LoRA). Leave empty to skip student generation/scoring.
STUDENT_MODEL="${STUDENT_MODEL:-}"
# Base SD3 for student pipe when STUDENT_MODEL is empty but RUN_STUDENT_BASE=1
SD3_BASE_MODEL="${SD3_BASE_MODEL:-${WORKSPACE_ROOT}/pretrained_models/stable-diffusion-3.5-medium}"
TAESD3_MODEL="${TAESD3_MODEL:-madebyollin/taesd3}"
SANA_SCHEDULER_REPO="${SANA_SCHEDULER_REPO:-Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers}"

OUTPUT_DIR="${OUTPUT_DIR:-}"
MAX_PROMPTS_PER_CATEGORY="${MAX_PROMPTS_PER_CATEGORY:-}"
HPS_CKPT="${HPS_CKPT:-${EVAL_DIR}/hpsv2/weights/HPS_v2.1_compressed.pt}"

TEACHER_STEPS="${TEACHER_STEPS:-25}"
STUDENT_STEPS="${STUDENT_STEPS:-4}"
SEED="${SEED:-8888}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${EVAL_DIR}"

pip install -q -r requirements_eval.txt
python -c "from hps_scorer import ensure_hpsv2_bpe; ensure_hpsv2_bpe()"

EXTRA=()
[[ -n "${OUTPUT_DIR}" ]] && EXTRA+=(--output_dir "${OUTPUT_DIR}")
[[ -n "${MAX_PROMPTS_PER_CATEGORY}" ]] && EXTRA+=(--max_prompts_per_category "${MAX_PROMPTS_PER_CATEGORY}")
[[ -n "${STUDENT_MODEL}" ]] && EXTRA+=(--student_model "${STUDENT_MODEL}")
[[ "${RUN_STUDENT_BASE:-0}" == "1" ]] && EXTRA+=(--run_student)

echo "=============================================="
echo "HPDv2 eval"
echo "  GPUs: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  NUM_GPUS=${NUM_GPUS}"
echo "  teacher: ${TEACHER_MODEL} (${TEACHER_STEPS} steps)"
echo "  student: ${STUDENT_MODEL:-<skip until finetuned>}"
echo "  HPS ckpt: ${HPS_CKPT}"
echo "  output: ${OUTPUT_DIR:-evaluation/outputs/<auto>}"
echo "  split: GPU0 -> anime+concept-art, GPU1 -> paintings+photo (2-GPU mode)"
echo "=============================================="

python run_hpdv2_eval.py \
  --num_gpus "${NUM_GPUS}" \
  --teacher_model "${TEACHER_MODEL}" \
  --sd3_base_model "${SD3_BASE_MODEL}" \
  --taesd3_model "${TAESD3_MODEL}" \
  --sana_scheduler_repo "${SANA_SCHEDULER_REPO}" \
  --teacher_steps "${TEACHER_STEPS}" \
  --student_steps "${STUDENT_STEPS}" \
  --seed "${SEED}" \
  --hps_checkpoint "${HPS_CKPT}" \
  "${EXTRA[@]}" \
  "$@"
