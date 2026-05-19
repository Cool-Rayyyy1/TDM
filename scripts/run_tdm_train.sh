#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/mnt/afs_zhangyunzhe}"
CONDA_ROOT="${CONDA_ROOT:-${WORKSPACE_ROOT}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-tdm}"
TDM_DIR="${TDM_DIR:-${WORKSPACE_ROOT}/TDM}"

PIXART_MODEL="${PIXART_MODEL:-${WORKSPACE_ROOT}/pretrained_models/PixArt-XL-2-1024-MS/PixArt-XL-2-1024-MS}"
PROMPT_JSONL="${PROMPT_JSONL:-${WORKSPACE_ROOT}/dataset/JourneyDB/data/train/train_anno/train_anno.jsonl}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${WORKSPACE_ROOT}/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

BSZ="${BSZ:-1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29503}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-12001}"
CFG="${CFG:-4.5}"
TOTAL_STEPS="${TOTAL_STEPS:-900}"
RESOLUTION="${RESOLUTION:-1024}"
CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-500}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
NUM_STUDENT_STEPS="${NUM_STUDENT_STEPS:-4}"
STUDENT_TIMESTEPS="${STUDENT_TIMESTEPS:-899-674-449-224}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/tdm_pixart1024_steps${STUDENT_TIMESTEPS}}"

LOG_DIR="${LOG_DIR:-${TDM_DIR}/logs}"
mkdir -p "${LOG_DIR}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train_steps${STUDENT_TIMESTEPS}_${RUN_ID}.log}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${TDM_DIR}"

EXTRA_ARGS=()
[[ -n "${RESUME_FROM_CHECKPOINT:-}" ]] && EXTRA_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
[[ -n "${MAX_TRAIN_SAMPLES:-}" ]] && EXTRA_ARGS+=(--max_train_samples "${MAX_TRAIN_SAMPLES}")

echo "=============================================="
echo "TDM train | ${NUM_STUDENT_STEPS}-step | anchors ${STUDENT_TIMESTEPS} | log ${LOG_FILE}"
echo "output: ${OUTPUT_DIR} -> ..._cfg${CFG}_totalstep${TOTAL_STEPS}-Huber"
echo "ckpt every ${CHECKPOINTING_STEPS} steps"
echo "=============================================="

CMD=(
  accelerate launch
  --main_process_port "${MAIN_PROCESS_PORT}"
  --num_processes "${NUM_PROCESSES}"
  --mixed_precision fp16
  train_tdm_demo.py
  --pretrained_model_name_or_path "${PIXART_MODEL}"
  --prompt_jsonl_path "${PROMPT_JSONL}"
  --train_batch_size="${BSZ}"
  --gradient_accumulation_steps=1
  --gradient_checkpointing
  --max_train_steps="${MAX_TRAIN_STEPS}"
  --learning_rate=2e-05
  --max_grad_norm=1
  --enable_xformers_memory_efficient_attention
  --use_8bit_adam
  --cfg "${CFG}"
  --total_steps "${TOTAL_STEPS}"
  --num_student_steps "${NUM_STUDENT_STEPS}"
  --lr_scheduler cosine_with_restarts
  --lr_warmup_steps 50
  --use_huber --use_separate
  --resolution "${RESOLUTION}"
  --checkpointing_steps "${CHECKPOINTING_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --output_dir "${OUTPUT_DIR}"
  "${EXTRA_ARGS[@]}"
  "$@"
)

stdbuf -oL -eL "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
