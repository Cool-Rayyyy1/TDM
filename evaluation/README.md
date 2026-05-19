# HPDv2 evaluation

Generate HPDv2 benchmark images with **teacher** (PixArt, 25 steps) and **student** (SD3, 4 steps), then score with offline **HPS v2.1**.

## Quick start

```bash
cd /mnt/afs_zhangyunzhe/TDM/evaluation
bash run_hpdv2_eval.sh
```

## Output layout

```
outputs/<run_id>/
  manifest.json
  images/
    teacher/
      anime/00000.jpg
      concept-art/
      paintings/
      photo/
    student/          # after STUDENT_MODEL is set
      ...
  scores/
    hps_v2.1_summary.json
```

## Multi-GPU (default 2 cards)

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash run_hpdv2_eval.sh
```

| GPU | Generation workload | HPS workload |
|-----|-------------------|--------------|
| 0 | anime + concept-art (1600 imgs) | anime + concept-art |
| 1 | paintings + photo (1600 imgs) | paintings + photo |

Single GPU: `CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 bash run_hpdv2_eval.sh`

### Rough runtime (1024², teacher only, 4×800 prompts)

Measured ~3 s/image for PixArt 25-step on one A100-class GPU.

| Stage | 1 GPU | 2 GPU |
|-------|-------|-------|
| Teacher 3200 imgs | ~2.7–3.3 h | ~1.4–1.7 h |
| HPS v2.1 scoring | ~5–10 min | ~3–6 min |
| **Total (teacher + HPS)** | **~3–3.5 h** | **~1.5–2 h** |

If student (SD3 4-step) is enabled later, add ~0.5–1 h per 3200 imgs on 2 GPUs (faster per image than teacher).

## Environment variables (`run_hpdv2_eval.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CUDA_VISIBLE_DEVICES` | `0,1` | Visible GPU ids |
| `NUM_GPUS` | `2` | Parallel workers |
| `TEACHER_MODEL` | `TDM/pretrained_models/.../PixArt-XL-2-1024-MS` | PixArt diffusers folder (symlink created from workspace `pretrained_models`) |
| `STUDENT_MODEL` | *(empty)* | Finetuned SD3 diffusers dir (global finetune, **no LoRA**). Empty = skip student |
| `SD3_BASE_MODEL` | `pretrained_models/stable-diffusion-3.5-medium` | Base SD3 when testing student pipe before finetune |
| `HPS_CKPT` | `evaluation/hpsv2/weights/HPS_v2.1_compressed.pt` | Offline HPS v2.1 weights |
| `MAX_PROMPTS_PER_CATEGORY` | *(all 800)* | Debug cap per category |
| `OUTPUT_DIR` | auto timestamp | Fixed output root |

When the student checkpoint is ready:

```bash
STUDENT_MODEL=/path/to/finetuned_sd3 bash run_hpdv2_eval.sh
```

## Student pipeline notes

- 4 inference steps, `guidance_scale=1.0`, `flow_shift=6`
- TAESD3 tiny VAE, DPM-Solver scheduler from Sana repo
- **No LoRA** (matches global finetune training)

## Scores

HPS v2.1 reports per-category means (**Animation**, **Concept-Art**, **Painting**, **Photo**) and **Average** in `scores/hps_v2.1_summary.json`.
