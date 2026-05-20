# HPDv2 evaluation

Generate HPDv2 benchmark images with **teacher** (PixArt-512, 25 steps) and **student** (TDM 4-step, `train_tdm_demo.generate_new`), then score with offline **HPS v2.1**.

## Quick start

```bash
cd /mnt/afs_zhangyunzhe/TDM/evaluation
bash run_hpdv2_eval.sh
```

Set model paths after 512 weights and a new checkpoint are ready:

```bash
export TEACHER_MODEL=/mnt/afs_zhangyunzhe/pretrained_models/PixArt-XL-2-512x512
export STUDENT_MODEL=/mnt/afs_zhangyunzhe/TDM/outputs/tdm_pixart512_cfg4.5_totalstep900-Huber/checkpoint-10000
bash run_hpdv2_eval.sh
```

## Output layout

```
outputs/<run_id>/
  manifest.json
  images/
    teacher/
      anime/00000.jpg
      ...
    student/
      ...
  scores/
    hps_v2.1_summary.json
```

## Multi-GPU (default 2 cards)

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash run_hpdv2_eval.sh
```

| GPU | Generation | HPS |
|-----|------------|-----|
| 0 | anime + concept-art | anime + concept-art |
| 1 | paintings + photo | paintings + photo |

## Environment variables (`run_hpdv2_eval.sh`)

| Variable | Default | Description |
|----------|---------|-------------|
| `TEACHER_MODEL` | `pretrained_models/PixArt-XL-2-512x512` | PixArt diffusers folder |
| `STUDENT_MODEL` | *(empty)* | TDM checkpoint with `unet/` |
| `STUDENT_BASE_MODEL` | same as teacher | PixArt base for student VAE/T5 |
| `STUDENT_TOTAL_STEPS` | `900` | Training `--total_steps` (4-step anchors: 899,674,449,224) |
| `STUDENT_STEPS` | `4` | Student inference steps |
| `OUTPUT_DIR` | auto timestamp | Resume: `bash run_hpdv2_eval.sh /path/to/run` |

Resolution defaults to **512×512** (PixArt-512).

## Student sampler

Uses `evaluation/tdm_student.py` → upstream `generate_new` with `DDPMScheduler`, **not** SD3. No LoRA.
