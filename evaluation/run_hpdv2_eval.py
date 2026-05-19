#!/usr/bin/env python
# coding=utf-8
"""HPDv2 benchmark: teacher (PixArt 25-step) vs student (SD3 4-step) + HPS v2.1 scoring."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

EVAL_ROOT = Path(__file__).resolve().parent
if str(EVAL_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_ROOT))
DEFAULT_PROMPTS_DIR = EVAL_ROOT / "hpdv2_benchmark" / "prompts"
DEFAULT_HPS_CKPT = EVAL_ROOT / "hpsv2" / "weights" / "HPS_v2.1_compressed.pt"
DEFAULT_OUTPUT_ROOT = EVAL_ROOT / "outputs"

# json filename -> folder name (hpsv2 benchmark layout) -> report label
CATEGORIES: Tuple[Tuple[str, str, str], ...] = (
    ("anime.json", "anime", "Animation"),
    ("concept-art.json", "concept-art", "Concept-Art"),
    ("paintings.json", "paintings", "Painting"),
    ("photo.json", "photo", "Photo"),
)


@dataclass
class CategoryData:
    folder: str
    label: str
    prompts: List[str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HPDv2 teacher/student generation and HPS v2.1 evaluation.")
    p.add_argument("--prompts_dir", type=Path, default=DEFAULT_PROMPTS_DIR)
    p.add_argument("--output_dir", type=Path, default=None, help="Run output root (default: evaluation/outputs/<timestamp>).")
    p.add_argument("--teacher_model", type=str, required=True, help="PixArt diffusers folder.")
    p.add_argument(
        "--student_model",
        type=str,
        default="",
        help="Finetuned SD3 diffusers folder (global finetune, no LoRA). Empty = skip student.",
    )
    p.add_argument(
        "--sd3_base_model",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Base SD3 when --student_model is not set but --run_student is used.",
    )
    p.add_argument("--taesd3_model", type=str, default="madebyollin/taesd3")
    p.add_argument(
        "--sana_scheduler_repo",
        type=str,
        default="Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers",
    )
    p.add_argument("--teacher_steps", type=int, default=25)
    p.add_argument("--student_steps", type=int, default=4)
    p.add_argument("--teacher_guidance", type=float, default=4.5)
    p.add_argument("--student_guidance", type=float, default=1.0)
    p.add_argument("--flow_shift", type=float, default=6.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--max_prompts_per_category", type=int, default=None, help="Debug cap per category.")
    p.add_argument("--skip_generate", action="store_true")
    p.add_argument("--skip_teacher", action="store_true")
    p.add_argument("--skip_student", action="store_true")
    p.add_argument("--skip_score", action="store_true")
    p.add_argument("--run_student", action="store_true", help="Generate with SD3 base (no finetune ckpt).")
    p.add_argument("--hps_checkpoint", type=Path, default=DEFAULT_HPS_CKPT)
    p.add_argument("--hps_batch_size", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Parallel GPUs for generation and HPS scoring (categories split round-robin).",
    )
    return p.parse_args()


def resolve_gpu_ids(num_gpus: int) -> List[int]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = [int(x) for x in visible.split(",") if x.strip() != ""]
    else:
        n = torch.cuda.device_count() if torch.cuda.is_available() else 0
        ids = list(range(n))
    if not ids:
        return [0]
    if num_gpus > len(ids):
        raise ValueError(f"--num_gpus={num_gpus} but only {len(ids)} GPU(s) in CUDA_VISIBLE_DEVICES={visible!r}")
    return ids[:num_gpus]


def split_categories_round_robin(categories: List[CategoryData], num_workers: int) -> List[List[CategoryData]]:
    buckets: List[List[CategoryData]] = [[] for _ in range(num_workers)]
    for i, cat in enumerate(categories):
        buckets[i % num_workers].append(cat)
    return [b for b in buckets if b]


def load_categories(prompts_dir: Path, max_per_category: Optional[int]) -> List[CategoryData]:
    out: List[CategoryData] = []
    for json_name, folder, label in CATEGORIES:
        path = prompts_dir / json_name
        with open(path, encoding="utf-8") as f:
            prompts = json.load(f)
        if max_per_category is not None:
            prompts = prompts[:max_per_category]
        out.append(CategoryData(folder=folder, label=label, prompts=prompts))
    return out


def image_path(root: Path, model_name: str, category: str, index: int) -> Path:
    return root / "images" / model_name / category / f"{index:05d}.jpg"


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def build_teacher_pipe(model_path: str, device: str):
    from diffusers import PixArtAlphaPipeline

    dtype = torch.float16
    pipe = PixArtAlphaPipeline.from_pretrained(model_path, torch_dtype=dtype)
    pipe = pipe.to(device)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def build_student_pipe(
    student_model: str,
    sd3_base_model: str,
    taesd3_model: str,
    sana_scheduler_repo: str,
    flow_shift: float,
    device: str,
):
    from diffusers import AutoencoderTiny, DPMSolverMultistepScheduler, StableDiffusion3Pipeline

    dtype = torch.float16
    load_path = student_model if student_model else sd3_base_model
    pipe = StableDiffusion3Pipeline.from_pretrained(load_path, torch_dtype=dtype)
    pipe.vae = AutoencoderTiny.from_pretrained(taesd3_model, torch_dtype=dtype)
    pipe.vae.config.shift_factor = 0.0
    pipe = pipe.to(device)
    sched = DPMSolverMultistepScheduler.from_pretrained(sana_scheduler_repo, subfolder="scheduler")
    sched.config["flow_shift"] = flow_shift
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(sched.config)
    if hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
    return pipe


def _pipe_device(pipe) -> str:
    if hasattr(pipe, "_execution_device"):
        return str(pipe._execution_device)
    if hasattr(pipe, "device"):
        return str(pipe.device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def generate_images(
    pipe,
    categories: List[CategoryData],
    out_root: Path,
    model_name: str,
    num_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
    negative_prompt: str = "",
) -> None:
    generator = torch.Generator(device=_pipe_device(pipe)).manual_seed(seed)
    for cat in categories:
        cat_dir = out_root / "images" / model_name / cat.folder
        cat_dir.mkdir(parents=True, exist_ok=True)
        for idx, prompt in enumerate(tqdm(cat.prompts, desc=f"{model_name}/{cat.folder}")):
            save_path = image_path(out_root, model_name, cat.folder, idx)
            if save_path.exists():
                continue
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                height=height,
                width=width,
                num_images_per_prompt=1,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            result.images[0].save(save_path, quality=95)


def _category_to_dict(cat: CategoryData) -> dict:
    return {"folder": cat.folder, "label": cat.label, "prompts": cat.prompts}


def _dict_to_category(d: dict) -> CategoryData:
    return CategoryData(folder=d["folder"], label=d["label"], prompts=d["prompts"])


def _generate_worker(
    gpu_id: int,
    categories_dicts: List[dict],
    out_root: str,
    model_kind: str,
    gen_cfg: dict,
) -> None:
    """Subprocess entry: one GPU, subset of categories."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    import torch as _torch

    out = Path(out_root)
    cats = [_dict_to_category(d) for d in categories_dicts]
    if model_kind == "teacher":
        pipe = build_teacher_pipe(gen_cfg["model_path"], "cuda")
        generate_images(
            pipe,
            cats,
            out,
            "teacher",
            gen_cfg["num_steps"],
            gen_cfg["guidance_scale"],
            gen_cfg["height"],
            gen_cfg["width"],
            gen_cfg["seed"],
        )
    else:
        pipe = build_student_pipe(
            gen_cfg.get("student_model", ""),
            gen_cfg["sd3_base_model"],
            gen_cfg["taesd3_model"],
            gen_cfg["sana_scheduler_repo"],
            gen_cfg["flow_shift"],
            "cuda",
        )
        generate_images(
            pipe,
            cats,
            out,
            "student",
            gen_cfg["num_steps"],
            gen_cfg["guidance_scale"],
            gen_cfg["height"],
            gen_cfg["width"],
            gen_cfg["seed"],
        )
    del pipe
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()


def _hps_worker(
    gpu_id: int,
    categories_tuples: List[Tuple[str, str, str]],
    prompts_dir: str,
    images_dir: str,
    checkpoint: str,
    batch_size: int,
) -> Dict[str, object]:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from hps_scorer import _score_categories_subset

    per_style, all_scores = _score_categories_subset(
        Path(prompts_dir),
        Path(images_dir),
        Path(checkpoint),
        categories_tuples,
        batch_size,
    )
    for label, stats in per_style.items():
        print(f"[GPU {gpu_id}] {label:<16} {stats['mean']:.4f}  (n={stats['count']})", flush=True)
    return {"per_category": per_style, "scores": all_scores}


def parallel_generate(
    gpu_ids: List[int],
    categories: List[CategoryData],
    out_root: Path,
    model_kind: str,
    gen_cfg: dict,
) -> None:
    buckets = split_categories_round_robin(categories, len(gpu_ids))
    if len(buckets) == 1:
        _generate_worker(
            gpu_ids[0],
            [_category_to_dict(c) for c in buckets[0]],
            str(out_root),
            model_kind,
            gen_cfg,
        )
        return

    ctx = mp.get_context("spawn")
    procs = []
    for gpu_id, bucket in zip(gpu_ids, buckets):
        p = ctx.Process(
            target=_generate_worker,
            args=(gpu_id, [_category_to_dict(c) for c in bucket], str(out_root), model_kind, gen_cfg),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"{model_kind} generation worker failed with exit code {p.exitcode}")


def parallel_hps_score(
    gpu_ids: List[int],
    prompts_dir: Path,
    images_dir: Path,
    checkpoint: Path,
    batch_size: int,
) -> Dict[str, object]:
    cat_list = list(CATEGORIES)
    buckets: List[List[Tuple[str, str, str]]] = [[] for _ in range(len(gpu_ids))]
    for i, cat in enumerate(cat_list):
        buckets[i % len(gpu_ids)].append(cat)

    if len(gpu_ids) == 1:
        return run_hps_benchmark(prompts_dir, images_dir, checkpoint, batch_size)

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(gpu_ids)) as pool:
        async_results = []
        for gpu_id, bucket in zip(gpu_ids, buckets):
            if not bucket:
                continue
            async_results.append(
                pool.apply_async(
                    _hps_worker,
                    (gpu_id, bucket, str(prompts_dir), str(images_dir), str(checkpoint), batch_size),
                )
            )
        partials = [r.get() for r in async_results]

    merged_style: Dict[str, Dict[str, float]] = {}
    all_scores: List[float] = []
    for part in partials:
        merged_style.update(part["per_category"])
        all_scores.extend(part["scores"])

    import numpy as np

    for label in ["Animation", "Concept-Art", "Painting", "Photo"]:
        if label in merged_style:
            s = merged_style[label]
            print(f"  {label:<16} {s['mean']:.4f}  (n={s['count']})")
    average = float(np.mean(all_scores)) if all_scores else float("nan")
    print(f"  {'Average':<16} {average:.4f}")
    return {"per_category": merged_style, "average": average, "num_images": len(all_scores)}


def run_hps_benchmark(
    prompts_dir: Path,
    images_model_dir: Path,
    checkpoint: Path,
    batch_size: int,
) -> Dict[str, object]:
    from hps_scorer import score_benchmark_folder

    return score_benchmark_folder(
        prompts_dir, images_model_dir, checkpoint, list(CATEGORIES), batch_size=batch_size
    )


def write_manifest(out_root: Path, categories: List[CategoryData], args: argparse.Namespace) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(),
        "num_gpus": args.num_gpus,
        "gpu_ids": resolve_gpu_ids(args.num_gpus),
        "teacher_model": args.teacher_model,
        "student_model": args.student_model or None,
        "sd3_base_model": args.sd3_base_model,
        "teacher_steps": args.teacher_steps,
        "student_steps": args.student_steps,
        "categories": {c.folder: {"label": c.label, "num_prompts": len(c.prompts)} for c in categories},
    }
    with open(out_root / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = DEFAULT_OUTPUT_ROOT / stamp
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    categories = load_categories(args.prompts_dir, args.max_prompts_per_category)
    write_manifest(out_root, categories, args)

    run_student = bool(args.student_model) or args.run_student
    if not args.student_model and args.run_student:
        print("[student] Using SD3 base weights (no finetuned checkpoint).")

    gpu_ids = resolve_gpu_ids(args.num_gpus)
    print(f"[GPUs] Using {len(gpu_ids)} device(s): {gpu_ids}")

    if not args.skip_generate:
        if not args.skip_teacher:
            print(f"[teacher] PixArt from {args.teacher_model} ({args.teacher_steps} steps)")
            teacher_cfg = {
                "model_path": args.teacher_model,
                "num_steps": args.teacher_steps,
                "guidance_scale": args.teacher_guidance,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
            }
            parallel_generate(gpu_ids, categories, out_root, "teacher", teacher_cfg)
            free_cuda()
            print("[teacher] Done.")

        if run_student and not args.skip_student:
            print(f"[student] SD3 (finetune={args.student_model or 'base only'}, {args.student_steps} steps)")
            student_cfg = {
                "student_model": args.student_model,
                "sd3_base_model": args.sd3_base_model,
                "taesd3_model": args.taesd3_model,
                "sana_scheduler_repo": args.sana_scheduler_repo,
                "flow_shift": args.flow_shift,
                "num_steps": args.student_steps,
                "guidance_scale": args.student_guidance,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
            }
            parallel_generate(gpu_ids, categories, out_root, "student", student_cfg)
            free_cuda()
            print("[student] Done.")
        elif not run_student:
            print("[student] Skipped (set STUDENT_MODEL when finetune checkpoint is ready).")

    if not args.skip_score:
        if not args.hps_checkpoint.is_file():
            raise FileNotFoundError(f"HPS checkpoint not found: {args.hps_checkpoint}")

        scores_dir = out_root / "scores"
        scores_dir.mkdir(parents=True, exist_ok=True)
        summary: Dict[str, object] = {"hps_version": "v2.1", "checkpoint": str(args.hps_checkpoint), "models": {}}

        teacher_img = out_root / "images" / "teacher"
        if teacher_img.is_dir():
            print("\n[HPS v2.1] Teacher")
            summary["models"]["teacher"] = parallel_hps_score(
                gpu_ids, args.prompts_dir, teacher_img, args.hps_checkpoint, args.hps_batch_size
            )

        student_img = out_root / "images" / "student"
        if student_img.is_dir() and any(student_img.iterdir()):
            print("\n[HPS v2.1] Student")
            summary["models"]["student"] = parallel_hps_score(
                gpu_ids, args.prompts_dir, student_img, args.hps_checkpoint, args.hps_batch_size
            )
        else:
            print("[HPS] Student images missing; skip student scoring.")

        out_json = scores_dir / "hps_v2.1_summary.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[HPS] Summary written to {out_json}")

    print(f"\nAll outputs under: {out_root}")


if __name__ == "__main__":
    main()
