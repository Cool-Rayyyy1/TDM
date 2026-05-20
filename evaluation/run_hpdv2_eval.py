#!/usr/bin/env python
# coding=utf-8
"""HPDv2 benchmark: teacher (PixArt 25-step) vs student (TDM 4-step) + HPS v2.1 scoring."""

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
        help="TDM student checkpoint dir (contains unet/). Empty = skip student.",
    )
    p.add_argument(
        "--student_base_model",
        type=str,
        default="",
        help="PixArt base for student VAE/T5 (default: same as --teacher_model).",
    )
    p.add_argument("--teacher_steps", type=int, default=25)
    p.add_argument("--student_steps", type=int, default=4)
    p.add_argument(
        "--student_total_steps",
        type=int,
        default=900,
        help="Training DDPM total_steps; 4-step anchors e.g. 899,674,449,224 when 900.",
    )
    p.add_argument("--teacher_guidance", type=float, default=4.5)
    p.add_argument(
        "--student_guidance",
        type=float,
        default=1.0,
        help="CFG in generate_new. Training 4step.jpg uses 1.0 (off). Teacher uses --teacher_guidance.",
    )
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--flow_shift", type=float, default=6.0)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=8888)
    p.add_argument("--max_prompts_per_category", type=int, default=None, help="Debug cap per category.")
    p.add_argument("--skip_generate", action="store_true", help="Skip all image generation.")
    p.add_argument(
        "--score_only",
        action="store_true",
        help="Only run HPS scoring (same as --skip_generate).",
    )
    p.add_argument(
        "--force_generate",
        action="store_true",
        help="Regenerate even when images/ folder is already complete.",
    )
    p.add_argument("--skip_teacher", action="store_true")
    p.add_argument("--skip_student", action="store_true")
    p.add_argument("--skip_score", action="store_true")
    p.add_argument("--run_student", action="store_true", help="Reserved; student requires --student_model checkpoint.")
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


def print_banner(title: str, lines: List[str]) -> None:
    print("\n" + "-" * 72, flush=True)
    print(title, flush=True)
    for line in lines:
        print(line, flush=True)
    print("-" * 72 + "\n", flush=True)


def check_model_images_complete(
    out_root: Path,
    model_name: str,
    categories: List[CategoryData],
) -> Tuple[bool, Dict[str, Dict[str, int]]]:
    """Return (all_complete, {folder: {expected, found, missing_indices_sample}})."""
    report: Dict[str, Dict[str, int]] = {}
    all_ok = True
    for cat in categories:
        expected = len(cat.prompts)
        found = 0
        missing = 0
        for idx in range(expected):
            if image_path(out_root, model_name, cat.folder, idx).is_file():
                found += 1
            else:
                missing += 1
        report[cat.folder] = {"expected": expected, "found": found, "missing": missing}
        if missing > 0:
            all_ok = False
    return all_ok, report


def format_image_check_report(model_name: str, report: Dict[str, Dict[str, int]], complete: bool) -> List[str]:
    lines = [f"Model: {model_name}  ->  {'COMPLETE' if complete else 'INCOMPLETE'}"]
    for folder, stats in report.items():
        lines.append(
            f"  {folder:<14} {stats['found']}/{stats['expected']} images"
            + ("" if stats["missing"] == 0 else f"  (missing {stats['missing']})")
        )
    return lines


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
            gen_cfg.get("negative_prompt", ""),
        )
    else:
        from tdm_student import build_tdm_student, generate_images_tdm_student

        pipe, scheduler, vae, sample_size = build_tdm_student(
            gen_cfg["student_base_model"],
            gen_cfg["student_model"],
            "cuda",
        )
        generate_images_tdm_student(
            pipe,
            scheduler,
            vae,
            sample_size,
            cats,
            out,
            total_steps=gen_cfg["student_total_steps"],
            num_inference_steps=gen_cfg["num_steps"],
            guidance_scale=gen_cfg["guidance_scale"],
            seed=gen_cfg["seed"],
            image_path_fn=image_path,
        )
        del pipe, scheduler, vae
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


def write_manifest(
    out_root: Path,
    categories: List[CategoryData],
    args: argparse.Namespace,
    compute_student_timesteps_fn=None,
) -> None:
    if compute_student_timesteps_fn is None:
        from tdm_student import compute_student_timesteps as compute_student_timesteps_fn
    manifest = {
        "created_at": datetime.now().isoformat(),
        "num_gpus": args.num_gpus,
        "gpu_ids": resolve_gpu_ids(args.num_gpus),
        "teacher_model": args.teacher_model,
        "student_model": args.student_model or None,
        "student_base_model": args.student_base_model or args.teacher_model,
        "student_total_steps": args.student_total_steps,
        "teacher_steps": args.teacher_steps,
        "student_steps": args.student_steps,
        "teacher_guidance": args.teacher_guidance,
        "student_guidance": args.student_guidance,
        "height": args.height,
        "width": args.width,
        "student_timesteps": list(
            compute_student_timesteps_fn(args.student_total_steps, args.student_steps)[0]
        ),
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
        print(f"[output] OUTPUT_DIR not set -> new run: {args.output_dir}")
    else:
        print(f"[output] Using OUTPUT_DIR={args.output_dir}")
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    categories = load_categories(args.prompts_dir, args.max_prompts_per_category)
    write_manifest(out_root, categories, args)

    if args.score_only:
        args.skip_generate = True

    run_student = bool(args.student_model)
    if args.run_student and not args.student_model:
        print("[student] --run_student ignored; set --student_model to a TDM checkpoint (unet/).")

    student_base = args.student_base_model or args.teacher_model
    gpu_ids = resolve_gpu_ids(args.num_gpus)
    print(f"[GPUs] Using {len(gpu_ids)} device(s): {gpu_ids}")

    teacher_complete, teacher_report = check_model_images_complete(out_root, "teacher", categories)
    student_complete = False
    student_report: Dict[str, Dict[str, int]] = {}
    if run_student:
        student_complete, student_report = check_model_images_complete(out_root, "student", categories)

    print_banner(
        "[RESUME CHECK] Image folders under " + str(out_root),
        format_image_check_report("teacher", teacher_report, teacher_complete)
        + (format_image_check_report("student", student_report, student_complete) if run_student else []),
    )

    if teacher_complete and (not run_student or student_complete):
        print_banner(
            "---------  ALL GENERATION COMPLETE — SKIP TO HPS SCORING  ---------",
            [
                f"output_dir: {out_root}",
                "Use --force_generate to regenerate images anyway.",
            ],
        )
        if not args.force_generate:
            args.skip_generate = True
    elif teacher_complete and not args.force_generate:
        print_banner(
            "---------  TEACHER COMPLETE — SKIP TEACHER GENERATION  ---------",
            [f"output_dir: {out_root}/images/teacher"],
        )

    if run_student and student_complete and not args.force_generate:
        print_banner(
            "---------  STUDENT COMPLETE — SKIP STUDENT GENERATION  ---------",
            [f"output_dir: {out_root}/images/student"],
        )

    if not args.skip_generate:
        if not args.skip_teacher and not (teacher_complete and not args.force_generate):
            print(f"[teacher] PixArt from {args.teacher_model} ({args.teacher_steps} steps)")
            teacher_cfg = {
                "model_path": args.teacher_model,
                "num_steps": args.teacher_steps,
                "guidance_scale": args.teacher_guidance,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
                "negative_prompt": args.negative_prompt,
            }
            parallel_generate(gpu_ids, categories, out_root, "teacher", teacher_cfg)
            free_cuda()
            teacher_complete, teacher_report = check_model_images_complete(out_root, "teacher", categories)
            print_banner(
                "---------  TEACHER GENERATION FINISHED  ---------",
                format_image_check_report("teacher", teacher_report, teacher_complete),
            )

        if run_student and not args.skip_student and not (student_complete and not args.force_generate):
            from tdm_student import (
                TAESD_VAE_ID,
                format_student_schedule_banner,
                print_training_vs_inference_diff,
            )
            from diffusers import DDPMScheduler

            sched_probe = DDPMScheduler.from_pretrained(student_base, subfolder="scheduler")
            print_training_vs_inference_diff(
                student_guidance=args.student_guidance,
                vae_name=f"AutoencoderTiny ({TAESD_VAE_ID})",
            )
            print(
                format_student_schedule_banner(
                    checkpoint=args.student_model,
                    base_model=student_base,
                    scheduler_name=sched_probe.__class__.__name__,
                    vae_name=f"AutoencoderTiny ({TAESD_VAE_ID})",
                    total_steps=args.student_total_steps,
                    num_inference_steps=args.student_steps,
                    cfg=args.student_guidance if args.student_guidance > 1.0 else None,
                    latent_size=64,
                    seed=args.seed,
                ),
                flush=True,
            )
            student_cfg = {
                "student_model": args.student_model,
                "student_base_model": student_base,
                "student_total_steps": args.student_total_steps,
                "num_steps": args.student_steps,
                "guidance_scale": args.student_guidance,
                "height": args.height,
                "width": args.width,
                "seed": args.seed,
            }
            parallel_generate(gpu_ids, categories, out_root, "student", student_cfg)
            free_cuda()
            student_complete, student_report = check_model_images_complete(out_root, "student", categories)
            print_banner(
                "---------  STUDENT GENERATION FINISHED  ---------",
                format_image_check_report("student", student_report, student_complete),
            )
        elif not run_student:
            print("[student] Skipped (set STUDENT_MODEL when finetune checkpoint is ready).")
    elif args.skip_generate:
        print("[generate] Skipped (--skip_generate / resume / --score_only).")

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
