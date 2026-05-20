"""Offline HPS v2.1 scoring helpers (fixes missing bpe asset in pip hpsv2)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import hpsv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_HPS_CKPT = EVAL_ROOT / "hpsv2" / "weights" / "HPS_v2.1_compressed.pt"
BPE_ASSET = EVAL_ROOT / "hpsv2" / "assets" / "bpe_simple_vocab_16e6.txt.gz"


def ensure_hpsv2_bpe() -> None:
    """Copy bundled BPE into hpsv2.open_clip if the pip package is missing it."""
    target = Path(hpsv2.__file__).resolve().parent / "src/open_clip/bpe_simple_vocab_16e6.txt.gz"
    if target.is_file():
        return
    if not BPE_ASSET.is_file():
        raise FileNotFoundError(
            f"Missing BPE vocab at {BPE_ASSET}. "
            "Download from open_clip or run: curl -L -o {BPE_ASSET} "
            "https://github.com/mlfoundations/open_clip/raw/main/src/open_clip/bpe_simple_vocab_16e6.txt.gz"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BPE_ASSET, target)


def _score_categories_subset(
    prompts_dir: Path,
    images_dir: Path,
    checkpoint: Path,
    categories: List[Tuple[str, str, str]],
    batch_size: int,
) -> Dict[str, object]:
    """
    Score images laid out as images_dir/<category>/{idx:05d}.jpg.

    categories: list of (json_name, folder_name, display_label)
    """
    ensure_hpsv2_bpe()
    from hpsv2.evaluation import BenchmarkDataset, collate_eval, initialize_model, model_dict
    from hpsv2.src.open_clip import get_tokenizer

    class QuietBenchmarkDataset(BenchmarkDataset):
        """Only include prompts whose images exist (no missing-image spam)."""

        def __init__(self, meta_file, image_folder, transforms, tokenizer):
            self.transforms = transforms
            self.image_folder = image_folder
            self.tokenizer = tokenizer
            self.open_image = Image.open
            import json
            import os

            with open(meta_file, "r", encoding="utf-8") as f:
                prompts = json.load(f)
            used_prompts, files = [], []
            for idx, prompt in enumerate(prompts):
                filename = os.path.join(self.image_folder, f"{idx:05d}.jpg")
                if os.path.exists(filename):
                    used_prompts.append(prompt)
                    files.append(filename)
            self.prompts = used_prompts
            self.files = files

    device = "cuda" if torch.cuda.is_available() else "cpu"
    initialize_model()
    model = model_dict["model"]
    preprocess_val = model_dict["preprocess_val"]
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    tokenizer = get_tokenizer("ViT-H-14")
    model = model.to(device)
    model.eval()

    per_style: Dict[str, Dict[str, float]] = {}
    all_scores: List[float] = []

    for json_name, folder, label in categories:
        meta_file = str(prompts_dir / json_name)
        image_folder = str(images_dir / folder)
        dataset = QuietBenchmarkDataset(meta_file, image_folder, preprocess_val, tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_eval)
        style_scores: List[float] = []
        with torch.no_grad():
            for images, texts in dataloader:
                images = images.to(device=device, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    outputs = model(images, texts)
                    logits = outputs["image_features"] @ outputs["text_features"].T * 100
                style_scores.extend(torch.diagonal(logits).cpu().tolist())
        mean_score = float(np.mean(style_scores)) if style_scores else float("nan")
        per_style[label] = {"mean": mean_score, "std": float(np.std(style_scores)), "count": len(style_scores)}
        all_scores.extend(style_scores)
        print(f"  {label:<16} {mean_score:.4f}  (n={len(style_scores)})")

    return per_style, all_scores


def score_benchmark_folder(
    prompts_dir: Path,
    images_dir: Path,
    checkpoint: Path,
    categories: List[Tuple[str, str, str]],
    batch_size: int = 20,
) -> Dict[str, object]:
    """Score all categories; returns summary with per_category + average."""
    per_style, all_scores = _score_categories_subset(
        prompts_dir, images_dir, checkpoint, categories, batch_size
    )
    for label, stats in per_style.items():
        print(f"  {label:<16} {stats['mean']:.4f}  (n={stats['count']})")
    average = float(np.mean(all_scores)) if all_scores else float("nan")
    print(f"  {'Average':<16} {average:.4f}")
    return {"per_category": per_style, "average": average, "num_images": len(all_scores)}
