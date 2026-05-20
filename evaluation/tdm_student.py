"""TDM student inference aligned with train_tdm_demo online sampling."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from PIL import Image

TDM_ROOT = Path(__file__).resolve().parent.parent
if str(TDM_ROOT) not in sys.path:
    sys.path.insert(0, str(TDM_ROOT))

from train_tdm_demo import generate_new  # noqa: E402

TAESD_VAE_ID = "madebyollin/taesd"


def compute_student_timesteps(total_steps: int, num_inference_steps: int) -> Tuple[List[int], List[int]]:
    stride = total_steps // num_inference_steps
    timesteps = [total_steps - 1 - i * stride for i in range(num_inference_steps)]
    next_timesteps = timesteps[1:] + [max(timesteps[-1] - stride, 0)]
    return timesteps, next_timesteps


def encode_prompt_training_style(tokenizer, text_encoder, prompt: str, device: str):
    """Same path as train_tdm_demo dataloader + text_encoder forward."""
    text_input = tokenizer(
        [prompt],
        max_length=120,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_input.input_ids.to(device)
    attention_mask = text_input.attention_mask.to(device).to(torch.float16)
    prompt_embeds = text_encoder(input_ids, return_dict=False, attention_mask=attention_mask)[0]
    return prompt_embeds, attention_mask


def decode_latents_training_style(vae, latents: torch.Tensor) -> Image.Image:
    """Same decode as train_tdm_demo 4step.jpg logging."""
    decoded = vae.decode(latents.to(vae.dtype) / vae.config.scaling_factor, return_dict=False)[0]
    decoded = decoded.clamp(-1, 1) * 0.5 + 0.5
    from torchvision.transforms.functional import to_pil_image

    return to_pil_image(decoded[0].cpu().float())


def format_student_schedule_banner(
    *,
    checkpoint: str,
    base_model: str,
    scheduler_name: str,
    vae_name: str,
    total_steps: int,
    num_inference_steps: int,
    cfg: Optional[float],
    latent_size: int,
    seed: int,
) -> str:
    timesteps, next_ts = compute_student_timesteps(total_steps, num_inference_steps)
    lines = [
        "",
        "=" * 72,
        "  TDM STUDENT INFERENCE CONFIG (training-aligned)",
        "=" * 72,
        f"  checkpoint       : {checkpoint}",
        f"  base (T5/sched)  : {base_model}",
        f"  VAE decode       : {vae_name}",
        f"  noise scheduler  : {scheduler_name}",
        f"  latent shape     : [1, 4, {latent_size}, {latent_size}]",
        f"  total_steps      : {total_steps}",
        f"  num_infer_steps  : {num_inference_steps}",
        f"  CFG in generate_new: {cfg if cfg is not None else 'OFF (same as training 4step.jpg)'}",
        f"  text encoding    : T5Tokenizer + T5Encoder (not pipeline.encode_prompt)",
        f"  seed             : {seed}",
        "-" * 72,
        "  Timestep anchors (T at each forward):",
    ]
    for i, t in enumerate(timesteps):
        lines.append(f"    step {i + 1}/{num_inference_steps}:  T = {t:4d}  ->  re-noise to T = {next_ts[i]}")
    lines.append("=" * 72)
    lines.append("")
    return "\n".join(lines)


def print_training_vs_inference_diff(*, student_guidance: float, vae_name: str) -> None:
    lines = [
        "",
        "-" * 72,
        "  TRAINING online sample (4step.jpg) vs current STUDENT inference",
        "-" * 72,
        "  Item                    | Training (train_tdm_demo)     | Student inference",
        "  ------------------------|------------------------------|---------------------------",
        "  generate_new CFG        | OFF (no add_cfg)              | "
        + ("OFF" if student_guidance <= 1.0 else f"ON cfg={student_guidance}"),
        "  VAE decode              | AutoencoderTiny (taesd)       | " + vae_name,
        "  Text encode             | tokenizer + text_encoder      | tokenizer + text_encoder",
        "  added_cond_kwargs       | resolution/aspect_ratio=None  | resolution/aspect_ratio=None",
        "  steps / total_steps     | 4 / 900                       | 4 / 900",
        "  Teacher (PixArt pipe)   | N/A                           | 25-step, cfg=4.5, full VAE",
        "-" * 72,
        "",
    ]
    print("\n".join(lines), flush=True)


def build_tdm_student(
    base_model: str,
    student_checkpoint: str,
    device: str = "cuda",
):
    from diffusers import AutoencoderTiny, DDPMScheduler, PixArtAlphaPipeline
    from safetensors.torch import load_file

    ckpt = Path(student_checkpoint)
    unet_dir = ckpt / "unet"
    if not unet_dir.is_dir():
        raise FileNotFoundError(f"Expected student unet/ under {ckpt}")

    dtype = torch.float16
    pipe = PixArtAlphaPipeline.from_pretrained(base_model, torch_dtype=dtype)
    state = load_file(unet_dir / "diffusion_pytorch_model.safetensors", device="cpu")
    missing, unexpected = pipe.transformer.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected keys loading student: {unexpected[:5]}...")
    pipe = pipe.to(device)

    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    vae = AutoencoderTiny.from_pretrained(TAESD_VAE_ID).to(device=device, dtype=dtype)

    if hasattr(pipe.transformer, "enable_xformers_memory_efficient_attention"):
        try:
            pipe.transformer.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    sample_size = pipe.transformer.config.sample_size
    return pipe, scheduler, vae, sample_size


@torch.no_grad()
def generate_student_image(
    pipe,
    scheduler,
    vae,
    sample_size: int,
    prompt: str,
    *,
    total_steps: int = 900,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int = 8888,
    device: str = "cuda",
) -> Image.Image:
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        1, 4, sample_size, sample_size, generator=generator, device=device, dtype=torch.float16
    )

    prompt_embeds, prompt_attention_mask = encode_prompt_training_style(
        pipe.tokenizer, pipe.text_encoder, prompt, device
    )

    add_cfg = None
    if guidance_scale > 1.0:
        uncond_embeds, uncond_mask = encode_prompt_training_style(
            pipe.tokenizer, pipe.text_encoder, "", device
        )
        add_cfg = {
            "uncond_attention_mask": uncond_mask,
            "uncond_prompt_embeds": uncond_embeds,
            "cfg": guidance_scale,
        }

    out_latents = generate_new(
        pipe.transformer,
        scheduler,
        noise,
        noise,
        prompt_embeds,
        prompt_attention_mask,
        steps=num_inference_steps,
        return_mid=False,
        total_steps=total_steps,
        add_cfg=add_cfg,
    )
    return decode_latents_training_style(vae, out_latents)


def generate_images_tdm_student(
    pipe,
    scheduler,
    vae,
    sample_size: int,
    categories,
    out_root: Path,
    *,
    total_steps: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    image_path_fn,
) -> None:
    from tqdm import tqdm

    for cat in categories:
        cat_dir = out_root / "images" / "student" / cat.folder
        cat_dir.mkdir(parents=True, exist_ok=True)
        for idx, prompt in enumerate(tqdm(cat.prompts, desc=f"student/{cat.folder}")):
            save_path = image_path_fn(out_root, "student", cat.folder, idx)
            if save_path.exists():
                continue
            img = generate_student_image(
                pipe,
                scheduler,
                vae,
                sample_size,
                prompt,
                total_steps=total_steps,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed + idx,
            )
            img.save(save_path, quality=95)
