"""
2D SD-style character image generation using Stable Diffusion (SDXL).
Input: text prompt (from prompt_generator).
Output: PNG in generated_2d/
"""
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from PIL import Image


# Default SDXL model (can be overridden for local/finetuned)
DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEFAULT_VAE_ID = "madebyollin/sdxl-vae-fp16-fix"

# Generation defaults
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 768  # portrait-ish for full body
DEFAULT_STEPS = 30
DEFAULT_CFG = 7.5
DEFAULT_SEED = 42


def get_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    vae_id: Optional[str] = DEFAULT_VAE_ID,
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load SDXL pipeline. Uses fp16 on GPU when available."""
    device = device or get_device()
    dtype = torch_dtype or (torch.float16 if device == "cuda" else torch.float32)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    )
    if vae_id:
        pipe.vae = AutoencoderKL.from_pretrained(vae_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def generate(
    prompt: str,
    output_path: str | Path,
    *,
    negative_prompt: str = (
        "ugly, blurry, low quality, distorted, deformed, "
        "multiple characters, cropped, bad anatomy, text, watermark"
    ),
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_CFG,
    seed: Optional[int] = DEFAULT_SEED,
    pipeline=None,
) -> str:
    """
    Generate one SD-style 2D character image and save to output_path.
    Returns absolute path to saved PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = get_device()
    if pipeline is None:
        pipeline = load_pipeline(device=device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    image: Image.Image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]

    out_file = output_path if output_path.suffix else output_path.with_suffix(".png")
    image.save(str(out_file), "PNG")
    return str(out_file.resolve())


def generate_from_prompt_file(
    prompt_path: str | Path,
    output_dir: str | Path = "generated_2d",
    output_name: str = "character.png",
    **kwargs,
) -> str:
    """
    Read prompt from text file (one prompt per file) and generate.
    Returns path to saved PNG.
    """
    prompt_path = Path(prompt_path)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    out = Path(output_dir) / output_name
    return generate(prompt, out, **kwargs)


if __name__ == "__main__":
    import sys
    from core.prompt_generator import generate_prompt

    # Example: generate from features.json
    features_path = sys.argv[1] if len(sys.argv) > 1 else "features/features.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "generated_2d"
    prompt = generate_prompt(features_path)
    out_path = generate(prompt, Path(out_dir) / "character.png")
    print(f"Saved: {out_path}")
