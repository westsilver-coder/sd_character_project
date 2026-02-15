"""
2D SD-style character image generation using Stable Diffusion 1.5.
Input: text prompt (from prompt_generator).
Output: PNG in generated_2d/
RTX 2080Ti (11GB) 안정 실행: 512x512, attention_slicing.
"""
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_STEPS = 25
DEFAULT_CFG = 7.5
DEFAULT_SEED = 42


def get_device() -> str:
    """Return 'cuda' if available else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
) -> StableDiffusionPipeline:
    device = device or get_device()

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    )

    pipe = pipe.to(device)

    # RTX 2080Ti 안정 세팅
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    return pipe



def generate(
    prompt: str,
    output_path: str | Path,
    *,
    negative_prompt: str | None = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_CFG,
    seed: Optional[int] = DEFAULT_SEED,
    pipeline: Optional[StableDiffusionPipeline] = None,
) -> str:
    """
    Generate one SD-style 2D character image and save to output_path.
    Returns absolute path to saved PNG.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if negative_prompt is None:
        from core.prompt_generator import get_negative_prompt
        negative_prompt = get_negative_prompt()

    device = get_device()
    if pipeline is None:
        pipeline = load_pipeline(device=device)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    image: Image.Image = pipeline(
        prompt,
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

    features_path = sys.argv[1] if len(sys.argv) > 1 else "features/features.json"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "generated_2d"
    prompt = generate_prompt(features_path)
    out_path = generate(prompt, Path(out_dir) / "character.png")
    print(f"Saved: {out_path}")
