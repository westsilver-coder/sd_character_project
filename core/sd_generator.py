"""
2D SD 스타일 캐릭터 생성 (SD 1.5).
입력 이미지를 init으로 쓰는 Img2Img로 닮은 캐릭터 생성. init_image 없으면 txt2img.
RTX 2080Ti (11GB) 안정: 512x512, attention_slicing.
"""
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image

DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"
DEFAULT_HEIGHT = 512
DEFAULT_WIDTH = 512
DEFAULT_STEPS = 25
DEFAULT_CFG = 7.5
DEFAULT_SEED = 42
# Img2Img: 입력 이미지 보존 정도. 낮을수록 닮음 유지, 높을수록 프롬프트 반영 큼
DEFAULT_STRENGTH = 0.55


def get_device() -> str:
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
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


def load_img2img_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    device: Optional[str] = None,
) -> StableDiffusionImg2ImgPipeline:
    device = device or get_device()
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    return pipe


def _preprocess_init_image(
    image_path: str | Path,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> Image.Image:
    """입력 사진을 512x512로 맞춤. 중앙 기준 크롭 후 리사이즈해서 얼굴이 들어가게."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Init image not found: {path}")
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if w == width and h == height:
        return img
    # 비율 유지한 채 짧은 쪽을 height/width에 맞추고, 중앙 크롭
    scale = min(width / w, height / h)
    nw, nh = int(w * scale), int(h * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    # 중앙에서 width x height 크롭
    left = (nw - width) // 2
    top = (nh - height) // 2
    left = max(0, min(left, nw - width))
    top = max(0, min(top, nh - height))
    img = img.crop((left, top, left + width, top + height))
    return img


def generate(
    prompt: str,
    output_path: str | Path,
    *,
    init_image: str | Path | None = None,
    strength: float = DEFAULT_STRENGTH,
    negative_prompt: str | None = None,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_CFG,
    seed: Optional[int] = DEFAULT_SEED,
    pipeline=None,
) -> str:
    """
    캐릭터 이미지 생성.
    init_image가 있으면 Img2Img로 입력 이미지의 얼굴·구도를 반영해 닮은 캐릭터 생성.
    없으면 txt2img만 사용.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if negative_prompt is None:
        from core.prompt_generator import get_negative_prompt
        negative_prompt = get_negative_prompt()

    device = get_device()
    use_img2img = init_image is not None

    if use_img2img:
        init_pil = _preprocess_init_image(init_image, width=width, height=height)
        if pipeline is None:
            pipeline = load_img2img_pipeline(device=device)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        image = pipeline(
            prompt=prompt,
            image=init_pil,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
    else:
        if pipeline is None:
            pipeline = load_pipeline(device=device)
        generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
        image = pipeline(
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
