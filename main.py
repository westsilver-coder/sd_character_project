"""
사진 기반 SD 스타일 3D 캐릭터 자동 생성 파이프라인

입력: input/user_photo.png (또는 지정 경로)
출력:
  - features/features.json
  - generated_2d/character.png
  - generated_3d/character.obj → character_sd.obj
  - renders/turntable.mp4
"""
import argparse
import os
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent


def run_pipeline(
    input_photo: str | Path,
    output_dir_2d: str | Path = None,
    output_dir_3d: str | Path = None,
    output_dir_renders: str | Path = None,
    features_dir: str | Path = None,
    skip_sd: bool = False,
    skip_3d: bool = False,
    skip_deform: bool = False,
    skip_render: bool = False,
    triposr_script: str | None = None,
    seed: int = 42,
) -> dict:
    """
    Run full pipeline: photo → features → prompt → 2D → 3D → SD deform → turntable.
    Returns dict with paths of generated files.
    """
    from core.feature_extractor import extract_and_save
    from core.prompt_generator import generate_prompt
    from core.sd_generator import generate
    from core.mesh_generator import image_to_mesh
    from core.deform_sd import load_deform_export
    from core.renderer import render_turntable

    input_photo = Path(input_photo)
    output_dir_2d = Path(output_dir_2d or ROOT / "generated_2d")
    output_dir_3d = Path(output_dir_3d or ROOT / "generated_3d")
    output_dir_renders = Path(output_dir_renders or ROOT / "renders")
    features_dir = Path(features_dir or ROOT / "features")

    result = {}

    # 1) Feature extraction
    features_path = extract_and_save(str(input_photo), features_dir)
    result["features"] = features_path

    # 2) Prompt generation
    prompt = generate_prompt(features_path)
    result["prompt"] = prompt

    # 3) 2D SD character
    if not skip_sd:
        img_2d = output_dir_2d / "character.png"
        generate(prompt, img_2d, seed=seed)
        result["image_2d"] = str(img_2d.resolve())
    else:
        img_2d = output_dir_2d / "character.png"
        if not img_2d.exists():
            raise FileNotFoundError(f"Skip SD but no existing image: {img_2d}")
        result["image_2d"] = str(img_2d.resolve())

    # 4) 2D → 3D mesh
    mesh_raw = output_dir_3d / "character.obj"
    if not skip_3d:
        image_to_mesh(result["image_2d"], mesh_raw, use_triposr_script=triposr_script)
    if not mesh_raw.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_raw}")
    result["mesh_raw"] = str(mesh_raw.resolve())

    # 5) SD proportion deformation
    mesh_sd = output_dir_3d / "character_sd.obj"
    if not skip_deform:
        load_deform_export(result["mesh_raw"], mesh_sd)
    else:
        mesh_sd = mesh_raw  # use raw mesh as final
    result["mesh_sd"] = str(Path(mesh_sd).resolve())

    # 6) Turntable video
    if not skip_render:
        video_path = output_dir_renders / "turntable.mp4"
        render_turntable(result["mesh_sd"], video_path)
        result["video"] = str(video_path.resolve())
    else:
        result["video"] = None

    return result


def main():
    parser = argparse.ArgumentParser(
        description="사진 기반 SD 스타일 3D 캐릭터 자동 생성"
    )
    parser.add_argument(
        "input_photo",
        nargs="?",
        default=ROOT / "input" / "user_photo.png",
        help="입력 인물 사진 경로",
    )
    parser.add_argument("--no-sd", action="store_true", help="2D 생성 생략 (기존 이미지 사용)")
    parser.add_argument("--no-3d", action="store_true", help="2D→3D 변환 생략")
    parser.add_argument("--no-deform", action="store_true", help="SD 비율 변형 생략")
    parser.add_argument("--no-render", action="store_true", help="터닝테이블 영상 생성 생략")
    parser.add_argument("--triposr", type=str, default=os.environ.get("TRIPOSR_SCRIPT"), help="TripoSR run.py 경로")
    parser.add_argument("--seed", type=int, default=42, help="SD 샘플링 시드")
    args = parser.parse_args()

    result = run_pipeline(
        input_photo=args.input_photo,
        skip_sd=args.no_sd,
        skip_3d=args.no_3d,
        skip_deform=args.no_deform,
        skip_render=args.no_render,
        triposr_script=args.triposr,
        seed=args.seed,
    )
    print("Pipeline finished.")
    for k, v in result.items():
        if v:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
