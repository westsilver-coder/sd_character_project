"""
Turntable rendering: 360° rotation of 3D mesh → mp4.
Rotates mesh per frame and uses trimesh save_image; imageio writes mp4.
Soft lighting, white or pastel background.
"""
from pathlib import Path

import numpy as np
import trimesh
import imageio
from imageio import get_writer


def _render_frame(
    mesh: trimesh.Trimesh,
    angle_deg: float,
    resolution: tuple = (512, 512),
    background: tuple = (255, 255, 255),
) -> np.ndarray:
    """Render one frame: rotate mesh around Y then render. Returns RGB uint8."""
    rotated = mesh.copy()
    angle_rad = np.deg2rad(angle_deg)
    R = trimesh.transformations.rotation_matrix(angle_rad, [0, 1, 0])
    rotated.apply_transform(R)
    scene = trimesh.Scene(rotated)
    try:
        png_bytes = scene.save_image(resolution=resolution, background=background)
        img = imageio.imread(png_bytes, format="PNG")
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] == 4:
            img = img[:, :, :3]
    except Exception:
        img = np.full((*resolution, 3), background, dtype=np.uint8)
    return np.asarray(img)


def render_turntable(
    mesh_path: str | Path,
    output_path: str | Path,
    *,
    duration_sec: float = 7.0,
    fps: int = 30,
    resolution: tuple = (512, 512),
    background_rgb: tuple = (255, 255, 255),
) -> str:
    """
    Render 360° turntable video of mesh. Save to output_path (e.g. renders/turntable.mp4).
    Returns path to saved mp4.
    """
    mesh_path = Path(mesh_path)
    output_path = Path(output_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found: {mesh_path}")

    mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = output_path if output_path.suffix else output_path.with_suffix(".mp4")

    n_frames = int(duration_sec * fps)
    angles = np.linspace(0, 360, n_frames, endpoint=False)

    with get_writer(str(out_file), fps=fps, codec="libx264", quality=8) as writer:
        for angle in angles:
            frame = _render_frame(
                mesh,
                angle,
                resolution=resolution,
                background=background_rgb,
            )
            writer.append_data(frame)

    return str(out_file.resolve())


if __name__ == "__main__":
    import sys
    mesh = sys.argv[1] if len(sys.argv) > 1 else "generated_3d/character_sd.obj"
    out = sys.argv[2] if len(sys.argv) > 2 else "renders/turntable.mp4"
    path = render_turntable(mesh, out)
    print(f"Saved: {path}")
