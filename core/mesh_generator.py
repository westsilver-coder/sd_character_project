"""
2D → 3D mesh generation.
- With TripoSR: set TRIPOSR_SCRIPT or use run_triposr_script() for real image→mesh.
- Without: exports a placeholder mesh so the rest of the pipeline runs.
Output: .obj (or .glb) in generated_3d/
"""
from pathlib import Path
from typing import Optional
import subprocess
import sys

import trimesh


def image_to_mesh(
    image_path: str | Path,
    output_path: str | Path,
    *,
    use_triposr_script: Optional[str] = None,
) -> str:
    """
    Convert 2D character image to 3D mesh.
    - If use_triposr_script is set (path to TripoSR run.py), runs that and copies result.
    - Otherwise exports a placeholder mesh so pipeline can run.
    Returns path to saved mesh file.
    """
    image_path = Path(image_path)
    output_path = Path(output_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    out_dir = output_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_path if output_path.suffix else output_path.with_suffix(".obj")

    if use_triposr_script:
        script = Path(use_triposr_script)
        if script.exists():
            triposr_out = run_triposr_script(script, image_path, out_dir)
            if triposr_out and Path(triposr_out).exists():
                Path(triposr_out).rename(out_file)
                return str(out_file.resolve())

    return _export_placeholder_mesh(out_file)


def run_triposr_script(
    script_path: Path,
    image_path: Path,
    output_dir: Path,
) -> Optional[str]:
    """
    Run TripoSR repo's run.py: python run.py <image> --output-dir <dir>.
    Returns path to generated mesh in output_dir, or None on failure.
    """
    try:
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                str(image_path),
                "--output-dir", str(output_dir),
            ],
            check=True,
            capture_output=True,
            timeout=120,
        )
        # TripoSR typically outputs .obj or .glb in output_dir
        for ext in (".obj", ".glb"):
            p = output_dir / (image_path.stem + ext)
            if p.exists():
                return str(p)
        for f in output_dir.iterdir():
            if f.suffix in (".obj", ".glb"):
                return str(f)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _export_placeholder_mesh(output_path: Path) -> str:
    """Export a simple humanoid-ish shape as placeholder when no 2D→3D model is used."""
    # Two boxes: head (larger), body (smaller) to suggest SD proportion
    head = trimesh.creation.box(extents=[0.5, 0.5, 0.5])
    head.apply_translation([0, 0, 0.4])
    body = trimesh.creation.box(extents=[0.4, 0.35, 0.5])
    body.apply_translation([0, 0, -0.15])
    mesh = head + body
    mesh.vertices -= mesh.vertices.mean(axis=0)
    out = output_path if output_path.suffix else output_path.with_suffix(".obj")
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))
    return str(out.resolve())


if __name__ == "__main__":
    import os
    img = sys.argv[1] if len(sys.argv) > 1 else "generated_2d/character.png"
    out = sys.argv[2] if len(sys.argv) > 2 else "generated_3d/character.obj"
    triposr = os.environ.get("TRIPOSR_SCRIPT")
    path = image_to_mesh(img, out, use_triposr_script=triposr)
    print(f"Saved mesh: {path}")
