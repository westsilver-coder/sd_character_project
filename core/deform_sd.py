"""
SD proportion deformation: scale head 1.5~1.7x, body 0.6~0.8x.
Splits mesh by Y (height) using bounding box, then applies vertex scaling.
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


# SD style proportions
DEFAULT_HEAD_SCALE = 1.6
DEFAULT_BODY_SCALE = 0.7
# Fraction of bbox height used as split between head and body (from top)
DEFAULT_HEAD_RATIO = 0.45


def _bbox_extents(mesh: trimesh.Trimesh) -> Tuple[np.ndarray, np.ndarray]:
    """Return (min_xyz, max_xyz) of mesh bounding box."""
    return mesh.bounds[0].copy(), mesh.bounds[1].copy()


def _split_head_body(
    mesh: trimesh.Trimesh,
    head_ratio: float = DEFAULT_HEAD_RATIO,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split vertices into head (above split Y) and body (below).
    Returns (head_mask, body_mask) boolean arrays of length n_vertices.
    """
    mn, mx = _bbox_extents(mesh)
    verts = mesh.vertices
    y_min, y_max = mn[1], mx[1]
    split_y = y_max - head_ratio * (y_max - y_min)
    head_mask = verts[:, 1] >= split_y
    body_mask = ~head_mask
    return head_mask, body_mask


def _scale_vertices_around_center(
    vertices: np.ndarray,
    mask: np.ndarray,
    scale: float,
    center: np.ndarray,
) -> np.ndarray:
    """Scale vertices where mask is True around center. Returns new vertex array."""
    out = vertices.copy()
    if not np.any(mask):
        return out
    out[mask] = center + scale * (out[mask] - center)
    return out


def deform_sd_proportions(
    mesh: trimesh.Trimesh,
    head_scale: float = DEFAULT_HEAD_SCALE,
    body_scale: float = DEFAULT_BODY_SCALE,
    head_ratio: float = DEFAULT_HEAD_RATIO,
) -> trimesh.Trimesh:
    """
    Apply SD-style proportion: scale head up, body down.
    Uses bounding box to split by height; head = top head_ratio of bbox.
    Returns a new mesh (copy).
    """
    mesh = mesh.copy()
    head_mask, body_mask = _split_head_body(mesh, head_ratio)
    mn, mx = _bbox_extents(mesh)
    center = (mn + mx) / 2
    # Head center: top part
    head_verts = mesh.vertices[head_mask]
    head_center = head_verts.mean(axis=0) if np.any(head_mask) else center
    body_verts = mesh.vertices[body_mask]
    body_center = body_verts.mean(axis=0) if np.any(body_mask) else center

    verts = mesh.vertices.copy()
    verts = _scale_vertices_around_center(verts, head_mask, head_scale, head_center)
    verts = _scale_vertices_around_center(verts, body_mask, body_scale, body_center)
    mesh.vertices = verts
    return mesh


def load_deform_export(
    input_path: str | Path,
    output_path: str | Path,
    head_scale: float = DEFAULT_HEAD_SCALE,
    body_scale: float = DEFAULT_BODY_SCALE,
    head_ratio: float = DEFAULT_HEAD_RATIO,
) -> str:
    """
    Load mesh from input_path, apply SD deformation, save to output_path.
    Returns path to saved file.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Mesh not found: {input_path}")

    mesh = trimesh.load(str(input_path), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = trimesh.util.concatenate(mesh)

    deformed = deform_sd_proportions(mesh, head_scale, body_scale, head_ratio)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = output_path if output_path.suffix else output_path.with_suffix(".obj")
    deformed.export(str(out))
    return str(out.resolve())


if __name__ == "__main__":
    import sys
    inp = sys.argv[1] if len(sys.argv) > 1 else "generated_3d/character.obj"
    out = sys.argv[2] if len(sys.argv) > 2 else "generated_3d/character_sd.obj"
    path = load_deform_export(inp, out)
    print(f"Saved: {path}")
