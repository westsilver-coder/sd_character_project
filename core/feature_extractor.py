"""
Identity Feature Extraction from a single face photo.
Extracts: face shape, hair color, skin tone, glasses, hair length (and optional gender hint).
Output: features.json
"""
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

# MediaPipe is optional; fallback to heuristic if not available
try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


# Face shape inference from aspect ratio (width/height of face bbox)
FACE_SHAPES = ["round", "oval", "long", "square"]
# Hair color buckets (dominant color → label)
HAIR_COLORS = [
    "black", "dark_brown", "brown", "light_brown", "blonde", "gray", "red", "other"
]
SKIN_TONES = ["very_light", "light", "medium", "tan", "dark"]
HAIR_LENGTHS = ["short", "medium", "long"]
# Clothing types (하의/원피스 등)
CLOTHING_LOWER_TYPES = ["dress", "skirt", "pants", "shorts", "top_only", "unknown"]
# 의상 색상 라벨 (디테일보다 특징적인 색 위주)
CLOTHING_COLORS = [
    "white", "black", "gray", "navy", "blue", "red", "pink", "green", "yellow",
    "orange", "brown", "beige", "purple", "mint", "lavender", "other"
]


def _load_image(path: str) -> np.ndarray:
    """Load image as BGR numpy array. EXIF 방향 적용해 세로/가로 보정."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input image not found: {path}")
    # PIL로 열어 EXIF 방향 보정 후 BGR로 변환 (cv2.imread는 EXIF 무시함)
    pil_img = Image.open(path).convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    img = np.array(pil_img)[:, :, ::-1]
    return img


def _get_face_bbox_mediapipe(image: np.ndarray) -> tuple[float, float, float, float] | None:
    """Get face bounding box using MediaPipe Face Detection."""
    if not HAS_MEDIAPIPE:
        return None
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_face = mp.solutions.face_detection
    with mp_face.FaceDetection(min_detection_confidence=0.5, model_selection=1) as detector:
        results = detector.process(rgb)
        if not results.detections:
            return None
        det = results.detections[0]
        bbox = det.location_data.relative_bounding_box
        h, w = image.shape[:2]
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        return (x, y, bw, bh)
    return None


def _get_face_bbox_heuristic(image: np.ndarray) -> tuple[int, int, int, int]:
    """Fallback: assume face in center 40% of image."""
    h, w = image.shape[:2]
    cw, ch = w // 5, h // 5
    x = w // 2 - cw
    y = h // 2 - ch * 2
    bw = 2 * cw
    bh = 2 * ch
    x = max(0, min(x, w - bw))
    y = max(0, min(y, h - bh))
    return (x, y, bw, bh)


def _infer_face_shape(bbox: tuple, image_shape: tuple) -> str:
    """Infer face shape from bounding box aspect ratio."""
    _, _, bw, bh = bbox
    if bh <= 0:
        return "oval"
    ratio = bw / bh
    if ratio > 1.05:
        return "round"
    if ratio < 0.85:
        return "long"
    return "oval"


def _dominant_color_region(
    image: np.ndarray, bbox: tuple,
    y_offset_ratio: float = 0.0, height_ratio: float = 0.4
) -> np.ndarray:
    """Sample region (e.g. top of head for hair) and get dominant color (BGR)."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    # region above face for hair
    y0 = max(0, int(y + y_offset_ratio * bh))
    y1 = min(h, int(y0 + height_ratio * bh))
    x0 = max(0, x)
    x1 = min(w, x + bw)
    if y1 <= y0 or x1 <= x0:
        return np.array([128, 128, 128])
    region = image[y0:y1, x0:x1]
    pixels = region.reshape(-1, 3)
    # k-means style: use median per channel (robust)
    return np.median(pixels, axis=0).astype(np.uint8)


def _bgr_to_hair_label(bgr: np.ndarray) -> str:
    """Map BGR dominant color to hair color label."""
    b, g, r = float(bgr[0]), float(bgr[1]), float(bgr[2])
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    sat = max(r, g, b) - min(r, g, b) if max(r, g, b) else 0
    if gray < 50:
        return "black"
    if gray < 90:
        return "dark_brown"
    if gray < 140:
        return "brown"
    if gray < 180:
        return "light_brown"
    if gray < 220 and sat < 60:
        return "gray"
    if r > g * 1.3 and r > b * 1.2:
        return "red"
    if gray > 180:
        return "blonde"
    return "other"


def _skin_tone_from_face(image: np.ndarray, bbox: tuple) -> str:
    """Sample cheek/forehead region and map to skin tone."""
    x, y, bw, bh = bbox
    h, w = image.shape[:2]
    # center-bottom of bbox (cheek area)
    cx = x + bw // 2
    cy = y + int(bh * 0.7)
    r = min(bw, bh) // 4
    x0 = max(0, cx - r)
    x1 = min(w, cx + r)
    y0 = max(0, cy - r)
    y1 = min(h, cy + r)
    if x1 <= x0 or y1 <= y0:
        return "light"
    region = image[y0:y1, x0:x1]
    mean_bgr = np.mean(region.reshape(-1, 3), axis=0)
    luminance = 0.299 * mean_bgr[2] + 0.587 * mean_bgr[1] + 0.114 * mean_bgr[0]
    if luminance < 80:
        return "dark"
    if luminance < 120:
        return "tan"
    if luminance < 160:
        return "medium"
    if luminance < 200:
        return "light"
    return "very_light"


def _detect_glasses_heuristic(image: np.ndarray, bbox: tuple) -> bool:
    """Simple heuristic: bright reflection or edge density in eye region."""
    x, y, bw, bh = bbox
    h, w = image.shape[:2]
    # eye band
    ey0 = y + int(bh * 0.25)
    ey1 = y + int(bh * 0.55)
    ex0 = max(0, x)
    ex1 = min(w, x + bw)
    if ey1 <= ey0 or ex1 <= ex0:
        return False
    eye_region = image[ey0:ey1, ex0:ex1]
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_ratio = np.sum(edges > 0) / max(edges.size, 1)
    # glasses often add more edges
    return edge_ratio > 0.12


def _estimate_hair_length(image: np.ndarray, bbox: tuple) -> str:
    """Rough estimate from visible area above shoulders (top of image vs face)."""
    x, y, bw, bh = bbox
    h, w = image.shape[:2]
    # face top to image top
    head_top_to_face = y
    face_height = bh
    if face_height <= 0:
        return "medium"
    ratio = head_top_to_face / face_height
    if ratio < 0.3:
        return "short"
    if ratio > 1.2:
        return "long"
    return "medium"


def _dominant_color_rect(
    image: np.ndarray,
    y0: int, y1: int,
    x0: int, x1: int,
) -> np.ndarray:
    """Sample rectangle region and return dominant BGR (median)."""
    h, w = image.shape[:2]
    y0, y1 = max(0, y0), min(h, y1)
    x0, x1 = max(0, x0), min(w, x1)
    if y1 <= y0 or x1 <= x0:
        return np.array([128, 128, 128])
    region = image[y0:y1, x0:x1]
    return np.median(region.reshape(-1, 3), axis=0).astype(np.uint8)


def _bgr_to_clothing_color_label(bgr: np.ndarray) -> str:
    """Map BGR to characteristic clothing color label."""
    b, g, r = float(bgr[0]), float(bgr[1]), float(bgr[2])
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    sat = max(r, g, b) - min(r, g, b) if max(r, g, b) else 0
    # White / black / gray
    if gray > 240 and sat < 30:
        return "white"
    if gray < 40 and sat < 40:
        return "black"
    if 60 < gray < 200 and sat < 50:
        return "gray"
    # Navy (dark blue)
    if b > r and b > g and gray < 100:
        return "navy"
    if b > r * 1.2 and b > g * 1.2:
        return "blue"
    if r > g * 1.3 and r > b * 1.2:
        if gray < 120:
            return "red"
        return "pink"
    if g > r and g > b:
        if gray > 180:
            return "mint"
        return "green"
    if r > g and g > b and gray > 140:
        return "orange"
    if r > 140 and g > 100 and b < 100:
        return "yellow"
    if 80 < gray < 160 and sat < 80:
        return "brown"
    if 160 < gray < 220 and sat < 60 and r >= g:
        return "beige"
    if b > r and r > g and gray < 180:
        return "purple"
    if 150 < gray < 220 and b > g and g > r:
        return "lavender"
    return "other"


def _sample_torso_region(image: np.ndarray, face_bbox: tuple) -> np.ndarray:
    """Sample upper body (가슷~배) 영역: 얼굴 bbox 바로 아래."""
    x, y, bw, bh = face_bbox
    h, w = image.shape[:2]
    # 얼굴 아래 ~ 1.5배 높이만큼 (상의 구간)
    y0 = y + bh
    y1 = min(h, y0 + int(bh * 1.8))
    x0 = max(0, x - bw // 2)
    x1 = min(w, x + bw + bw // 2)
    return _dominant_color_rect(image, y0, y1, x0, x1)


def _sample_lower_body_region(image: np.ndarray, face_bbox: tuple) -> np.ndarray | None:
    """Sample lower body (하의) 영역. 전신이 보일 때만 의미 있음."""
    x, y, bw, bh = face_bbox
    h, w = image.shape[:2]
    # 얼굴 아래 2bh 이상 있어야 하의 영역으로 간주
    torso_bottom = y + bh + int(bh * 1.8)
    if torso_bottom >= h - 20:
        return None
    y0 = torso_bottom
    y1 = h
    x0 = max(0, w // 4)
    x1 = min(w, 3 * w // 4)
    return _dominant_color_rect(image, y0, y1, x0, x1)


def _color_distance_bgr(c1: np.ndarray, c2: np.ndarray) -> float:
    """Simple L2 distance in BGR (0~255 scale)."""
    return float(np.sqrt(np.sum((c1.astype(float) - c2.astype(float)) ** 2)))


def _infer_clothing_type(
    image: np.ndarray,
    face_bbox: tuple,
    upper_bgr: np.ndarray,
    lower_bgr: np.ndarray | None,
) -> str:
    """
    Infer clothing type: dress, skirt, pants, shorts, top_only.
    - dress: 상하 색이 거의 동일하고, 위아래 경계가 분리되지 않은 경우에만 (엄격).
    - skirt: 하의 영역이 충분히 길어서 다리가 보일 수 있을 때만 (다리 노출 가능).
    - 모르겠으면 pants 통일.
    """
    x, y, bw, bh = face_bbox
    h, w = image.shape[:2]
    torso_bottom = y + bh + int(bh * 1.8)
    if torso_bottom >= h - 30:
        return "top_only"
    if lower_bgr is None:
        return "pants"
    lower_y0 = torso_bottom
    lower_y1 = h
    lower_h = lower_y1 - lower_y0
    lower_w = w
    # dress: 색이 거의 동일할 뿐 아니라, 상하가 분리되지 않은 느낌일 때만 (엄격)
    dist = _color_distance_bgr(upper_bgr, lower_bgr)
    if dist < 22 and lower_h >= bh * 2.0:
        return "dress"
    # skirt: 하의 구간이 길어서 치마 아래 다리가 보일 수 있을 때만 (다리 노출)
    if lower_h >= bh * 2.2:
        aspect = lower_w / max(lower_h, 1)
        if aspect > 1.05:
            return "skirt"
    # 짧은 하의
    if lower_h < bh * 1.3:
        return "shorts"
    return "pants"


def extract_features(image_path: str) -> dict[str, Any]:
    """
    Extract identity features from a single face photo.
    Returns a dict suitable for features.json and prompt generation.
    """
    image = _load_image(image_path)
    bbox_mp = _get_face_bbox_mediapipe(image)
    if bbox_mp is not None:
        x, y, bw, bh = bbox_mp
        bbox = (max(0, x), max(0, y), max(1, bw), max(1, bh))
    else:
        bbox = _get_face_bbox_heuristic(image)

    face_shape = _infer_face_shape(bbox, image.shape)
    hair_region_color = _dominant_color_region(
        image, bbox, y_offset_ratio=-0.5, height_ratio=0.5
    )
    hair_color = _bgr_to_hair_label(hair_region_color)
    skin_tone = _skin_tone_from_face(image, bbox)
    glasses = _detect_glasses_heuristic(image, bbox)
    hair_length = _estimate_hair_length(image, bbox)

    # 의상: 상의 색, 하의 색, 의상 종류
    torso_bgr = _sample_torso_region(image, bbox)
    clothing_upper_color = _bgr_to_clothing_color_label(torso_bgr)
    lower_bgr = _sample_lower_body_region(image, bbox)
    clothing_lower_color: str | None = None
    if lower_bgr is not None:
        clothing_lower_color = _bgr_to_clothing_color_label(lower_bgr)
    clothing_type = _infer_clothing_type(image, bbox, torso_bgr, lower_bgr)

    return {
        "face_shape": face_shape,
        "hair_color": hair_color,
        "skin_tone": skin_tone,
        "glasses": bool(glasses),
        "hair_length": hair_length,
        "clothing_upper_color": clothing_upper_color,
        "clothing_lower_color": clothing_lower_color,
        "clothing_type": clothing_type,
    }


def extract_and_save(image_path: str, output_dir: str | Path) -> str:
    """
    Extract features and save to features/features.json.
    Returns path to saved JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    features = extract_features(image_path)
    out_path = output_dir / "features.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    return str(out_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m core.feature_extractor <input_photo.png> [output_dir]")
        sys.exit(1)
    img_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "features"
    path = extract_and_save(img_path, out_dir)
    print(f"Saved: {path}")
