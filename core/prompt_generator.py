"""
Rule-based prompt generation from extracted features (features.json).
Produces a single string suitable for Stable Diffusion (SD 1.5) to generate
a cute SD-style full-body 3D character.
"""
import json
from pathlib import Path
from typing import Any


# Base prompt parts
BASE_PARTS = [
    "A cute super-deformed 3D character",
    "big head small body",
    "toy-like",
    "smooth surface",
    "pastel soft shading",
    "full body",
    "high quality 3D render",
]

FACE_SHAPE_PROMPT = {
    "round": "round face",
    "oval": "oval face",
    "long": "slightly long face",
    "square": "soft square face",
}

HAIR_PROMPT = {
    "black": "black hair",
    "dark_brown": "dark brown hair",
    "brown": "brown hair",
    "light_brown": "light brown hair",
    "blonde": "blonde hair",
    "gray": "gray hair",
    "red": "red hair",
    "other": "natural hair",
}

SKIN_PROMPT = {
    "very_light": "very light skin tone",
    "light": "light skin tone",
    "medium": "medium skin tone",
    "tan": "tan skin tone",
    "dark": "dark skin tone",
}

# 의상 색상 → SD 프롬프트
CLOTHING_COLOR_PROMPT = {
    "white": "white",
    "black": "black",
    "gray": "gray",
    "navy": "navy blue",
    "blue": "blue",
    "red": "red",
    "pink": "pink",
    "green": "green",
    "yellow": "yellow",
    "orange": "orange",
    "brown": "brown",
    "beige": "beige",
    "purple": "purple",
    "mint": "mint",
    "lavender": "lavender",
    "other": "colored",
}

# 의상 종류 → SD 프롬프트
CLOTHING_TYPE_PROMPT = {
    "dress": "wearing a dress",
    "skirt": "wearing a skirt",
    "pants": "wearing pants",
    "shorts": "wearing shorts",
    "top_only": "wearing top",
    "unknown": "wearing casual clothes",
}


def load_features(features_path: str | Path) -> dict[str, Any]:
    """Load features from JSON file."""
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(features: dict[str, Any]) -> str:
    """
    Build a single positive prompt string from feature dict.
    """
    parts = list(BASE_PARTS)

    face_shape = features.get("face_shape", "oval")
    parts.append(FACE_SHAPE_PROMPT.get(face_shape, "oval face"))

    hair_color = features.get("hair_color", "brown")
    hair_length = features.get("hair_length", "medium")
    hair_desc = f"{hair_length} {HAIR_PROMPT.get(hair_color, 'natural hair')}"
    parts.append(hair_desc)

    skin_tone = features.get("skin_tone", "light")
    parts.append(SKIN_PROMPT.get(skin_tone, "light skin tone"))

    if features.get("glasses", False):
        parts.append("wearing glasses")
    else:
        parts.append("no glasses")

    # 의상: 상의 색 + 하의(종류·색)
    upper_color = features.get("clothing_upper_color", "other")
    parts.append(f"{CLOTHING_COLOR_PROMPT.get(upper_color, 'colored')} top")
    lower_color = features.get("clothing_lower_color")
    clothing_type = features.get("clothing_type", "unknown")
    if clothing_type == "dress":
        parts.append("wearing a dress")
    elif clothing_type == "skirt":
        c = CLOTHING_COLOR_PROMPT.get(lower_color or upper_color, "colored")
        parts.append(f"{c} skirt")
    elif clothing_type == "pants":
        c = CLOTHING_COLOR_PROMPT.get(lower_color or "other", "colored")
        parts.append(f"{c} pants")
    elif clothing_type == "shorts":
        c = CLOTHING_COLOR_PROMPT.get(lower_color or "other", "colored")
        parts.append(f"{c} shorts")
    elif clothing_type == "top_only":
        parts.append("upper body only visible")
    else:
        parts.append(CLOTHING_TYPE_PROMPT.get(clothing_type, "wearing casual clothes"))

    return ",\n".join(parts)


def generate_prompt(features_path: str | Path) -> str:
    """
    Load features from path and return generated prompt.
    """
    features = load_features(features_path)
    return build_prompt(features)


def generate_and_print(features_path: str | Path) -> None:
    """Convenience: generate and print prompt."""
    prompt = generate_prompt(features_path)
    print(prompt)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "features/features.json"
    generate_and_print(path)
