"""
Rule-based prompt generation from extracted features (features.json).
SD 1.5용: 애니/웹툰 느낌, 적당한 비율, 구체적 헤어/의상, 갈색 눈 기본.
성별·의상·헤어는 CLI(overrides)로 지정 가능. 지정 안 하면 features 사용.
"""
import json
from pathlib import Path
from typing import Any

# CLI에서 쓸 수 있는 옵션 값 (overrides)
GENDER_CHOICES = ("male", "female", "person")
CLOTHING_TYPE_CHOICES = ("auto", "dress", "skirt", "pants", "shorts", "top_only")
HAIR_LENGTH_CHOICES = ("auto", "short", "medium", "long")


# 스타일: 과한 SD/인형 느낌 X, 애니·웹툰 정도 비율
BASE_PARTS = [
    "anime style full body character",
    "webtoon style proportions",
    "moderate head to body ratio, not chibi",
    "standing pose",
    "clean neutral background",
    "3D render style",
    "soft lighting",
]

# 얼굴: 형태 구별 안 함, 예쁘게/잘 나오게
FACE_PROMPT = "pretty face, well-proportioned facial features"

# 눈: 모르면 갈색 기본
EYE_PROMPT = "brown eyes"

# 헤어 색상
HAIR_COLOR_PROMPT = {
    "black": "black hair",
    "dark_brown": "dark brown hair",
    "brown": "brown hair",
    "light_brown": "light brown hair",
    "blonde": "blonde hair",
    "gray": "gray hair",
    "red": "red hair",
    "other": "natural hair color",
}

# 남성 헤어스타일 (구체적)
HAIR_STYLE_MALE = {
    "short": "short neat hair, natural look",
    "medium": "medium length hair, natural wave",
    "long": "long hair, natural wave",
}
# 여성 헤어스타일 (구체적): 숏컷, 앞머리 유무, 긴머리/중단발/어깨까지 등
HAIR_STYLE_FEMALE = {
    "short": "short cut hair, natural hair texture",
    "medium": "shoulder length hair, natural hair texture",
    "long": "long hair, natural hair texture, fine hair details",
}
# 공통: 덩어리 방지
HAIR_QUALITY = "detailed hair, natural hair strands, not blocky"

# 피부
SKIN_PROMPT = {
    "very_light": "very light skin tone",
    "light": "light skin tone",
    "medium": "medium skin tone",
    "tan": "tan skin tone",
    "dark": "dark skin tone",
}

# 의상 색상
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


def load_features(features_path: str | Path) -> dict[str, Any]:
    """Load features from JSON file."""
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(f"Features file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _apply_overrides(features: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    """overrides에 있는 키만 features 값을 덮어쓴 새 dict 반환. auto는 덮어쓰지 않음."""
    if not overrides:
        return dict(features)
    out = dict(features)
    for k, v in overrides.items():
        if v is None or v == "auto":
            continue
        out[k] = v
    return out


def build_prompt(features: dict[str, Any], overrides: dict[str, Any] | None = None) -> str:
    """
    Build positive prompt. 성별은 overrides["gender"]로만 지정 (CLI에서 필수).
    의상/헤어는 overrides로 지정 가능, 없으면 features 사용.
    """
    merged = _apply_overrides(features, overrides)
    parts = list(BASE_PARTS)

    # 성인 + 성별 (CLI에서만 지정, 기본 person)
    gender = merged.get("gender", "person")
    if gender == "male":
        parts.append("adult man")
    elif gender == "female":
        parts.append("adult woman")
    else:
        parts.append("adult person")

    parts.append(FACE_PROMPT)
    parts.append(EYE_PROMPT)

    # 헤어: 색 + 길이별 구체 스타일 + 덩어리 방지
    hair_color = merged.get("hair_color", "brown")
    hair_length = merged.get("hair_length", "medium")
    parts.append(HAIR_COLOR_PROMPT.get(hair_color, "natural hair color"))
    if gender == "male":
        parts.append(HAIR_STYLE_MALE.get(hair_length, "short neat hair, natural look"))
    else:
        parts.append(HAIR_STYLE_FEMALE.get(hair_length, "natural hair texture"))
    parts.append(HAIR_QUALITY)

    skin_tone = merged.get("skin_tone", "light")
    parts.append(SKIN_PROMPT.get(skin_tone, "light skin tone"))

    if merged.get("glasses", False):
        parts.append("wearing glasses")
    else:
        parts.append("no glasses")

    # 의상: 항상 상의+하의 명시 (overrides 또는 features)
    upper_color = merged.get("clothing_upper_color", "other")
    lower_color = merged.get("clothing_lower_color")
    clothing_type = merged.get("clothing_type", "pants")
    uc = CLOTHING_COLOR_PROMPT.get(upper_color, "colored")
    lc = CLOTHING_COLOR_PROMPT.get(lower_color or "other", "colored")

    if clothing_type == "dress":
        parts.append(f"wearing {uc} one-piece dress, fully clothed")
    elif clothing_type == "skirt":
        parts.append(f"wearing {uc} top and {lc} skirt, legs visible, fully clothed")
    elif clothing_type == "pants":
        parts.append(f"wearing {uc} shirt and {lc} pants, fully clothed")
    elif clothing_type == "shorts":
        parts.append(f"wearing {uc} shirt and {lc} shorts, fully clothed")
    elif clothing_type == "top_only":
        parts.append(f"wearing {uc} top, fully clothed")
    else:
        parts.append(f"wearing {uc} shirt and {lc} pants, fully clothed")

    return ", ".join(parts)


def get_negative_prompt() -> str:
    """강화된 네거티브 프롬프트: 큰 대가리, 덩어리 머리, 인형/요정, 상의 누락 등."""
    return (
        "ugly, blurry, low quality, distorted, deformed, "
        "multiple characters, cropped, bad anatomy, text, watermark, "
        "huge head, oversized head, big head, bloated face, round blob head, "
        "chunky hair, blocky hair, hair blob, "
        "doll, fairy, wings, tutu, child, baby, toddler, "
        "topless, nude, shirtless, no shirt, exposed chest, "
        "ugly face, deformed face, bad proportions, "
        "extreme chibi, super deformation, toy like"
    )


def generate_prompt(
    features_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> str:
    """Load features and return positive prompt string. overrides로 성별·의상·헤어 지정 가능."""
    features = load_features(features_path)
    return build_prompt(features, overrides)


def generate_and_print(
    features_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Convenience: generate and print prompt."""
    prompt = generate_prompt(features_path, overrides)
    print(prompt)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "features/features.json"
    generate_and_print(path)
