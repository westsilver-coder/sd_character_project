# 사진 기반 SD 스타일 3D 캐릭터 자동 생성 시스템

사용자 사진을 입력하면, 해당 인물의 특징을 반영한 **귀여운 SD(슈퍼 디포르메) 스타일 3D 캐릭터**를 자동 생성하는 Python 파이프라인입니다.

- **목표**: 정확한 3D 재구성이 아니라, “사진 속 인물을 닮았다고 느껴지는” SD 스타일 3D 캐릭터 생성  
- **언어**: Python 3.10  
- **환경**: conda 권장, SSH GPU 서버 (root 불필요)

---

## 파이프라인 개요

```
input photo → feature extraction → features.json
     → prompt generation → Stable Diffusion 2D
     → 2D→3D mesh → SD proportion deformation
     → Turntable rendering → 3D avatar + mp4
```

1. **Feature Extraction**: 얼굴형, 머리색, 피부톤, 안경, 머리 길이, **의상 색상·종류**(상의색, 하의색, dress/skirt/pants 등) → `features/features.json`
2. **Prompt Generation**: 특징 기반 SD 프롬프트 자동 생성 (규칙 기반)
3. **2D 생성**: Stable Diffusion 1.5로 전신 SD 스타일 이미지 → `generated_2d/` (512×512, RTX 2080Ti 안정)
4. **2D→3D**: 이미지 → mesh (.obj). TripoSR 연동 시 실제 복원, 없으면 placeholder
5. **SD 변형**: head scale 1.5~1.7, body scale 0.6~0.8 (geometry 수정)
6. **Turntable**: 360° 회전 mp4 → `renders/`

---

## 코드 읽기 순서 (파이프라인 이해용)

데이터 흐름대로 보면 이해하기 쉽습니다.

| 순서 | 파일 | 보는 목적 |
|------|------|-----------|
| **1** | `main.py` | 전체 단계(1→2→…→6)와 입출력 경로 한눈에 파악 |
| **2** | `core/feature_extractor.py` | 입력 사진 → 얼굴 bbox, 머리/피부/의상 색, 안경/머리길이, 의상 종류 → `features.json` |
| **3** | `core/prompt_generator.py` | `features.json` → 규칙으로 SD용 프롬프트 문자열 생성 |
| **4** | `core/sd_generator.py` | 프롬프트 → SD 1.5로 2D 전신 SD 캐릭터 이미지 생성 (512×512) |
| **5** | `core/mesh_generator.py` | 2D 이미지 → TripoSR(선택) 또는 placeholder → .obj mesh |
| **6** | `core/deform_sd.py` | mesh를 head/body 구간으로 나누어 SD 비율(헤드 확대, 바디 축소) 적용 |
| **7** | `core/renderer.py` | mesh 360° 회전 프레임 생성 → imageio로 mp4 저장 |

**요약**: `main.py`로 흐름 잡은 뒤, **2→3→4→5→6→7** 순서로 각 단계의 입력/출력만 따라가면 됩니다.

---

## 입력 이미지 가이드 (제한 사항)

현재 파이프라인은 **입력 사진 1장**만 사용합니다.

### 장수·구도

- **1장**만 사용합니다. (전신/정면 필수 아님)
- **얼굴이 나온 사진**이면 됩니다. Feature extraction이 **얼굴 영역**을 기준으로 하므로, 전신이 아니어도 됩니다.
- **전신이 보이면** 상의/하의 색상과 의상 종류(원피스·치마·바지 등)를 추출해 프롬프트에 반영합니다. 상반신만 있으면 상의 색만 쓰고 `top_only`로 처리됩니다.
- 단, **2D 생성 단계(SD 1.5)**는 “full body” 프롬프트로 전신 캐릭터를 그리므로, 입력이 상반신/얼굴만 있어도 **출력만 전신**이 됩니다.

### 추천 조건 (결과 품질을 위해)

- **얼굴이 정면에 가깝게** 보이는 사진: MediaPipe 얼굴 검출이 안정적입니다.
- **얼굴이 가려지지 않은 것**: 마스크, 손, 머리카락으로 눈·코·입이 크게 가려지면 bbox/특징 추출이 부정확해질 수 있습니다.
- **해상도**: 너무 작지 않게 (예: 한 변 256px 이상). 얼굴이 50px 이상 정도로 보이면 좋습니다.

### 조명

- 코드 상 **조명 조건에 대한 별도 제한은 없습니다**.
- 다만 머리색·피부톤은 **픽셀 평균/중앙값**으로 추정하므로, **과도한 역광·극단적 그림자**는 색 추정을 틀리게 할 수 있습니다. **적당히 밝고, 얼굴이 잘 보이는** 조명을 추천합니다.

### 사용하지 않는 것

- **여러 장 입력**: 현재는 1장만 읽습니다. 멀티뷰/멀티포즈 미지원.
- **정면 전신 필수 아님**: 전신이어도 되고, 상반신/얼굴만 있어도 동작합니다.

---

## 디렉터리 구조

```
sd_character_project/
├── input/              # 입력 사진 (예: user_photo.png)
├── features/            # features.json
├── generated_2d/       # SD 2D 캐릭터 PNG
├── generated_3d/       # mesh .obj (원본, SD 변형)
├── renders/            # turntable.mp4
├── core/
│   ├── feature_extractor.py
│   ├── prompt_generator.py
│   ├── sd_generator.py
│   ├── mesh_generator.py
│   ├── deform_sd.py
│   └── renderer.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 설치 (conda)

```bash
conda create -n sd_char python=3.10 -y
conda activate sd_char
cd sd_character_project
pip install -r requirements.txt
```

- **GPU**: CUDA 사용 시 `torch`는 CUDA 버전으로 설치  
- **TripoSR** (선택): 실제 2D→3D 시 [TripoSR](https://github.com/VAST-AI-Research/TripoSR) 저장소 클론 후 `run.py` 경로를 `--triposr` 또는 환경변수 `TRIPOSR_SCRIPT`로 지정

---

## 사용법

### 전체 파이프라인 (한 번에)

```bash
# 입력 사진을 input/user_photo.png 에 두고
python main.py

# 또는 경로 지정
python main.py path/to/photo.jpg
```

### 단계별 실행

```bash
# 1) 특징 추출
python -m core.feature_extractor input/user_photo.png features

# 2) 프롬프트 생성 (확인용)
python -m core.prompt_generator features/features.json

# 3) 2D SD 이미지 생성
python -m core.sd_generator   # features/features.json → generated_2d/character.png

# 4) 2D → 3D mesh (TripoSR 없으면 placeholder)
python -m core.mesh_generator generated_2d/character.png generated_3d/character.obj

# 5) SD 비율 변형
python -m core.deform_sd generated_3d/character.obj generated_3d/character_sd.obj

# 6) 터닝테이블 영상
python -m core.renderer generated_3d/character_sd.obj renders/turntable.mp4
```

### 옵션 (main.py)

**성별·의상·헤어 (사진이 애매할 때 지정)**

- `--gender {male,female,person}` : 캐릭터 성별. **필수 지정 권장** (사진으로 추정하지 않음). 기본: person
- `--clothing-type {auto,dress,skirt,pants,shorts,top_only}` : 의상 종류. auto=사진에서 추출
- `--upper-color`, `--lower-color` : 상의/하의 색 (white, black, navy, blue 등)
- `--hair-length {auto,short,medium,long}` : 머리 길이. auto=사진에서 추출
- `--hair-color` : 머리 색 (black, dark_brown, brown, blonde, red, gray 등)

**파이프라인**

- `--no-sd` : 2D 생성 생략 (이미 있는 `generated_2d/character.png` 사용)
- `--no-3d` : 2D→3D 생략
- `--no-deform` : SD 비율 변형 생략
- `--no-render` : mp4 생성 생략
- `--triposr /path/to/run.py` : TripoSR run.py 경로
- `--seed 42` : SD 샘플링 시드

---

## MVP 성공 조건

- [x] 입력 사진 1장 → 3D SD 캐릭터 생성 파이프라인 동작  
- [x] 특징이 프롬프트에 자동 반영  
- [x] 3D mesh 생성 (placeholder 또는 TripoSR)  
- [x] head/body 비율 SD 스타일로 geometry 변형  
- [x] Turntable mp4 생성  

---

## 기술 스택

| 용도 | 라이브러리 |
|------|------------|
| 특징 추출 | mediapipe, opencv-python, scikit-learn |
| 프롬프트 | 규칙 기반 (JSON → 문자열) |
| 2D 생성 | diffusers (SD 1.5), torch, transformers, accelerate |
| 2D→3D | TripoSR(선택) / trimesh placeholder |
| 3D 변형/렌더 | trimesh, imageio, imageio-ffmpeg |

---

## 확장 아이디어 (추후)

- Gradio 웹 UI  
- 캐릭터 스타일 프리셋  
- 여러 표정 자동 생성  
- 게임용 glb export 최적화  

---

## 한 문장 요약

**“사용자 사진을 입력하면, 해당 인물의 특징을 반영한 귀여운 SD 스타일 3D 캐릭터를 자동 생성해주는 시스템”**
