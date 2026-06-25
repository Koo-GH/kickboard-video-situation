# 킥보드 주행 영상 상황 인식 시스템

전동 킥보드 주행 영상을 분석하여 안전 상황을 자동 분류하는 시스템입니다.

## 환경 구성

### Windows (GTX 1080, VRAM 8GB)

```bash
# 1. 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\activate

# 2. torch 설치 (CUDA 12.1 기준)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. 나머지 패키지 설치
pip install -r requirements-dev.txt

# 4. 환경변수 설정
copy .env.example .env
```

### GPU 서버 (A100 80GB, Docker)

```bash
# Docker 컨테이너 내부에서
# torch는 이미 설치되어 있음

# 1. GitHub에서 코드 클론
git clone https://github.com/<your-repo>/kickboard-video-situation.git
cd kickboard-video-situation

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
```

## 모델 다운로드 및 서버 전송

```bash
# Windows에서 모델 다운로드
python scripts/download_model.py --model Qwen/Qwen2.5-VL-3B-Instruct   # 로컬용
python scripts/download_model.py --model Qwen/Qwen2.5-VL-7B-Instruct   # 서버용

# scp로 서버 전송 (Git Bash 또는 WSL)
scp -r ./models/Qwen2.5-VL-7B-Instruct cvlab308@10.10.15.112:/workspace/models/
```

## 실행

### Mock 모드 (GPU 없이 파이프라인 테스트)

```bash
python scripts/run_inference.py --video data/samples/sample.mp4 --model mock
```

### Qwen2.5-VL 실제 추론 (Windows, 3B 모델)

```bash
python scripts/run_inference.py \
  --video data/samples/sample.mp4 \
  --model qwen2_5_vl \
  --model-path ./models/Qwen2.5-VL-3B-Instruct
```

### Qwen2.5-VL 실제 추론 (GPU 서버, 7B 모델)

```bash
python scripts/run_inference.py \
  --video data/samples/sample.mp4 \
  --model qwen2_5_vl \
  --model-path /workspace/models/Qwen2.5-VL-7B-Instruct
```

## 테스트

```bash
pytest tests/ -v
```

## 출력 형식

```json
{
  "primary_situation": "pedestrian_risk",
  "secondary_situations": ["roadway_riding", "no_helmet"],
  "risk_level": "high",
  "confidence": 0.78,
  "evidence": ["전방에 보행자 근접", "차도에서 주행 중"],
  "recommended_action": "감속 또는 정지 필요",
  "needs_human_review": false
}
```

## 프로젝트 구조

```
kickboard_project/
├── src/
│   ├── config.py            # 설정 관리
│   ├── video/               # 영상 처리
│   │   ├── loader.py
│   │   ├── sampler.py
│   │   └── metadata.py
│   ├── models/              # 모델 인터페이스
│   │   ├── base.py
│   │   ├── mock_model.py
│   │   └── qwen_vl.py
│   ├── analysis/            # 분류 및 스키마
│   │   └── schema.py
│   └── utils/
│       └── json_utils.py
├── tests/                   # 단위 테스트
├── scripts/
│   ├── run_inference.py     # CLI
│   └── download_model.py    # 모델 다운로드
├── data/
│   ├── samples/             # 입력 영상
│   └── outputs/             # 결과 JSON
└── models/                  # 모델 가중치 (git 제외)
```
