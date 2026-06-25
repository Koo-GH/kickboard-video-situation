# 킥보드 주행 영상 상황 인식 프로젝트 요구사항서

## 1. 프로젝트 개요

본 프로젝트는 전동 킥보드 주행 영상을 입력으로 받아, 현재 킥보드 운전자가 어떤 상황에 처해 있는지 자동으로 구분하는 시스템을 구현하는 것을 목표로 한다.  
구현은 ANTIGRAVITY 기반 개발 환경에서 진행하며, 가능한 한 무료로 사용 가능한 오픈소스 비전-언어 모델 또는 비디오 LLM을 활용한다.

본 문서는 ANTIGRAVITY 에이전트가 개발 환경을 구성하고, MVP를 구현하고, 테스트까지 수행할 수 있도록 요구사항과 작업 지시를 정리한 문서이다.

---

## 2. 목표

### 2.1 최종 목표

킥보드 주행 영상에서 다음과 같은 상황을 자동 분류한다.

- 정상 주행
- 보행자 접근 또는 충돌 위험
- 차량 접근 또는 충돌 위험
- 장애물 접근
- 교차로 또는 횡단보도 진입
- 급정지 또는 위험 회피 상황
- 넘어짐, 사고, 전도 의심
- 도로 상태 불량 또는 노면 위험
- 역주행, 인도 주행, 차도 주행 등 주행 위치 관련 상황
- 헬멧 미착용 등 보호장구 관련 위험 상황
- 영상 품질 저하 등으로 인한 사람 검토 필요 상황

### 2.2 MVP 목표

초기 버전에서는 실시간 스트리밍이 아니라, 저장된 짧은 영상 파일을 입력으로 받아 상황을 분류한다.

MVP 입력과 출력은 다음과 같다.

- 입력: `.mp4`, `.mov`, `.avi` 형식의 킥보드 주행 영상
- 출력: 상황 분류 결과 JSON 및 사람이 읽을 수 있는 요약 문장

예상 출력 예시는 다음과 같다.

```json
{
  "primary_situation": "pedestrian_risk",
  "secondary_situations": ["roadway_riding", "no_helmet"],
  "risk_level": "high",
  "confidence": 0.78,
  "evidence": [
    "영상 전방에 보행자가 가까이 있음",
    "킥보드가 차도 영역에서 주행 중임",
    "운전자의 헬멧 착용이 확인되지 않음"
  ],
  "recommended_action": "감속 또는 정지 필요"
}
```

---

## 3. 개발 원칙

1. 무료 또는 오픈소스 모델을 우선 사용한다.
2. GPU가 없는 환경에서도 최소한의 테스트가 가능해야 한다.
3. 모델 추론 결과는 반드시 구조화된 JSON으로 반환한다.
4. 영상 분석 결과는 “왜 그렇게 판단했는지” 근거를 포함해야 한다.
5. 안전 관련 판단이므로, 확신이 낮으면 무리하게 단정하지 않고 `needs_review`로 표시한다. 단, `unknown` 라벨은 사용하지 않는다.
6. 차도 주행, 헬멧 미착용, 인도 주행 등은 법적 판정이 아니라 영상 기반 안전 상황 라벨로만 취급한다.
7. 초기 버전에서는 법적 판단이 아니라 “상황 인식 및 위험도 추정”으로 범위를 제한한다.
8. 개인정보 보호를 위해 얼굴, 번호판 등 민감 정보 저장은 최소화한다.

---

## 4. 추천 기술 스택

### 4.1 개발 환경

- OS: Ubuntu 22.04 이상 권장
- Python: 3.10 이상
- 패키지 관리: `uv` 또는 `pip`
- 가상환경: `.venv`
- IDE/Agent: ANTIGRAVITY
- 버전관리: Git

### 4.2 주요 라이브러리

- `torch`
- `transformers`
- `accelerate`
- `opencv-python`
- `decord` 또는 `moviepy`
- `pydantic`
- `fastapi`
- `uvicorn`
- `python-dotenv`
- `rich`
- `pytest`

### 4.3 모델 후보

초기 후보는 다음 순서로 검토한다.

#### 1순위: Qwen2.5-VL 계열

- 장점: 이미지/비디오 이해, 객체 위치 추론, 긴 영상 이해에 강점이 있음
- 권장 모델:
  - `Qwen/Qwen2.5-VL-3B-Instruct`
  - `Qwen/Qwen2.5-VL-7B-Instruct`
- 사용 조건:
  - 로컬 GPU가 부족하면 3B 또는 양자화 버전 사용
  - 7B 이상은 VRAM 요구량 확인 필요

#### 2순위: VideoLLaMA3 계열

- 장점: 비디오 이해 전용 모델 계열
- 권장 모델:
  - `DAMO-NLP-SG/VideoLLaMA3-2B`
  - `DAMO-NLP-SG/VideoLLaMA3-7B`
- 사용 조건:
  - 설치 의존성이 다소 무거울 수 있으므로 별도 브랜치에서 검토

#### 3순위: Video-LLaVA / LLaVA-Video 계열

- 장점: 연구 및 프로토타입 용도로 접근성이 좋음
- 사용 조건:
  - 라이선스와 상업적 사용 가능 여부를 반드시 확인
  - 모델별 입력 포맷과 프레임 처리 방식 확인

---

## 5. 시스템 구조

### 5.1 전체 파이프라인

```text
영상 파일 입력
  ↓
영상 메타데이터 추출
  ↓
프레임 샘플링
  ↓
프레임 또는 클립 단위 전처리
  ↓
비전-언어 모델 추론
  ↓
상황 분류 JSON 파싱
  ↓
위험도 산정
  ↓
결과 저장 및 리포트 생성
```

### 5.2 모듈 구조

권장 디렉터리 구조는 다음과 같다.

```text
kickboard-video-situation/
├── README.md
├── requirements.md
├── pyproject.toml
├── .env.example
├── data/
│   ├── samples/
│   └── outputs/
├── src/
│   ├── main.py
│   ├── config.py
│   ├── video/
│   │   ├── loader.py
│   │   ├── sampler.py
│   │   └── metadata.py
│   ├── models/
│   │   ├── base.py
│   │   ├── qwen_vl.py
│   │   └── videollama.py
│   ├── analysis/
│   │   ├── prompts.py
│   │   ├── classifier.py
│   │   ├── schema.py
│   │   └── risk.py
│   ├── api/
│   │   └── server.py
│   └── utils/
│       ├── logging.py
│       └── json_utils.py
├── tests/
│   ├── test_sampler.py
│   ├── test_schema.py
│   └── test_classifier_mock.py
└── scripts/
    ├── run_inference.py
    └── benchmark_samples.py
```

---

## 6. 상황 라벨 정의

### 6.1 기본 라벨

`unknown` 라벨은 기본 라벨에서 제외한다. 모델이 판단을 회피하지 않도록, 입력 영상에서 가장 가까운 상황 라벨을 반드시 선택하게 한다. 단, 사람이 직접 검토해야 할 정도로 영상 품질이 낮거나 판단 근거가 부족한 경우에는 `needs_review`만 사용한다.

| 라벨 | 설명 | 위험도 기본값 |
|---|---|---|
| `normal_riding` | 특이 위험 없이 정상 주행 | low |
| `roadway_riding` | 차도에서 킥보드를 주행 중인 상황 | medium |
| `no_helmet` | 운전자가 헬멧을 착용하지 않은 것으로 보이는 상황 | high |
| `pedestrian_risk` | 보행자가 가까이 있거나 진행 경로와 겹침 | high |
| `vehicle_risk` | 차량이 가까이 있거나 충돌 가능성이 있음 | high |
| `obstacle_risk` | 전방 또는 측면에 장애물이 있음 | medium |
| `intersection_crosswalk` | 교차로, 횡단보도, 진입로 등 복잡 구간 | medium |
| `sudden_stop_or_evasion` | 급정지, 급회피, 불안정한 움직임 | high |
| `fall_or_crash` | 넘어짐, 충돌, 사고 의심 | critical |
| `poor_road_condition` | 포트홀, 턱, 젖은 노면 등 위험 노면 | medium |
| `sidewalk_riding` | 인도 또는 보행 공간 주행 의심 | medium |
| `wrong_way_or_lane_issue` | 역주행, 차선 이탈, 부적절한 주행 방향 | high |
| `needs_review` | 영상 품질 저하, 가림, 야간 등으로 사람이 검토해야 하는 상황 | medium |

라벨 선택 원칙:

- `primary_situation`에는 위 라벨 중 하나를 반드시 선택한다.
- 여러 상황이 동시에 보이면 가장 위험도가 높은 상황을 `primary_situation`으로 선택한다.
- `roadway_riding`, `no_helmet`, `sidewalk_riding`처럼 동시에 성립할 수 있는 라벨은 `secondary_situations`에 함께 포함한다.
- `unknown`은 사용하지 않는다.
- 판단이 매우 어려운 경우에도 `unknown` 대신 `needs_review`를 사용한다.

### 6.2 위험도 단계

```text
low      : 일반 주행 상황
medium   : 주의가 필요한 상황
high     : 즉시 감속 또는 회피가 필요한 상황
critical : 사고 또는 사고 직전으로 보이는 상황
```

위험도 선택 원칙:

- `roadway_riding`, `sidewalk_riding`, `intersection_crosswalk`, `obstacle_risk`, `poor_road_condition`, `needs_review`는 기본적으로 `medium` 이상으로 본다.
- `no_helmet`, `pedestrian_risk`, `vehicle_risk`, `wrong_way_or_lane_issue`, `sudden_stop_or_evasion`는 기본적으로 `high` 이상으로 본다.
- `fall_or_crash`는 기본적으로 `critical`로 본다.
- `unknown` 위험도는 사용하지 않는다. 판단이 어려우면 `risk_level`은 `medium`, `primary_situation`은 `needs_review`로 둔다.

### 6.3 다중 라벨 처리 원칙

킥보드 주행 영상에서는 한 장면에 여러 상황이 동시에 나타날 수 있다. 예를 들어 운전자가 차도에서 헬멧 없이 주행하면서 차량이 접근하는 경우, `vehicle_risk`를 `primary_situation`으로 선택하고 `roadway_riding`, `no_helmet`을 `secondary_situations`에 포함한다.

우선순위는 다음과 같다.

1. 사고 또는 사고 직전: `fall_or_crash`, `sudden_stop_or_evasion`
2. 즉각적인 충돌 위험: `pedestrian_risk`, `vehicle_risk`
3. 보호장구 및 주행 위치 위험: `no_helmet`, `roadway_riding`, `sidewalk_riding`, `wrong_way_or_lane_issue`
4. 환경 위험: `obstacle_risk`, `poor_road_condition`, `intersection_crosswalk`
5. 특이 위험 없음: `normal_riding`
6. 영상 품질 또는 가림 문제: `needs_review`

`normal_riding`은 다른 위험 라벨과 함께 사용하지 않는다. 위험 또는 위반 의심 상황이 하나라도 명확히 보이면 `normal_riding` 대신 해당 위험 라벨을 선택한다.

---

## 7. 모델 프롬프트 요구사항

모델은 자유 서술이 아니라 반드시 JSON 형식으로만 응답해야 한다.

### 7.1 기본 프롬프트

```text
당신은 전동 킥보드 주행 영상을 분석하는 교통 안전 상황 인식 모델입니다.
영상 속 킥보드 운전자의 현재 상황을 판단하세요.

다음 라벨 중 가장 적절한 primary_situation 하나를 선택하세요. unknown 라벨은 사용하지 마세요.
- normal_riding
- roadway_riding
- no_helmet
- pedestrian_risk
- vehicle_risk
- obstacle_risk
- intersection_crosswalk
- sudden_stop_or_evasion
- fall_or_crash
- poor_road_condition
- sidewalk_riding
- wrong_way_or_lane_issue
- needs_review

추가로 관찰되는 상황은 secondary_situations 배열에 넣으세요.
차도 주행이 보이면 roadway_riding을 포함하세요.
헬멧 미착용이 보이면 no_helmet을 포함하세요.
위험도는 low, medium, high, critical 중 하나로 선택하세요. unknown 위험도는 사용하지 마세요.
확신도는 0.0부터 1.0 사이 숫자로 표시하세요.
근거는 영상에서 관찰 가능한 사실만 작성하세요.
법적 판단이나 과장된 표현은 피하세요.
확신이 낮거나 영상 품질이 낮으면 unknown 대신 needs_review를 사용하세요.

반드시 아래 JSON 스키마만 출력하세요.

{
  "primary_situation": "",
  "secondary_situations": [],
  "risk_level": "",
  "confidence": 0.0,
  "evidence": [],
  "recommended_action": "",
  "needs_human_review": false
}
```

---

## 8. 데이터 처리 요구사항

### 8.1 입력 영상

- 지원 포맷: `.mp4`, `.mov`, `.avi`
- 권장 길이: 5초 ~ 60초
- 너무 긴 영상은 일정 간격으로 클립 분할
- 해상도는 모델 입력 제한에 맞게 리사이즈

### 8.2 프레임 샘플링

MVP에서는 다음 방식 중 하나를 사용한다.

1. 균등 샘플링
   - 예: 1초당 1~2프레임
2. 이벤트 기반 샘플링
   - optical flow 또는 장면 변화량이 큰 구간 우선
3. 혼합 방식
   - 균등 샘플링 + 급격한 움직임 구간 추가 샘플링

초기 구현은 균등 샘플링으로 시작한다.

### 8.3 출력 저장

분석 결과는 다음 파일로 저장한다.

```text
data/outputs/{video_filename}.json
data/outputs/{video_filename}_summary.md
```

---

## 9. API 요구사항

MVP 이후 FastAPI 서버를 제공한다.

### 9.1 엔드포인트

#### `POST /analyze-video`

영상 파일을 업로드하면 분석 결과를 반환한다.

응답 예시:

```json
{
  "video_id": "sample_001",
  "primary_situation": "vehicle_risk",
  "secondary_situations": ["roadway_riding", "no_helmet"],
  "risk_level": "high",
  "confidence": 0.81,
  "evidence": ["전방 오른쪽에서 차량이 근접", "킥보드 진행 방향과 차량 경로가 가까움"],
  "recommended_action": "감속 및 주변 확인 필요",
  "needs_human_review": false
}
```

#### `GET /health`

서버 상태를 확인한다.

```json
{
  "status": "ok"
}
```

---

## 10. ANTIGRAVITY 작업 지시

ANTIGRAVITY 에이전트는 아래 순서대로 작업한다.

### 10.1 1단계: 프로젝트 초기화

1. 새 Python 프로젝트를 생성한다.
2. 위 디렉터리 구조를 만든다.
3. `.venv` 가상환경을 구성한다.
4. `pyproject.toml` 또는 `requirements.txt`를 작성한다.
5. `README.md`에 실행 방법을 작성한다.

### 10.2 2단계: 영상 처리 모듈 구현

1. `src/video/loader.py`에 영상 로딩 함수를 구현한다.
2. `src/video/sampler.py`에 균등 프레임 샘플링 함수를 구현한다.
3. `src/video/metadata.py`에 FPS, 길이, 해상도 추출 함수를 구현한다.
4. 샘플 영상 없이도 테스트 가능한 mock 테스트를 작성한다.

### 10.3 3단계: 출력 스키마 구현

1. `src/analysis/schema.py`에 Pydantic 모델을 정의한다.
2. 라벨과 위험도 enum을 정의한다.
3. `unknown` 라벨 또는 `unknown` 위험도가 출력되면 스키마 검증에서 실패하도록 한다.
4. JSON 파싱 실패 시 재시도 또는 에러 처리 로직을 만든다.

### 10.4 4단계: 모델 추론 모듈 구현

1. `src/models/base.py`에 공통 인터페이스를 정의한다.
2. `src/models/qwen_vl.py`에 Qwen2.5-VL 기반 추론 클래스를 구현한다.
3. GPU가 없을 경우 mock model로 전체 파이프라인을 테스트할 수 있게 한다.
4. 모델 출력은 반드시 `SituationAnalysis` 스키마로 검증한다.

### 10.5 5단계: CLI 구현

`scripts/run_inference.py`를 구현한다.

실행 예시:

```bash
python scripts/run_inference.py --video data/samples/sample.mp4 --model qwen2_5_vl --output data/outputs
```

### 10.6 6단계: API 구현

1. `src/api/server.py`에 FastAPI 서버를 구현한다.
2. `/health`와 `/analyze-video` 엔드포인트를 만든다.
3. 업로드 파일을 임시 저장 후 분석하고 결과를 반환한다.

### 10.7 7단계: 테스트 및 검증

1. 단위 테스트를 작성한다.
2. 최소 3개 이상의 샘플 영상으로 수동 검증한다.
3. 모델 출력 JSON이 항상 스키마를 만족하는지 확인한다.
4. 실패 사례를 `data/outputs/error_cases.md`에 기록한다.

---

## 11. 품질 기준

### 11.1 기능 기준

- 영상 파일을 정상적으로 읽을 수 있어야 한다.
- 일정 간격으로 프레임을 추출할 수 있어야 한다.
- 모델 추론 결과를 JSON으로 받을 수 있어야 한다.
- JSON 결과가 스키마 검증을 통과해야 한다.
- `roadway_riding`, `no_helmet` 등 다중 라벨이 secondary_situations에 반영되어야 한다.
- `unknown` 라벨 또는 `unknown` 위험도가 결과에 포함되지 않아야 한다.
- CLI에서 단일 영상 분석이 가능해야 한다.
- API에서 영상 업로드 분석이 가능해야 한다.

### 11.2 성능 기준

MVP 기준 성능 목표는 다음과 같다.

- 30초 이하 영상 1개 분석 시간이 로컬 GPU 환경에서 60초 이내
- CPU 환경에서는 mock 또는 소형 모델 테스트만 보장
- JSON 파싱 성공률 95% 이상
- 위험 상황 샘플에 대해 사람이 납득할 수 있는 근거 1개 이상 제공

### 11.3 안전 기준

- 분석 결과는 참고용이며, 법적 판단이나 처벌 판단으로 사용하지 않는다.
- 사람 얼굴, 차량 번호판 등 개인정보를 별도 저장하지 않는다.
- 모델 확신도가 낮으면 `needs_review` 또는 `needs_human_review: true`로 표시한다.
- `unknown`으로 회피하지 않고, 영상 품질 문제와 판단 불가 상황은 `needs_review`로 통일한다.
- `critical` 결과는 항상 `needs_human_review: true`로 처리한다.
- `no_helmet`은 얼굴 식별이 아니라 헬멧 착용 여부만 판단한다.

---

## 12. 환경 변수

`.env.example` 파일 예시:

```env
MODEL_NAME=Qwen/Qwen2.5-VL-3B-Instruct
DEVICE=auto
FRAME_SAMPLE_RATE=1
MAX_FRAMES=32
OUTPUT_DIR=data/outputs
LOG_LEVEL=INFO
```

---

## 13. 예상 구현 코드 인터페이스

### 13.1 분석 결과 스키마

```python
from enum import Enum
from pydantic import BaseModel, Field

class SituationLabel(str, Enum):
    normal_riding = "normal_riding"
    roadway_riding = "roadway_riding"
    no_helmet = "no_helmet"
    pedestrian_risk = "pedestrian_risk"
    vehicle_risk = "vehicle_risk"
    obstacle_risk = "obstacle_risk"
    intersection_crosswalk = "intersection_crosswalk"
    sudden_stop_or_evasion = "sudden_stop_or_evasion"
    fall_or_crash = "fall_or_crash"
    poor_road_condition = "poor_road_condition"
    sidewalk_riding = "sidewalk_riding"
    wrong_way_or_lane_issue = "wrong_way_or_lane_issue"
    needs_review = "needs_review"

class RiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"

class SituationAnalysis(BaseModel):
    primary_situation: SituationLabel
    secondary_situations: list[SituationLabel] = Field(default_factory=list)
    risk_level: RiskLevel
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list)
    recommended_action: str
    needs_human_review: bool = False
```

### 13.2 모델 인터페이스

```python
from abc import ABC, abstractmethod
from pathlib import Path
from src.analysis.schema import SituationAnalysis

class VideoSituationModel(ABC):
    @abstractmethod
    def analyze(self, video_path: Path) -> SituationAnalysis:
        pass
```

---

## 14. 평가 방법

### 14.1 정성 평가

각 샘플 영상에 대해 다음 항목을 사람이 확인한다.

- 상황 라벨이 적절한가?
- 위험도 판단이 과장되거나 축소되지 않았는가?
- 근거가 실제 영상에서 관찰 가능한 내용인가?
- 불확실한 상황에서 `unknown`이 아니라 `needs_review`를 사용했는가?

### 14.2 정량 평가

라벨링된 데이터가 확보되면 다음 지표를 계산한다.

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- 위험 상황 탐지 Recall

특히 `pedestrian_risk`, `vehicle_risk`, `fall_or_crash`, `roadway_riding`, `no_helmet` 라벨은 Recall을 우선 관리한다.

---

## 15. 향후 확장 계획

1. 실시간 웹캠 또는 블랙박스 스트리밍 분석
2. YOLO 계열 객체 탐지 모델과 결합
3. optical flow 기반 급정지/급회피 탐지
4. GPS, IMU, 속도 센서 데이터 결합
5. 사고 전후 구간 자동 하이라이트 생성
6. 라벨링 도구 연동
7. 모델별 정확도 및 속도 벤치마크 자동화
8. 온디바이스 추론 가능성 검토

---

## 16. ANTIGRAVITY 에이전트 완료 조건

ANTIGRAVITY 에이전트는 다음 조건을 만족하면 작업을 완료한 것으로 간주한다.

1. 프로젝트 디렉터리와 기본 파일이 생성되어 있다.
2. `README.md`에 설치 및 실행 방법이 작성되어 있다.
3. 영상 프레임 샘플링 코드가 동작한다.
4. mock model 기반으로 CLI 분석이 성공한다.
5. 실제 비전-언어 모델 연동 코드 초안이 존재한다.
6. Pydantic 스키마 검증이 적용되어 있다.
7. `pytest`가 통과한다.
8. 샘플 출력 JSON이 `data/outputs/`에 생성된다.

---

## 17. 첫 실행 명령 예시

```bash
git init
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch transformers accelerate opencv-python decord pydantic fastapi uvicorn python-dotenv rich pytest
python scripts/run_inference.py --video data/samples/sample.mp4 --model mock
pytest
```

---

## 18. 주의사항

- 모델이 “보이는 것처럼 추정”하는 내용을 사실처럼 단정하지 않도록 한다.
- 법규 위반 여부는 국가, 지역, 도로 조건에 따라 달라질 수 있으므로 본 시스템은 법적 판정 기능을 제공하지 않는다.
- 실제 서비스에 적용하기 전에는 충분한 데이터셋과 전문가 검증이 필요하다.
- 오픈소스 모델 사용 시 모델, 코드, 데이터셋 각각의 라이선스를 확인한다.
