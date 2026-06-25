"""
Mock 모델 (GPU 없는 환경 또는 파이프라인 테스트용)
실제 추론 없이 고정된 예시 결과를 반환한다.
"""
import random
from pathlib import Path
from src.models.base import VideoSituationModel
from src.analysis.schema import SituationAnalysis, SituationLabel, RiskLevel


_MOCK_RESPONSES = [
    SituationAnalysis(
        primary_situation=SituationLabel.normal_riding,
        secondary_situations=[],
        risk_level=RiskLevel.low,
        confidence=0.85,
        evidence=["영상에서 특이한 위험 요소가 관찰되지 않음", "주행 속도가 안정적임"],
        recommended_action="현재 주행 유지",
        needs_human_review=False,
    ),
    SituationAnalysis(
        primary_situation=SituationLabel.pedestrian_risk,
        secondary_situations=[SituationLabel.roadway_riding, SituationLabel.no_helmet],
        risk_level=RiskLevel.high,
        confidence=0.78,
        evidence=[
            "영상 전방에 보행자가 가까이 있음",
            "킥보드가 차도 영역에서 주행 중임",
            "운전자의 헬멧 착용이 확인되지 않음",
        ],
        recommended_action="감속 또는 정지 필요",
        needs_human_review=False,
    ),
    SituationAnalysis(
        primary_situation=SituationLabel.vehicle_risk,
        secondary_situations=[SituationLabel.roadway_riding],
        risk_level=RiskLevel.high,
        confidence=0.72,
        evidence=["전방 오른쪽에서 차량이 근접", "킥보드 진행 방향과 차량 경로가 가까움"],
        recommended_action="감속 및 주변 확인 필요",
        needs_human_review=False,
    ),
    SituationAnalysis(
        primary_situation=SituationLabel.needs_review,
        secondary_situations=[],
        risk_level=RiskLevel.medium,
        confidence=0.40,
        evidence=["영상 품질이 낮아 상황 판단이 어려움"],
        recommended_action="사람이 직접 영상을 검토해야 함",
        needs_human_review=True,
    ),
]


class MockModel(VideoSituationModel):
    """
    테스트용 Mock 모델.
    실제 추론 없이 사전 정의된 결과를 반환한다.
    """

    def __init__(self, fixed_index: int | None = None):
        """
        Args:
            fixed_index: 고정 응답 인덱스. None이면 랜덤 선택.
        """
        self._fixed_index = fixed_index

    @property
    def model_name(self) -> str:
        return "mock"

    def analyze(self, video_path: Path) -> SituationAnalysis:
        """Mock 분석 결과 반환 (실제 영상을 읽지 않음)"""
        if self._fixed_index is not None:
            idx = self._fixed_index % len(_MOCK_RESPONSES)
        else:
            idx = random.randint(0, len(_MOCK_RESPONSES) - 1)
        return _MOCK_RESPONSES[idx]
