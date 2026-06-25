"""
모델 공통 인터페이스 (추상 기반 클래스)
"""
from abc import ABC, abstractmethod
from pathlib import Path
from src.analysis.schema import SituationAnalysis


class VideoSituationModel(ABC):
    """비디오 상황 분석 모델의 공통 인터페이스"""

    @abstractmethod
    def analyze(self, video_path: Path) -> SituationAnalysis:
        """
        영상을 분석하여 상황 분류 결과를 반환한다.

        Args:
            video_path: 영상 파일 경로

        Returns:
            SituationAnalysis: 검증된 분석 결과
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """모델 이름 반환"""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"
