"""
테스트: Mock 모델 기반 분류기 파이프라인 검증
"""
import pytest
from pathlib import Path
from unittest.mock import patch

from src.models.mock_model import MockModel
from src.analysis.schema import SituationAnalysis, SituationLabel, RiskLevel


def test_mock_model_returns_situation_analysis():
    """MockModel이 SituationAnalysis를 반환하는지 확인"""
    model = MockModel(fixed_index=0)
    result = model.analyze(Path("dummy.mp4"))
    assert isinstance(result, SituationAnalysis)


def test_mock_model_all_responses_valid():
    """MockModel의 모든 응답이 스키마를 통과하는지 확인"""
    from src.models.mock_model import _MOCK_RESPONSES
    for resp in _MOCK_RESPONSES:
        assert isinstance(resp, SituationAnalysis)
        assert resp.confidence >= 0.0
        assert resp.confidence <= 1.0
        assert resp.primary_situation is not None


def test_mock_model_name():
    model = MockModel()
    assert model.model_name == "mock"


def test_mock_model_fixed_index():
    """fixed_index로 항상 같은 결과가 나오는지 확인"""
    model = MockModel(fixed_index=1)
    result1 = model.analyze(Path("a.mp4"))
    result2 = model.analyze(Path("b.mp4"))
    assert result1.primary_situation == result2.primary_situation
    assert result1.risk_level == result2.risk_level


def test_mock_model_no_unknown_label():
    """Mock 모델 결과에 'unknown' 라벨이 없는지 확인"""
    from src.models.mock_model import _MOCK_RESPONSES
    for resp in _MOCK_RESPONSES:
        assert resp.primary_situation.value != "unknown"
        for s in resp.secondary_situations:
            assert s.value != "unknown"
        assert resp.risk_level.value != "unknown"
