"""
테스트: Pydantic 스키마 검증
"""
import pytest
from pydantic import ValidationError
from src.analysis.schema import SituationAnalysis, SituationLabel, RiskLevel


def test_valid_normal_riding():
    result = SituationAnalysis(
        primary_situation=SituationLabel.normal_riding,
        secondary_situations=[],
        risk_level=RiskLevel.low,
        confidence=0.9,
        evidence=["정상 주행 확인"],
        recommended_action="현재 주행 유지",
    )
    assert result.primary_situation == SituationLabel.normal_riding
    assert result.needs_human_review is False


def test_critical_forces_needs_human_review():
    """critical 위험도는 자동으로 needs_human_review=True"""
    result = SituationAnalysis(
        primary_situation=SituationLabel.fall_or_crash,
        secondary_situations=[],
        risk_level=RiskLevel.critical,
        confidence=0.91,
        evidence=["넘어짐 감지"],
        recommended_action="즉시 멈춤",
        needs_human_review=False,  # 명시적으로 False 넣어도 True로 강제
    )
    assert result.needs_human_review is True


def test_normal_riding_with_secondary_situations_fails():
    """normal_riding + secondary_situations 조합은 유효하지 않음"""
    with pytest.raises(ValidationError):
        SituationAnalysis(
            primary_situation=SituationLabel.normal_riding,
            secondary_situations=[SituationLabel.no_helmet],
            risk_level=RiskLevel.low,
            confidence=0.5,
            evidence=[],
            recommended_action="없음",
        )


def test_confidence_out_of_range_fails():
    """confidence는 0.0~1.0 범위를 벗어나면 실패"""
    with pytest.raises(ValidationError):
        SituationAnalysis(
            primary_situation=SituationLabel.vehicle_risk,
            secondary_situations=[],
            risk_level=RiskLevel.high,
            confidence=1.5,  # 범위 초과
            evidence=[],
            recommended_action="감속",
        )


def test_multi_label_secondary_situations():
    """다중 라벨이 secondary_situations에 정상 등록되는지 확인"""
    result = SituationAnalysis(
        primary_situation=SituationLabel.vehicle_risk,
        secondary_situations=[
            SituationLabel.roadway_riding,
            SituationLabel.no_helmet,
        ],
        risk_level=RiskLevel.high,
        confidence=0.81,
        evidence=["전방 차량 접근"],
        recommended_action="감속 및 주변 확인",
    )
    assert SituationLabel.roadway_riding in result.secondary_situations
    assert SituationLabel.no_helmet in result.secondary_situations


def test_to_summary_output():
    """to_summary()가 문자열을 반환하는지 확인"""
    result = SituationAnalysis(
        primary_situation=SituationLabel.pedestrian_risk,
        secondary_situations=[SituationLabel.no_helmet],
        risk_level=RiskLevel.high,
        confidence=0.78,
        evidence=["보행자 근접"],
        recommended_action="감속",
    )
    summary = result.to_summary()
    assert isinstance(summary, str)
    assert "pedestrian_risk" in summary
    assert "no_helmet" in summary
