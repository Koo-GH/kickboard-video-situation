"""
출력 스키마 정의
Pydantic v2 기반. unknown 라벨/위험도를 허용하지 않는다.
"""
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, model_validator


class SituationLabel(str, Enum):
    normal_riding = "normal_riding"
    roadway_riding = "roadway_riding"
    no_helmet = "no_helmet"
    multiple_riders = "multiple_riders"
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

    @model_validator(mode="after")
    def validate_critical_review(self) -> "SituationAnalysis":
        """critical 결과는 반드시 needs_human_review=True"""
        if self.risk_level == RiskLevel.critical:
            self.needs_human_review = True
        return self

    @model_validator(mode="after")
    def validate_normal_riding_exclusive(self) -> "SituationAnalysis":
        """normal_riding은 다른 위험 라벨과 함께 쓰지 않는다."""
        if self.primary_situation == SituationLabel.normal_riding:
            if self.secondary_situations:
                raise ValueError(
                    "normal_riding은 secondary_situations와 함께 사용할 수 없습니다."
                )
        return self

    def to_summary(self) -> str:
        """사람이 읽을 수 있는 요약 문장 생성"""
        lines = [
            f"[상황] {self.primary_situation.value}",
            f"[위험도] {self.risk_level.value} | [신뢰도] {self.confidence:.2f}",
        ]
        if self.secondary_situations:
            secs = ", ".join(s.value for s in self.secondary_situations)
            lines.append(f"[추가상황] {secs}")
        if self.evidence:
            lines.append("[근거]")
            for e in self.evidence:
                lines.append(f"  - {e}")
        lines.append(f"[권장조치] {self.recommended_action}")
        if self.needs_human_review:
            lines.append("[!] 사람 검토 필요")
        return "\n".join(lines)
