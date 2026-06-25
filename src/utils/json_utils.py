"""
JSON 파싱 유틸리티
모델 출력에서 JSON을 추출하고 파싱 실패 시 재시도 로직을 제공한다.
"""
import json
import re
from typing import Any


def extract_json_from_text(text: str) -> dict[str, Any]:
    """
    텍스트에서 JSON 블록을 추출한다.
    모델이 설명과 함께 JSON을 출력하는 경우를 처리한다.
    """
    # 1. 코드 블록 안의 JSON 시도
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if code_block:
        return json.loads(code_block.group(1))

    # 2. 텍스트에서 중괄호로 감싸진 JSON 추출
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        return json.loads(brace_match.group())

    # 3. 전체 텍스트를 JSON으로 파싱 시도
    return json.loads(text.strip())


def safe_parse_json(text: str, max_retries: int = 3) -> dict[str, Any] | None:
    """
    안전한 JSON 파싱. 실패해도 예외를 올리지 않고 None을 반환한다.

    Args:
        text: 파싱할 텍스트
        max_retries: 정규식 정리 후 재시도 횟수

    Returns:
        파싱된 딕셔너리 또는 None
    """
    for attempt in range(max_retries):
        try:
            return extract_json_from_text(text)
        except (json.JSONDecodeError, AttributeError):
            if attempt == 0:
                # 1차 정리: 줄바꿈 제거 후 재시도
                text = " ".join(text.split())
            elif attempt == 1:
                # 2차 정리: 마지막 쉼표 제거 후 재시도 (trailing comma)
                text = re.sub(r",\s*([}\]])", r"\1", text)
    return None
