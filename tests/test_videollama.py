"""
테스트: VideoLLaMA3Model 인터페이스 및 모크 연동 테스트
"""
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.models.videollama import VideoLLaMA3Model
from src.analysis.schema import SituationAnalysis


@patch("transformers.AutoProcessor.from_pretrained")
@patch("transformers.AutoModelForCausalLM.from_pretrained")
def test_videollama_init_and_analyze(mock_model_from_pretrained, mock_processor_from_pretrained):
    """VideoLLaMA3Model의 초기화 및 추론이 모크를 통해 정상 호출되는지 테스트"""
    # 모크 설정
    mock_processor = MagicMock()
    mock_processor_from_pretrained.return_value = mock_processor
    
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.dtype = "float32"
    mock_model_from_pretrained.return_value = mock_model
    
    # 모델 출력물 모킹
    mock_processor.batch_decode.return_value = [
        """{
            "primary_situation": "normal_riding",
            "secondary_situations": [],
            "risk_level": "low",
            "confidence": 0.9,
            "evidence": ["정상 주행"],
            "recommended_action": "현재 주행 유지",
            "needs_human_review": false
        }"""
    ]
    
    # 모델 인스턴스화
    model = VideoLLaMA3Model(model_path="./dummy_model", device="cpu")
    assert model.model_name == "videollama3:dummy_model"
    
    # analyze 실행
    with patch("pathlib.Path.exists", return_value=True):
        result = model.analyze(Path("dummy.mp4"))
        
    assert isinstance(result, SituationAnalysis)
    assert result.primary_situation.value == "normal_riding"
