"""
Qwen2.5-VL 모델 추론 모듈
GPU 서버 또는 Windows GTX 1080에서 실행된다.
로컬 경로에서 모델을 로드한다 (HuggingFace 연결 불필요).
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from src.models.base import VideoSituationModel
from src.analysis.schema import SituationAnalysis, SituationLabel, RiskLevel
from src.utils.json_utils import safe_parse_json
from src.analysis.prompts import SYSTEM_PROMPT


FALLBACK_RESPONSE = SituationAnalysis(
    primary_situation=SituationLabel.needs_review,
    secondary_situations=[],
    risk_level=RiskLevel.medium,
    confidence=0.30,
    evidence=["모델 응답 파싱 실패로 사람 검토 필요"],
    recommended_action="영상을 직접 확인하세요.",
    needs_human_review=True,
)


class QwenVLModel(VideoSituationModel):
    """
    Qwen2.5-VL 기반 영상 상황 분석 모델.
    로컬 경로에서 모델을 로드하여 오프라인 동작한다.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        n_frames: int = 8,
    ):
        """
        Args:
            model_path: 로컬 모델 경로. None이면 환경변수 MODEL_PATH 사용.
            device: 'auto', 'cuda', 'cpu'
            max_new_tokens: 최대 생성 토큰 수
            n_frames: 영상에서 샘플링할 프레임 수
        """
        from src.config import settings

        self._model_path = Path(model_path or settings.model_path)
        self._device = device or settings.device
        self._max_new_tokens = max_new_tokens
        self._n_frames = n_frames
        self._model = None
        self._processor = None

        self._load_model()

    def _load_model(self):
        """모델과 프로세서를 로컬 경로에서 로드한다."""
        try:
            import torch
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        except ImportError as e:
            raise ImportError(
                f"torch 또는 transformers가 설치되지 않았습니다: {e}\n"
                "pip install torch transformers accelerate"
            ) from e

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"모델 경로를 찾을 수 없습니다: {self._model_path}\n"
                "python scripts/download_model.py 로 먼저 다운로드하세요."
            )

        from rich.console import Console
        console = Console()
        console.print(f"[cyan]모델 로딩: {self._model_path}[/cyan]")

        self._processor = AutoProcessor.from_pretrained(
            str(self._model_path),
            local_files_only=True,   # HuggingFace 연결 없이 로컬만 사용
        )
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self._model_path),
            torch_dtype=torch.float16,
            device_map=self._device,
            local_files_only=True,
        )
        console.print(f"[green]모델 로드 완료[/green]")

    @property
    def model_name(self) -> str:
        return f"qwen2_5_vl:{self._model_path.name}"

    def analyze(self, video_path: Path) -> SituationAnalysis:
        """영상을 분석하여 상황 분류 결과를 반환한다."""
        from qwen_vl_utils import process_vision_info
        import torch

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        # 메시지 구성 (비디오 입력)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_path),
                        "max_pixels": 360 * 420,
                        "nframes": self._n_frames,
                    },
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            }
        ]

        # 입력 전처리
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        # 추론
        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=self._max_new_tokens
            )
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return self._parse_response(output_text)

    def _parse_response(self, text: str) -> SituationAnalysis:
        """모델 텍스트 출력을 SituationAnalysis로 변환한다."""
        parsed = safe_parse_json(text)
        if parsed is None:
            return FALLBACK_RESPONSE

        try:
            return SituationAnalysis(**parsed)
        except Exception:
            return FALLBACK_RESPONSE
