"""
VideoLLaMA3 모델 추론 모듈
로컬 경로에서 모델을 로드하여 오프라인 동작하거나 원격에서 가중치를 로드하여 작동한다.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

# VideoLLaMA3 compatibility patch for newer transformers versions
try:
    import transformers.image_utils
    if not hasattr(transformers.image_utils, "VideoInput"):
        from transformers.video_utils import VideoInput
        transformers.image_utils.VideoInput = VideoInput
except ImportError:
    pass

try:
    import transformers.dynamic_module_utils as dynamic_utils
    if hasattr(dynamic_utils, "get_class_from_dynamic_module"):
        original_get_class = dynamic_utils.get_class_from_dynamic_module

        def patched_get_class(*args, **kwargs):
            class_obj = original_get_class(*args, **kwargs)
            if class_obj and getattr(class_obj, "__name__", None) == "Videollama3Qwen2Processor":
                original_get_args = class_obj._get_arguments_from_pretrained
                
                @classmethod
                def robust_get_args(cls, pretrained_model_name_or_path, *args_inner, **kwargs_inner):
                    # Newer transformers pass processor_dict as a 2nd positional argument,
                    # but VideoLLaMA3's custom processor takes only (pretrained_model_name_or_path, **kwargs)
                    return original_get_args(pretrained_model_name_or_path, **kwargs_inner)
                
                class_obj._get_arguments_from_pretrained = robust_get_args
            return class_obj

        dynamic_utils.get_class_from_dynamic_module = patched_get_class
except Exception:
    pass

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


class VideoLLaMA3Model(VideoSituationModel):
    """
    VideoLLaMA3 기반 영상 상황 분석 모델.
    로컬 경로 또는 HuggingFace Hub에서 모델을 로드하여 동작한다.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        max_new_tokens: int = 512,
        n_frames: int = 16,
    ):
        """
        Args:
            model_path: 로컬 모델 경로. None이면 기본 "DAMO-NLP-SG/VideoLLaMA3-2B" 레포 사용.
            device: 'auto', 'cuda', 'cpu'
            max_new_tokens: 최대 생성 토큰 수
            n_frames: 영상에서 샘플링할 최대 프레임 수
        """
        self._model_path = Path(model_path) if model_path else None
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._n_frames = n_frames
        self._model = None
        self._processor = None

        self._load_model()

    def _load_model(self):
        """모델과 프로세서를 로드한다."""
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        path_str = str(self._model_path) if self._model_path else "DAMO-NLP-SG/VideoLLaMA3-2B"

        from rich.console import Console
        console = Console()
        console.print(f"[cyan]VideoLLaMA3 모델 로딩: {path_str}[/cyan]")

        local_files = False
        if self._model_path and self._model_path.exists():
            local_files = True

        self._processor = AutoProcessor.from_pretrained(
            path_str,
            trust_remote_code=True,
            local_files_only=local_files,
        )

        # GPU 가속 및 정밀도 설정
        if torch.cuda.is_available() and self._device != "cpu":
            dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
            device_map = "auto" if self._device == "auto" else self._device
        else:
            dtype = torch.float32
            attn_impl = "sdpa"
            device_map = None

        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                path_str,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation=attn_impl,
                local_files_only=local_files,
            )
        except Exception as e:
            # flash_attention_2 미지원/미설치 대비 fallback
            console.print(f"[yellow]attn_impl='{attn_impl}' 실패 ({e}). 기본 sdpa로 재시도합니다.[/yellow]")
            self._model = AutoModelForCausalLM.from_pretrained(
                path_str,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
                attn_implementation="sdpa",
                local_files_only=local_files,
            )

        console.print(f"[green]VideoLLaMA3 모델 로드 완료[/green]")

    @property
    def model_name(self) -> str:
        return f"videollama3:{self._model_path.name if self._model_path else '2B'}"

    def analyze(self, video_path: Path) -> SituationAnalysis:
        """영상을 분석하여 상황 분류 결과를 반환한다."""
        import torch

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

        # VideoLLaMA3 대화 입력 구성
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": {
                            "video_path": str(video_path),
                            "fps": 1,
                            "max_frames": self._n_frames,
                        }
                    },
                    {"type": "text", "text": SYSTEM_PROMPT},
                ],
            }
        ]

        # 입력 전처리
        inputs = self._processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        
        # 디바이스 이동
        inputs = inputs.to(self._model.device)

        # pixel_values 정밀도 일치
        if "pixel_values" in inputs and inputs["pixel_values"] is not None:
            inputs["pixel_values"] = inputs["pixel_values"].to(self._model.dtype)

        # 추론
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
            )

        # 디코딩
        output_text = self._processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()

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
