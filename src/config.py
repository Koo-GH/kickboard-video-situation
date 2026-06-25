"""
설정 관리 모듈
환경변수 또는 .env 파일에서 설정을 로드한다.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 모델 설정
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    model_path: str = "./models/Qwen2.5-VL-3B-Instruct"
    device: str = "auto"

    # 프레임 샘플링
    frame_sample_rate: int = 1   # 초당 프레임 수
    max_frames: int = 32

    # 출력
    output_dir: str = "data/outputs"

    # 로깅
    log_level: str = "INFO"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def model_local_path(self) -> Path:
        return Path(self.model_path)


settings = Settings()
