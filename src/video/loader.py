"""
영상 로더 모듈
"""
from pathlib import Path
import cv2

SUPPORTED_FORMATS = {".mp4", ".mov", ".avi", ".mkv"}


def validate_video_path(video_path: Path | str) -> Path:
    """영상 파일 경로를 검증하고 Path 객체로 반환한다."""
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    if video_path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"지원하지 않는 형식: {video_path.suffix}. "
            f"지원 형식: {', '.join(SUPPORTED_FORMATS)}"
        )

    return video_path


def is_valid_video(video_path: Path | str) -> bool:
    """영상 파일이 정상적으로 열리는지 확인한다."""
    try:
        path = validate_video_path(video_path)
        cap = cv2.VideoCapture(str(path))
        ok = cap.isOpened()
        cap.release()
        return ok
    except (FileNotFoundError, ValueError):
        return False
