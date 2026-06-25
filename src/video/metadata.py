"""
영상 메타데이터 추출
"""
from pathlib import Path
from dataclasses import dataclass
import cv2


@dataclass
class VideoMetadata:
    path: Path
    fps: float
    total_frames: int
    width: int
    height: int
    duration_sec: float

    def __str__(self) -> str:
        return (
            f"VideoMetadata(fps={self.fps:.2f}, frames={self.total_frames}, "
            f"size={self.width}x{self.height}, duration={self.duration_sec:.1f}s)"
        )


def extract_metadata(video_path: Path | str) -> VideoMetadata:
    """영상 파일에서 메타데이터를 추출한다."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = total_frames / fps if fps > 0 else 0.0
    finally:
        cap.release()

    return VideoMetadata(
        path=video_path,
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_sec=duration_sec,
    )
