"""
프레임 샘플링 모듈
균등 샘플링 방식으로 프레임을 추출한다.
"""
from pathlib import Path
import cv2
import numpy as np
from src.video.metadata import VideoMetadata


def sample_frames_uniform(
    video_path: Path | str,
    metadata: VideoMetadata,
    sample_rate: int = 1,
    max_frames: int = 32,
) -> list[np.ndarray]:
    """
    균등 간격으로 프레임을 샘플링한다.

    Args:
        video_path: 영상 파일 경로
        metadata: VideoMetadata 객체
        sample_rate: 초당 추출할 프레임 수 (기본 1fps)
        max_frames: 최대 프레임 수 제한

    Returns:
        RGB numpy array 리스트 (H, W, 3)
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")

    try:
        # 추출할 프레임 인덱스 계산
        step = max(1, int(metadata.fps / sample_rate))
        frame_indices = list(range(0, metadata.total_frames, step))

        # max_frames 제한 적용 (균등하게 선택)
        if len(frame_indices) > max_frames:
            indices_arr = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)
            frame_indices = [frame_indices[i] for i in indices_arr]

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # BGR → RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
    finally:
        cap.release()

    return frames


def sample_frames_count(
    video_path: Path | str,
    n_frames: int = 8,
) -> list[np.ndarray]:
    """
    영상에서 n_frames개를 균등하게 샘플링한다.
    (모델 입력이 프레임 수를 지정할 때 사용)
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 파일을 열 수 없습니다: {video_path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()

        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
    finally:
        cap.release()

    return frames
