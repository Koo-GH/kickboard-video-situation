"""
테스트: 프레임 샘플러 (영상 파일 없이 mock 테스트)
"""
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.video.sampler import sample_frames_uniform, sample_frames_count
from src.video.metadata import VideoMetadata


def make_mock_metadata(fps=30.0, total_frames=300, width=1280, height=720) -> VideoMetadata:
    return VideoMetadata(
        path=Path("dummy.mp4"),
        fps=fps,
        total_frames=total_frames,
        width=width,
        height=height,
        duration_sec=total_frames / fps,
    )


def make_fake_frame(width=1280, height=720) -> np.ndarray:
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


@patch("cv2.VideoCapture")
def test_sample_frames_uniform_basic(mock_cap_class):
    """균등 샘플링이 max_frames 이하로 반환되는지 확인"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, make_fake_frame())
    mock_cap_class.return_value = mock_cap

    meta = make_mock_metadata(fps=30.0, total_frames=300)
    frames = sample_frames_uniform(
        Path("dummy.mp4"), meta, sample_rate=1, max_frames=16
    )

    assert len(frames) <= 16
    assert all(isinstance(f, np.ndarray) for f in frames)
    assert all(f.shape[2] == 3 for f in frames)


@patch("cv2.VideoCapture")
def test_sample_frames_count_exact(mock_cap_class):
    """count 기반 샘플링이 정확히 n_frames를 반환하는지 확인"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 90  # total frames
    mock_cap.read.return_value = (True, make_fake_frame())
    mock_cap_class.return_value = mock_cap

    frames = sample_frames_count(Path("dummy.mp4"), n_frames=8)
    assert len(frames) == 8


@patch("cv2.VideoCapture")
def test_sample_frames_uniform_max_frames_limit(mock_cap_class):
    """max_frames 제한이 실제로 동작하는지 확인"""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, make_fake_frame())
    mock_cap_class.return_value = mock_cap

    meta = make_mock_metadata(fps=30.0, total_frames=3000)
    frames = sample_frames_uniform(
        Path("dummy.mp4"), meta, sample_rate=1, max_frames=32
    )
    assert len(frames) <= 32
