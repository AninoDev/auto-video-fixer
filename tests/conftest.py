"""Auto Video Fixer - Test fixtures and helpers."""

import subprocess

import pytest


@pytest.fixture
def tmp_video_file(tmp_path):
    """Create a temporary test video file using FFmpeg."""
    video_path = tmp_path / "test_video.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=2:size=320x240:rate=24",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )

    return str(video_path)


@pytest.fixture
def tmp_video_directory(tmp_path):
    """Create a temporary directory with test video files."""
    videos = []
    for i in range(3):
        video_path = tmp_path / f"video_{i}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=24",
                "-c:v",
                "libx264",
                str(video_path),
            ],
            capture_output=True,
            check=True,
        )
        videos.append(str(video_path))

    return tmp_path, videos


@pytest.fixture
def multi_scene_video(tmp_path):
    """Create a small 3-scene test video (~6s) with hard cuts between visually
    distinct patterns, for scene-detection/scene-mode tests. Scene boundaries
    are at ~2s and ~4s.
    """
    video_path = tmp_path / "multi_scene.mp4"
    segments = []
    for i, src in enumerate(
        [
            "testsrc2=size=320x240:rate=24",
            "color=c=blue:size=320x240:rate=24",
            "smptebars=size=320x240:rate=24",
        ]
    ):
        seg_path = tmp_path / f"seg_{i}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                src,
                "-t",
                "2",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(seg_path),
            ],
            capture_output=True,
            check=True,
        )
        segments.append(seg_path)

    list_path = tmp_path / "concat_list.txt"
    list_path.write_text("".join(f"file '{p}'\n" for p in segments))
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        capture_output=True,
        check=True,
    )
    return str(video_path)


@pytest.fixture
def sample_config(tmp_path):
    """Provide a sample configuration for testing, isolated from the user's real config file."""
    from autovideofixer.config import Config

    config = Config(tmp_path / "nonexistent.yaml")
    config.set(1, "general", "max_concurrent_jobs")
    config.set(18, "encoding", "crf")
    return config


@pytest.fixture
def sample_video_info():
    """Provide sample video metadata for testing."""
    return {
        "filepath": "/test/video.mp4",
        "filename": "video.mp4",
        "duration": 120.5,
        "resolution": (1920, 1080),
        "width": 1920,
        "height": 1080,
        "framerate": 30.0,
        "has_video": True,
        "has_audio": True,
        "is_hdr": False,
        "video_codec": "h264",
        "audio_codecs": ["aac"],
        "bit_rate": 5000000,
    }
