"""Real-ffmpeg tests for REQUIREMENTS.md § 22 (normalize_codecs + container safety).

Every ffmpeg command here runs against the real binary on tiny synthetic
lavfi media and asserts on ffprobe output (codec, container, frame count) --
the project's mocked ffmpeg tests have hidden broken invocations three times.

Run with: pytest tests/integration/test_normalize_codecs_e2e.py -v -m integration
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from autovideofixer.config import Config
from autovideofixer.core.pipeline import Pipeline
from autovideofixer.core.stages.base import StageStatus
from autovideofixer.core.stages.normalize_codecs import NormalizeCodecsStage
from autovideofixer.core.stages.retime import RetimeStage
from autovideofixer.core.stages.upscale import UpscaleStage

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not installed",
    ),
]


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


def _probe(path: str) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-show_entries",
            "stream=codec_type,codec_name,pix_fmt,nb_read_frames,color_transfer,channel_layout"
            ":format=format_name,duration",
            "-of", "json", path,
        ],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    data = json.loads(out)
    data["video"] = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    data["audio"] = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    return data


def _make_webm(path: str, acodec: str, *, vf: str = "null", pix_fmt: str = "yuv420p") -> None:
    _ff(
        "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=2",
        "-f", "lavfi", "-i", "sine=f=440:d=2",
        "-vf", f"{vf},format={pix_fmt}", "-c:v", "libvpx-vp9", "-b:v", "300k",
        "-c:a", acodec, "-shortest", path,
    )  # fmt: skip


def _stage(tmp_path) -> NormalizeCodecsStage:
    return NormalizeCodecsStage(Config(tmp_path / "nonexistent.yaml"))


@pytest.mark.parametrize("acodec", ["libopus", "libvorbis"])
def test_vp9_webm_normalized_to_h264_aac(tmp_path, acodec):
    src = str(tmp_path / "in.webm")
    _make_webm(src, acodec)
    out = str(tmp_path / "out.mkv")
    result = _stage(tmp_path).execute(src, out)
    assert result.status == StageStatus.COMPLETED, result.error
    p = _probe(out)
    assert p["video"]["codec_name"] == "h264"
    assert p["video"]["pix_fmt"] == "yuv420p"
    assert p["audio"]["codec_name"] == "aac"
    assert int(p["video"]["nb_read_frames"]) == int(_probe(src)["video"]["nb_read_frames"]) == 48
    assert "matroska" in p["format"]["format_name"]
    assert [s["copied"] for s in result.metadata["streams"]] == [False, False]


def test_rgb_vp9_source_is_normalized(tmp_path):
    # VP9 stores RGB input as gbrp with matrix "gbr"; carrying that tag onto the
    # yuv420p libx264 encode made ffmpeg refuse to open the output.
    src = str(tmp_path / "rgb.webm")
    _make_webm(src, "libvorbis", pix_fmt="gbrp")
    assert _probe(src)["video"]["pix_fmt"] == "gbrp"
    out = str(tmp_path / "out.mkv")
    result = _stage(tmp_path).execute(src, out)
    assert result.status == StageStatus.COMPLETED, result.error
    p = _probe(out)
    assert p["video"]["codec_name"] == "h264"
    assert p["video"]["pix_fmt"] == "yuv420p"
    assert int(p["video"]["nb_read_frames"]) == 48


def test_h264_aac_is_skipped_with_no_file(tmp_path):
    src = str(tmp_path / "in.mp4")
    _ff(
        "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=1", "-f", "lavfi", "-i", "sine=d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src,
    )  # fmt: skip
    out = str(tmp_path / "out.mkv")
    result = _stage(tmp_path).execute(src, out)
    assert result.status == StageStatus.SKIPPED
    assert not os.path.exists(out)


def test_audio_only_transcode_copies_video(tmp_path):
    src = str(tmp_path / "in.mkv")
    _ff(
        "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=1", "-f", "lavfi", "-i", "sine=d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "libvorbis", "-shortest", src,
    )  # fmt: skip
    out = str(tmp_path / "out.mkv")
    result = _stage(tmp_path).execute(src, out)
    assert result.status == StageStatus.COMPLETED, result.error
    assert result.metadata["streams"][0]["copied"] is True
    p = _probe(out)
    assert (p["video"]["codec_name"], p["audio"]["codec_name"]) == ("h264", "aac")
    assert int(p["video"]["nb_read_frames"]) == 24


def test_ten_bit_hdr_tags_preserved_and_surround_layout_kept(tmp_path):
    src = str(tmp_path / "hdr.webm")
    _ff(
        "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=2",
        "-f", "lavfi", "-i", "sine=f=440:d=2",
        "-af", "pan=5.1|FL=c0|FR=c0|FC=c0|LFE=c0|BL=c0|BR=c0",
        "-vf", "format=yuv420p10le,setparams=color_primaries=bt2020:color_trc=smpte2084"
        ":colorspace=bt2020nc:range=tv",
        "-c:v", "libvpx-vp9", "-b:v", "300k", "-c:a", "libopus", "-shortest", src,
    )  # fmt: skip
    out = str(tmp_path / "out.mkv")
    result = _stage(tmp_path).execute(src, out)
    assert result.status == StageStatus.COMPLETED, result.error
    p = _probe(out)
    assert p["video"]["codec_name"] == "h264"
    assert p["video"]["pix_fmt"] == "yuv420p10le"
    assert p["video"]["color_transfer"] == "smpte2084"
    assert p["audio"]["codec_name"] == "aac"
    assert p["audio"]["channel_layout"] == "5.1"


def test_odd_dimension_source_is_accepted(tmp_path):
    src = str(tmp_path / "odd.mkv")
    _ff("-f", "lavfi", "-i", "testsrc2=s=321x241:r=24:d=1,format=yuv444p", "-c:v", "ffv1", src)
    assert _probe(src)["video"]["codec_name"] == "ffv1"
    out = str(tmp_path / "out.mkv")
    result = _stage(tmp_path).execute(src, out)
    assert result.status == StageStatus.COMPLETED, result.error
    p = _probe(out)
    assert p["video"]["codec_name"] == "h264"
    assert int(p["video"]["nb_read_frames"]) == 24


def test_vfr_retime_output_keeps_its_dropped_frames(tmp_path):
    """REQUIREMENTS.md § 12.4: -fps_mode passthrough must not re-conform a VFR
    intermediate to CFR (that would re-insert retime's 72 dropped frames)."""
    src = str(tmp_path / "padded.webm")
    _ff(
        "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=2", "-f", "lavfi", "-i", "sine=d=2",
        "-vf", "fps=60", "-c:v", "libvpx-vp9", "-b:v", "500k", "-c:a", "libopus",
        "-shortest", src,
    )  # fmt: skip
    assert int(_probe(src)["video"]["nb_read_frames"]) == 120
    config = Config(tmp_path / "nonexistent.yaml")
    retimed = str(tmp_path / "retimed.mkv")
    rt = RetimeStage(config).execute(src, retimed)
    assert rt.status == StageStatus.COMPLETED, rt.skipped_reason or rt.error
    assert int(_probe(retimed)["video"]["nb_read_frames"]) == 48
    out = str(tmp_path / "norm.mkv")
    result = NormalizeCodecsStage(config).execute(retimed, out)
    assert result.status == StageStatus.COMPLETED, result.error
    p = _probe(out)
    assert int(p["video"]["nb_read_frames"]) == 48  # NOT 120
    assert p["audio"]["codec_name"] == "aac"
    assert abs(float(p["format"]["duration"]) - 2.0) < 0.1


class TestPromoteTempToOutput:
    """``Pipeline._promote_temp_to_output``: an .mkv temp landing on an .mp4
    output must be a REAL MP4 holding MP4-standard codecs (§ 22.3)."""

    def _pipeline(self, tmp_path) -> Pipeline:
        return Pipeline(Config(tmp_path / "nonexistent.yaml"))

    def _mkv(self, path: str, acodec: str, vcodec: str = "libx264") -> None:
        extra = ["-pix_fmt", "yuv420p"] if vcodec == "libx264" else []
        _ff(
            "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=1",
            "-f", "lavfi", "-i", "sine=d=1", "-c:v", vcodec, *extra, "-c:a", acodec,
            "-shortest", path,
        )  # fmt: skip

    def test_mkv_to_mp4_is_a_real_mp4_copy(self, tmp_path):
        temp, out = str(tmp_path / ".avf_x.mkv"), str(tmp_path / "o.mp4")
        self._mkv(temp, "aac")
        landed = self._pipeline(tmp_path)._promote_temp_to_output(temp, out)
        assert landed == out and not os.path.exists(temp)
        p = _probe(out)
        assert "mp4" in p["format"]["format_name"]
        assert (p["video"]["codec_name"], p["audio"]["codec_name"]) == ("h264", "aac")
        assert int(p["video"]["nb_read_frames"]) == 24

    def test_vorbis_audio_is_transcoded_not_smuggled_into_mp4(self, tmp_path):
        temp, out = str(tmp_path / ".avf_x.mkv"), str(tmp_path / "o.mp4")
        self._mkv(temp, "libvorbis")
        landed = self._pipeline(tmp_path)._promote_temp_to_output(temp, out)
        assert landed == out
        p = _probe(out)
        assert "mp4" in p["format"]["format_name"]
        assert p["video"]["codec_name"] == "h264"  # video still copied
        assert p["audio"]["codec_name"] == "aac"

    def test_non_mp4_video_codec_is_transcoded(self, tmp_path):
        temp, out = str(tmp_path / ".avf_x.mkv"), str(tmp_path / "o.mp4")
        self._mkv(temp, "aac", vcodec="ffv1")
        landed = self._pipeline(tmp_path)._promote_temp_to_output(temp, out)
        assert landed == out
        p = _probe(out)
        assert (p["video"]["codec_name"], p["audio"]["codec_name"]) == ("h264", "aac")
        assert int(p["video"]["nb_read_frames"]) == 24

    def test_h264_mkv_to_webm_becomes_vp9_opus(self, tmp_path):
        temp, out = str(tmp_path / ".avf_x.mkv"), str(tmp_path / "o.webm")
        self._mkv(temp, "aac")
        landed = self._pipeline(tmp_path)._promote_temp_to_output(temp, out)
        assert landed == out
        p = _probe(out)
        assert (p["video"]["codec_name"], p["audio"]["codec_name"]) == ("vp9", "opus")
        assert int(p["video"]["nb_read_frames"]) == 24

    def test_same_extension_is_a_plain_move(self, tmp_path):
        temp, out = str(tmp_path / ".avf_x.mkv"), str(tmp_path / "o.mkv")
        self._mkv(temp, "libvorbis")
        landed = self._pipeline(tmp_path)._promote_temp_to_output(temp, out)
        assert landed == out and not os.path.exists(temp)
        assert _probe(out)["audio"]["codec_name"] == "vorbis"  # untouched

    def test_unremuxable_temp_keeps_honest_extension(self, tmp_path):
        temp, out = str(tmp_path / ".avf_x.mkv"), str(tmp_path / "o.mp4")
        with open(temp, "wb") as f:
            f.write(b"not a real container")
        landed = self._pipeline(tmp_path)._promote_temp_to_output(temp, out)
        assert landed == str(tmp_path / "o.mkv")  # never an MKV named .mp4
        assert not os.path.exists(out)


def test_upscale_resize_scratch_keeps_destination_container(tmp_path):
    """The resize scratch used to be a hardcoded .mp4 replaced onto a .mkv
    destination (MP4 bytes in a .mkv-named file; Vorbis audio rejected)."""
    src = str(tmp_path / "in.mkv")
    _ff(
        "-f", "lavfi", "-i", "testsrc2=s=320x240:r=24:d=1", "-f", "lavfi", "-i", "sine=d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "libvorbis", "-shortest", src,
    )  # fmt: skip
    out = str(tmp_path / "out.mkv")
    stage = UpscaleStage(Config(tmp_path / "nonexistent.yaml"))
    import time

    result = stage._resize_to_exact(src, out, 160, 120, time.time())
    assert result.status == StageStatus.COMPLETED, result.error
    p = _probe(out)
    assert "matroska" in p["format"]["format_name"]
    assert p["audio"]["codec_name"] == "vorbis"
    assert not [f for f in os.listdir(tmp_path) if "resize_tmp" in f]
