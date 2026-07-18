"""Tests for FFmpeg utilities."""

import os
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from autovideofixer.core.ffmpeg_utils import (
    ProbeResult,
    StreamInfo,
    _parse_fps,
    detect_hardware_acceleration,
    generate_temp_path,
    get_ffmpeg_path,
    get_ffprobe_path,
    probe,
    resolve_hwaccel,
    run_ffmpeg,
)


class TestParseFps:
    """Test the avg_frame_rate-first framerate selection rule.

    avg_frame_rate is the honest "real frame count / real duration" rate.
    r_frame_rate ("tbr") is ffmpeg's declared/theoretical rate and can be a
    multiple of the true average for VFR or YouTube-origin sources (e.g.
    avg_frame_rate=29.64 vs r_frame_rate=59.94 for the same file). Any code
    that reads r_frame_rate where it should read avg_frame_rate will
    mis-time piped/raw video relative to real frame count.
    """

    def test_prefers_avg_frame_rate_over_r_frame_rate(self):
        # Real-world VFR/YouTube-origin case: r_frame_rate (tbr) is ~2x the
        # true average rate.
        s = {"avg_frame_rate": "2964/100", "r_frame_rate": "60000/1001"}
        assert _parse_fps(s) == pytest.approx(29.64, abs=0.01)

    def test_falls_back_to_r_frame_rate_when_avg_undefined(self):
        # ffprobe emits "0/0" for avg_frame_rate when duration is unknown.
        s = {"avg_frame_rate": "0/0", "r_frame_rate": "30/1"}
        assert _parse_fps(s) == pytest.approx(30.0)

    def test_falls_back_to_r_frame_rate_when_avg_missing(self):
        s = {"r_frame_rate": "24/1"}
        assert _parse_fps(s) == pytest.approx(24.0)

    def test_matching_rates_simple_case(self):
        s = {"avg_frame_rate": "30/1", "r_frame_rate": "30/1"}
        assert _parse_fps(s) == pytest.approx(30.0)

    def test_no_rate_info_returns_zero(self):
        s = {"avg_frame_rate": "0/0", "r_frame_rate": "0/0"}
        assert _parse_fps(s) == 0.0


class TestFFmpegDetection:
    """Test FFmpeg binary detection."""

    def test_get_ffmpeg_path(self):
        """Test finding FFmpeg binary."""
        path = get_ffmpeg_path()
        assert path is not None
        assert os.path.isfile(path)

    def test_get_ffprobe_path(self):
        """Test finding FFprobe binary."""
        path = get_ffprobe_path()
        assert path is not None
        assert os.path.isfile(path)

    def test_ffmpeg_version(self):
        """Test FFmpeg version detection."""
        path = get_ffmpeg_path()
        result = subprocess.run([path, "-version"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "ffmpeg version" in result.stdout.lower()


class TestProbe:
    """Test media file probing."""

    @pytest.mark.integration
    def test_probe_valid_video(self, tmp_path):
        """Test probing a valid video file (integration test)."""
        import subprocess

        test_file = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=1:size=320x240:rate=24",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=1",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(test_file),
            ],
            capture_output=True,
            check=True,
        )

        result = probe(str(test_file))

        assert isinstance(result, ProbeResult)
        assert result.filename == "test.mp4"
        assert result.duration > 0
        assert result.has_video is True
        assert result.has_audio is True
        assert result.resolution == (320, 240)
        assert result.framerate > 0

    @pytest.mark.integration
    def test_probe_video_info(self, tmp_path):
        """Test extracting video information (integration test)."""
        import subprocess

        test_file = tmp_path / "test.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "testsrc=duration=2:size=640x480:rate=30",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                str(test_file),
            ],
            capture_output=True,
            check=True,
        )

        result = probe(str(test_file))

        assert result.resolution == (640, 480)
        assert abs(result.framerate - 30.0) < 1.0
        assert result.format_name == "mov,mp4,m4a,3gp,3g2,mj2"

    def test_probe_nonexistent_file(self):
        """Test probing a nonexistent file."""
        with pytest.raises(Exception):
            probe("/nonexistent/file.mp4")

    def test_probe_uses_v_error_not_quiet(self):
        """REQUIREMENTS.md § 6.3: probe() must request "-v error" (not the old
        "-v quiet") so a failing probe's RuntimeError carries an actually
        informative ffprobe stderr."""
        with (
            patch("autovideofixer.core.ffmpeg_utils.get_ffprobe_path", return_value="ffprobe"),
            patch("autovideofixer.core.ffmpeg_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout='{"format": {}, "streams": []}', stderr=""
            )
            probe("/some/file.mp4")

        cmd = mock_run.call_args[0][0]
        v_index = cmd.index("-v")
        assert cmd[v_index + 1] == "error"

    def test_probe_failure_stderr_surfaced_in_exception(self):
        with (
            patch("autovideofixer.core.ffmpeg_utils.get_ffprobe_path", return_value="ffprobe"),
            patch("autovideofixer.core.ffmpeg_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="moov atom not found")
            with pytest.raises(RuntimeError, match="moov atom not found"):
                probe("/some/file.mp4")

    def test_probe_success_with_stderr_warnings_is_surfaced_on_result(self):
        """A successful (rc=0) probe can still have non-empty stderr (ffprobe
        warnings) -- callers read it back via ProbeResult.stderr /
        to_info_dict()["probe_stderr"]."""
        with (
            patch("autovideofixer.core.ffmpeg_utils.get_ffprobe_path", return_value="ffprobe"),
            patch("autovideofixer.core.ffmpeg_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout='{"format": {}, "streams": []}',
                stderr="Non-monotonous DTS in output stream",
            )
            result = probe("/some/file.mp4")

        assert result.stderr == "Non-monotonous DTS in output stream"
        assert result.to_info_dict()["probe_stderr"] == "Non-monotonous DTS in output stream"

    def test_probe_success_with_empty_stderr(self):
        with (
            patch("autovideofixer.core.ffmpeg_utils.get_ffprobe_path", return_value="ffprobe"),
            patch("autovideofixer.core.ffmpeg_utils.subprocess.run") as mock_run,
        ):
            mock_run.return_value = MagicMock(
                returncode=0, stdout='{"format": {}, "streams": []}', stderr=""
            )
            result = probe("/some/file.mp4")

        assert result.stderr == ""
        assert result.to_info_dict()["probe_stderr"] == ""


class TestStreamInfo:
    """Test stream information."""

    def test_stream_info_properties(self):
        """Test StreamInfo properties."""
        stream = StreamInfo(
            index=0,
            codec_type="video",
            codec_name="h264",
            width=1920,
            height=1080,
            fps=30.0,
        )

        assert stream.is_video is True
        assert stream.is_audio is False
        assert stream.codec_name == "h264"

    def test_audio_stream_info(self):
        """Test audio stream properties."""
        stream = StreamInfo(
            index=1,
            codec_type="audio",
            codec_name="aac",
            channels=2,
            sample_rate=48000,
        )

        assert stream.is_audio is True
        assert stream.is_video is False


class TestHardwareAcceleration:
    """Test hardware acceleration detection."""

    def test_detect_hardware_acceleration(self):
        """Test detecting available hardware acceleration."""
        hwaccels = detect_hardware_acceleration()
        assert isinstance(hwaccels, list)
        # May be empty if no GPU available
        for accel in hwaccels:
            assert isinstance(accel, str)

    def test_resolve_hwaccel_auto(self):
        """Test auto resolution of hardware acceleration."""
        result = resolve_hwaccel("auto")
        assert isinstance(result, str)
        assert result in ["auto", "cuda", "vaapi", "qsv", "none", ...]

    def test_resolve_hwaccel_none(self):
        """Test disabling hardware acceleration."""
        result = resolve_hwaccel("none")
        assert result == "none"


class TestStdinDetachment:
    """No ffmpeg/ffprobe spawn in this module may inherit the caller's tty
    on stdin -- ffmpeg raw-modes a controlling terminal to poll for
    interactive keys and doesn't restore it if killed/backgrounded/crashed.
    Every spawn site here must pass ``-nostdin`` (ffmpeg only -- ffprobe has
    no such flag) AND ``stdin=subprocess.DEVNULL`` (belt-and-suspenders, and
    the only lever available to ffprobe).
    """

    def test_run_ffmpeg_central_runner_detaches_stdin(self):
        """`run_ffmpeg()` is the central choke point nearly every ffmpeg-
        spawning stage routes through -- fixing it here covers all of them.
        """
        with patch("autovideofixer.core.ffmpeg_utils.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = None
            mock_proc.returncode = 0
            mock_proc.stderr = iter([])
            mock_popen.return_value = mock_proc

            run_ffmpeg(["-i", "in.mp4", "out.mp4"], capture_stderr=False)

            assert mock_popen.called
            args, kwargs = mock_popen.call_args
            cmd = args[0]
            assert "-nostdin" in cmd
            # -nostdin must appear before any -i/output args, matching the
            # spec's "early, before -i" placement.
            assert cmd.index("-nostdin") < cmd.index("-i")
            assert kwargs["stdin"] == subprocess.DEVNULL

    def test_probe_detaches_stdin_no_nostdin_flag(self):
        """ffprobe has no -nostdin flag -- only stdin=DEVNULL applies."""
        with patch("autovideofixer.core.ffmpeg_utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout='{"format": {}, "streams": []}')
            probe("in.mp4")

            args, kwargs = mock_run.call_args
            cmd = args[0]
            assert "-nostdin" not in cmd
            assert kwargs["stdin"] == subprocess.DEVNULL

    def test_detect_hardware_acceleration_detaches_stdin(self):
        with patch("autovideofixer.core.ffmpeg_utils.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Hardware acceleration:\ncuda\n")
            detect_hardware_acceleration()

            args, kwargs = mock_run.call_args
            cmd = args[0]
            assert "-nostdin" in cmd
            assert kwargs["stdin"] == subprocess.DEVNULL


class TestRunFfmpegTimeout:
    """run_ffmpeg(timeout=None) must mean "wait forever", not "timeout
    immediately" or coerce to some other default -- most stages now resolve
    their timeout via BaseStage.stage_timeout(), which returns None whenever
    neither stages.<name>.timeout nor pipeline.stage_timeout is set (the
    default, see config.py's DEFAULTS)."""

    def test_timeout_none_passed_through_to_proc_wait(self):
        with patch("autovideofixer.core.ffmpeg_utils.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = None
            mock_proc.returncode = 0
            mock_proc.stderr = iter([])
            mock_popen.return_value = mock_proc

            run_ffmpeg(["-i", "in.mp4", "out.mp4"], timeout=None, capture_stderr=False)

            # subprocess.Popen.wait(timeout=None) blocks indefinitely rather
            # than raising -- this is the "no timeout" behavior this feature
            # relies on, not a special-cased branch in run_ffmpeg itself.
            mock_proc.wait.assert_called_once_with(timeout=None)

    def test_timeout_numeric_passed_through_to_proc_wait(self):
        with patch("autovideofixer.core.ffmpeg_utils.subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.wait.return_value = None
            mock_proc.returncode = 0
            mock_proc.stderr = iter([])
            mock_popen.return_value = mock_proc

            run_ffmpeg(["-i", "in.mp4", "out.mp4"], timeout=42, capture_stderr=False)

            mock_proc.wait.assert_called_once_with(timeout=42)


class TestPathGeneration:
    """Test temporary path generation."""

    def test_generate_temp_path(self, tmp_path):
        """Test generating temporary file paths."""
        input_path = str(tmp_path / "input.mp4")

        temp_path = generate_temp_path(str(tmp_path), input_path, suffix="_test")

        # Intermediate temp files always use .mkv regardless of the input's
        # container, since stages hardcode codecs that aren't valid in every
        # container (e.g. libx264 in a .webm/VP9 container).
        assert temp_path.endswith("_test.mkv")
        assert str(tmp_path) in temp_path
        assert ".avf_" in temp_path

    def test_generate_temp_path_webm_input_still_mkv(self, tmp_path):
        """A .webm input's intermediate temp path must not inherit .webm."""
        input_path = str(tmp_path / "input.webm")

        temp_path = generate_temp_path(str(tmp_path), input_path)

        assert temp_path.endswith(".mkv")

    def test_generate_temp_path_uniqueness(self, tmp_path):
        """Test that generated paths are unique."""
        input_path = str(tmp_path / "input.mp4")

        paths = set()
        for _ in range(10):
            temp_path = generate_temp_path(str(tmp_path), input_path)
            paths.add(temp_path)

        assert len(paths) == 10  # All paths should be unique
