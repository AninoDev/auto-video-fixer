"""Unit tests for the normalize_codecs stage (REQUIREMENTS.md § 22.2) and the
other P1 pieces that need no real ffmpeg: ``default_order``/plan membership, the
CLI flag, the crop default, bit-depth parsing and the container-safety helpers.

``probe()`` and ``run_ffmpeg()`` are mocked here to pin the EXACT ffmpeg arg
list per case. The mocked-ffmpeg lesson (AGENTS.md) applies: the real-ffmpeg
counterparts live in ``tests/integration/test_normalize_codecs_e2e.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from autovideofixer.cli.cli import main
from autovideofixer.config import Config
from autovideofixer.core.ffmpeg_utils import (
    ProbeResult,
    StreamInfo,
    _parse_probe_result,
    pix_fmt_bit_depth,
    sibling_temp_path,
)
from autovideofixer.core.pipeline import DEFAULT_STAGE_ORDER, Job, Pipeline
from autovideofixer.core.stages.base import StageStatus, get_stage
from autovideofixer.core.stages.normalize_codecs import NormalizeCodecsStage


def _config(tmp_path, **overrides) -> Config:
    config = Config(tmp_path / "nonexistent.yaml")
    for key, value in overrides.items():
        config.set(value, "stages", "normalize_codecs", key)
    return config


def _video(codec="h264", pix_fmt="yuv420p", w=320, h=240, **kw) -> StreamInfo:
    return StreamInfo(
        index=0, codec_type="video", codec_name=codec, pixel_format=pix_fmt, width=w, height=h, **kw
    )


def _audio(codec="aac", channels=2, index=1, layout="stereo") -> StreamInfo:
    return StreamInfo(
        index=index, codec_type="audio", codec_name=codec, channels=channels, channel_layout=layout
    )


def _probe(*streams: StreamInfo) -> ProbeResult:
    return ProbeResult(filepath="in.mkv", filename="in.mkv", duration=2.0, streams=list(streams))


def _run(tmp_path, probe_result, config=None, ffmpeg_rc=0):
    """Execute the stage with mocked probe/ffmpeg; returns (result, argv-or-None)."""
    stage = NormalizeCodecsStage(config or _config(tmp_path))
    out = tmp_path / "out.mkv"
    proc = MagicMock(returncode=ffmpeg_rc, stderr="boom")

    def fake_run(args, **kwargs):
        if ffmpeg_rc == 0:
            out.write_bytes(b"x")
        return proc

    with (
        patch("autovideofixer.core.stages.normalize_codecs.probe", return_value=probe_result),
        patch(
            "autovideofixer.core.stages.normalize_codecs.run_ffmpeg", side_effect=fake_run
        ) as run,
    ):
        result = stage.execute("in.mkv", str(out))
    argv = run.call_args.args[0] if run.called else None
    return result, argv


class TestRegistrationAndDefaults:
    def test_registered(self):
        assert get_stage("normalize_codecs") is NormalizeCodecsStage

    def test_default_order_has_it_right_after_retime(self, tmp_path):
        order = Config(tmp_path / "x.yaml").get("pipeline", "default_order")
        assert order[order.index("retime") + 1] == "normalize_codecs"
        assert order.index("normalize_codecs") < order.index("deinterlace")

    def test_pipeline_constant_matches_config(self, tmp_path):
        order = Config(tmp_path / "x.yaml").get("pipeline", "default_order")
        assert list(order) == DEFAULT_STAGE_ORDER

    def test_max_stages_covers_default_order(self, tmp_path):
        config = Config(tmp_path / "x.yaml")
        assert config.get("pipeline", "max_stages") >= len(config.get("pipeline", "default_order"))

    def test_enabled_by_default_with_spec_keys(self, tmp_path):
        cfg = Config(tmp_path / "x.yaml").get("stages", "normalize_codecs")
        assert cfg["enabled"] is True
        assert cfg["video_codecs_allowed"] == ["h264"]
        assert cfg["audio_codecs_allowed"] == ["aac"]
        assert cfg["video_encoder"] == "libx264"
        assert cfg["temp_crf"] == 16
        assert cfg["preset"] == "medium"
        assert cfg["audio_codec"] == "aac"
        assert cfg["audio_bitrate"] == "256k"
        assert "timeout" in cfg

    def test_in_auto_determined_plan_after_retime(self, tmp_path):
        pipeline = Pipeline(Config(tmp_path / "x.yaml"))
        job = Job(input_path=str(tmp_path / "in.mp4"), output_path=str(tmp_path / "o.mp4"))
        info = {"resolution": (640, 480), "framerate": 30, "is_hdr": False}
        stages = pipeline.auto_determine_stages(job, info)
        ordered = pipeline.optimize_stage_order(stages)
        assert "normalize_codecs" in stages
        assert ordered.index("normalize_codecs") == ordered.index("retime") + 1

    def test_disabled_not_in_plan(self, tmp_path):
        config = _config(tmp_path, enabled=False)
        pipeline = Pipeline(config)
        job = Job(input_path=str(tmp_path / "in.mp4"), output_path=str(tmp_path / "o.mp4"))
        stages = pipeline.auto_determine_stages(job, {"resolution": (64, 48), "framerate": 30})
        assert "normalize_codecs" not in stages

    def test_crop_on_by_default_and_in_plan(self, tmp_path):
        config = Config(tmp_path / "x.yaml")
        assert config.get("stages", "crop", "enabled") is True
        pipeline = Pipeline(config)
        job = Job(input_path=str(tmp_path / "in.mp4"), output_path=str(tmp_path / "o.mp4"))
        stages = pipeline.auto_determine_stages(job, {"resolution": (64, 48), "framerate": 30})
        assert "crop" in stages


class TestShouldRun:
    def test_disabled(self, tmp_path):
        stage = NormalizeCodecsStage(_config(tmp_path, enabled=False))
        assert stage.should_run({})[0] is False

    def test_runs_when_info_absent(self, tmp_path):
        # Partial info defers to execute()'s authoritative probe.
        assert NormalizeCodecsStage(_config(tmp_path)).should_run({})[0] is True

    def test_skips_when_probe_info_says_all_allowed(self, tmp_path):
        stage = NormalizeCodecsStage(_config(tmp_path))
        info = {"selected_video_codec": "h264", "selected_audio_codec": "aac"}
        ok, reason = stage.should_run(info)
        assert ok is False and "allowed" in reason

    @pytest.mark.parametrize(
        "v,a", [("vp9", "aac"), ("h264", "opus"), ("av1", "vorbis"), ("h264", "flac")]
    )
    def test_runs_when_any_codec_unusual(self, tmp_path, v, a):
        stage = NormalizeCodecsStage(_config(tmp_path))
        info = {"selected_video_codec": v, "selected_audio_codec": a}
        assert stage.should_run(info)[0] is True

    def test_no_audio_video_h264_skips(self, tmp_path):
        stage = NormalizeCodecsStage(_config(tmp_path))
        info = {"selected_video_codec": "h264", "selected_audio_codec": ""}
        assert stage.should_run(info)[0] is False

    def test_no_streams_at_all(self, tmp_path):
        stage = NormalizeCodecsStage(_config(tmp_path))
        assert stage.should_run({"has_video": False, "has_audio": False})[0] is False

    def test_force_enabled_overrides_disabled_config(self, tmp_path):
        stage = NormalizeCodecsStage(_config(tmp_path, enabled=False))
        stage._force_enabled = True
        assert stage.should_run({})[0] is True


class TestExecuteArgs:
    def test_copy_all_skips_without_ffmpeg(self, tmp_path):
        result, argv = _run(tmp_path, _probe(_video("h264"), _audio("aac")))
        assert result.status == StageStatus.SKIPPED
        assert argv is None
        assert [s["copied"] for s in result.metadata["streams"]] == [True, True]

    def test_video_only_transcode_exact_args(self, tmp_path):
        result, argv = _run(tmp_path, _probe(_video("vp9"), _audio("aac")))
        assert result.status == StageStatus.COMPLETED
        assert argv == [
            "-i", "in.mkv",
            "-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-fps_mode", "passthrough", "-y", str(tmp_path / "out.mkv"),
        ]  # fmt: skip
        assert result.metadata["streams"][0]["action"] == "transcoded (vp9->libx264)"
        assert result.metadata["streams"][1]["action"] == "copied (aac)"

    def test_audio_only_transcode_exact_args(self, tmp_path):
        result, argv = _run(tmp_path, _probe(_video("h264"), _audio("opus", 2)))
        assert result.status == StageStatus.COMPLETED
        assert argv == [
            "-i", "in.mkv",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "256k",
            "-fps_mode", "passthrough", "-y", str(tmp_path / "out.mkv"),
        ]  # fmt: skip
        assert result.metadata["streams"][1]["action"] == "transcoded (opus->aac)"

    def test_both_transcoded_vorbis(self, tmp_path):
        result, argv = _run(tmp_path, _probe(_video("vp9"), _audio("vorbis")))
        assert "libx264" in argv and "aac" in argv and "copy" not in argv

    def test_ten_bit_uses_yuv420p10le_and_carries_color(self, tmp_path):
        v = _video(
            "vp9",
            "yuv420p10le",
            color_primaries="bt2020",
            color_transfer="smpte2084",
            color_space="bt2020nc",
            color_range="tv",
        )
        result, argv = _run(tmp_path, _probe(v, _audio("opus")))
        assert argv[argv.index("-pix_fmt") + 1] == "yuv420p10le"
        assert argv[argv.index("-color_primaries") + 1] == "bt2020"
        assert argv[argv.index("-color_trc") + 1] == "smpte2084"
        assert argv[argv.index("-colorspace") + 1] == "bt2020nc"
        assert argv[argv.index("-color_range") + 1] == "tv"
        assert result.metadata["streams"][0]["bit_depth"] == 10

    def test_unset_color_fields_not_passed(self, tmp_path):
        v = _video("vp9", color_primaries="unknown", color_transfer="", color_space="unspecified")
        _, argv = _run(tmp_path, _probe(v))
        for flag in ("-color_primaries", "-color_trc", "-colorspace", "-color_range"):
            assert flag not in argv

    def test_rgb_source_drops_matrix_and_range_but_keeps_primaries(self, tmp_path):
        # RGB VP9 (gbrp, matrix "gbr") becomes YUV on re-encode: libx264 rejects
        # -colorspace gbr, and RGB full range would mislabel the YUV output.
        v = _video(
            "vp9",
            "gbrp",
            color_primaries="bt709",
            color_transfer="bt709",
            color_space="gbr",
            color_range="pc",
        )
        _, argv = _run(tmp_path, _probe(v))
        assert "-colorspace" not in argv
        assert "-color_range" not in argv
        assert argv[argv.index("-color_primaries") + 1] == "bt709"
        assert argv[argv.index("-pix_fmt") + 1] == "yuv420p"

    def test_vfr_passthrough_flag_in_every_ffmpeg_case(self, tmp_path):
        for streams in (
            _probe(_video("vp9"), _audio("aac")),
            _probe(_video("h264"), _audio("opus")),
            _probe(_video("av1")),
        ):
            _, argv = _run(tmp_path, streams)
            i = argv.index("-fps_mode")
            assert argv[i + 1] == "passthrough"
            assert i > argv.index("-i")  # an OUTPUT option, not before -i

    def test_odd_dimensions_cropped_even(self, tmp_path):
        _, argv = _run(tmp_path, _probe(_video("vp9", w=321, h=241)))
        assert argv[argv.index("-vf") + 1] == "crop=trunc(iw/2)*2:trunc(ih/2)*2"

    def test_even_dimensions_no_filter(self, tmp_path):
        _, argv = _run(tmp_path, _probe(_video("vp9", w=320, h=240)))
        assert "-vf" not in argv

    def test_no_map_flags_default_selection_preserved(self, tmp_path):
        _, argv = _run(tmp_path, _probe(_video("vp9"), _audio("opus")))
        assert "-map" not in argv

    def test_video_only_input(self, tmp_path):
        _, argv = _run(tmp_path, _probe(_video("vp9")))
        assert "-c:a" not in argv

    def test_audio_only_input(self, tmp_path):
        _, argv = _run(tmp_path, _probe(_audio("opus", index=0)))
        assert "-c:v" not in argv and "aac" in argv

    def test_custom_allowed_lists_and_encoder(self, tmp_path):
        config = _config(
            tmp_path,
            video_codecs_allowed=["h264", "hevc"],
            audio_codecs_allowed=["aac", "opus"],
            temp_crf=20,
            preset="fast",
            audio_bitrate="128k",
        )
        result, argv = _run(tmp_path, _probe(_video("hevc"), _audio("opus")), config)
        assert result.status == StageStatus.SKIPPED
        result, argv = _run(tmp_path, _probe(_video("vp9"), _audio("flac")), config)
        assert argv[argv.index("-crf") + 1] == "20"
        assert argv[argv.index("-preset") + 1] == "fast"
        assert argv[argv.index("-b:a") + 1] == "128k"

    def test_default_selection_mirrors_ffmpeg(self, tmp_path):
        # cover art (attached pic) is never the "real" video; the audio with
        # the most channels is the one ffmpeg picks.
        cover = _video("mjpeg", w=1000, h=1000, is_attached_pic=True)
        real = StreamInfo(index=1, codec_type="video", codec_name="h264", width=320, height=240)
        stereo = _audio("opus", 2, index=2)
        surround = _audio("aac", 6, index=3, layout="5.1")
        result, argv = _run(tmp_path, _probe(cover, real, stereo, surround))
        assert result.status == StageStatus.SKIPPED  # h264 + aac(5.1) chosen

    def test_ffmpeg_failure_is_failed(self, tmp_path):
        result, _ = _run(tmp_path, _probe(_video("vp9")), ffmpeg_rc=1)
        assert result.status == StageStatus.FAILED
        assert "boom" in result.error

    def test_probe_failure_is_failed(self, tmp_path):
        stage = NormalizeCodecsStage(_config(tmp_path))
        with patch(
            "autovideofixer.core.stages.normalize_codecs.probe", side_effect=RuntimeError("bad")
        ):
            result = stage.execute("in.mkv", str(tmp_path / "o.mkv"))
        assert result.status == StageStatus.FAILED

    def test_no_output_path_fails(self, tmp_path):
        result = NormalizeCodecsStage(_config(tmp_path)).execute("in.mkv", None)
        assert result.status == StageStatus.FAILED

    def test_timeout_resolution(self, tmp_path):
        config = _config(tmp_path, timeout=42)
        stage = NormalizeCodecsStage(config)
        assert stage.stage_timeout() == 42


class TestProbeExtensions:
    def test_bit_depth_from_pix_fmt(self):
        assert pix_fmt_bit_depth("yuv420p") == 8
        assert pix_fmt_bit_depth("yuv420p10le") == 10
        assert pix_fmt_bit_depth("yuv422p12le") == 12
        assert pix_fmt_bit_depth("p010le") == 10
        assert pix_fmt_bit_depth("nv12") == 8
        assert pix_fmt_bit_depth("gbrp10le") == 10
        assert pix_fmt_bit_depth("gray") == 8
        assert pix_fmt_bit_depth("gray10le") == 10
        assert pix_fmt_bit_depth("rgb48le") == 16
        assert pix_fmt_bit_depth("") == 0
        assert pix_fmt_bit_depth("", 10) == 10
        assert pix_fmt_bit_depth("unknownfmt", 10) == 10

    def test_probe_parses_string_fields_and_attached_pic(self):
        data = {
            "streams": [
                {
                    "codec_type": "video",
                    "codec_name": "mjpeg",
                    "width": 500,
                    "height": 500,
                    "disposition": {"attached_pic": 1},
                },
                {
                    "codec_type": "video",
                    "codec_name": "vp9",
                    "width": 320,
                    "height": 240,
                    "pix_fmt": "yuv420p10le",
                    "bits_per_raw_sample": "10",  # ffprobe emits STRINGS
                    "color_primaries": "bt2020",
                    "color_transfer": "smpte2084",
                    "disposition": {"attached_pic": 0},
                },
                {
                    "codec_type": "audio",
                    "codec_name": "opus",
                    "channels": 6,
                    "channel_layout": "5.1",
                },
            ],
            "format": {"duration": "2.0"},
        }
        pr = _parse_probe_result(data, "x.webm")
        assert pr.streams[1].bits_per_raw_sample == 10
        assert pr.streams[0].is_attached_pic and not pr.streams[1].is_attached_pic
        assert pr.selected_video_stream.codec_name == "vp9"
        info = pr.to_info_dict()
        assert info["bit_depth"] == 10
        assert info["pix_fmt"] == "yuv420p10le"
        assert info["color_primaries"] == "bt2020"
        assert info["color_trc"] == "smpte2084"
        assert info["selected_video_codec"] == "vp9"
        assert info["selected_audio_codec"] == "opus"

    def test_sibling_temp_path_keeps_extension(self):
        assert sibling_temp_path("/t/a.mkv", "x_tmp") == "/t/a.x_tmp.mkv"
        assert sibling_temp_path("/t/a.mp4", "x_tmp") == "/t/a.x_tmp.mp4"
        assert sibling_temp_path("/t/a", "x_tmp") == "/t/a.x_tmp.mkv"


class TestCliFlag:
    def _effective_enabled(self, tmp_path, *flags, cfg_text: str = "{}\n") -> bool:
        cfg = tmp_path / "empty.yaml"
        cfg.write_text(cfg_text)
        video = tmp_path / "v.mp4"
        video.write_bytes(b"\x00")
        captured: dict = {}

        def fake_pipeline(config):
            captured["config"] = config
            return MagicMock()

        with patch("autovideofixer.cli.cli.Pipeline", side_effect=fake_pipeline):
            result = CliRunner().invoke(
                main, ["--config", str(cfg), "process", str(video), "--dry-run", *flags]
            )
        assert result.exit_code == 0, result.output
        return captured["config"].get("stages", "normalize_codecs", "enabled")

    def test_default_on(self, tmp_path):
        assert self._effective_enabled(tmp_path) is True

    def test_no_flag_disables(self, tmp_path):
        assert self._effective_enabled(tmp_path, "--no-normalize-codecs") is False

    def test_flag_enables_over_config_file_false(self, tmp_path):
        assert (
            self._effective_enabled(
                tmp_path,
                "--normalize-codecs",
                cfg_text="stages:\n  normalize_codecs:\n    enabled: false\n",
            )
            is True
        )

    def test_help_lists_flag(self):
        out = CliRunner().invoke(main, ["process", "--help"]).output
        assert "--normalize-codecs" in out and "--no-normalize-codecs" in out
