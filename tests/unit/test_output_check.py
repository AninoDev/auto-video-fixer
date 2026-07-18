"""Tests for core/output_check.py -- REQUIREMENTS.md § 6.2 existing-output
spec-check helpers (pure functions, no ffmpeg/pipeline dependency)."""

from __future__ import annotations

from types import SimpleNamespace

from autovideofixer.core.output_check import (
    OutputTargets,
    check_output_spec,
    effective_output_targets,
    effective_target_bounds,
    resolution_satisfies,
)


class FakeConfig:
    """Minimal ``Config``-shaped stub: ``.get(*keys, default=...)`` over a
    plain nested dict, matching ``Config.get()``'s exact signature/behavior."""

    def __init__(self, data: dict):
        self._data = data

    def get(self, *keys, default=None):
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k, default)
        return node


def make_job(stages=None, stage_overrides=None):
    return SimpleNamespace(stages=stages or [], stage_overrides=stage_overrides or {})


class TestEffectiveTargetBounds:
    def test_landscape_input_keeps_target_orientation(self):
        assert effective_target_bounds(1920, 1080, 1920, 1080) == (1920, 1080)

    def test_portrait_input_rotates_target(self):
        assert effective_target_bounds(1080, 1920, 1920, 1080) == (1080, 1920)

    def test_square_input_uses_shorter_edge(self):
        assert effective_target_bounds(500, 500, 1920, 1080) == (1080, 1080)

    def test_keep_aspect_ratio_false_returns_target_unrotated(self):
        assert effective_target_bounds(1080, 1920, 1920, 1080, keep_aspect_ratio=False) == (
            1920,
            1080,
        )


class TestResolutionSatisfies:
    def test_rotated_resolution_satisfies(self):
        # A 1080x1918 output vs. a [1920, 1080] target: rotated bounds are
        # 1080x1920, and 1918 is within the 1.05 threshold of 1920.
        assert resolution_satisfies(1080, 1918, 1920, 1080) is True

    def test_few_px_short_exact_aspect_satisfies(self):
        # 1072x1908 vs. a rotated 1080x1920 bound is ~1.0075x -- within 1.05.
        assert resolution_satisfies(1072, 1908, 1920, 1080) is True

    def test_genuine_upscale_need_does_not_satisfy(self):
        assert resolution_satisfies(540, 960, 1920, 1080) is False

    def test_zero_resolution_falls_back_to_non_rotated_compare(self):
        assert resolution_satisfies(0, 0, 1920, 1080) is False


class TestFramerateEpsilon:
    def test_2997_satisfies_30_target(self):
        targets = OutputTargets(target_framerate=30.0)
        matches, reasons = check_output_spec({"resolution": (0, 0), "framerate": 29.97}, targets)
        assert matches is True
        assert reasons == []

    def test_genuinely_lower_framerate_mismatches(self):
        targets = OutputTargets(target_framerate=60.0)
        matches, reasons = check_output_spec({"resolution": (0, 0), "framerate": 30.0}, targets)
        assert matches is False
        assert any("framerate" in r for r in reasons)


class TestCodecFamilyNormalization:
    def test_libx265_satisfies_hevc_target(self):
        targets = OutputTargets(video_codec="hevc")
        matches, _ = check_output_spec(
            {"resolution": (0, 0), "framerate": 0, "video_codec": "hevc"}, targets
        )
        assert matches is True

    def test_libx264_target_vs_hevc_existing_mismatches(self):
        targets = OutputTargets(video_codec="libx264")
        matches, reasons = check_output_spec(
            {"resolution": (0, 0), "framerate": 0, "video_codec": "hevc"}, targets
        )
        assert matches is False
        assert any("vcodec" in r for r in reasons)

    def test_audio_codec_family_match(self):
        targets = OutputTargets(audio_codec="aac")
        matches, _ = check_output_spec(
            {
                "resolution": (0, 0),
                "framerate": 0,
                "audio_codecs": ["aac"],
            },
            targets,
        )
        assert matches is True


class TestUnspecifiedTargetsNeverMismatch:
    def test_empty_targets_always_match(self):
        targets = OutputTargets()
        matches, reasons = check_output_spec(
            {"resolution": (10, 10), "framerate": 1.0, "video_codec": "mpeg2video"}, targets
        )
        assert matches is True
        assert reasons == []


class TestCorruptExistingOutput:
    def test_none_existing_info_is_unreadable_mismatch(self):
        matches, reasons = check_output_spec(None, OutputTargets(container="mp4"))
        assert matches is False
        assert reasons == ["unreadable"]


class TestEffectiveOutputTargets:
    def test_target_format_only_from_general_target_format(self):
        config = FakeConfig({"general": {"target_format": "MKV"}})
        targets = effective_output_targets(config, make_job())
        assert targets.container == "mkv"

    def test_output_container_alone_is_not_a_specified_target(self):
        config = FakeConfig({"general": {"output_container": "mp4"}})
        targets = effective_output_targets(config, make_job())
        assert targets.container is None

    def test_resolution_target_from_quality_target(self):
        config = FakeConfig({"quality": {"quality_target": {"target_resolution": [1920, 1080]}}})
        targets = effective_output_targets(config, make_job())
        assert (targets.target_width, targets.target_height) == (1920, 1080)

    def test_framerate_target_only_when_interpolate_would_run(self):
        config = FakeConfig({"quality": {"quality_target": {"target_framerate": 60.0}}})
        job = make_job(stages=["encode"])  # interpolate NOT requested
        targets = effective_output_targets(config, job)
        assert targets.target_framerate is None

        job_with_interp = make_job(stages=["interpolate", "encode"])
        targets2 = effective_output_targets(config, job_with_interp)
        assert targets2.target_framerate == 60.0

    def test_encode_override_wins_over_encoding_preset(self):
        config = FakeConfig({"encoding": {"video_codec": "libx264"}})
        job = make_job(stage_overrides={"encode": {"codec": "libx265"}})
        targets = effective_output_targets(config, job)
        assert targets.video_codec == "libx265"

    def test_no_encode_settings_specified_leaves_codec_unspecified(self):
        config = FakeConfig({})
        targets = effective_output_targets(config, make_job())
        assert targets.video_codec is None
        assert targets.audio_codec is None

    def test_stages_encode_config_is_not_a_specified_target(self):
        # Base stages.encode.* config never reaches EncodeStage.execute()'s
        # codec kwargs (only job/occurrence overrides do), so it must not be
        # treated as a target -- it would flag mismatches the pipeline can
        # never resolve (endless rename/reprocess churn).
        config = FakeConfig({"stages": {"encode": {"codec": "libx265", "audio_codec": "opus"}}})
        targets = effective_output_targets(config, make_job())
        assert targets.video_codec is None
        assert targets.audio_codec is None
