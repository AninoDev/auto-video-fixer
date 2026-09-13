"""ffprobe's numeric stream fields must be coerced to int, not passed through raw.

`ffprobe -show_streams` returns numbers as JSON STRINGS ("44100", "2",
"128000"), but `StreamInfo` annotates them as `int` and callers do arithmetic
and comparisons on them. `sample_rate`/`channels`/`bit_rate` were assigned
straight from the probe dict while every neighbouring numeric field went
through `_safe_int`.

The failure this caused was invisible to unit tests and immediate on real
input: SpeedStage's `asetrate` audio path crashed with

    '<=' not supported between instances of 'str' and 'int'

because its guard compares the probed sample rate against 0. Tests passed ints
directly, so only an end-to-end run surfaced it.
"""

from __future__ import annotations

from autovideofixer.core.ffmpeg_utils import _parse_probe_result


def _probe_payload(**audio_overrides):
    audio = {
        "index": 1,
        "codec_type": "audio",
        "codec_name": "aac",
        # Exactly how ffprobe emits them -- strings, not numbers.
        "sample_rate": "44100",
        "channels": "2",
        "bit_rate": "128000",
    }
    audio.update(audio_overrides)
    return {
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30/1",
            },
            audio,
        ],
        "format": {"format_name": "mov,mp4", "duration": "4.0"},
    }


class TestAudioNumericFieldsAreInts:
    def test_sample_rate_is_int(self):
        result = _parse_probe_result(_probe_payload(), "in.mp4")
        rate = result.audio_streams[0].sample_rate

        assert isinstance(rate, int), f"sample_rate came back as {type(rate).__name__}"
        assert rate == 44100

    def test_channels_is_int(self):
        result = _parse_probe_result(_probe_payload(), "in.mp4")
        assert isinstance(result.audio_streams[0].channels, int)
        assert result.audio_streams[0].channels == 2

    def test_bit_rate_is_int(self):
        result = _parse_probe_result(_probe_payload(), "in.mp4")
        assert isinstance(result.audio_streams[0].bit_rate, int)
        assert result.audio_streams[0].bit_rate == 128000

    def test_sample_rate_supports_numeric_comparison(self):
        """The exact operation that crashed the asetrate path."""
        rate = _parse_probe_result(_probe_payload(), "in.mp4").audio_streams[0].sample_rate

        assert rate > 0
        assert not rate <= 0
        assert round(rate * 0.25) == 11025

    def test_missing_fields_default_to_zero(self):
        payload = _probe_payload()
        for key in ("sample_rate", "channels", "bit_rate"):
            payload["streams"][1].pop(key)
        stream = _parse_probe_result(payload, "in.mp4").audio_streams[0]

        assert stream.sample_rate == 0
        assert stream.channels == 0
        assert stream.bit_rate == 0

    def test_malformed_values_degrade_to_zero_rather_than_raising(self):
        """ffprobe emits "N/A" for unknown values on some inputs."""
        stream = _parse_probe_result(
            _probe_payload(sample_rate="N/A", channels="", bit_rate="N/A"), "in.mp4"
        ).audio_streams[0]

        assert stream.sample_rate == 0
        assert stream.channels == 0
        assert stream.bit_rate == 0
