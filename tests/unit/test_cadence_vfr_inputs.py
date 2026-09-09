"""Cadence analysis must distinguish a GENUINE VFR input from a CFR encode of
a VFR original (REQUIREMENTS.md § 12.1/§ 12.3).

These are two different inputs that look similar in the container:

  A. Native VFR -- a direct phone/screen capture. Irregular frame gaps, but
     NO duplicate frames. `retime` must SKIP: there is nothing to remove.
  B. VFR transcoded to CFR -- e.g. a YouTube download. The publisher padded
     the irregular original up to a constant rate by DUPLICATING frames, so
     the duplicates are real and `retime` must run.

The trap: `total_frames` was derived as `duration * encoded_fps`, and
`encoded_fps` comes from `avg_frame_rate`, which § 12.3 documents as
unreliable for VFR. A genuine 123-frame VFR capture advertising 60fps over 3s
therefore "measured" 180 total frames and reported a 0.317 duplicate ratio for
a file with ZERO duplicates -- sending an input that needed nothing through a
full needless re-encode, and pinning the CFR deliverable to a meaningless mean
rate. The total is now MEASURED via a second `showinfo` placed before
`mpdecimate` in the same decode pass.
"""

from __future__ import annotations

import subprocess

import pytest

from autovideofixer.config import Config
from autovideofixer.core.cadence import _parse_pts_times, analyze_cadence


@pytest.fixture
def vfr_pair(tmp_path):
    """(native_vfr, vfr_as_cfr) -- same content, two container representations.

    The native VFR file is built by dropping a non-uniform subset of a 60fps
    source while keeping real timestamps, so gaps are irregular and no two
    adjacent frames are duplicates. The CFR file is that same file padded back
    up to a constant 60fps, which is exactly what a publisher does.
    """
    src = tmp_path / "src60.mp4"
    native = tmp_path / "vfr_native.mkv"
    as_cfr = tmp_path / "vfr_as_cfr.mp4"

    def run(args):
        subprocess.run(args, capture_output=True, check=True)

    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=60",
            "-t",
            "3",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(src),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-vf",
            r"select='not(eq(mod(n\,7),3))*not(eq(mod(n\,5),1))'",
            "-fps_mode",
            "vfr",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            str(native),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(native),
            "-fps_mode",
            "cfr",
            "-r",
            "60",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(as_cfr),
        ]
    )
    return str(native), str(as_cfr)


@pytest.fixture
def fresh_config():
    return Config(config_path="/nonexistent/fresh.yaml")


class TestVfrVersusPaddedVfr:
    def test_native_vfr_reports_no_duplicates_and_skips(self, vfr_pair, fresh_config):
        """A real VFR capture has nothing to decimate -- must not be 'padded'."""
        native, _ = vfr_pair
        analysis = analyze_cadence(native, fresh_config)

        assert analysis.total_frames == analysis.unique_frames, (
            "total was not measured from the decode -- avg_frame_rate invented "
            f"duplicates: total={analysis.total_frames} unique={analysis.unique_frames}"
        )
        assert analysis.duplicate_ratio == pytest.approx(0.0, abs=1e-6)
        assert analysis.is_padded is False, (
            "a genuine VFR input would be sent through a needless full re-encode"
        )

    def test_vfr_transcoded_to_cfr_reports_real_duplicates(self, vfr_pair, fresh_config):
        """The same content padded to CFR has genuine duplicates to remove."""
        _, as_cfr = vfr_pair
        analysis = analyze_cadence(as_cfr, fresh_config)

        assert analysis.total_frames > analysis.unique_frames
        assert analysis.duplicate_ratio > 0.1
        assert analysis.is_padded is True

    def test_same_content_recovers_the_same_unique_frame_count(self, vfr_pair, fresh_config):
        """Both representations describe identical content, so the recovered
        timeline must have the same number of real frames."""
        native, as_cfr = vfr_pair

        assert (
            analyze_cadence(native, fresh_config).unique_frames
            == analyze_cadence(as_cfr, fresh_config).unique_frames
        )


class TestShowinfoInstanceParsing:
    """Pure parsing -- no ffmpeg needed."""

    SAMPLE = (
        "[Parsed_showinfo_0 @ 0x1] n:0 pts:0 pts_time:0 duration_time:0.016\n"
        "[Parsed_showinfo_0 @ 0x1] n:1 pts:1 pts_time:0.016 duration_time:0.016\n"
        "[Parsed_showinfo_0 @ 0x1] n:2 pts:2 pts_time:0.033 duration_time:0.016\n"
        "[Parsed_showinfo_2 @ 0x2] n:0 pts:0 pts_time:0 duration_time:0.033\n"
        "[Parsed_showinfo_2 @ 0x2] n:1 pts:2 pts_time:0.033 duration_time:0.033\n"
    )

    def test_instance_zero_is_every_decoded_frame(self):
        assert _parse_pts_times(self.SAMPLE, instance=0) == [0.0, 0.016, 0.033]

    def test_instance_two_is_only_survivors(self):
        assert _parse_pts_times(self.SAMPLE, instance=2) == [0.0, 0.033]

    def test_no_instance_matches_everything(self):
        assert len(_parse_pts_times(self.SAMPLE)) == 5

    def test_unknown_instance_yields_empty(self):
        assert _parse_pts_times(self.SAMPLE, instance=7) == []

    def test_malformed_text_never_raises(self):
        assert _parse_pts_times("", instance=0) == []
        assert _parse_pts_times("garbage\nno timestamps here", instance=0) == []
