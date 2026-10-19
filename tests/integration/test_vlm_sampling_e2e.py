"""Real-ffmpeg tests for REQUIREMENTS.md § 25 (VLM frame sampling).

All media is generated here with ffmpeg / numpy on tiny clips; the assertions decode the JPEGs
that would be sent to the model and check what is actually in them (dimensions, which source
picture they came from), because the mocked-frame tests cannot see seek, timestamp or
resize bugs.

Run with: pytest tests/integration/test_vlm_sampling_e2e.py -v -m integration
"""

from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from autovideofixer.core import vlm_sampling as vs_module
from autovideofixer.core.vlm_sampling import (
    FFmpegFrameSource,
    SamplingConfig,
    VisionLimits,
    sample_vlm_frames,
    sample_vlm_frames_multi,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not installed",
    ),
]


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True)


def _picture(seed: int, size=(320, 180)) -> np.ndarray:
    """A busy, distinctive picture per seed (so different seeds are far apart in luma)."""
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = rng.integers(20, 235, 3)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, w - 20)), int(rng.integers(0, h - 20))
        x1, y1 = x0 + int(rng.integers(20, w // 2)), y0 + int(rng.integers(20, h // 2))
        cv2.rectangle(
            img, (x0, y0), (min(x1, w - 1), min(y1, h - 1)), rng.integers(0, 255, 3).tolist(), -1
        )
    return img


def _hold_clip(path, holds, *, fps=None, size=(320, 180), vfr=False, crf=20) -> None:
    """Video that shows picture ``seed`` for ``seconds`` each: ``holds = [(seed, seconds)]``.

    ``vfr=True`` keeps one frame per picture with its real duration (a genuine variable-frame-
    rate file); otherwise the pictures are repeated to a constant ``fps`` and libx264-encoded,
    so repeated frames are *not* bit-identical after decoding (the realistic duplicate case).
    """
    d = os.path.dirname(path)
    lines = []
    for i, (seed, secs) in enumerate(holds):
        png = os.path.join(d, f"hold_{os.path.basename(path)}_{i}.png")
        cv2.imwrite(png, _picture(seed, size))
        lines.append(f"file '{png}'\nduration {secs}\n")
    lines.append(f"file '{png}'\n")  # concat demuxer: last file repeated, duration is ignored
    total = f"{sum(secs for _, secs in holds):.3f}"  # ... so cut a CFR clip to its real length
    lst = path + ".txt"
    with open(lst, "w") as f:
        f.writelines(lines)
    if vfr:
        _ff("-f", "concat", "-safe", "0", "-i", lst, "-fps_mode", "vfr", "-c:v", "libx264",
            "-crf", str(crf), "-pix_fmt", "yuv420p", path)  # fmt: skip
    else:
        _ff("-f", "concat", "-safe", "0", "-i", lst, "-t", total, "-vf", f"fps={fps or 30}", "-c:v",
            "libx264", "-crf", str(crf), "-pix_fmt", "yuv420p", path)  # fmt: skip


def _sample(path, cfg=None, vision=None, start=0.0, end=None):
    return sample_vlm_frames(
        str(path), cfg or SamplingConfig(), vision or VisionLimits(), start, end
    )


def _decode(paths):
    imgs = [cv2.imread(p) for p in paths]
    assert all(i is not None for i in imgs)
    return imgs


def _which_picture(img, seeds, size=(320, 180)):
    """Index in ``seeds`` of the source picture ``img`` (any size) looks most like."""
    best, best_d = None, 1.0
    for i, seed in enumerate(seeds):
        ref = cv2.resize(
            _picture(seed, size), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA
        )
        d = float(np.abs(ref.astype(np.int16) - img.astype(np.int16)).mean() / 255)
        if d < best_d:
            best, best_d = i, d
    assert best_d < 0.06, f"frame matches no source picture (closest diff {best_d:.3f})"
    return best


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("vlm_sampling_media")
    m = {}
    # A held for 5.5 s then three short pictures: 4 samples of an 8 s clip land A, A, A, C ...
    m["dup"] = str(d / "dup.mp4")
    _hold_clip(m["dup"], [(1, 5.5), (2, 0.8), (3, 0.8), (4, 0.9)], fps=30)
    # True VFR: 5 frames with wildly different durations.
    m["vfr"] = str(d / "vfr.mp4")
    _hold_clip(m["vfr"], [(10, 0.2), (11, 3.0), (12, 0.1), (13, 2.0), (14, 1.7)], vfr=True)
    m["three"] = str(d / "three.mp4")
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=3:d=1", "-c:v", "libx264", m["three"])
    m["one"] = str(d / "one.mp4")
    _ff("-f", "lavfi", "-i", "testsrc2=s=320x180:r=1:d=1", "-c:v", "libx264", m["one"])
    m["static"] = str(d / "static.mp4")
    _ff("-f", "lavfi", "-i", "color=c=0x3060a0:s=320x180:r=30:d=6", "-c:v", "libx264", m["static"])
    m["hd"] = str(d / "hd.mp4")
    _ff("-f", "lavfi", "-i", "testsrc2=s=1280x720:r=30:d=30", "-c:v", "libx264", m["hd"])
    m["small"] = str(d / "small.mp4")
    _ff("-f", "lavfi", "-i", "testsrc2=s=160x90:r=30:d=30", "-c:v", "libx264", m["small"])
    m["ts"] = str(d / "long_gop.ts")  # 1 keyframe in 6 s: input seeking can't work on MPEG-TS
    _ff(
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=320x180:r=25:d=6",
        "-c:v",
        "libx264",
        "-f",
        "mpegts",
        m["ts"],
    )
    return m


def test_duplicate_frames_are_dropped_and_refilled(media):
    cfg = SamplingConfig(min_frames=4, max_frames=4, interval_sec=2.0)
    sampled = _sample(media["dup"], cfg)
    try:
        r = sampled.report
        imgs = _decode(sampled.paths)
        # The centres (1, 3, 5, 7 s) hit A, A, A, C. Two A's are dropped and the refill finds
        # B and D, so the four frames are four *different* source pictures.
        assert r.frames == len(imgs) == 4
        assert r.dropped_duplicates >= 2
        assert sorted(_which_picture(i, [1, 2, 3, 4]) for i in imgs) == [0, 1, 2, 3]
        # ... in time order
        assert [_which_picture(i, [1, 2, 3, 4]) for i in imgs] == [0, 1, 2, 3]
        assert r.attempts <= 16
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_dedupe_off_sends_the_repeats(media):
    cfg = SamplingConfig(min_frames=4, max_frames=4, interval_sec=2.0, dedupe_consecutive=False)
    sampled = _sample(media["dup"], cfg)
    try:
        imgs = _decode(sampled.paths)
        assert len(imgs) == 4 and sampled.report.dropped_duplicates == 0
        pics = [_which_picture(i, [1, 2, 3, 4]) for i in imgs]
        assert pics.count(0) >= 3
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_three_frame_clip_sends_at_most_three_distinct_frames(media):
    sampled = _sample(media["three"], SamplingConfig())  # min_frames=4 by default
    try:
        imgs = _decode(sampled.paths)
        assert 1 <= len(imgs) <= 3
        assert len(imgs) == 3  # all three real frames are distinct and reachable
        for a in range(len(imgs)):
            for b in range(a + 1, len(imgs)):
                assert not np.array_equal(imgs[a], imgs[b])
        assert sampled.report.frames == 3
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_single_frame_clip(media):
    sampled = _sample(media["one"])
    try:
        assert len(sampled.paths) == 1
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_vfr_samples_the_frame_on_screen_at_each_position(media):
    # Frames: P10 [0,0.2) P11 [0.2,3.2) P12 [3.2,3.32) P13 [3.32,5.32) P14 [5.32,7.0) and a
    # repeat of P14 at 7.0 (the concat demuxer's closing frame)
    # 3 centres over ~7 s: ~1.2 s (P11), ~3.5 s (P13), ~5.8 s (P14). An index*fps seek (or
    # OpenCV's CAP_PROP_POS_MSEC, which maps time to a frame number with the average fps) would
    # return P13/P14/P14-ish frames here, seconds off.
    cfg = SamplingConfig(min_frames=3, max_frames=3, interval_sec=2.4)
    sampled = _sample(media["vfr"], cfg)
    try:
        pics = [_which_picture(i, [10, 11, 12, 13, 14]) for i in _decode(sampled.paths)]
        assert pics == [1, 3, 4]
        assert sampled.report.dropped_duplicates == 0
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_vfr_never_sends_the_same_frame_twice(media):
    # 5 real frames; ask for 8: positions pile onto the long holds, but each frame is sent once.
    cfg = SamplingConfig(min_frames=8, max_frames=8, interval_sec=1.0)
    sampled = _sample(media["vfr"], cfg)
    try:
        pics = [_which_picture(i, [10, 11, 12, 13, 14]) for i in _decode(sampled.paths)]
        assert len(pics) == len(set(pics)) <= 5
        assert pics == sorted(pics)
        assert sampled.report.dropped_duplicates > 0
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_static_clip_ends_below_min_frames(media):
    sampled = _sample(media["static"], SamplingConfig())
    try:
        assert len(sampled.paths) == 1
        assert sampled.report.dropped_duplicates >= 1
        assert sampled.report.attempts <= 4 * 4
        assert any("distinct" in n for n in sampled.report.notes)
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_scene_range_only_samples_inside_the_range(media):
    # [3.4, 5.2) is entirely P13: every position is the same frame -> exactly one frame.
    cfg = SamplingConfig(min_frames=3, max_frames=3, interval_sec=0.5)
    sampled = _sample(media["vfr"], cfg, start=3.4, end=5.2)
    try:
        pics = [_which_picture(i, [10, 11, 12, 13, 14]) for i in _decode(sampled.paths)]
        assert pics == [3]
        assert (sampled.report.start_time, sampled.report.end_time) == (3.4, 5.2)
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)
    # [0.25, 3.25): P11 (on screen at 0.25) and P12 are visible; P10 and later pictures never.
    sampled = _sample(media["vfr"], cfg, start=0.25, end=3.25)
    try:
        pics = {_which_picture(i, [10, 11, 12, 13, 14]) for i in _decode(sampled.paths)}
        assert pics <= {1, 2} and 1 in pics
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_scene_range_on_cfr_input_picks_segment_centres(media):
    # dup.mp4: [5.5, 6.3) is B, [6.3, 7.1) is C, [7.1, 8.0) is D. Range [5.6, 8.0) with 3
    # frames -> centres 6.0 (B), 6.8 (C), 7.6 (D).
    cfg = SamplingConfig(min_frames=3, max_frames=3, interval_sec=0.8)
    sampled = _sample(media["dup"], cfg, start=5.6, end=8.0)
    try:
        pics = [_which_picture(i, [1, 2, 3, 4]) for i in _decode(sampled.paths)]
        assert pics == [1, 2, 3]
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


@pytest.mark.parametrize(
    ("vision", "expected"),
    [
        (VisionLimits(max_long_side_px=320), (320, 180)),
        (VisionLimits(max_long_side_px=1280), (1280, 720)),  # at the cap: unchanged
        (VisionLimits(max_long_side_px=4000), (1280, 720)),  # never upscaled
        (VisionLimits(), (1280, 720)),
        (VisionLimits(max_megapixels=0.1), None),
    ],
)
def test_resolution_caps_are_honoured_in_the_jpegs(media, vision, expected):
    sampled = _sample(media["hd"], SamplingConfig(), vision)
    try:
        imgs = _decode(sampled.paths)
        sizes = {(i.shape[1], i.shape[0]) for i in imgs}
        assert len(sizes) == 1
        w, h = sizes.pop()
        assert (w, h) == (sampled.report.width, sampled.report.height)
        if expected:
            assert (w, h) == expected
        else:  # megapixel cap
            assert w * h <= 0.1e6 * 1.01 and abs(w / h - 16 / 9) < 0.02
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_small_source_is_not_upscaled(media):
    sampled = _sample(media["small"], SamplingConfig(), VisionLimits(max_long_side_px=720))
    try:
        assert {(i.shape[1], i.shape[0]) for i in _decode(sampled.paths)} == {(160, 90)}
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_tile_budget_is_respected_by_the_real_frames(media):
    vision = VisionLimits(
        max_long_side_px=720,
        tile_width=448,
        tile_height=448,
        max_tiles_per_call=6,
        tokens_per_tile=2,
    )
    cfg = SamplingConfig(balance="spatial")
    sampled = _sample(media["hd"], cfg, vision)
    try:
        r = sampled.report
        imgs = _decode(sampled.paths)
        w, h = imgs[0].shape[1], imgs[0].shape[0]
        tiles = -(-w // 448) * -(-h // 448)
        assert r.frames == len(imgs) == 4
        assert (r.width, r.height) == (w, h)
        assert r.tiles_per_frame == tiles and r.est_tiles == 4 * tiles <= 6
        assert r.est_vision_tokens == 4 * tiles * 2
        assert max(w, h) <= 720
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_temporal_balance_trades_size_for_frames(media):
    vision = VisionLimits(tile_width=448, tile_height=448, max_tiles_per_call=8)
    spatial = _sample(media["hd"], SamplingConfig(balance="spatial"), vision)
    temporal = _sample(media["hd"], SamplingConfig(balance="temporal"), vision)
    try:
        assert temporal.report.frames > spatial.report.frames
        assert temporal.report.width < spatial.report.width
        for s in (spatial, temporal):
            assert s.report.est_tiles <= 8
    finally:
        shutil.rmtree(spatial.tmp_dir, ignore_errors=True)
        shutil.rmtree(temporal.tmp_dir, ignore_errors=True)


def test_auto_balance_runs_the_motion_probe(media):
    vision = VisionLimits(tile_width=448, tile_height=448, max_tiles_per_call=8)
    sampled = _sample(media["hd"], SamplingConfig(balance="auto"), vision)
    try:
        assert 0 < sampled.report.temporal_pref < 1
        assert sampled.report.est_tiles <= 8
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_max_frames_per_call_is_a_hard_limit(media):
    sampled = _sample(media["hd"], SamplingConfig(), VisionLimits(max_frames_per_call=2))
    try:
        assert len(sampled.paths) == 2
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_jpeg_quality_is_applied(media):
    lo = _sample(
        media["hd"], SamplingConfig(min_frames=1, max_frames=1), VisionLimits(jpeg_quality=20)
    )
    hi = _sample(
        media["hd"], SamplingConfig(min_frames=1, max_frames=1), VisionLimits(jpeg_quality=95)
    )
    try:
        assert os.path.getsize(lo.paths[0]) < os.path.getsize(hi.paths[0])
        assert lo.report.jpeg_quality == 20
    finally:
        shutil.rmtree(lo.tmp_dir, ignore_errors=True)
        shutil.rmtree(hi.tmp_dir, ignore_errors=True)


def test_mpegts_without_a_seek_index_still_works(media):
    cfg = SamplingConfig(min_frames=4, max_frames=4, interval_sec=1.5)
    sampled = _sample(media["ts"], cfg)
    try:
        imgs = _decode(sampled.paths)
        assert len(imgs) == 4
        assert len({img.tobytes() for img in imgs}) == 4
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_rotated_video_reports_display_size(media, tmp_path):
    out = str(tmp_path / "rot.mp4")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", media["small"],
         "-c", "copy", out],
        capture_output=True,
    )  # fmt: skip
    if r.returncode != 0:
        pytest.skip("ffmpeg lacks -display_rotation")
    sampled = _sample(out, SamplingConfig(min_frames=1, max_frames=1))
    try:
        img = _decode(sampled.paths)[0]
        assert (img.shape[1], img.shape[0]) == (sampled.report.width, sampled.report.height)
        assert img.shape[0] > img.shape[1]  # portrait after rotation
    finally:
        shutil.rmtree(sampled.tmp_dir, ignore_errors=True)


def test_missing_or_garbage_file_returns_empty(tmp_path):
    assert _sample(tmp_path / "nope.mp4").paths == []
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video" * 100)
    assert _sample(junk).paths == []


# --- § 27: several endpoints, one probe, one decode per frame -----------------------------


def _cleanup(*sampled_list):
    for s in sampled_list:
        shutil.rmtree(s.tmp_dir, ignore_errors=True)


@pytest.mark.parametrize("clip", ["dup", "vfr", "ts", "hd"])
def test_multi_endpoint_sizes_and_frames_match_single_endpoint_runs(media, clip):
    cfg = SamplingConfig(min_frames=4, max_frames=4, interval_sec=1.5)
    visions = [VisionLimits(max_long_side_px=96), VisionLimits(max_long_side_px=192)]
    multi = sample_vlm_frames_multi(media[clip], cfg, visions)
    singles = [sample_vlm_frames(media[clip], cfg, v) for v in visions]
    try:
        for m, s, v in zip(multi, singles, visions, strict=True):
            mi, si = _decode(m.paths), _decode(s.paths)
            assert len(mi) == len(si) == 4
            assert {(i.shape[1], i.shape[0]) for i in mi} == {(i.shape[1], i.shape[0]) for i in si}
            assert max(mi[0].shape[:2]) == v.max_long_side_px
            assert (m.report.width, m.report.height) == (s.report.width, s.report.height)
            # the same instants are sampled: the shared decode is just a cheaper way to get them
            assert [round(float(i.mean()), 0) for i in mi] == pytest.approx(
                [round(float(i.mean()), 0) for i in si], abs=3.0
            )
    finally:
        _cleanup(*multi, *singles)


def test_multi_endpoint_decodes_each_frame_once_with_real_ffmpeg(media, monkeypatch):
    instances = []

    class Spy(FFmpegFrameSource):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            instances.append(self)

    monkeypatch.setattr(vs_module, "FFmpegFrameSource", Spy)
    cfg = SamplingConfig(min_frames=5, max_frames=5, interval_sec=1.0)
    visions = [VisionLimits(max_long_side_px=320), VisionLimits(max_long_side_px=640)] * 2
    multi = sample_vlm_frames_multi(media["hd"], cfg, visions)
    try:
        (source,) = instances
        assert [len(m.paths) for m in multi] == [5, 5, 5, 5]
        assert source.decodes == 5  # four endpoints, five distinct frames, five ffmpeg runs
        sizes = [{(i.shape[1], i.shape[0]) for i in _decode(m.paths)} for m in multi]
        assert sizes == [{(320, 180)}, {(640, 360)}, {(320, 180)}, {(640, 360)}]
    finally:
        _cleanup(*multi)


def test_multi_endpoint_scene_range_and_rotation(media, tmp_path):
    cfg = SamplingConfig(min_frames=3, max_frames=3, interval_sec=1.0)
    visions = [VisionLimits(max_long_side_px=64), VisionLimits(max_long_side_px=128)]
    multi = sample_vlm_frames_multi(media["hd"], cfg, visions, 10.0, 20.0)
    try:
        assert all(10.0 <= m.report.start_time and m.report.end_time <= 20.0 for m in multi)
        assert [len(m.paths) for m in multi] == [3, 3]
    finally:
        _cleanup(*multi)
    out = str(tmp_path / "rot.mp4")
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-display_rotation", "90", "-i", media["small"],
         "-c", "copy", out],
        capture_output=True,
    )  # fmt: skip
    if r.returncode != 0:
        pytest.skip("ffmpeg lacks -display_rotation")
    multi = sample_vlm_frames_multi(out, SamplingConfig(min_frames=1, max_frames=1), visions)
    try:
        for m in multi:
            img = _decode(m.paths)[0]
            assert img.shape[0] > img.shape[1]  # portrait after rotation, at each endpoint's size
            assert (img.shape[1], img.shape[0]) == (m.report.width, m.report.height)
    finally:
        _cleanup(*multi)
