"""Real-ffmpeg tests for REQUIREMENTS.md § 28 (analysis stages through ``Pipeline``).

Scene detection and the VLM are faked (the E2E notes in § 28 cover the real stub-server run);
everything else -- probing, the stage loop, speed, encode, the subtitle muxing / burn-in and
filter-graph path escaping -- is real ffmpeg on a 6 s generated clip.

Run with: pytest tests/integration/test_analysis_stages_e2e.py -v -m integration
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from autovideofixer.config import Config
from autovideofixer.core import analysis as an
from autovideofixer.core.pipeline import Pipeline

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
        reason="ffmpeg/ffprobe not installed",
    ),
]

SCENES = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]
NASTY = "we ird: 'dir' [x],y=z;w"


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("clip") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6", "-c:v", "libx264", "-c:a", "aac",
         "-shortest", str(path)],
        check=True, capture_output=True,
    )  # fmt: skip
    return str(path)


def _scenes(*a, **k):
    return [an.SceneEvent(s, e, "scene_change", 0.3) for s, e in SCENES]


def _scene_vlm(self, filepath, scene, **kw):
    return {
        "summary": f"Scene starting at {scene.start_time:g} seconds",
        "tags": [], "objects": [], "rating": None, "media_type": None,
        "rating_reasoning": None, "extra": {}, "call": None, "sampling": None,
    }  # fmt: skip


def _run(tmp_path, clip, stages, out_name="out.mp4", **cfg):
    config = Config(tmp_path / "none.yaml")
    config.set(True, "analysis", "vlm", "enabled")
    config.set("scenes", "stages", "analyze", "scope")
    config.set(True, "general", "overwrite")
    for key, value in cfg.items():
        config.set(value, *key.split("."))
    pipe = Pipeline(config)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    job = pipe.add_job(clip, str(out / out_name))
    job.stages = list(stages)
    with (
        patch("autovideofixer.core.analysis._detect_scene_changes", _scenes),
        patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", _scene_vlm),
    ):
        result = pipe.execute_job(job)
    return result, out / out_name


def _probe(path, entries="stream=codec_name,codec_type"):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", f"{entries}:format=duration",
         "-of", "json", path],
        check=True, capture_output=True, text=True,
    ).stdout  # fmt: skip
    return json.loads(out)


def _seconds(stamp):
    h, m, s = stamp.replace(",", ".").split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _cues(srt_text):
    return [
        (
            _seconds(m.group(1)),
            _seconds(m.group(2)),
        )
        for m in re.finditer(r"(\d\d:\d\d:\d\d,\d{3}) --> (\d\d:\d\d:\d\d,\d{3})", srt_text)
    ]  # fmt: skip


def _codecs(path, kind):
    return [s["codec_name"] for s in _probe(path)["streams"] if s["codec_type"] == kind]


def test_sidecar_modes_write_files_and_never_delete_the_source(tmp_path, clip):
    before = os.stat(clip).st_mtime_ns
    result, out = _run(tmp_path, clip, ["analyze", "scene_subtitles", "analysis_sidecar", "encode"])
    assert result.success, result.errors
    assert os.path.exists(clip) and os.stat(clip).st_mtime_ns == before
    assert out.exists() and (out.parent / "out.srt").exists()
    side = json.loads((out.parent / "out.mp4.avf-analysis.json").read_text())
    assert side["analysis"]["scenes_detected"] == 3
    assert len(_cues((out.parent / "out.srt").read_text())) == 3
    assert result.input_info["analysis"]["scope"] == "scenes"
    assert "analysis" not in result.output_info
    assert not [p for p in os.listdir(os.path.dirname(clip)) if p.startswith(".avf_")]


@pytest.mark.parametrize("container,codec", [("mp4", "mov_text"), ("mkv", "subrip")])
def test_soft_track_is_muxed_without_a_second_video_generation(tmp_path, clip, container, codec):
    result, out = _run(
        tmp_path, clip, ["analyze", "scene_subtitles", "encode"], f"out.{container}",
        **{"stages.scene_subtitles.mode": "soft"},
    )  # fmt: skip
    assert result.success, result.errors
    assert _codecs(str(out), "subtitle") == [codec]
    assert _codecs(str(out), "video") == ["h264"] and not (out.parent / "out.srt").exists()
    assert abs(float(_probe(str(out))["format"]["duration"]) - 6.0) < 0.2
    sub = result.stage_results["scene_subtitles"]
    assert sub.metadata["method"] == "soft" and sub.metadata["subtitle_codec"] == (
        "mov_text" if container == "mp4" else "srt"
    )
    # the soft stage itself is a stream copy: its temp (an .mkv) kept the source video codec/frames
    count = lambda p: int(  # noqa: E731
        subprocess.run(
            ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
             "stream=nb_read_frames", "-of", "csv=p=0", p], capture_output=True, text=True,
        ).stdout.strip()
    )  # fmt: skip
    assert count(str(out)) == count(clip)


def test_soft_into_webm_degrades_to_a_sidecar(tmp_path, clip, caplog):
    with caplog.at_level("WARNING"):
        result, out = _run(
            tmp_path, clip, ["analyze", "scene_subtitles", "encode"], "out.webm",
            **{"stages.scene_subtitles.mode": "soft"},
        )  # fmt: skip
    assert "falling back to mode=sidecar" in caplog.text
    assert (out.parent / "out.srt").exists()
    assert result.stage_results["scene_subtitles"].metadata["method"] == "sidecar"


def _bottom_diff(a, b):
    def frame(path):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
        ok, img = cap.read()
        cap.release()
        assert ok
        return img.astype(np.int16)

    fa, fb = frame(a), frame(b)
    h = fa.shape[0]
    return float(np.abs(fa[h * 3 // 4 :] - fb[h * 3 // 4 :]).mean()), float(
        np.abs(fa[: h // 2] - fb[: h // 2]).mean()
    )


def test_burn_draws_text_with_a_hostile_temp_dir_path_and_stays_vfr_safe(tmp_path, clip):
    hostile = tmp_path / NASTY
    hostile.mkdir()
    plain, plain_out = (
        _run(tmp_path / "p", clip, ["encode"]) if (tmp_path / "p").mkdir() is None else (None, None)
    )
    assert plain.success, plain.errors
    result, out = _run(
        tmp_path, clip, ["analyze", "scene_subtitles", "encode"],
        **{"stages.scene_subtitles.mode": "burn", "general.temp_dir": str(hostile)},
    )  # fmt: skip
    assert result.success, result.errors
    meta = result.stage_results["scene_subtitles"].metadata
    assert meta["method"] == "burn"
    bottom, top = _bottom_diff(str(out), str(plain_out))
    assert bottom > 0.5, "no text was drawn in the lower part of the frame"
    assert bottom > top * 2
    assert _codecs(str(out), "subtitle") == [] and not (out.parent / "out.srt").exists()
    assert not list(hostile.iterdir())  # the SRT scratch dir was cleaned up


def test_speed_after_analysis_maps_cue_times_onto_the_output(tmp_path, clip):
    result, out = _run(
        tmp_path, clip, ["analyze", "speed", "scene_subtitles", "analysis_sidecar", "encode"],
        **{"stages.speed.factor": 2.0},
    )  # fmt: skip
    assert result.success, result.errors
    duration = float(_probe(str(out))["format"]["duration"])
    assert abs(duration - 3.0) < 0.2
    assert result.input_info["analysis_time_scale"] == pytest.approx(0.5)
    cues = _cues((out.parent / "out.srt").read_text())
    assert [round(c[0], 2) for c in cues] == [0.0, 1.0, 2.0]
    assert cues[-1][1] <= duration + 0.05 and cues[-1][1] == pytest.approx(3.0, abs=0.1)
    side = json.loads((out.parent / "out.mp4.avf-analysis.json").read_text())
    sc = side["analysis"]["scenes"][2]
    assert (sc["start_time"], sc["end_time"]) == (4.0, 6.0)
    assert (sc["output_start_time"], sc["output_end_time"]) == (2.0, 3.0)


def test_speed_before_analysis_needs_no_mapping(tmp_path, clip):
    # analyze after speed: the analysed file IS the sped-up one, so the scale stays 1.0. (An
    # explicit stage list is still ordered by default_order, so use a per-occurrence order.)
    config_order = ["speed", "analyze", "scene_subtitles", "encode"]
    result, out = _run(
        tmp_path, clip, config_order,
        **{"stages.speed.factor": 2.0, "pipeline.default_order": config_order},
    )  # fmt: skip
    assert result.success, result.errors
    assert result.input_info["analysis_time_scale"] == 1.0
    cues = _cues((out.parent / "out.srt").read_text())
    # the faked scenes still say 0-6 s, clamped to the 3 s file
    assert cues[-1][1] <= 3.05


def test_vlm_down_is_fail_open_for_the_whole_chain(tmp_path, clip, caplog):
    def down(self, filepath, scene, **kw):
        return {"summary": "", "tags": [], "objects": [], "rating": None, "media_type": None,
                "rating_reasoning": None, "extra": {}, "call": None, "sampling": None}  # fmt: skip

    config = Config(tmp_path / "none.yaml")
    config.set(True, "analysis", "vlm", "enabled")
    config.set("scenes", "stages", "analyze", "scope")
    pipe = Pipeline(config)
    job = pipe.add_job(clip, str(tmp_path / "o.mp4"))
    job.stages = ["analyze", "scene_subtitles", "analysis_sidecar", "encode"]
    with (
        patch("autovideofixer.core.analysis._detect_scene_changes", _scenes),
        patch.object(an.VideoAnalyzer, "run_vlm_analysis_for_scene", down),
        caplog.at_level("WARNING"),
    ):
        result = pipe.execute_job(job)
    assert result.success and os.path.exists(tmp_path / "o.mp4")
    assert result.stage_results["scene_subtitles"].status.value == "skipped"
    assert not (tmp_path / "o.srt").exists()
