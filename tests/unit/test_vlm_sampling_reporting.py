"""§ 25.4: the sampling report reaches analyze's console, CSV and log, the scene-mode
stats, and the sampler's temp dir is always cleaned up. Provider/ffmpeg are faked here; the
real-media and real-HTTP paths are in tests/integration/test_vlm_sampling_e2e.py."""

from __future__ import annotations

import csv
import os
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from autovideofixer.cli.cli import (
    _ANALYZE_CSV_FIELDS,
    _analysis_to_csv_row,
    _print_analysis_result,
    main,
)
from autovideofixer.config import Config
from autovideofixer.core import analysis as an
from autovideofixer.core.vlm_sampling import SampledFrames, SamplingReport


def _sampled(tmp_path, n=3, **report):
    d = tmp_path / "samples"
    d.mkdir(exist_ok=True)
    paths = []
    for i in range(n):
        p = d / f"frame_{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff")
        paths.append(str(p))
    rep = SamplingReport(
        frames=n, dropped_duplicates=2, width=720, height=405, target_frames=4,
        tiles_per_frame=4, est_tiles=4 * n, est_vision_tokens=8 * n, **report,
    )  # fmt: skip
    return SampledFrames(paths=paths, tmp_dir=str(d), report=rep)


def _cfg(tmp_path):
    cfg = Config(tmp_path / "n.yaml")
    cfg.set("llamacpp", "analysis", "endpoints", "default_vlm", "provider")
    cfg.set("http://127.0.0.1:1/x", "analysis", "endpoints", "default_vlm", "api_url")
    return cfg


class TestAnalyzerPlumbing:
    def test_whole_video_result_carries_sampling_and_cleans_up(self, tmp_path):
        sampled = _sampled(tmp_path)
        with (
            patch("autovideofixer.core.analysis.sample_vlm_frames", return_value=sampled) as smp,
            patch("autovideofixer.core.analysis._dispatch_vlm", return_value={"summary": "s"}),
        ):
            res = an.VideoAnalyzer(_cfg(tmp_path)).run_vlm_analysis("v.mp4", max_sample_frames=6)
        assert res["summary"] == "s"
        assert res["sampling"]["frames"] == 3 and res["sampling"]["dropped_duplicates"] == 2
        assert res["sampling"]["frame_size"] == "720x405"
        assert res["sampling"]["est_vision_tokens"] == 24
        assert not os.path.exists(sampled.tmp_dir)  # temp dir removed after the call
        sampling_cfg = smp.call_args[0][1]
        assert sampling_cfg.max_frames == 6 and sampling_cfg.min_frames == 4

    def test_temp_dir_removed_even_when_the_provider_raises(self, tmp_path):
        sampled = _sampled(tmp_path)
        with (
            patch("autovideofixer.core.analysis.sample_vlm_frames", return_value=sampled),
            patch("autovideofixer.core.analysis._dispatch_vlm", side_effect=RuntimeError("x")),
            pytest.raises(RuntimeError),
        ):
            an.VideoAnalyzer(_cfg(tmp_path)).run_vlm_analysis("v.mp4")
        assert not os.path.exists(sampled.tmp_dir)

    def test_scene_call_uses_the_scene_range_and_reports(self, tmp_path):
        sampled = _sampled(tmp_path, n=2)
        scene = an.SceneEvent(start_time=4.0, end_time=9.0, event_type="x")
        with (
            patch("autovideofixer.core.analysis.sample_vlm_frames", return_value=sampled) as smp,
            patch("autovideofixer.core.analysis._dispatch_vlm", return_value={"summary": "s"}),
        ):
            res = an.VideoAnalyzer(_cfg(tmp_path)).run_vlm_analysis_for_scene("v.mp4", scene)
        assert smp.call_args[0][3:5] == (4.0, 9.0)
        assert res["sampling"]["frames"] == 2
        assert not os.path.exists(sampled.tmp_dir)

    def test_empty_scene_never_samples(self, tmp_path):
        scene = an.SceneEvent(start_time=4.0, end_time=4.0, event_type="x")
        with patch("autovideofixer.core.analysis.sample_vlm_frames") as smp:
            res = an.VideoAnalyzer(_cfg(tmp_path)).run_vlm_analysis_for_scene("v.mp4", scene)
        smp.assert_not_called()
        assert res["summary"] == ""

    def test_endpoint_vision_limits_reach_the_sampler(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.set({"max_long_side_px": 720, "jpeg_quality": 70}, "analysis", "endpoints",
                "default_vlm", "vision")  # fmt: skip
        with (
            patch(
                "autovideofixer.core.analysis.sample_vlm_frames", return_value=_sampled(tmp_path)
            ) as smp,
            patch("autovideofixer.core.analysis._dispatch_vlm", return_value={}),
        ):
            an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
        vision = smp.call_args[0][2]
        assert vision.max_long_side_px == 720 and vision.jpeg_quality == 70

    def test_bad_sampling_config_is_a_clear_error_before_any_work(self, tmp_path):
        cfg = _cfg(tmp_path)
        cfg.set(0, "analysis", "vlm", "sampling", "interval_sec")
        with patch("autovideofixer.core.analysis.sample_vlm_frames") as smp:
            with pytest.raises(ValueError, match="interval_sec"):
                an.VideoAnalyzer(cfg).run_vlm_analysis("v.mp4")
            smp.assert_not_called()

    def test_analyze_stores_sampling_on_the_analysis(self, tmp_path):
        from unittest.mock import MagicMock

        analyzer = an.VideoAnalyzer(_cfg(tmp_path))
        info = MagicMock(filename="v.mp4", duration=30.0, resolution=(640, 360), framerate=30.0)
        with (
            patch("autovideofixer.core.ffmpeg_utils.probe", return_value=info),
            patch.object(
                analyzer,
                "run_vlm_analysis",
                return_value={"summary": "s", "sampling": {"frames": 5, "frame_size": "1x1"}},
            ) as run,
        ):
            a = analyzer.analyze("v.mp4", include_vlm=True, include_events=False,
                                 min_sample_frames=2)  # fmt: skip
        assert a.vlm_sampling == {"frames": 5, "frame_size": "1x1"}
        assert run.call_args[1]["min_sample_frames"] == 2


class TestCliReporting:
    def _analysis(self, sampling):
        return an.VideoAnalysis(
            filepath="/x/v.mp4", filename="v.mp4", duration=30.0, resolution=(1920, 1080),
            framerate=30.0, vlm_summary="a summary", vlm_sampling=sampling,
        )  # fmt: skip

    def test_csv_columns_exist_and_are_filled(self):
        for col in ("vlm_frames", "vlm_dropped_duplicates", "vlm_frame_size",
                    "vlm_est_vision_tokens"):  # fmt: skip
            assert col in _ANALYZE_CSV_FIELDS
        row = _analysis_to_csv_row(
            self._analysis(
                {
                    "frames": 6,
                    "dropped_duplicates": 2,
                    "frame_size": "720x405",
                    "est_vision_tokens": 48,
                }  # fmt: skip
            )
        )
        assert row["vlm_frames"] == 6 and row["vlm_dropped_duplicates"] == 2
        assert row["vlm_frame_size"] == "720x405" and row["vlm_est_vision_tokens"] == 48
        assert set(row) == set(_ANALYZE_CSV_FIELDS)

    def test_csv_blank_without_vlm_and_tokens_blank_without_tiles(self):
        row = _analysis_to_csv_row(self._analysis(None))
        assert row["vlm_frames"] == "" and row["vlm_est_vision_tokens"] == ""
        row = _analysis_to_csv_row(
            self._analysis(
                {
                    "frames": 4,
                    "dropped_duplicates": 0,
                    "frame_size": "1x1",
                    "est_vision_tokens": None,
                }
            )  # fmt: skip
        )
        assert row["vlm_frames"] == 4 and row["vlm_est_vision_tokens"] == ""

    def test_console_shows_frames_duplicates_size_and_tokens(self, capsys):
        from autovideofixer.cli import cli as cli_mod

        cli_mod.console.width = 200
        _print_analysis_result(
            self._analysis(
                {
                    "frames": 6,
                    "dropped_duplicates": 2,
                    "frame_size": "720x405",
                    "est_tiles": 12,
                    "est_vision_tokens": 24,
                }  # fmt: skip
            ),
            full_output=False,
        )
        out = capsys.readouterr().out
        assert "VLM Frames Sent" in out and "6" in out
        assert "VLM Duplicates Dropped" in out
        assert "720x405" in out
        assert "12 / 24" in out

    def test_console_omits_tokens_without_a_tile_size(self, capsys):
        from autovideofixer.cli import cli as cli_mod

        cli_mod.console.width = 200
        _print_analysis_result(
            self._analysis(
                {
                    "frames": 6,
                    "dropped_duplicates": 0,
                    "frame_size": "720x405",
                    "est_tiles": None,
                    "est_vision_tokens": None,
                }
            ),  # fmt: skip
            full_output=False,
        )
        out = capsys.readouterr().out
        assert "VLM Frames Sent" in out and "Tokens" not in out

    def test_cli_end_to_end_with_fakes_writes_csv(self, tmp_path):
        cfgp = tmp_path / "c.yaml"
        cfgp.write_text("config_version: 2\n")
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"x")
        csvp = tmp_path / "out.csv"
        analysis = self._analysis({"frames": 5, "dropped_duplicates": 1, "frame_size": "640x360",
                                   "est_tiles": None, "est_vision_tokens": None})  # fmt: skip

        def fake(self, filepath, **kw):
            fake.kw = kw
            return analysis

        with (
            patch("autovideofixer.cli.cli.is_video_file", return_value=True),
            patch("autovideofixer.core.analysis.VideoAnalyzer.analyze", fake),
        ):
            r = CliRunner().invoke(
                main,
                ["--config", str(cfgp), "analyze", str(vid), "--vlm", "--csv", str(csvp),
                 "--min-sample-frames", "2", "--max-sample-frames", "3"],
            )  # fmt: skip
        assert r.exit_code == 0, r.output
        assert fake.kw["min_sample_frames"] == 2 and fake.kw["max_sample_frames"] == 3
        rows = list(csv.DictReader(csvp.open()))
        assert rows[0]["vlm_frames"] == "5" and rows[0]["vlm_frame_size"] == "640x360"

    def test_bad_sampling_flag_combo_is_a_usage_error(self, tmp_path):
        cfgp = tmp_path / "c.yaml"
        cfgp.write_text("config_version: 2\n")
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"x")
        with patch("autovideofixer.cli.cli.is_video_file", return_value=True):
            r = CliRunner().invoke(
                main,
                ["--config", str(cfgp), "analyze", str(vid), "--vlm", "--min-sample-frames", "9",
                 "--max-sample-frames", "3"],
            )  # fmt: skip
        assert r.exit_code == 2 and "min_frames" in r.output


class TestSceneModeStats:
    def test_collect_scene_vlm_summaries_forwards_sampling(self, tmp_path):
        from autovideofixer.core import scenes as sc

        scene = an.SceneEvent(start_time=0.0, end_time=5.0, event_type="x")
        with patch.object(
            an.VideoAnalyzer,
            "run_vlm_analysis_for_scene",
            return_value={"summary": "s", "tags": [], "sampling": {"frames": 4}},
        ):
            out = sc._collect_scene_vlm_summaries("v.mp4", [scene], Config(tmp_path / "n.yaml"))
        assert out[0]["sampling"] == {"frames": 4}
