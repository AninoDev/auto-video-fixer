"""Tests for core/reporting.py (REQUIREMENTS.md § 6.4/6.5/6.6): per-stage
classification, media-info formatting, stage-timing aggregation, and the
structured JSON run report. All pure functions -- no live pipeline runs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from autovideofixer.config import Config
from autovideofixer.core.pipeline import JobResult
from autovideofixer.core.reporting import (
    aggregate_stage_timing,
    base_stage_name,
    build_json_report,
    build_run_meta,
    classify_stage,
    format_media_info_lines,
    job_summary_line,
    job_to_json,
    run_classification_aggregate,
    run_outcome_aggregate,
    stage_table_rows,
    write_json_report,
)
from autovideofixer.core.stages.base import StageResult, StageStatus


def _config() -> Config:
    return Config(Path(tempfile.mkdtemp()) / "nonexistent.yaml")


class TestClassifyStage:
    def test_failed_status_wins_regardless_of_metadata(self):
        sr = StageResult(status=StageStatus.FAILED, metadata={"method": "ai"})
        assert classify_stage(sr) == "failed"

    def test_skipped_status_wins_regardless_of_metadata(self):
        sr = StageResult(
            status=StageStatus.SKIPPED,
            metadata={"method": "ai", "ai_fallback_used": True},
            skipped_reason="disabled",
        )
        assert classify_stage(sr) == "skipped"

    def test_completed_ai_method(self):
        sr = StageResult(status=StageStatus.COMPLETED, metadata={"method": "ai"})
        assert classify_stage(sr) == "ran-ai"

    def test_completed_traditional_method(self):
        sr = StageResult(status=StageStatus.COMPLETED, metadata={"method": "traditional"})
        assert classify_stage(sr) == "ran-traditional"

    def test_completed_analysis_method_string_is_traditional(self):
        """Analysis-type stages (crop/detect) report their own method string
        (e.g. "cropdetect", "ffprobe") rather than "ai"/"traditional" --
        classify_stage() must still bucket them as ran-traditional."""
        sr = StageResult(status=StageStatus.COMPLETED, metadata={"method": "cropdetect"})
        assert classify_stage(sr) == "ran-traditional"

    def test_completed_no_method_key_is_traditional(self):
        sr = StageResult(status=StageStatus.COMPLETED, metadata={})
        assert classify_stage(sr) == "ran-traditional"

    def test_fallback_marker_precedence_over_method_traditional(self):
        """ai_fallback_used takes precedence -- this is the "AI failed, fell
        back" bucket, distinct from "chose traditional outright"."""
        sr = StageResult(
            status=StageStatus.COMPLETED,
            metadata={"method": "traditional", "ai_fallback_used": True, "ai_fallback_reason": "x"},
        )
        assert classify_stage(sr) == "ran-traditional-fallback"

    def test_fallback_marker_precedence_over_missing_method(self):
        sr = StageResult(
            status=StageStatus.COMPLETED,
            metadata={"ai_fallback_used": True},
        )
        assert classify_stage(sr) == "ran-traditional-fallback"


class TestBaseStageName:
    def test_plain_label(self):
        assert base_stage_name("upscale") == "upscale"

    def test_occurrence_label(self):
        assert base_stage_name("upscale#2") == "upscale"
        assert base_stage_name("deblock#3") == "deblock"


class TestStageTableRows:
    def test_flattens_stage_results(self):
        stage_results = {
            "upscale": StageResult(
                status=StageStatus.COMPLETED,
                metadata={"method": "ai"},
                duration_sec=1.5,
            ),
            "crop": StageResult(
                status=StageStatus.SKIPPED,
                skipped_reason="no border",
            ),
        }
        rows = stage_table_rows(stage_results)
        assert rows[0]["label"] == "upscale"
        assert rows[0]["classification"] == "ran-ai"
        assert rows[0]["duration_sec"] == 1.5
        assert rows[1]["classification"] == "skipped"
        assert rows[1]["skipped_reason"] == "no border"


class TestJobSummaryLine:
    def test_includes_outcome_and_timing(self):
        jr = JobResult(
            input_path="/x/in.mp4",
            outcome="completed",
            job_wall_ms=123.4,
            processing_ms=100.0,
        )
        line = job_summary_line(jr)
        assert "in.mp4" in line
        assert "outcome=completed" in line
        assert "job_wall_ms=123.4" in line
        assert "processing_ms=100.0" in line

    def test_includes_reprocessed_and_scene_stats(self):
        jr = JobResult(
            input_path="/x/in.mp4",
            outcome="completed",
            reprocessed_mismatch=True,
            scene_stats={"total": 5, "kept": 4, "dropped": 1, "dropped_detail": []},
        )
        line = job_summary_line(jr)
        assert "reprocessed" in line
        assert "4/5" in line

    def test_includes_skip_reason(self):
        jr = JobResult(input_path="/x/in.mp4", outcome="skipped", skip_reason="output-exists")
        line = job_summary_line(jr)
        assert "sub-reason=output-exists" in line


class TestRunAggregates:
    def test_classification_aggregate_counts_across_jobs(self):
        jr1 = JobResult(
            input_path="/a.mp4",
            outcome="completed",
            stage_results={
                "upscale": StageResult(status=StageStatus.COMPLETED, metadata={"method": "ai"}),
                "encode": StageResult(
                    status=StageStatus.COMPLETED, metadata={"method": "traditional"}
                ),
            },
        )
        jr2 = JobResult(
            input_path="/b.mp4",
            outcome="failed",
            stage_results={
                "upscale": StageResult(status=StageStatus.FAILED, error="boom"),
            },
        )
        counts = run_classification_aggregate([jr1, jr2])
        assert counts == {"ran-ai": 1, "ran-traditional": 1, "failed": 1}

    def test_outcome_aggregate(self):
        jr1 = JobResult(input_path="/a.mp4", outcome="completed")
        jr2 = JobResult(input_path="/b.mp4", outcome="skipped")
        jr3 = JobResult(input_path="/c.mp4", outcome="failed")
        counts = run_outcome_aggregate([jr1, jr2, jr3])
        assert counts == {"completed": 1, "skipped": 1, "failed": 1}


class TestFormatMediaInfoLines:
    def test_input_only(self):
        info = {
            "resolution": (1920, 1080),
            "framerate": 29.97,
            "duration": 12.5,
            "bit_rate": "5000000",
            "video_codec": "h264",
            "audio_codecs": ["aac"],
        }
        lines = format_media_info_lines(info, None, "/tmp/nonexistent-input.mp4", None)
        joined = "; ".join(lines)
        assert "1920x1080" in joined
        assert "h264" in joined
        assert "aac" in joined
        assert "->" not in joined

    def test_input_vs_output_comparison(self):
        in_info = {"resolution": (1920, 1080), "framerate": 30.0, "video_codec": "h264"}
        out_info = {"resolution": (3840, 2160), "framerate": 60.0, "video_codec": "libx265"}
        lines = format_media_info_lines(in_info, out_info, "/tmp/no-in.mp4", "/tmp/no-out.mp4")
        joined = "; ".join(lines)
        assert "1920x1080 -> 3840x2160" in joined
        assert "h264 -> libx265" in joined


class TestAggregateStageTiming:
    def test_divisor_is_videos_that_ran_the_stage_not_total(self):
        """3 jobs; one fails before "upscale" even runs -- upscale's average
        divisor must be 2 (the two jobs that actually ran it), not 3."""
        jr1 = JobResult(
            input_path="/a.mp4",
            outcome="completed",
            stage_results={
                "detect": StageResult(
                    status=StageStatus.COMPLETED, metadata={"method": "ffprobe"}, duration_sec=1.0
                ),
                "upscale": StageResult(
                    status=StageStatus.COMPLETED, metadata={"method": "ai"}, duration_sec=2.0
                ),
            },
        )
        jr2 = JobResult(
            input_path="/b.mp4",
            outcome="completed",
            stage_results={
                "detect": StageResult(
                    status=StageStatus.COMPLETED, metadata={"method": "ffprobe"}, duration_sec=1.0
                ),
                "upscale": StageResult(
                    status=StageStatus.COMPLETED, metadata={"method": "ai"}, duration_sec=4.0
                ),
            },
        )
        jr3 = JobResult(
            input_path="/c.mp4",
            outcome="failed",
            stage_results={
                "detect": StageResult(
                    status=StageStatus.FAILED, error="probe explosion", duration_sec=0.1
                ),
                # upscale never ran for this job (earlier stage failed) --
                # simply absent from stage_results.
            },
        )
        agg = aggregate_stage_timing([jr1, jr2, jr3])
        assert agg["counts"]["upscale"] == 2
        assert agg["averages"]["upscale"] == 3000.0  # (2000+4000)/2 ms
        assert agg["counts"]["detect"] == 2  # only the 2 successful detects
        assert len(agg["failed"]) == 1
        assert agg["failed"][0]["stage"] == "detect"
        assert agg["failed"][0]["video"] == "/c.mp4"

    def test_failed_executions_excluded_from_totals_and_averages(self):
        jr = JobResult(
            input_path="/a.mp4",
            outcome="completed",
            stage_results={
                "encode": StageResult(status=StageStatus.FAILED, error="oops", duration_sec=2.0),
            },
        )
        agg = aggregate_stage_timing([jr])
        assert "encode" not in agg["totals"]
        assert "encode" not in agg["averages"]
        assert agg["failed"][0]["duration_ms"] == 2000.0

    def test_skipped_executions_contribute_nothing(self):
        jr = JobResult(
            input_path="/a.mp4",
            outcome="completed",
            stage_results={
                "hdr": StageResult(status=StageStatus.SKIPPED, skipped_reason="not HDR"),
            },
        )
        agg = aggregate_stage_timing([jr])
        assert "hdr" not in agg["totals"]
        assert "hdr" not in agg["counts"]
        assert agg["failed"] == []

    def test_occurrence_labels_roll_up_to_base_stage_name(self):
        jr = JobResult(
            input_path="/a.mp4",
            outcome="completed",
            stage_results={
                "deblock": StageResult(
                    status=StageStatus.COMPLETED, metadata={"method": "ai"}, duration_sec=1.0
                ),
                "deblock#2": StageResult(
                    status=StageStatus.COMPLETED, metadata={"method": "ai"}, duration_sec=3.0
                ),
            },
        )
        agg = aggregate_stage_timing([jr])
        assert agg["counts"]["deblock"] == 2
        assert agg["totals"]["deblock"] == 4000.0


class TestBuildRunMeta:
    def test_includes_version_timestamps_and_redacted_settings(self):
        from datetime import datetime, timezone

        config = _config()
        config.set("secret-value", "analysis", "vlm", "api_key")
        config.set("llava-custom", "analysis", "vlm", "model")
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        finished = datetime(2026, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

        meta = build_run_meta("9.9.9", config, started, finished)

        assert meta["avf_version"] == "9.9.9"
        assert meta["started_at"] == started.isoformat()
        assert meta["finished_at"] == finished.isoformat()
        assert meta["elapsed_ms"] == 1000.0
        assert meta["effective_settings"]["analysis"]["vlm"]["api_key"] == "***"
        assert meta["effective_settings"]["analysis"]["vlm"]["model"] == "llava-custom"


class TestJsonReport:
    def test_job_to_json_has_no_aggregates(self):
        jr = JobResult(
            input_path="/a.mp4",
            output_path="/a_out.mp4",
            outcome="completed",
            stage_results={
                "encode": StageResult(
                    status=StageStatus.COMPLETED, metadata={"method": "traditional"}
                ),
            },
        )
        doc = job_to_json(jr)
        assert "totals" not in doc
        assert "averages" not in doc
        assert doc["stages"][0]["label"] == "encode"
        assert doc["stages"][0]["classification"] == "ran-traditional"

    def test_build_json_report_structure_and_no_top_level_aggregates(self):
        jr = JobResult(input_path="/a.mp4", outcome="completed")
        run_meta = {"avf_version": "1.2.3"}
        report = build_json_report(run_meta, [jr])
        assert set(report.keys()) == {"run", "jobs"}
        assert "totals" not in report
        assert "averages" not in report
        assert report["run"] == run_meta
        assert len(report["jobs"]) == 1

    def test_secrets_redacted_in_stage_metadata(self):
        jr = JobResult(
            input_path="/a.mp4",
            outcome="completed",
            stage_results={
                "detect": StageResult(
                    status=StageStatus.COMPLETED,
                    metadata={"method": "ffprobe", "api_key": "sekrit"},
                ),
            },
        )
        doc = job_to_json(jr)
        assert doc["stages"][0]["metadata"]["api_key"] == "***"

    def test_non_serializable_metadata_value_survives_via_str(self, tmp_path):
        class Weird:
            def __str__(self):
                return "weird-repr"

        jr = JobResult(
            input_path="/a.mp4",
            outcome="completed",
            stage_results={
                "detect": StageResult(
                    status=StageStatus.COMPLETED,
                    metadata={"method": "ffprobe", "blob": Weird()},
                ),
            },
        )
        report = build_json_report({"avf_version": "1.0"}, [jr])
        out_path = tmp_path / "report.json"
        write_json_report(str(out_path), report)

        written = json.loads(out_path.read_text())
        assert written["jobs"][0]["stages"][0]["metadata"]["blob"] == "weird-repr"

    def test_write_json_report_creates_parent_dirs(self, tmp_path):
        out_path = tmp_path / "nested" / "dir" / "report.json"
        write_json_report(str(out_path), {"run": {}, "jobs": []})
        assert out_path.exists()
        assert json.loads(out_path.read_text()) == {"run": {}, "jobs": []}

    def test_iso_timestamps_in_run_meta_round_trip(self):
        from datetime import datetime, timezone

        config = _config()
        started = datetime(2026, 5, 4, 3, 2, 1, tzinfo=timezone.utc)
        finished = datetime(2026, 5, 4, 3, 2, 2, tzinfo=timezone.utc)
        meta = build_run_meta("1.0.0", config, started, finished)
        # Round-trips through fromisoformat without raising.
        assert datetime.fromisoformat(meta["started_at"]) == started
        assert datetime.fromisoformat(meta["finished_at"]) == finished
